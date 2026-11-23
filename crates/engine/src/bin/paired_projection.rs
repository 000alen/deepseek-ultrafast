//! Compare two four-row FFN router projections to one eight-row projection.
//! The router is replicated across TP ranks. This isolated, single-GPU diagnostic
//! uses every target layer's checkpoint weight and the engine's dense BF16 GEMV.
//! It measures the projection alone, excluding routing, experts, KV and collectives.
use anyhow::{Result, ensure};
use deepseek_ultrafast_cuda::{self as cu, Buffer, Event, GraphExec, Stream};
use deepseek_ultrafast_engine::ops::{self, GemvOpts, WFmt, Weight};
use deepseek_ultrafast_kernels::Kernels;
use deepseek_ultrafast_model::{formats, st::Checkpoint};
use half::bf16;
use std::path::Path;

#[path = "paired_projection/parity.rs"]
mod parity;
use parity::{LANE_ROWS, ROWS};
const USAGE: &str = "paired_projection CHECKPOINT [GPU_ORDINAL] [REPLAYS_PER_SAMPLE]";

struct Gate {
    tiles: Buffer,
    cols: usize,
    experts: usize,
}

impl Gate {
    fn project(
        &self,
        kernels: &Kernels,
        stream: &Stream,
        x: u64,
        y: u64,
        rows: usize,
    ) -> Result<()> {
        let weight = Weight {
            fmt: WFmt::Bf16,
            w: self.tiles.ptr,
            s: 0,
            n: self.experts,
            k: self.cols,
        };
        ops::gemv(
            kernels,
            stream,
            &weight,
            x,
            self.cols,
            rows,
            y,
            self.experts,
            &GemvOpts {
                out_f32: true,
                ..Default::default()
            },
        )
    }
}

fn capture(
    kernels: &Kernels,
    stream: &Stream,
    gates: &[Gate],
    x: &Buffer,
    out: &Buffer,
    rows: usize,
) -> Result<GraphExec> {
    stream.begin_capture()?;
    let enqueue = (|| -> Result<()> {
        for (layer, gate) in gates.iter().enumerate() {
            for row in (0..ROWS).step_by(rows) {
                let src = x.ptr + (row * gate.cols * size_of::<u16>()) as u64;
                let dst = out.ptr + ((layer * ROWS + row) * gate.experts * size_of::<f32>()) as u64;
                gate.project(kernels, stream, src, dst, rows)?;
            }
        }
        Ok(())
    })();
    let graph = stream.end_capture();
    enqueue?;
    let graph = graph?.instantiate()?;
    graph.upload(stream)?;
    stream.synchronize()?;
    Ok(graph)
}

fn time_us(graph: &GraphExec, stream: &Stream, replays: usize) -> Result<f32> {
    let start = Event::timed()?;
    let end = Event::timed()?;
    start.record(stream)?;
    for _ in 0..replays {
        graph.launch(stream)?;
    }
    end.record(stream)?;
    end.synchronize()?;
    let us = start.elapsed_ms(&end)? * 1000. / replays as f32;
    ensure!(us.is_finite() && us > 0., "invalid GPU elapsed time");
    Ok(us)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!((1..=3).contains(&args.len()), "{USAGE}");
    let ckpt = Path::new(&args[0]);
    let ordinal = args.get(1).map(|x| x.parse()).transpose()?.unwrap_or(0);
    let replays = args.get(2).map(|x| x.parse()).transpose()?.unwrap_or(64);
    ensure!((1..=10000).contains(&replays), "replays must be 1..=10000");
    cu::init()?;
    let _device = cu::Device::new(ordinal)?;
    let cc = cu::device_cc(ordinal)?;
    cu::profile::init(cc, 1);
    let cfg = deepseek_ultrafast_engine::load_config(ckpt)?;
    let checkpoint = Checkpoint::open(ckpt)?;
    let (cols, experts) = (cfg.dim, cfg.n_routed_experts);
    ensure!(
        cols >= ROWS && cols % 32 == 0 && experts > 0 && experts % 64 == 0 && cfg.n_layers > 0,
        "unsupported gate shape"
    );
    let kernels = Kernels::load()?;
    let stream = Stream::new()?;

    // Distinct one-hot columns let us check *every* expert in each lane against
    // the original checkpoint, independently of the tiler and GPU implementation.
    let column = |row: usize| row * (cols - 1) / (ROWS - 1);
    let mut probes = vec![0u16; ROWS * cols];
    for row in 0..ROWS {
        probes[row * cols + column(row)] = bf16::from_f32(1.).to_bits();
    }
    let mut gates = Vec::new();
    let mut expected = Vec::new();
    for layer in 0..cfg.n_layers {
        let name = format!("layers.{layer}.ffn.gate.weight");
        let bytes = checkpoint.get(&name, "BF16", &[experts, cols])?;
        ensure!(
            bytes.len() == experts * cols * 2,
            "wrong byte count in {name}"
        );
        let values: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|x| u16::from_le_bytes([x[0], x[1]]))
            .collect();
        for row in 0..ROWS {
            expected.extend(
                (0..experts).map(|e| bf16::from_bits(values[e * cols + column(row)]).to_f32()),
            );
        }
        gates.push(Gate {
            tiles: Buffer::from_host(&formats::tile_bf16(&values, experts, cols))?,
            cols,
            experts,
        });
    }
    let input = Buffer::from_host(&probes)?;
    let poison = vec![f32::NAN; expected.len()];
    let serial_out = Buffer::from_host(&poison)?;
    let packed_out = Buffer::from_host(&poison)?;
    let serial = capture(&kernels, &stream, &gates, &input, &serial_out, LANE_ROWS)?;
    let packed = capture(&kernels, &stream, &gates, &input, &packed_out, ROWS)?;
    for (arm, graph, output) in [
        ("serial", &serial, &serial_out),
        ("packed", &packed, &packed_out),
    ] {
        graph.launch(&stream)?;
        stream.synchronize()?;
        let actual = output.download::<f32>(expected.len())?;
        parity::compare(arm, &expected, &actual, experts, 0.).map_err(anyhow::Error::msg)?;
    }

    // Dense, reproducible BF16 activations for timing and cross-shape parity.
    let mut state = 0x84e8_36c1u32;
    for x in &mut probes {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        *x = bf16::from_f32((state as i32) as f32 / 2147483648.).to_bits();
    }
    input.upload(&probes)?;
    serial_out.upload(&poison)?;
    packed_out.upload(&poison)?;
    serial.launch(&stream)?;
    packed.launch(&stream)?;
    stream.synchronize()?;
    let serial_values = serial_out.download::<f32>(expected.len())?;
    let packed_values = packed_out.download::<f32>(expected.len())?;
    let max_abs_error = parity::compare("dense", &serial_values, &packed_values, experts, 1e-4)
        .map_err(anyhow::Error::msg)?;
    let mut samples = Vec::new();
    for round in 0..4 {
        let (serial_us, packed_us) = if round % 2 == 0 {
            (
                time_us(&serial, &stream, replays)?,
                time_us(&packed, &stream, replays)?,
            )
        } else {
            let packed_us = time_us(&packed, &stream, replays)?;
            (time_us(&serial, &stream, replays)?, packed_us)
        };
        samples.push(serde_json::json!({"serial_us":serial_us, "packed_us":packed_us, "ratio":serial_us / packed_us}));
    }
    println!(
        "{}",
        serde_json::json!({"diagnostic":"ffn_gate_projection", "gpu":ordinal, "cc":cc, "layers":gates.len(), "columns":cols, "experts":experts, "replays_per_sample":replays, "lane_max_abs_error":max_abs_error, "samples":samples})
    );
    Ok(())
}
