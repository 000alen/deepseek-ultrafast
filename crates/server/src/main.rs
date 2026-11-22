use deepseek_ultrafast_server::{api, chat};
use deepseek_ultrafast_server::generation::{DEFAULT_MAX_JOB_BYTES, DEFAULT_MAX_JOBS, JobIngress};
mod engine;
mod host_budget;
mod peer;
mod publisher;

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::Arc;

const USAGE: &str = "serve [--ckpt DIR] [--gpus 0,1,..] [--max-seq N] [--chunk N] [--host H] [--port P] [--model-name NAME] [--max-num-seqs N] [--max-num-decode-seqs N] [--max-num-active-seqs N] [--max-num-batched-tokens N] [--kv-cache-memory-bytes N] [--hot-kv-cache-memory-bytes N] [--kv-copy-mib N] [--packed-drafts N] [--busy-prefill-ms N] [--prefill-schedule-interval N] [--max-prefill-wait-ms N] [--decode-max-gap-ms N] [--decode-min-tokens-per-sec N] [--standard-pool-max-output N] [--max-outstanding-jobs N] [--max-outstanding-bytes N] [--shutdown-grace-seconds N]";

fn main() -> std::process::ExitCode {
    std::panic::set_hook(Box::new(|info| deepseek_ultrafast_logging::termination(&info.to_string())));
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            deepseek_ultrafast_logging::termination(&format!("{error:#}"));
            std::process::ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let mut ea = engine::EngineArgs {
        ckpt: PathBuf::from("/models"),
        gpus: (0..8).collect(),
        max_seq: 1048576,
        chunk: 8192,
        batch: engine::BatchConfig::default(),
    };
    let (mut host, mut port, mut model) = ("0.0.0.0".to_string(), 8000u16, "deepseek-v4.1-flash".to_string());
    let (mut max_jobs, mut max_job_bytes) = (DEFAULT_MAX_JOBS, DEFAULT_MAX_JOB_BYTES);
    let mut standard_pool_max_output = None;
    let mut shutdown_grace = 45u64;
    let v: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < v.len() {
        let val = || v.get(i + 1).cloned().with_context(|| format!("{} needs a value\nusage: {USAGE}", v[i]));
        match v[i].as_str() {
            "--ckpt" => ea.ckpt = val()?.into(),
            "--gpus" => ea.gpus = val()?.split(',').map(|x| x.parse()).collect::<Result<_, _>>()?,
            "--max-seq" => ea.max_seq = val()?.parse()?,
            "--chunk" => ea.chunk = val()?.parse()?,
            "--max-num-seqs" => ea.batch.max_num_seqs = val()?.parse()?,
            "--max-num-decode-seqs" => ea.batch.max_num_decode_seqs = Some(val()?.parse()?),
            "--max-num-active-seqs" => ea.batch.max_num_active_seqs = Some(val()?.parse()?),
            "--max-num-batched-tokens" => ea.batch.max_target_rows = val()?.parse()?,
            "--kv-cache-memory-bytes" => ea.batch.kv_bytes = Some(val()?.parse()?),
            "--hot-kv-cache-memory-bytes" => ea.batch.hot_kv_bytes = val()?.parse()?,
            "--kv-copy-mib" => ea.batch.kv_copy_mib = val()?.parse()?,
            "--packed-drafts" => ea.batch.drafts = val()?.parse()?,
            "--busy-prefill-ms" => ea.batch.busy_prefill_ms = val()?.parse()?,
            "--prefill-schedule-interval" => ea.batch.busy_prefill_interval = val()?.parse()?,
            "--max-prefill-wait-ms" => ea.batch.max_prefill_wait_ms = val()?.parse()?,
            "--decode-max-gap-ms" => ea.batch.decode_max_gap_ms = Some(val()?.parse()?),
            "--decode-min-tokens-per-sec" => ea.batch.decode_min_tokens_per_sec = Some(val()?.parse()?),
            "--standard-pool-max-output" => standard_pool_max_output = Some(val()?.parse()?),
            "--max-outstanding-jobs" => max_jobs = val()?.parse()?,
            "--max-outstanding-bytes" => max_job_bytes = val()?.parse()?,
            "--shutdown-grace-seconds" => shutdown_grace = val()?.parse()?,
            "--host" => host = val()?,
            "--port" => port = val()?.parse()?,
            "--model-name" => model = val()?,
            "-h" | "--help" => {
                println!("usage: {USAGE}");
                return Ok(());
            }
            x => anyhow::bail!("unknown arg {x}\nusage: {USAGE}"),
        }
        i += 2;
    }
    anyhow::ensure!(!ea.gpus.is_empty() && ea.gpus.iter().enumerate().all(|(i, gpu)| !ea.gpus[..i].contains(gpu)), "--gpus must name distinct GPUs");
    ea.batch.validate(ea.max_seq, ea.chunk)?;
    anyhow::ensure!(max_jobs > 0 && max_job_bytes > 0, "outstanding request limits must be positive");
    if let Some(output) = standard_pool_max_output {
        anyhow::ensure!(output > 0 && output <= deepseek_ultrafast_server::generation::max_new_limit(1, ea.max_seq),
            "standard pool output limit must fit the model context");
    }
    deepseek_ultrafast_model::st::require_checkpoint(&ea.ckpt)?;
    deepseek_ultrafast_cuda::init()?;
    deepseek_ultrafast_cuda::profile::init(deepseek_ultrafast_cuda::device_cc(ea.gpus[0] as i32)?, ea.gpus.len());
    unsafe {
        std::env::set_var("CUDNN_LOGLEVEL_DBG", "0");
        std::env::set_var("OPENCV_LOG_LEVEL", "SILENT");
    }
    let _logging = deepseek_ultrafast_logging::init()?;
    let rt = tokio::runtime::Runtime::new()?;
    let peer_options = peer::Options::new(&ea)?;
    let tok = Arc::new(tokenizers::Tokenizer::from_file(ea.ckpt.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?);
    deepseek_ultrafast_server::token_cache::register(tok.clone());
    let peer_options = peer_options.with_token_text(Arc::new(deepseek_ultrafast_server::token_text::token_text(&tok)));
    let eos = tok.token_to_id(chat::EOS).context("tokenizer has no end-of-sentence token")? as i32;
    let max_seq = ea.max_seq;

    let (jobs, rx) = JobIngress::channel(max_jobs, max_job_bytes);
    let jobs = if let Some(max_output) = standard_pool_max_output { jobs.with_standard_pool(max_output) } else { jobs };
    let load = jobs.load_reporter();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let runtime = rt.handle().clone();
    let eng = std::thread::spawn(move || engine::run(ea, rx, ready_tx, peer_options, runtime, load));
    let peer = match ready_rx.recv() {
        Ok(peer) => peer,
        Err(_) => return eng.join().map_err(|_| anyhow::anyhow!("engine thread panicked"))?,
    };

    let app = Arc::new(api::App { tok, jobs: jobs.clone(), model, max_seq, eos });
    rt.block_on(async move {
        let ready = deepseek_ultrafast_server::readiness::Readiness::default();
        let result: Result<()> = async {
            let addr = format!("{host}:{port}");
            let listener = tokio::net::TcpListener::bind(&addr).await.with_context(|| format!("bind {addr}"))?;
            ready.set_ready();
            let shutdown_ready = ready.clone();
            let shutdown_jobs = jobs.clone();
            let result = axum::serve(listener, api::router_with_readiness(app, ready.clone())).with_graceful_shutdown(async move {
                let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("SIGTERM handler");
                tokio::select! { _ = tokio::signal::ctrl_c() => {}, _ = term.recv() => {} }
                shutdown_ready.stop();
                shutdown_jobs.close();
                tokio::spawn(async move {
                    tokio::time::sleep(std::time::Duration::from_secs(shutdown_grace)).await;
                    shutdown_jobs.cancel_outstanding();
                });
            }).await;
            result?;
            Ok(())
        }.await;
        ready.stop();
        jobs.close();
        let engine_result = tokio::task::spawn_blocking(move || {
            eng.join().map_err(|_| anyhow::anyhow!("engine thread panicked"))?
        }).await;
        peer.shutdown().await;
        result?;
        engine_result??;
        Ok(())
    })
}
