use super::{LayerCache, Rank};
use anyhow::{Result, ensure};

/// Contiguous logical rows actually written to one cache. A branch write invalidates
/// the old suffix even when its device pages are still mapped.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct WrittenRows {
    first: usize,
    end: usize,
}

impl WrittenRows {
    fn covers(self, first: usize, end: usize) -> bool {
        first == end || (self.first <= first && first < end && end <= self.end)
    }

    fn appended(self, start: usize, end: usize, ring_rows: usize) -> Self {
        let first = if self.first <= start && start <= self.end {
            self.first
        } else {
            start
        };
        Self {
            first: if ring_rows == 0 {
                first
            } else {
                first.max(end.saturating_sub(ring_rows))
            },
            end,
        }
    }

    fn truncated(self, end: usize) -> Self {
        let end = end.min(self.end);
        if end <= self.first {
            Self::default()
        } else {
            Self { end, ..self }
        }
    }
}

impl LayerCache {
    pub(super) fn truncate(&self, end: usize) {
        self.window_live.set(self.window_live.get().truncated(end));
        self.history_live
            .set(self.history_live.get().truncated(end));
        if self.comp.is_some() {
            self.compact_live
                .set(self.compact_live.get().min(end / self.compress_ratio));
        }
    }

    pub(super) fn check_window(&self, first: usize, end: usize) -> Result<()> {
        ensure!(
            self.window_live.get().covers(first, end),
            "attention requires unwritten or expired window rows [{first}, {end})"
        );
        Ok(())
    }

    pub(super) fn check_history(&self, start: usize, end: usize) -> Result<()> {
        let ratio = self.compress_ratio;
        // Only the first completed group in a chunk can need earlier raw rows.
        if self.comp.is_some() {
            ensure!(
                start / ratio <= self.compact_live.get(),
                "compressed KV prefix has unwritten groups"
            );
        }
        if self.kv_hist.is_some() && start / ratio != end / ratio {
            ensure!(
                self.history_live.get().covers(start - start % ratio, start),
                "compressor requires unwritten or expired group rows"
            );
        }
        Ok(())
    }

    pub(super) fn record_window(&self, start: usize, end: usize) {
        self.window_live
            .set(self.window_live.get().appended(start, end, self.ring_rows));
    }

    pub(super) fn record_history(&self, start: usize, end: usize) {
        if self.comp.is_some() {
            self.compact_live.set(end / self.compress_ratio);
        }
        if self.kv_hist.is_some() {
            self.history_live
                .set(self.history_live.get().appended(start, end, self.hist_rows));
        }
    }
}

pub(super) fn snapshot_floor(caches: &[LayerCache]) -> usize {
    caches
        .iter()
        .map(|c| {
            let window = if c.on_wire {
                c.window_live.get().first
            } else {
                0
            };
            let history = if c.kv_hist.is_some() {
                c.history_live.get().first
            } else {
                0
            };
            window.max(history)
        })
        .max()
        .unwrap_or(0)
}

impl Rank {
    pub(crate) fn raw_tail_floor(&self) -> usize {
        if self.packed_rows == 0 {
            self.tail.raw_floor.get()
        } else {
            snapshot_floor(&self.caches)
        }
    }

    pub(crate) fn restore_window_coverage(&self, prefix: usize, floor: usize) {
        self.tail.raw_floor.set(floor);
        let restored = WrittenRows {
            first: floor,
            end: prefix,
        };
        for c in self.caches.iter().chain(&self.mtp_caches) {
            c.window_live.set(if c.on_wire {
                restored
            } else {
                WrittenRows::default()
            });
            c.history_live.set(if c.kv_hist.is_some() {
                restored
            } else {
                WrittenRows::default()
            });
            if c.comp.is_some() {
                c.compact_live.set(prefix / c.compress_ratio);
            }
        }
    }

    pub(crate) fn invalidate_restoring_slot(&self) {
        for cache in self.caches.iter().chain(&self.mtp_caches) {
            cache.truncate(0);
        }
        self.tail.raw_floor.set(usize::MAX);
        self.tail.state.set(Default::default());
    }

    fn check_cache_coverage(&self, end: usize, first: usize, include_decoder: bool) -> Result<()> {
        self.check_kv_mapping(end)?;
        ensure!(
            first <= end && first >= self.raw_tail_floor(),
            "invalid retained KV interval"
        );
        // The incumbent dense decoder advances its position inside a CUDA graph.
        if self.packed_rows == 0 {
            return Ok(());
        }
        for c in self.caches.iter().chain(&self.mtp_caches) {
            if include_decoder || c.on_wire {
                ensure!(
                    end <= c.window_live.get().end,
                    "requested KV prefix exceeds written window rows"
                );
                c.check_window(first, end)?;
            }
            if c.comp.is_some() {
                ensure!(
                    end / c.compress_ratio <= c.compact_live.get(),
                    "requested KV prefix exceeds written compressed groups"
                );
            }
            if c.kv_hist.is_some() {
                ensure!(
                    end <= c.history_live.get().end,
                    "requested KV prefix exceeds written history"
                );
                ensure!(
                    c.history_live.get().covers(first, end),
                    "requested compressor history is unavailable"
                );
            }
        }
        Ok(())
    }

    pub(crate) fn check_snapshot_coverage(&self, end: usize, first: usize) -> Result<()> {
        self.check_cache_coverage(end, first, false)
    }

    pub(crate) fn check_all_window_coverage(&self, end: usize, first: usize) -> Result<()> {
        self.check_cache_coverage(end, first, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_windows_track_replay_branch_and_debug_bounds_independently() {
        let restored = WrittenRows {
            first: 2000,
            end: 9000,
        };
        let encoder = restored.truncated(8400).appended(8400, 9100, 8208);
        let decoder = WrittenRows::default().appended(8972, 9100, 144);
        assert!(encoder.covers(3000, 9100));
        assert!(!decoder.covers(3000, 3010)); // fits the ring's byte count but refers to expired positions
        assert!(decoder.covers(8972, 9100));
        let verify = decoder.appended(9100, 9109, 144);
        let branch = verify.truncated(9101).appended(9101, 9102, 144);
        assert!(branch.covers(8974, 9102));
        assert!(!branch.covers(9102, 9108));
        assert!(!branch.covers(9102, 9110)); // mapping ahead does not extend valid rows
    }
}
