use deepseek_ultrafast_kv_transfer::{ImageSpan, Prompt, SAVED_TAIL, TAIL_READ, image_prefix};
use std::sync::Arc;

/// An idle rank-local slot may be reused only for matching prompt tokens and images
/// and only while its retained history is sufficient for encoder replay.
pub(super) struct HotPrefix {
    pub session: String,
    pub tokens: Arc<Vec<i32>>,
    pub images: Vec<ImageSpan>,
    /// Includes any provisional decode rows that might have overwritten a ring.
    pub mapped_tokens: usize,
    pub raw_tail_floor: usize,
    pub bytes: usize,
    pub retired_at: u64,
}

impl HotPrefix {
    pub(super) fn reusable(&self, session: &str, prompt: Prompt<'_>, through: usize) -> usize {
        if self.session != session || self.mapped_tokens > through {
            return 0;
        }
        let common = self
            .tokens
            .iter()
            .zip(prompt.tokens)
            .take_while(|(a, b)| a == b)
            .count()
            .min(prompt.tokens.len().saturating_sub(1));
        let prefix = image_prefix(common, &self.images, prompt.images);
        let tail_start = self
            .raw_tail_floor
            .max(self.mapped_tokens.saturating_sub(SAVED_TAIL));
        if tail_start != 0 && prefix < tail_start + TAIL_READ {
            return 0;
        }
        prefix
    }
}

pub(super) struct HotPrefixes {
    slots: Vec<Option<HotPrefix>>,
    bytes: usize,
}

impl HotPrefixes {
    pub(super) fn new(slots: usize) -> Self {
        Self {
            slots: (0..slots).map(|_| None).collect(),
            bytes: 0,
        }
    }

    pub(super) fn get(&self, slot: usize) -> Option<&HotPrefix> {
        self.slots[slot].as_ref()
    }

    pub(super) fn remove(&mut self, slot: usize) -> Option<HotPrefix> {
        let entry = self.slots[slot].take()?;
        self.bytes -= entry.bytes;
        Some(entry)
    }

    pub(super) fn insert(&mut self, slot: usize, entry: HotPrefix) {
        debug_assert!(self.slots[slot].is_none(), "only an active slot can retire");
        self.bytes += entry.bytes;
        self.slots[slot] = Some(entry);
    }

    pub(super) fn discard_session(&mut self, session: &str, evictions: &mut Vec<usize>) {
        for slot in 0..self.slots.len() {
            if self.get(slot).is_some_and(|entry| entry.session == session) {
                self.remove(slot);
                evictions.push(slot);
            }
        }
    }

    pub(super) fn evict_to(&mut self, budget: usize, evictions: &mut Vec<usize>) {
        while self.bytes > budget {
            let slot = self
                .slots
                .iter()
                .enumerate()
                .filter_map(|(slot, entry)| entry.as_ref().map(|entry| (entry.retired_at, slot)))
                .min()
                .expect("nonempty cache has a slot")
                .1;
            self.remove(slot);
            evictions.push(slot);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(session: &str, mapped_tokens: usize, retired_at: u64) -> HotPrefix {
        HotPrefix {
            session: session.into(),
            tokens: Arc::new(vec![1; mapped_tokens]),
            images: vec![],
            mapped_tokens,
            raw_tail_floor: 0,
            bytes: 128,
            retired_at,
        }
    }

    #[test]
    fn retains_the_replay_tail_and_requires_room_for_existing_pages() {
        let mapped = SAVED_TAIL + 64;
        let mut prefix = entry("coding", mapped, 0);
        let mut turns = vec![1; mapped + 1];
        assert_eq!(
            prefix.reusable("coding", Prompt::text(&turns), mapped + 1),
            mapped
        );
        assert_eq!(
            prefix.reusable("other", Prompt::text(&turns), mapped + 1),
            0
        );
        assert_eq!(
            prefix.reusable("coding", Prompt::text(&turns), mapped - 1),
            0
        );
        turns[64 + TAIL_READ - 1] = 2;
        assert_eq!(
            prefix.reusable("coding", Prompt::text(&turns), mapped + 1),
            0
        );
        turns[64 + TAIL_READ - 1] = 1;
        prefix.raw_tail_floor = 512;
        turns[700] = 2;
        assert_eq!(
            prefix.reusable("coding", Prompt::text(&turns), mapped + 1),
            0
        );
        turns[700] = 1;
        prefix.images.push(ImageSpan {
            start: 12,
            end: 16,
            height: 2,
            width: 2,
            digest: [0; 32],
        });
        assert_eq!(
            prefix.reusable("coding", Prompt::text(&turns), mapped + 1),
            0
        );
    }

    #[test]
    fn evicts_old_prefixes_to_make_room_for_active_reservations() {
        let mut cache = HotPrefixes::new(3);
        cache.insert(0, entry("a", 128, 3));
        cache.insert(2, entry("b", 128, 1));
        let mut evictions = Vec::new();
        cache.evict_to(128, &mut evictions);
        assert_eq!(evictions, vec![2]);
        cache.discard_session("a", &mut evictions);
        assert_eq!(evictions, vec![2, 0]);
        assert_eq!(cache.bytes, 0);
    }
}
