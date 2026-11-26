//! How the app cuts a recording into windows for the speech model (`src/media/speech.rs`):
//! windows of at most 28 s (or the model's limit), each ending at the quietest 40 ms of its last
//! three seconds, except the last. Both engines get exactly these windows.

use crate::audio::RATE;
use std::ops::Range;

const RATE_US: usize = RATE as usize;

/// The quietest 40 ms near the end of a full window, as a cut position.
pub fn boundary(pcm: &[f32]) -> usize {
    let frame = RATE_US / 25;
    if pcm.len() < 4 * RATE_US {
        return pcm.len();
    }
    let start = pcm.len() - 3 * RATE_US;
    let (quietest, _) = pcm[start..]
        .chunks_exact(frame)
        .enumerate()
        .map(|(i, values)| (i, values.iter().map(|v| v * v).sum::<f32>()))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();
    start + quietest * frame + frame / 2
}

/// The window length in samples: `max_ms`, or the model's limit when that's shorter.
pub fn window_samples(max_ms: u32, model_max_ms: u32) -> usize {
    let ms = if model_max_ms > 0 {
        max_ms.min(model_max_ms)
    } else {
        max_ms
    };
    ms as usize * RATE_US / 1000
}

/// The windows of a recording of `pcm`, in order, covering it exactly.
pub fn windows(pcm: &[f32], window: usize) -> Vec<Range<usize>> {
    let window = window.max(1);
    let mut out = Vec::new();
    let mut offset = 0;
    while offset < pcm.len() {
        let end = (offset + window).min(pcm.len());
        let count = if end < pcm.len() {
            boundary(&pcm[offset..end])
        } else {
            end - offset
        };
        out.push(offset..offset + count);
        offset += count;
    }
    out
}

/// Whether the app skips a window: every sample exactly zero.
pub fn silent(pcm: &[f32]) -> bool {
    pcm.iter().all(|s| *s == 0.)
}
