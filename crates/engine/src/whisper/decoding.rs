//! The decisions around Whisper's decoder, as transcribe.cpp makes them
//! (`arch/whisper/model.cpp` there, itself a port of transformers' `generate`): which logits
//! are masked (suppressed tokens, the timestamp rules), how a token is chosen (greedy at
//! temperature 0, sampled above), when a temperature's result is rejected (compression ratio,
//! average log-probability), and how timestamps cut the result into segments.

/// Ids below this are text; Whisper's specials start here (`<|endoftext|>`).
pub const TEXT_END: i32 = 50257;

/// The vocabulary's layout around timestamps.
#[derive(Clone, Copy)]
pub struct Specials {
    pub eos: i32,
    pub no_timestamps: i32,
    /// The first timestamp token (`<|0.00|>`), one past `<|notimestamps|>`.
    pub timestamp_begin: i32,
    pub vocab: i32,
}

impl Specials {
    pub fn is_timestamp(&self, id: i32) -> bool {
        id >= self.timestamp_begin && id < self.vocab
    }
}

/// transformers' `WhisperTimeStampLogitsProcessor` as the reference applies it to one row of
/// logits, given the tokens generated so far in this window: no `<|notimestamps|>`;
/// timestamps in pairs (after text then a timestamp, only a timestamp or the end; after two
/// timestamps, only text); never decreasing; the first token a timestamp of at most
/// `max_initial` steps (1 s); and a timestamp forced whenever their total probability beats
/// the best text token's.
pub fn timestamp_rules(logits: &mut [f32], generated: &[i32], s: &Specials, max_initial: usize) {
    let (begin, vocab) = (s.timestamp_begin as usize, s.vocab as usize);
    logits[s.no_timestamps as usize] = f32::NEG_INFINITY;

    let last_ts = generated.last().is_some_and(|&t| s.is_timestamp(t));
    let penult_ts = generated.len() < 2 || s.is_timestamp(generated[generated.len() - 2]);
    if last_ts {
        if penult_ts {
            logits[begin..vocab].fill(f32::NEG_INFINITY);
        } else {
            logits[..(s.eos as usize).min(vocab)].fill(f32::NEG_INFINITY);
        }
    }

    if let Some(&last) = generated.iter().rev().find(|&&t| s.is_timestamp(t)) {
        let low = if last_ts && !penult_ts {
            last
        } else {
            last + 1
        };
        let high = (low as usize).min(vocab);
        if high > begin {
            logits[begin..high].fill(f32::NEG_INFINITY);
        }
    }

    if generated.is_empty() {
        logits[..begin].fill(f32::NEG_INFINITY);
        let last_allowed = begin + max_initial;
        if last_allowed + 1 < vocab {
            logits[last_allowed + 1..vocab].fill(f32::NEG_INFINITY);
        }
    }

    let max_ts = logits[begin..vocab]
        .iter()
        .fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let mut ts_logsumexp = f32::NEG_INFINITY;
    if max_ts.is_finite() {
        let sum: f64 = logits[begin..vocab]
            .iter()
            .filter(|v| v.is_finite())
            .map(|&v| ((v - max_ts) as f64).exp())
            .sum();
        if sum > 0. {
            ts_logsumexp = max_ts + sum.ln() as f32;
        }
    }
    let max_text = logits[..begin]
        .iter()
        .fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    if ts_logsumexp > max_text {
        logits[..begin].fill(f32::NEG_INFINITY);
    }
}

/// The best token (the first of equals) and its log-probability: the reference's
/// `sample_argmax_and_logprob`, with the log of the partition summed in f64.
pub fn argmax_logprob(logits: &[f32]) -> (usize, f32) {
    let mut best = 0;
    let mut best_l = logits[0];
    let mut max_l = if best_l.is_finite() {
        best_l
    } else {
        f32::NEG_INFINITY
    };
    for (i, &l) in logits.iter().enumerate().skip(1) {
        if l > best_l {
            best_l = l;
            best = i;
        }
        if l.is_finite() && l > max_l {
            max_l = l;
        }
    }
    (best, log_softmax_at(logits, best_l, max_l))
}

/// `l − log Σ exp(logits)`, given the largest finite logit.
fn log_softmax_at(logits: &[f32], l: f32, max_l: f32) -> f32 {
    if !max_l.is_finite() {
        return f32::NEG_INFINITY;
    }
    let sum: f64 = logits
        .iter()
        .filter(|v| v.is_finite())
        .map(|&v| ((v - max_l) as f64).exp())
        .sum();
    if sum <= 0. {
        return f32::NEG_INFINITY;
    }
    l - (max_l + sum.ln() as f32)
}

/// The two best logits' gap, for the trace: how near a decision was to going the other way.
pub fn top2_margin(logits: &[f32]) -> (usize, f32) {
    let (mut first, mut second) = ((0, f32::NEG_INFINITY), f32::NEG_INFINITY);
    for (i, &l) in logits.iter().enumerate() {
        if l > first.1 {
            second = first.1;
            first = (i, l);
        } else if l > second {
            second = l;
        }
    }
    (first.0, first.1 - second)
}

/// A token sampled from `softmax(logits / temperature)` and its log-probability as the
/// reference scores it (`logprob_of_token_hf`: the logits *multiplied* by the temperature,
/// transformers' convention).
pub fn sample(logits: &[f32], temperature: f32, rng: &mut Rng) -> (usize, f32) {
    let max_l = logits
        .iter()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let id = if !max_l.is_finite() {
        0
    } else {
        let probs: Vec<f64> = logits
            .iter()
            .map(|&l| {
                if l.is_finite() {
                    (((l - max_l) / temperature) as f64).exp()
                } else {
                    0.
                }
            })
            .collect();
        let sum: f64 = probs.iter().sum();
        let r = rng.uniform() * sum;
        let mut acc = 0.;
        probs
            .iter()
            .position(|p| {
                acc += p;
                r < acc
            })
            .unwrap_or(logits.len() - 1)
    };
    let scaled: Vec<f32> = logits.iter().map(|l| l * temperature).collect();
    let max_s = scaled
        .iter()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    (id, log_softmax_at(&scaled, scaled[id], max_s))
}

/// SplitMix64: the sampler's random numbers. The reference seeds its sampler from the
/// operating system's entropy, so its results above temperature 0 can't be reproduced by
/// anyone; this one is seeded the same every window, so ours can.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    /// A uniform value in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as f64 / 18_446_744_073_709_551_616.
    }
}

/// transformers' `_retrieve_compression_ratio`: the tokens as little-endian 16-bit values,
/// deflated as the reference's miniz does it (`mz_compress`: zlib framing, its default level,
/// which is greedy parsing with 128 probes), raw length over compressed length. A ratio above
/// 2.4 means the decoder is repeating itself.
pub fn compression_ratio(tokens: &[i32], vocab: i32) -> f32 {
    if tokens.is_empty() {
        return 0.;
    }
    let width = ((vocab as f64).log2() / 8.).floor() as usize + 1;
    let raw: Vec<u8> = tokens
        .iter()
        .flat_map(|&t| (t as u32).to_le_bytes().into_iter().take(width))
        .collect();
    use miniz_oxide::deflate::core::{
        CompressorOxide, TDEFLFlush, TDEFLStatus, compress, create_comp_flags_from_zip_params,
    };
    // `MZ_DEFAULT_COMPRESSION` (−1), the default window, the default strategy.
    let mut compressor = CompressorOxide::new(create_comp_flags_from_zip_params(-1, 15, 0));
    let mut out = vec![0u8; raw.len() * 2 + 128];
    let (status, _, written) = compress(&mut compressor, &raw, &mut out, TDEFLFlush::Finish);
    if status != TDEFLStatus::Done || written == 0 {
        return 0.;
    }
    raw.len() as f32 / written as f32
}

/// One stretch of a window's text between two timestamps, in milliseconds from the window's
/// start.
pub struct Segment {
    pub t0_ms: i64,
    pub t1_ms: i64,
    /// Indices into the generated tokens.
    pub tokens: std::ops::Range<usize>,
}

/// transformers' `_retrieve_segment` as the reference ports it: the generated tokens cut at
/// each pair of consecutive timestamps into segments, and how many mel frames the window
/// advances (all of them, unless the text ended on a pair of timestamps before the window's
/// end: then the decoder starts again at the last timestamp).
pub fn segments(
    generated: &[i32],
    s: &Specials,
    offset_ms: i64,
    frames: usize,
) -> (Vec<Segment>, usize) {
    let n = generated.len();
    let ts = |i: usize| s.is_timestamp(generated[i]);
    let mut slices: Vec<usize> = (0..n.saturating_sub(1))
        .filter(|&i| ts(i) && ts(i + 1))
        .map(|i| i + 1)
        .collect();
    let single_ending = n >= 2 && !ts(n - 2) && ts(n - 1);
    let mut out = Vec::new();
    if slices.is_empty() {
        let last_pos = (0..n)
            .rev()
            .find(|&i| ts(i) && generated[i] != s.timestamp_begin)
            .map_or(frames as i64 / 2, |i| {
                (generated[i] - s.timestamp_begin) as i64
            });
        if n > 0 {
            out.push(Segment {
                t0_ms: offset_ms,
                t1_ms: offset_ms + last_pos * 20,
                tokens: 0..n,
            });
        }
        return (out, frames);
    }
    if single_ending {
        slices.push(n);
    } else {
        *slices.last_mut().unwrap() += 1;
    }
    let mut last = 0;
    for (k, &current) in slices.iter().enumerate() {
        let is_last = k + 1 == slices.len();
        if current <= last || !ts(last) {
            last = current;
            continue;
        }
        let end = if !is_last || single_ending {
            current - 1
        } else {
            current - 2
        };
        if end < last || end >= n || !ts(end) {
            last = current;
            continue;
        }
        let t0 = offset_ms + (generated[last] - s.timestamp_begin) as i64 * 20;
        let t1 = offset_ms + (generated[end] - s.timestamp_begin) as i64 * 20;
        out.push(Segment {
            t0_ms: t0,
            t1_ms: t1.max(t0),
            tokens: last..current,
        });
        last = current;
    }
    let advance = if single_ending {
        frames
    } else if last >= 2 && last - 2 < n && ts(last - 2) {
        (generated[last - 2] - s.timestamp_begin) as usize * 2
    } else {
        frames
    };
    (out, advance)
}
