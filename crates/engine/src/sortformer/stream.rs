//! The streaming state between chunks (NeMo's `StreamingSortformerState`, batch of one): a
//! speaker cache of up to `spkcache_len` frames chosen to represent every speaker heard so far,
//! a FIFO of the most recent frames, and a running mean of the frames judged silent. The
//! frames are pre-encode embeddings (before the x-scaling); each chunk is encoded joined to
//! them, `[speaker cache | FIFO | chunk]`, so the model can tell which speaker is which.
//!
//! An exact port of transcribe.cpp's `arch/sortformer/stream.cpp` (itself NeMo's
//! `SortformerModules.streaming_update`, sync branch, and `_compress_spkcache`), in f32 and
//! in the same order of operations, so the same predictions pick the same frames.

use crate::{Trace, cpu::Mat};

/// The streaming operating point. Lengths are in 80 ms diarization frames.
#[derive(Debug, Clone)]
pub struct StreamParams {
    pub chunk_len: usize,
    pub chunk_left_context: usize,
    pub chunk_right_context: usize,
    pub fifo_len: usize,
    pub spkcache_len: usize,
    pub spkcache_update_period: usize,
    pub spkcache_sil_frames_per_spk: usize,
    pub sil_threshold: f32,
    pub pred_score_threshold: f32,
    pub scores_boost_latest: f32,
    pub strong_boost_rate: f32,
    pub weak_boost_rate: f32,
    pub min_pos_scores_rate: f32,
}

impl StreamParams {
    /// The "very high latency" preset the app runs (`SortformerPreset::VeryHighLatency`):
    /// 27.2 s chunks seeing 3.2 s ahead, a FIFO of 40 frames. As in the reference, a named
    /// preset in `TRANSCRIBE_SORTFORMER_STREAM_PRESET` and the per-field
    /// `TRANSCRIBE_SORTFORMER_STREAM_*` variables override it (validation hooks: the `small`
    /// preset forces several compressions on a short clip).
    pub fn very_high_latency() -> StreamParams {
        let mut p = StreamParams {
            chunk_len: 340,
            chunk_left_context: 1,
            chunk_right_context: 40,
            fifo_len: 40,
            spkcache_len: 188,
            spkcache_update_period: 300,
            spkcache_sil_frames_per_spk: 3,
            sil_threshold: 0.2,
            pred_score_threshold: 0.25,
            scores_boost_latest: 0.05,
            strong_boost_rate: 0.75,
            weak_boost_rate: 1.5,
            min_pos_scores_rate: 0.5,
        };
        let var = |name: &str| std::env::var(name).ok().filter(|v| !v.is_empty());
        if let Some(name) = var("TRANSCRIBE_SORTFORMER_STREAM_PRESET") {
            // (chunk_len, chunk_right_context, fifo_len, spkcache_update_period, spkcache_len)
            let preset = match name.as_str() {
                "very_high_latency" => Some((340, 40, 40, 300, 188)),
                "high_latency" => Some((124, 1, 124, 124, 188)),
                "low_latency" => Some((6, 7, 188, 144, 188)),
                "small" => Some((20, 1, 10, 20, 24)),
                _ => None,
            };
            if let Some((chunk, rc, fifo, period, cache)) = preset {
                p.chunk_len = chunk;
                p.chunk_right_context = rc;
                p.fifo_len = fifo;
                p.spkcache_update_period = period;
                p.spkcache_len = cache;
            }
        }
        let int = |name: &str, field: &mut usize| {
            if let Some(v) = var(name).and_then(|v| v.trim().parse().ok()) {
                *field = v;
            }
        };
        int("TRANSCRIBE_SORTFORMER_STREAM_CHUNK_LEN", &mut p.chunk_len);
        int("TRANSCRIBE_SORTFORMER_STREAM_FIFO_LEN", &mut p.fifo_len);
        int(
            "TRANSCRIBE_SORTFORMER_STREAM_SPKCACHE_LEN",
            &mut p.spkcache_len,
        );
        int(
            "TRANSCRIBE_SORTFORMER_STREAM_UPDATE_PERIOD",
            &mut p.spkcache_update_period,
        );
        int(
            "TRANSCRIBE_SORTFORMER_STREAM_RC",
            &mut p.chunk_right_context,
        );
        int("TRANSCRIBE_SORTFORMER_STREAM_LC", &mut p.chunk_left_context);
        p
    }
}

/// Frames of embeddings (`dim` each) and their predictions (`speakers` each), row by row.
#[derive(Debug, Clone, Default)]
struct Frames {
    embs: Vec<f32>,
    preds: Vec<f32>,
}

pub struct State {
    speakers: usize,
    dim: usize,
    /// The speaker cache's embeddings; its predictions are kept only from the first
    /// compression on (NeMo's `spkcache_preds` is `None` until then).
    spkcache: Vec<f32>,
    spkcache_preds: Option<Vec<f32>>,
    fifo: Frames,
    mean_sil_emb: Vec<f32>,
    n_sil_frames: i64,
    /// Every chunk's predictions, in order.
    pub total_preds: Vec<f32>,
    compressions: usize,
}

/// Whether the reference would dump its compression decisions (`TRANSCRIBE_SORTFORMER_COMPRESS_DUMP`),
/// so ours line up with them.
fn compress_dump() -> bool {
    std::env::var("TRANSCRIBE_SORTFORMER_COMPRESS_DUMP").is_ok_and(|v| !v.is_empty())
}

impl State {
    pub fn new(speakers: usize, dim: usize) -> State {
        State {
            speakers,
            dim,
            spkcache: Vec::new(),
            spkcache_preds: None,
            fifo: Frames::default(),
            mean_sil_emb: vec![0.; dim],
            n_sil_frames: 0,
            total_preds: Vec::new(),
            compressions: 0,
        }
    }

    pub fn spkcache_frames(&self) -> usize {
        self.spkcache.len() / self.dim
    }

    pub fn fifo_frames(&self) -> usize {
        self.fifo.embs.len() / self.dim
    }

    /// What the next chunk is encoded with: `[speaker cache | FIFO | chunk]` embeddings.
    pub fn concat(&self, chunk: &Mat) -> Mat {
        assert_eq!(chunk.cols, self.dim);
        let rows = self.spkcache_frames() + self.fifo_frames() + chunk.rows;
        let mut data = Vec::with_capacity(rows * self.dim);
        data.extend_from_slice(&self.spkcache);
        data.extend_from_slice(&self.fifo.embs);
        data.extend_from_slice(&chunk.data);
        Mat::from_vec(rows, self.dim, data)
    }

    /// After a chunk: `chunk` its embeddings (with `lc` frames of left and `rc` of right
    /// context), `preds` the predictions for the whole concatenation [`concat`](Self::concat)
    /// made from them. The chunk's middle frames join the FIFO; when it overflows, its oldest
    /// frames move to the speaker cache (and, if silent, into the silence profile), and an
    /// overfull cache is compressed. The chunk's predictions are kept.
    pub fn update(
        &mut self,
        params: &StreamParams,
        chunk: &Mat,
        preds: &Mat,
        (lc, rc): (usize, usize),
        trace: &mut dyn Trace,
    ) {
        let (n_spk, dim) = (self.speakers, self.dim);
        let s = self.spkcache_frames();
        let f = self.fifo_frames();
        let c = chunk.rows - lc - rc;
        assert_eq!(preds.rows, s + f + chunk.rows);

        // The FIFO's predictions are refreshed from this chunk's.
        self.fifo.preds = preds.data[s * n_spk..(s + f) * n_spk].to_vec();
        let chunk_preds = &preds.data[(s + f + lc) * n_spk..(s + f + lc + c) * n_spk];
        self.fifo
            .embs
            .extend_from_slice(&chunk.data[lc * dim..(lc + c) * dim]);
        self.fifo.preds.extend_from_slice(chunk_preds);

        let mut popped = 0;
        if f + c > params.fifo_len {
            let pop = params
                .spkcache_update_period
                .max((c + f).saturating_sub(params.fifo_len))
                .min(f + c);
            popped = pop;
            let pop_embs: Vec<f32> = self.fifo.embs.drain(..pop * dim).collect();
            let pop_preds: Vec<f32> = self.fifo.preds.drain(..pop * n_spk).collect();
            self.silence_profile(params.sil_threshold, &pop_embs, &pop_preds);
            self.spkcache.extend_from_slice(&pop_embs);
            if let Some(cache_preds) = &mut self.spkcache_preds {
                cache_preds.extend_from_slice(&pop_preds);
            }
            if self.spkcache_frames() > params.spkcache_len {
                if self.spkcache_preds.is_none() {
                    // First compression: the cache's predictions from this chunk's.
                    let mut seeded = preds.data[..s * n_spk].to_vec();
                    seeded.extend_from_slice(&pop_preds);
                    self.spkcache_preds = Some(seeded);
                }
                self.compress(params, trace);
            }
        }
        trace.event(
            "sortformer.update",
            serde_json::json!({
                "spkcache": s, "fifo": f, "chunk": c, "lc": lc, "rc": rc, "popped": popped,
                "spkcache_after": self.spkcache_frames(), "fifo_after": self.fifo_frames(),
                "sil_frames": self.n_sil_frames,
            }),
        );
        self.total_preds.extend_from_slice(chunk_preds);
    }

    /// NeMo's `_get_silence_profile`: the running mean of the embeddings of frames whose
    /// speaker probabilities sum below `threshold`.
    fn silence_profile(&mut self, threshold: f32, embs: &[f32], preds: &[f32]) {
        let n_spk = self.speakers;
        let silent: Vec<bool> = preds
            .chunks_exact(n_spk)
            .map(|p| p.iter().fold(0f32, |s, v| s + v) < threshold)
            .collect();
        let count = silent.iter().filter(|s| **s).count() as i64;
        if count == 0 {
            return;
        }
        let old = self.n_sil_frames;
        let new = old + count;
        let denom = new.max(1) as f32;
        for (d, mean) in self.mean_sil_emb.iter_mut().enumerate() {
            let mut sum = 0f32;
            for (frame, _) in embs
                .chunks_exact(self.dim)
                .zip(&silent)
                .filter(|(_, s)| **s)
            {
                sum += frame[d];
            }
            *mean = (*mean * old as f32 + sum) / denom;
        }
        self.n_sil_frames = new;
    }

    /// NeMo's `_compress_spkcache`: the speaker cache (more than `spkcache_len` frames) down
    /// to `spkcache_len`, keeping for each speaker the frames that most confidently belong to
    /// it alone (recent frames slightly favoured, a few strongest boosted), plus a few silence
    /// frames per speaker (the silence profile's mean embedding); a pick there's no frame for
    /// becomes silence too.
    ///
    /// Its checkpoints (`compress.<NNN>.input_preds`, `.frame_idx`, `.is_disabled`,
    /// `.spkcache_preds`) can each be injected: the reference's input predictions test the
    /// selection alone, its picks the gathering.
    fn compress(&mut self, params: &StreamParams, trace: &mut dyn Trace) {
        let (n_spk, dim) = (self.speakers, self.dim);
        let n = self.spkcache_frames();
        let len = params.spkcache_len;
        let index = self.compressions;
        let mut preds = self
            .spkcache_preds
            .take()
            .expect("seeded before compressing");
        compress_point(trace, index, "input_preds", &mut preds, &[n, n_spk]);
        let (mut frame_idx, mut disabled) = select(&preds, n_spk, params);
        compress_point(trace, index, "frame_idx", &mut frame_idx, &[len]);
        compress_point(trace, index, "is_disabled", &mut disabled, &[len]);

        let mut new_embs = vec![0f32; len * dim];
        let mut new_preds = vec![0f32; len * n_spk];
        for (j, (&f, &off)) in frame_idx.iter().zip(&disabled).enumerate() {
            let emb = &mut new_embs[j * dim..(j + 1) * dim];
            if off != 0. {
                emb.copy_from_slice(&self.mean_sil_emb);
            } else {
                let f = f as usize;
                emb.copy_from_slice(&self.spkcache[f * dim..(f + 1) * dim]);
                new_preds[j * n_spk..(j + 1) * n_spk]
                    .copy_from_slice(&preds[f * n_spk..(f + 1) * n_spk]);
            }
        }
        compress_point(
            trace,
            index,
            "spkcache_preds",
            &mut new_preds,
            &[len, n_spk],
        );
        trace.event(
            "sortformer.compress",
            serde_json::json!({
                "index": index,
                "frames": n,
                "silence": disabled.iter().filter(|d| **d != 0.).count(),
                "picked": frame_idx.iter().zip(&disabled)
                    .map(|(f, d)| if *d != 0. { -1 } else { *f as i64 }).collect::<Vec<_>>(),
            }),
        );
        self.compressions += 1;
        self.spkcache = new_embs;
        self.spkcache_preds = Some(new_preds);
    }
}

/// A compression checkpoint, `compress.<index>.<what>`: injected if the trace injects it,
/// recorded only when the reference would dump it too.
fn compress_point(
    trace: &mut dyn Trace,
    index: usize,
    what: &str,
    data: &mut Vec<f32>,
    shape: &[usize],
) {
    let name = format!("compress.{index:03}.{what}");
    if let Some((_, injected)) = trace.inject(&name, shape) {
        if injected.len() == data.len() {
            *data = injected;
        } else {
            eprintln!(
                "injecting {name}: {} values for {}; not used",
                injected.len(),
                data.len()
            );
        }
    }
    if compress_dump() && trace.wants(&name) {
        trace.tensor(&name, shape, data);
    }
}

/// Which frames a compression keeps, given the cache's predictions (frames × speakers): for
/// each of `spkcache_len` slots in frame order, its frame (as f32, the reference's dump) and
/// whether it's disabled (1: a silence slot, filled with the mean silent embedding).
fn select(preds: &[f32], n_spk: usize, params: &StreamParams) -> (Vec<f32>, Vec<f32>) {
    let n = preds.len() / n_spk;
    let len = params.spkcache_len;
    let sil = params.spkcache_sil_frames_per_spk;
    let per_spk = (len / n_spk) as i64 - sil as i64;
    let rate = |r: f32| (per_spk as f32 * r).floor() as i64;
    let (strong, weak, min_pos) = (
        rate(params.strong_boost_rate),
        rate(params.weak_boost_rate),
        rate(params.min_pos_scores_rate),
    );
    let mut scores = log_pred_scores(preds, n_spk, params.pred_score_threshold);
    disable_low_scores(preds, &mut scores, n_spk, min_pos);
    if params.scores_boost_latest > 0. {
        for v in &mut scores[len * n_spk..] {
            *v += params.scores_boost_latest;
        }
    }
    boost_topk(&mut scores, n_spk, strong, 2.);
    boost_topk(&mut scores, n_spk, weak, 1.);

    // `sil` rows of +inf per speaker after the frames, then the `len` best (speaker, frame)
    // pairs over the speaker-major flattening, ties to the lower flat index.
    let frames = n + sil;
    let value = |flat: usize| {
        let (s, i) = (flat / frames, flat % frames);
        if i < n {
            scores[i * n_spk + s]
        } else {
            f32::INFINITY
        }
    };
    let mut flat: Vec<usize> = (0..n_spk * frames).collect();
    let k = len.min(flat.len());
    top(&mut flat, k, value);
    /// A pick with no frame behind it (`max_index` in NeMo, which sorts last).
    const NONE: usize = usize::MAX;
    let mut picks: Vec<usize> = flat[..k]
        .iter()
        .map(|&f| {
            if value(f) == f32::NEG_INFINITY {
                NONE
            } else {
                f
            }
        })
        .chain(std::iter::repeat_n(NONE, len - k))
        .collect();
    picks.sort_unstable();
    picks
        .iter()
        .map(|&p| {
            let frame = (p != NONE).then_some(p % frames).filter(|&i| i < n);
            (
                frame.unwrap_or(0) as f32,
                f32::from(u8::from(frame.is_none())),
            )
        })
        .unzip()
}

/// NeMo's `_get_log_pred_scores`: per frame and speaker, how much more likely the frame is
/// that speaker alone than anything else, `log p − log(1 − p) + Σₛ log(1 − pₛ) − log ½`, with
/// probabilities clamped to at least `threshold`.
fn log_pred_scores(preds: &[f32], n_spk: usize, threshold: f32) -> Vec<f32> {
    let log_half = 0.5f32.ln();
    let mut scores = vec![0f32; preds.len()];
    for (p, out) in preds
        .chunks_exact(n_spk)
        .zip(scores.chunks_exact_mut(n_spk))
    {
        let mut l1sum = 0f32;
        for &v in p {
            l1sum += (1. - v).max(threshold).ln();
        }
        for (&v, o) in p.iter().zip(out) {
            let lp = v.max(threshold).ln();
            let l1 = (1. - v).max(threshold).ln();
            *o = lp - l1 + l1sum - log_half;
        }
    }
    scores
}

/// NeMo's `_disable_low_scores`: frames where a speaker isn't speaking (p ≤ ½) can't be
/// picked for it; nor, for a speaker with at least `min_pos` positively scored frames, its
/// speech frames that scored no better than zero (overlap).
fn disable_low_scores(preds: &[f32], scores: &mut [f32], n_spk: usize, min_pos: i64) {
    for (p, s) in preds.iter().zip(scores.iter_mut()) {
        if *p <= 0.5 {
            *s = f32::NEG_INFINITY;
        }
    }
    for spk in 0..n_spk {
        let positive = scores
            .iter()
            .skip(spk)
            .step_by(n_spk)
            .filter(|s| **s > 0.)
            .count() as i64;
        if positive >= min_pos {
            for (p, s) in preds.iter().zip(scores.iter_mut()).skip(spk).step_by(n_spk) {
                if *p > 0.5 && *s <= 0. {
                    *s = f32::NEG_INFINITY;
                }
            }
        }
    }
}

/// NeMo's `_boost_topk_scores`: each speaker's `k` best frames (ties to the earlier frame)
/// get `−scale · log ½` added.
fn boost_topk(scores: &mut [f32], n_spk: usize, k: i64, scale: f32) {
    if k <= 0 {
        return;
    }
    let delta = scale * 0.5f32.ln();
    let frames = scores.len() / n_spk;
    let k = (k as usize).min(frames);
    let mut order: Vec<usize> = Vec::with_capacity(frames);
    for spk in 0..n_spk {
        order.clear();
        order.extend(0..frames);
        top(&mut order, k, |i| scores[i * n_spk + spk]);
        for &i in &order[..k] {
            scores[i * n_spk + spk] -= delta;
        }
    }
}

/// Puts the `k` best of `items` first, best first: by `value` descending, ties to the lower
/// item (ATen's CPU `topk`, and the reference's `partial_sort` with that order).
fn top(items: &mut [usize], k: usize, value: impl Fn(usize) -> f32) {
    let order = |a: &usize, b: &usize| {
        let (va, vb) = (value(*a), value(*b));
        vb.partial_cmp(&va)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    };
    if k == 0 {
        return;
    }
    if k < items.len() {
        items.select_nth_unstable_by(k - 1, order);
    }
    items[..k].sort_unstable_by(order);
}
