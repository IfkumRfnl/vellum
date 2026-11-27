//! Multi-head self-attention with relative positions (Transformer-XL style, NeMo's
//! `RelPositionMultiHeadAttention`), over the whole window.
//!
//! The score of query `i` for key `j` is `(q_i + u)·k_j + (q_i + v)·p_{i−j}`, scaled by
//! 1/√d_head, where `p_r` is the projected encoding of relative offset `r`. The reference gets
//! the `p_{i−j}` term with the usual "rel shift" (pad, reshape, slice); here it is plain index
//! arithmetic: offset `i − j` is row `T − 1 − i + j` of the encodings, so query `i` reads the
//! contiguous rows `T − 1 − i ..= 2T − 2 − i`.
//!
//! On the int8 path the attention itself is the reference's ([`crate::cpu::flash`]): its flash
//! attention's kernels, the position term scaled and rounded to f16 as its mask, keys and
//! values in f16 where the reference keeps them so.

use super::rounding::{Layer, rounds_like_ggml};
use crate::{
    Error, Result,
    cpu::{Mat, Rows, axpy, flash::Head, products, vector},
    gguf::Gguf,
};
use rayon::prelude::*;

/// Queries per attention task: their scores (16 × T f32) and the rows they read stay in L1/L2.
const QUERY_BLOCK: usize = 16;

pub struct SelfAttention {
    heads: usize,
    q: Layer,
    k: Layer,
    v: Layer,
    out: Layer,
    /// The positional projection (no bias).
    pos: Layer,
    /// The learned content and position biases added to the queries, `[head][d_head]`.
    bias_u: Vec<f32>,
    bias_v: Vec<f32>,
    /// Whether the reference's flash attention takes keys and values in f16.
    kv_f16: bool,
}

impl SelfAttention {
    pub fn load(file: &Gguf, name: &str, heads: usize, kv_f16: bool) -> Result<SelfAttention> {
        let q = Layer::load(file, &format!("{name}.linear_q"))?;
        let bias_u = vector(file, &format!("{name}.pos_bias_u"))?;
        let bias_v = vector(file, &format!("{name}.pos_bias_v"))?;
        if q.rows() % heads != 0 || bias_u.len() != q.rows() || bias_v.len() != q.rows() {
            return Err(Error::Gguf(format!(
                "{name}: {heads} heads don't fit {} features",
                q.rows()
            )));
        }
        Ok(SelfAttention {
            heads,
            q,
            k: Layer::load(file, &format!("{name}.linear_k"))?,
            v: Layer::load(file, &format!("{name}.linear_v"))?,
            out: Layer::load(file, &format!("{name}.linear_out"))?,
            pos: Layer::load(file, &format!("{name}.linear_pos"))?,
            bias_u,
            bias_v,
            kv_f16,
        })
    }

    /// `x` (frames × d_model, already layer-normed) attending to itself; `pos` holds the
    /// encodings of offsets `T − 1` down to `−(T − 1)`.
    pub fn forward(&self, x: &Mat, pos: &Mat) -> Mat {
        let frames = x.rows;
        assert_eq!(
            pos.rows,
            2 * frames - 1,
            "positional encodings for {frames} frames"
        );
        let d = self.q.rows();
        let dh = d / self.heads;
        let scale = 1. / (dh as f32).sqrt();
        let q = self.q.forward(x);
        let k = by_head(&self.k.forward(x), self.heads);
        let v = by_head(&self.v.forward(x), self.heads);
        let p = by_head(&self.pos.forward(pos), self.heads);
        if rounds_like_ggml() {
            return self.out.forward(&self.flash(&q, k, v, &p, frames));
        }

        // Per head and block of queries: the content scores against every key and the
        // positional ones against the rows the block's queries read (one row more per query
        // than one query reads), as matrix products, each score still exactly `dot`.
        let blocks = frames.div_ceil(QUERY_BLOCK);
        let parts: Vec<(usize, usize, Vec<f32>)> = (0..self.heads * blocks)
            .into_par_iter()
            .map(|task| {
                let (h, b) = (task / blocks, task % blocks);
                let queries = b * QUERY_BLOCK..((b + 1) * QUERY_BLOCK).min(frames);
                let n = queries.len();
                let mut qu = vec![0f32; n * dh];
                let mut qv = vec![0f32; n * dh];
                for (a, i) in queries.clone().enumerate() {
                    let qi = &q.row(i)[h * dh..][..dh];
                    for (k, q) in qi.iter().enumerate() {
                        qu[a * dh + k] = q + self.bias_u[h * dh + k];
                        qv[a * dh + k] = q + self.bias_v[h * dh + k];
                    }
                }
                let keys = &k[h * frames * dh..][..frames * dh];
                let values = &v[h * frames * dh..][..frames * dh];
                // Offsets i − j for j = 0, 1, …: rows T − 1 − i onwards; the block's last query
                // starts first.
                let first = frames - queries.end;
                let span = frames + n - 1;
                let offsets = &p[(h * (2 * frames - 1) + first) * dh..][..span * dh];
                let mut content = vec![0f32; n * frames];
                let mut position = vec![0f32; n * span];
                products(
                    &qu.chunks_exact(dh).collect::<Vec<_>>(),
                    Rows::F32(keys),
                    dh,
                    &mut content,
                );
                products(
                    &qv.chunks_exact(dh).collect::<Vec<_>>(),
                    Rows::F32(offsets),
                    dh,
                    &mut position,
                );
                let mut out = vec![0f32; n * dh];
                for a in 0..n {
                    let scores = &mut content[a * frames..][..frames];
                    let shifted = &position[a * span + (n - 1 - a)..][..frames];
                    let mut max = f32::NEG_INFINITY;
                    for (s, p) in scores.iter_mut().zip(shifted) {
                        *s = (*s + p) * scale;
                        max = max.max(*s);
                    }
                    let mut sum = 0f64;
                    for s in scores.iter_mut() {
                        *s = (*s - max).exp();
                        sum += *s as f64;
                    }
                    let inv = (1. / sum) as f32;
                    let o = &mut out[a * dh..][..dh];
                    for (s, value) in scores.iter().zip(values.chunks_exact(dh)) {
                        axpy(o, s * inv, value);
                    }
                }
                (h, queries.start, out)
            })
            .collect();
        let mut context = Mat::zeros(frames, d);
        for (h, start, out) in parts {
            for (a, row) in out.chunks_exact(dh).enumerate() {
                context.row_mut(start + a)[h * dh..][..dh].copy_from_slice(row);
            }
        }
        self.out.forward(&context)
    }
}

impl SelfAttention {
    /// The context vectors (frames × d_model) as the reference's flash attention makes them:
    /// per head and block of queries, the position scores as above, scaled and rounded to f16
    /// (its mask), then [`Head::attend`] for each query.
    fn flash(&self, q: &Mat, k: Vec<f32>, v: Vec<f32>, p: &[f32], frames: usize) -> Mat {
        let d = q.cols;
        let dh = d / self.heads;
        let scale = 1. / (dh as f32).sqrt();
        let block = frames * dh;
        let heads: Vec<Head> = k
            .par_chunks(block)
            .zip(v.par_chunks(block))
            .map(|(k, v)| Head::new(k.to_vec(), v.to_vec(), dh, self.kv_f16))
            .collect();
        let blocks = frames.div_ceil(QUERY_BLOCK);
        let parts: Vec<(usize, usize, Vec<f32>)> = (0..self.heads * blocks)
            .into_par_iter()
            .map(|task| {
                let (h, b) = (task / blocks, task % blocks);
                let queries = b * QUERY_BLOCK..((b + 1) * QUERY_BLOCK).min(frames);
                let n = queries.len();
                let mut qu = vec![0f32; n * dh];
                let mut qv = vec![0f32; n * dh];
                for (a, i) in queries.clone().enumerate() {
                    let qi = &q.row(i)[h * dh..][..dh];
                    for (k, q) in qi.iter().enumerate() {
                        qu[a * dh + k] = q + self.bias_u[h * dh + k];
                        qv[a * dh + k] = q + self.bias_v[h * dh + k];
                    }
                }
                let first = frames - queries.end;
                let span = frames + n - 1;
                let offsets = &p[(h * (2 * frames - 1) + first) * dh..][..span * dh];
                let mut position = vec![0f32; n * span];
                products(
                    &qv.chunks_exact(dh).collect::<Vec<_>>(),
                    Rows::F32(offsets),
                    dh,
                    &mut position,
                );
                let mut mask = vec![0f32; frames];
                let mut out = vec![0f32; n * dh];
                for a in 0..n {
                    let shifted = &position[a * span + (n - 1 - a)..][..frames];
                    for (m, bd) in mask.iter_mut().zip(shifted) {
                        *m = half::f16::from_f32(bd * scale).to_f32();
                    }
                    heads[h].attend(&qu[a * dh..][..dh], &mask, scale, &mut out[a * dh..][..dh]);
                }
                (h, queries.start, out)
            })
            .collect();
        let mut context = Mat::zeros(frames, d);
        for (h, start, out) in parts {
            for (a, row) in out.chunks_exact(dh).enumerate() {
                context.row_mut(start + a)[h * dh..][..dh].copy_from_slice(row);
            }
        }
        context
    }
}

/// A frames × (heads · d_head) matrix regrouped head by head: `[head][frame][d_head]`, so each
/// head's keys (or values, or encodings) are contiguous.
fn by_head(x: &Mat, heads: usize) -> Vec<f32> {
    let dh = x.cols / heads;
    let mut out = vec![0f32; x.data.len()];
    out.par_chunks_mut(x.rows * dh)
        .enumerate()
        .for_each(|(h, head)| {
            for (t, row) in head.chunks_exact_mut(dh).enumerate() {
                row.copy_from_slice(&x.row(t)[h * dh..][..dh]);
            }
        });
    out
}
