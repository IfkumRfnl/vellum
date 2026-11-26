//! Multi-head self-attention with relative positions (Transformer-XL style, NeMo's
//! `RelPositionMultiHeadAttention`), over the whole window.
//!
//! The score of query `i` for key `j` is `(q_i + u)·k_j + (q_i + v)·p_{i−j}`, scaled by
//! 1/√d_head, where `p_r` is the projected encoding of relative offset `r`. The reference gets
//! the `p_{i−j}` term with the usual "rel shift" (pad, reshape, slice); here it is plain index
//! arithmetic: offset `i − j` is row `T − 1 − i + j` of the encodings, so query `i` reads the
//! contiguous rows `T − 1 − i ..= 2T − 2 − i`.

use super::rounding::Layer;
use crate::{
    Error, Result,
    cpu::{Mat, dot, vector},
    gguf::Gguf,
};
use rayon::prelude::*;

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
}

impl SelfAttention {
    pub fn load(file: &Gguf, name: &str, heads: usize) -> Result<SelfAttention> {
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

        let mut context = Mat::zeros(frames, d);
        context.data.par_chunks_mut(d).enumerate().for_each_init(
            || (vec![0f32; frames], vec![0f32; dh], vec![0f32; dh]),
            |(scores, qu, qv), (i, out)| {
                for h in 0..self.heads {
                    let qi = &q.row(i)[h * dh..][..dh];
                    for (((u, w), q), (bu, bv)) in qu
                        .iter_mut()
                        .zip(qv.iter_mut())
                        .zip(qi)
                        .zip(self.bias_u[h * dh..].iter().zip(&self.bias_v[h * dh..]))
                    {
                        *u = q + bu;
                        *w = q + bv;
                    }
                    let keys = &k[h * frames * dh..][..frames * dh];
                    let values = &v[h * frames * dh..][..frames * dh];
                    // Offsets i − j for j = 0, 1, …: rows T − 1 − i onwards.
                    let offsets = &p[(h * (2 * frames - 1) + frames - 1 - i) * dh..][..frames * dh];
                    let mut max = f32::NEG_INFINITY;
                    for ((s, key), offset) in scores
                        .iter_mut()
                        .zip(keys.chunks_exact(dh))
                        .zip(offsets.chunks_exact(dh))
                    {
                        *s = (dot(qu, key) + dot(qv, offset)) * scale;
                        max = max.max(*s);
                    }
                    let mut sum = 0f64;
                    for s in scores.iter_mut() {
                        *s = (*s - max).exp();
                        sum += *s as f64;
                    }
                    let inv = (1. / sum) as f32;
                    let o = &mut out[h * dh..][..dh];
                    for (s, value) in scores.iter().zip(values.chunks_exact(dh)) {
                        let a = s * inv;
                        for (o, v) in o.iter_mut().zip(value) {
                            *o = a.mul_add(*v, *o);
                        }
                    }
                }
            },
        );
        self.out.forward(&context)
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
