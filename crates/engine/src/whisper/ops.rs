//! What Whisper's encoder and decoder share: layer norms, linear layers sized for one row or
//! thousands, and attention over keys and values kept in f32 or f16.

use crate::{
    Result,
    cpu::{
        Linear, Mat, Weight,
        int8::{Dense, mad_f16, scale_f16},
        layer_norm, products, vector,
    },
    gguf::Gguf,
};
use half::f16;
use rayon::prelude::*;
use std::sync::OnceLock;

/// The reference's `kLayerNormEps`.
const EPS: f32 = 1e-5;

pub struct Norm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Norm {
    pub fn load(file: &Gguf, name: &str) -> Result<Norm> {
        Ok(Norm {
            weight: vector(file, &format!("{name}.weight"))?,
            bias: vector(file, &format!("{name}.bias"))?,
        })
    }

    pub fn apply(&self, x: &Mat) -> Mat {
        layer_norm(x, &self.weight, &self.bias, EPS)
    }
}

/// GELU over `x` in place: as ggml computes it on the int8 path, exactly (erf in f64) on the
/// f32 oracle.
pub fn gelu_rows(x: &mut Mat, like_ggml: bool) {
    if like_ggml {
        x.map(crate::cpu::gelu_like_ggml);
    } else {
        x.map(crate::cpu::gelu);
    }
}

/// Output rows per task when a layer multiplies only a few input rows.
const ROWS_PER_TASK: usize = 64;

/// `x · Wᵀ + b`. The int8 products take any number of rows themselves. In f32, many rows go
/// through [`Linear::forward`] (weights expanded per panel, cache blocked); a few (decoder
/// steps) stream the stored weights once, in tasks of output rows, with the same sums either
/// way (see [`products`]).
pub fn linear(layer: &Dense, x: &Mat) -> Mat {
    let Dense::Exact(layer) = layer else {
        return layer.forward(x);
    };
    if ggml_rounding() && matches!(layer.weight, Weight::Q8(_)) {
        return exact_linear(layer, &to_q8(x));
    }
    exact_linear(layer, x)
}

/// Whether `VELLUM_GGML_ROUNDING=1` is set: an experiment on the f32 oracle
/// (`VELLUM_CPU_GEMM=f32`) that rounds each Q8_0 layer's input to Q8_0 blocks first, as ggml's
/// CPU matmuls do (an f16 scale per 32 values, the values `round(x / scale)`, ties to even),
/// in plain code: the int8 products' reference. It also turns on the emulation of ggml's CPU
/// flash attention on f16 keys and values ([`attend`]).
pub fn ggml_rounding() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VELLUM_GGML_ROUNDING").is_ok_and(|v| v == "1"))
}

/// Each block of 32 values of each row as ggml's `quantize_row_q8_0` leaves it, as f32.
pub fn to_q8(x: &Mat) -> Mat {
    let mut x = x.clone();
    x.data.par_chunks_mut(32).for_each(|v| {
        let amax = v.iter().fold(0f32, |m, v| m.max(v.abs()));
        let id = if amax != 0. { 127. / amax } else { 0. };
        let scale = f16::from_f32(amax / 127.).to_f32();
        for x in v {
            *x = scale * (*x * id).round_ties_even();
        }
    });
    x
}

fn exact_linear(layer: &Linear, x: &Mat) -> Mat {
    if x.rows > 16 {
        return layer.forward(x);
    }
    let inputs: Vec<&[f32]> = (0..x.rows).map(|t| x.row(t)).collect();
    let cols = layer.cols;
    let tasks = layer.rows.div_ceil(ROWS_PER_TASK);
    let blocks: Vec<Vec<f32>> = (0..tasks)
        .into_par_iter()
        .map(|task| {
            let range = task * ROWS_PER_TASK..((task + 1) * ROWS_PER_TASK).min(layer.rows);
            let mut block = vec![0f32; x.rows * range.len()];
            products(&inputs, layer.weight.rows(range, cols), cols, &mut block);
            block
        })
        .collect();
    let mut out = Mat::zeros(x.rows, layer.rows);
    for (task, block) in blocks.iter().enumerate() {
        let start = task * ROWS_PER_TASK;
        let n = block.len() / x.rows.max(1);
        for t in 0..x.rows {
            let row = &mut out.row_mut(t)[start..start + n];
            row.copy_from_slice(&block[t * n..(t + 1) * n]);
            if let Some(bias) = &layer.bias {
                for (v, b) in row.iter_mut().zip(&bias[start..start + n]) {
                    *v += b;
                }
            }
        }
    }
    out
}

/// Keys or values of one layer, by head: `heads` blocks of `capacity` rows of `head_dim`.
/// The decoder's caches hold f16 values, as the reference keeps them for a quantized model
/// (its `kv_type` AUTO), stored as f32 so attention reads them without converting;
/// language detection's are f32, as its prefill graph computes them.
pub struct HeadRows {
    values: Vec<f32>,
    /// Whether values are rounded to f16 as they're written.
    f16: bool,
    pub heads: usize,
    pub capacity: usize,
    pub head_dim: usize,
}

impl HeadRows {
    pub fn new(heads: usize, capacity: usize, head_dim: usize, f16: bool) -> HeadRows {
        HeadRows {
            values: vec![0.; heads * capacity * head_dim],
            f16,
            heads,
            capacity,
            head_dim,
        }
    }

    /// Rows `at..` from `x` (one row per position, heads side by side).
    pub fn write(&mut self, at: usize, x: &Mat) {
        assert!(at + x.rows <= self.capacity, "a cache of {}", self.capacity);
        let (hd, cap) = (self.head_dim, self.capacity);
        for t in 0..x.rows {
            for h in 0..self.heads {
                let src = &x.row(t)[h * hd..(h + 1) * hd];
                let dst = &mut self.values[(h * cap + at + t) * hd..][..hd];
                dst.copy_from_slice(src);
                if self.f16 {
                    for v in dst {
                        *v = f16::from_f32(*v).to_f32();
                    }
                }
            }
        }
    }

    fn is_f16(&self) -> bool {
        self.f16
    }

    /// The first `n` rows of head `h`.
    fn head(&self, h: usize, n: usize) -> &[f32] {
        &self.values[h * self.capacity * self.head_dim..(h * self.capacity + n) * self.head_dim]
    }
}

/// Multi-head attention of the rows of `q` (queries × heads·head_dim) over the first `n`
/// positions of `keys` and `values`, query `i` seeing positions `..visible(i)`. Scores are
/// `q·k / √head_dim`, softmaxed in f32 with their sum in f64, and the values weighted and
/// summed before dividing by it. Against f16 keys the query is rounded to f16 first, as ggml's
/// CPU flash attention does.
pub fn attend(
    q: &Mat,
    keys: &HeadRows,
    values: &HeadRows,
    n: usize,
    visible: impl Fn(usize) -> usize + Sync,
) -> Mat {
    let (heads, hd) = (keys.heads, keys.head_dim);
    let scale = 1. / (hd as f32).sqrt();
    let round_q = keys.is_f16();
    if round_q && ggml_rounding() {
        return attend_ggml(q, keys, values, n, visible);
    }
    let per_head: Vec<Mat> = (0..heads)
        .into_par_iter()
        .map(|h| {
            let k = keys.head(h, n);
            let v = values.head(h, n);
            let mut out = Mat::zeros(q.rows, hd);
            let mut query = vec![0f32; hd];
            let mut scores = vec![0f32; n];
            for i in 0..q.rows {
                query.copy_from_slice(&q.row(i)[h * hd..(h + 1) * hd]);
                if round_q {
                    for x in &mut query {
                        *x = f16::from_f32(*x).to_f32();
                    }
                }
                let seen = visible(i).min(n);
                let s = &mut scores[..seen];
                products(&[&query], crate::cpu::Rows::F32(&k[..seen * hd]), hd, s);
                let max = s.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x * scale));
                let mut sum = 0f64;
                for x in s.iter_mut() {
                    *x = (*x * scale - max).exp();
                    sum += *x as f64;
                }
                let row = out.row_mut(i);
                for (j, &p) in s.iter().enumerate() {
                    for (o, &vv) in row.iter_mut().zip(&v[j * hd..(j + 1) * hd]) {
                        *o += p * vv;
                    }
                }
                let inv = (1. / sum) as f32;
                for o in row.iter_mut() {
                    *o *= inv;
                }
            }
            out
        })
        .collect();
    let mut out = Mat::zeros(q.rows, heads * hd);
    for (h, part) in per_head.iter().enumerate() {
        for i in 0..q.rows {
            out.row_mut(i)[h * hd..(h + 1) * hd].copy_from_slice(part.row(i));
        }
    }
    out
}

/// The reference's threads (`default_n_threads`: the CPUs, at most 8), which decide how its
/// flash attention splits a single query's keys.
const GGML_THREADS: usize = 8;

/// [`attend`] on f16 keys and values as ggml's CPU flash attention computes it (the
/// `VELLUM_GGML_ROUNDING` experiment on the f32 oracle; on the int8 path it measured further
/// from the reference than the exact attention): one key at a time with a running maximum, the weighted
/// values summed in an f16 accumulator (rounded after every multiply-add and rescale). A
/// single query over 512 keys or more is split into one run per thread (over the cache
/// padded to a multiple of 256, as the reference's cross cache is), joined in f32.
fn attend_ggml(
    q: &Mat,
    keys: &HeadRows,
    values: &HeadRows,
    n: usize,
    visible: impl Fn(usize) -> usize + Sync,
) -> Mat {
    let (heads, hd) = (keys.heads, keys.head_dim);
    let scale = 1. / (hd as f32).sqrt();
    let padded = if q.rows == 1 && n >= 512 {
        n.div_ceil(256) * 256
    } else {
        n
    };
    let split = q.rows == 1 && padded >= 512;
    let chunk = if split {
        padded.div_ceil(GGML_THREADS)
    } else {
        padded
    };
    let per_head: Vec<Mat> = (0..heads)
        .into_par_iter()
        .map(|h| {
            let k = keys.head(h, n);
            let v = values.head(h, n);
            let mut out = Mat::zeros(q.rows, hd);
            for i in 0..q.rows {
                let query: Vec<f32> = q.row(i)[h * hd..(h + 1) * hd]
                    .iter()
                    .map(|x| f16::from_f32(*x).to_f32())
                    .collect();
                let seen = visible(i).min(n);
                let (mut m_all, mut s_all) = (f32::NEG_INFINITY, 0f32);
                let mut acc_all = vec![0f32; hd];
                for start in (0..padded).step_by(chunk) {
                    let end = (start + chunk).min(seen);
                    let (mut m, mut s) = (f32::NEG_INFINITY, 0f32);
                    // An f16 accumulator, held as the f32 values of its f16s.
                    let mut acc = vec![0f32; hd];
                    for j in start..end {
                        let score = crate::cpu::dot(&query, &k[j * hd..(j + 1) * hd]) * scale;
                        let (mut ms, mut vs) = (1f32, 1f32);
                        if score > m {
                            ms = (m - score).exp();
                            m = score;
                            scale_f16(&mut acc, ms);
                        } else {
                            vs = (score - m).exp();
                        }
                        mad_f16(&mut acc, &v[j * hd..(j + 1) * hd], vs);
                        s = s * ms + vs;
                    }
                    if s == 0. {
                        continue;
                    }
                    let m_new = m_all.max(m);
                    let (old, new) = ((m_all - m_new).exp(), (m - m_new).exp());
                    for (a, b) in acc_all.iter_mut().zip(&acc) {
                        *a = *a * old + b * new;
                    }
                    s_all = s_all * old + s * new;
                    m_all = m_new;
                }
                let inv = if s_all == 0. { 0. } else { 1. / s_all };
                for (o, a) in out.row_mut(i).iter_mut().zip(&acc_all) {
                    *o = a * inv;
                }
            }
            out
        })
        .collect();
    let mut out = Mat::zeros(q.rows, heads * hd);
    for (h, part) in per_head.iter().enumerate() {
        for i in 0..q.rows {
            out.row_mut(i)[h * hd..(h + 1) * hd].copy_from_slice(part.row(i));
        }
    }
    out
}
