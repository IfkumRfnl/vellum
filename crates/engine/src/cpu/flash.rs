//! Attention of one head as the reference's CPU computes it: ggml's flash attention
//! (`ggml_compute_forward_flash_attn_ext_f16`), which its conformer uses by default (Parakeet's
//! and Sortformer's encoders). A mask, when there is one, is added to the scaled scores (the
//! conformer's relative-position scores, scaled and rounded to f16 as it casts them); keys and
//! values are rounded to f16 when the reference keeps them so (Parakeet: its `kv_type` AUTO picks
//! f16 for quantized weights; streaming Sortformer keeps f32).
//!
//! ggml takes one of two kernels. With 64 queries or more (`GGML_FA_TILE_Q`), the tiled one:
//! keys in tiles of 64, each score one fused multiply-add per dimension in order, scaled, the
//! mask added; an online softmax over the tiles with ggml's vector `expf`; the values summed in
//! f32. With fewer, one query at a time over every key: the query rounded to the keys' type,
//! the softmax updated key by key with the library's `expf`, the values summed in an f16
//! accumulator when they're f16. Both are reproduced here operation by operation, except the
//! order of the f32 additions in the short kernel's dot products (about 1e-7).

use super::{
    axpy, dot, fma_columns,
    int8::{mad_f16, scale_f16},
    softmax_like_ggml,
};
use half::f16;

/// Keys per tile of the tiled kernel, and the fewest queries that take it.
pub const TILE: usize = 64;

/// One head's keys and values (`frames` rows of `dh` each), rounded as the reference holds them,
/// and its keys again tile by tile, transposed (`[tile][dimension][64 keys]`, zero-padded), as
/// the tiled kernel reads them.
pub struct Head {
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    tiles: Vec<f32>,
    frames: usize,
    dh: usize,
    f16: bool,
}

impl Head {
    pub fn new(mut keys: Vec<f32>, mut values: Vec<f32>, dh: usize, f16: bool) -> Head {
        if f16 {
            for v in keys.iter_mut().chain(values.iter_mut()) {
                *v = half::f16::from_f32(*v).to_f32();
            }
        }
        let frames = keys.len() / dh;
        let mut tiles = Vec::new();
        if frames >= TILE {
            tiles = vec![0f32; frames.div_ceil(TILE) * dh * TILE];
            for (j, key) in keys.chunks_exact(dh).enumerate() {
                let tile = &mut tiles[(j / TILE) * dh * TILE..][..dh * TILE];
                for (d, k) in key.iter().enumerate() {
                    tile[d * TILE + j % TILE] = *k;
                }
            }
        }
        Head {
            keys,
            values,
            tiles,
            frames,
            dh,
            f16,
        }
    }

    /// Query `q` (`dh` values, f32) over every key, with `mask` (one value per key, or none
    /// when empty) added to its scaled scores; the weighted sum of the values into `out`.
    pub fn attend(&self, q: &[f32], mask: &[f32], scale: f32, out: &mut [f32]) {
        if self.frames >= TILE {
            self.tiled(q, mask, scale, out);
        } else {
            self.one_by_one(q, mask, scale, out);
        }
    }

    /// `ggml_compute_forward_flash_attn_ext_tiled`, for one query.
    fn tiled(&self, q: &[f32], mask: &[f32], scale: f32, out: &mut [f32]) {
        let dh = self.dh;
        let (mut sum, mut max) = (0f32, f32::NEG_INFINITY);
        out.fill(0.);
        let mut scores = [0f32; TILE];
        for (t, tile) in self.tiles.chunks_exact(dh * TILE).enumerate() {
            let start = t * TILE;
            let keys = (self.frames - start).min(TILE);
            scores.fill(0.);
            fma_columns(&mut scores, tile, q);
            for (j, s) in scores[..keys].iter_mut().enumerate() {
                *s *= scale;
                if let Some(m) = mask.get(start + j) {
                    *s += m;
                }
            }
            scores[keys..].fill(f32::NEG_INFINITY);
            let tile_max = scores.iter().fold(f32::NEG_INFINITY, |m, &s| m.max(s));
            if tile_max == f32::NEG_INFINITY {
                continue;
            }
            let new = max.max(tile_max);
            if new > max {
                let ms = (max - new).exp();
                out.iter_mut().for_each(|v| *v *= ms);
                sum *= ms;
            }
            max = new;
            sum = (sum as f64 + softmax_like_ggml(&mut scores, new)) as f32;
            let values = &self.values[start * dh..(start + keys) * dh];
            fma_columns(out, values, &scores[..keys]);
        }
        let inv = if sum == 0. { 0. } else { 1. / sum };
        out.iter_mut().for_each(|v| *v *= inv);
    }

    /// `ggml_compute_forward_flash_attn_ext_f16_one_chunk`, for one query.
    fn one_by_one(&self, q: &[f32], mask: &[f32], scale: f32, out: &mut [f32]) {
        let dh = self.dh;
        let rounded: Vec<f32>;
        let q = if self.f16 {
            rounded = q.iter().map(|v| f16::from_f32(*v).to_f32()).collect();
            &rounded
        } else {
            q
        };
        let (mut sum, mut max) = (0f32, f32::NEG_INFINITY);
        out.fill(0.);
        for (j, (key, value)) in self
            .keys
            .chunks_exact(dh)
            .zip(self.values.chunks_exact(dh))
            .enumerate()
        {
            let s = dot(q, key) * scale + mask.get(j).copied().unwrap_or(0.);
            let (mut ms, mut vs) = (1f32, 1f32);
            if s > max {
                ms = (max - s).exp();
                max = s;
                if self.f16 {
                    scale_f16(out, ms);
                } else {
                    out.iter_mut().for_each(|v| *v *= ms);
                }
            } else {
                vs = (s - max).exp();
            }
            if self.f16 {
                mad_f16(out, value, vs);
            } else {
                axpy(out, vs, value);
            }
            sum = sum * ms + vs;
        }
        let inv = if sum == 0. { 0. } else { 1. / sum };
        out.iter_mut().for_each(|v| *v *= inv);
    }
}
