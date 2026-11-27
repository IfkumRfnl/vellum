//! Whisper's encoder as transcribe.cpp builds it (`arch/whisper/encoder.cpp` there): two 1-D
//! convolutions of width 3 (the second of stride 2) with GELU, learned positions added, then
//! pre-LN transformer blocks (self-attention over all 1500 frames, a GELU feed-forward) and a
//! final layer norm. Checkpoints: `enc.mel.in` (taken by the caller), `enc.conv1.out`,
//! `enc.conv2.out`, `enc.pos_emb`, `enc.embed.out`, `enc.block.<L>.out`, `enc.final`.
//!
//! Matrix products are, for Large v3 (Turbo), the CPU's int8 products: each Q8_0 layer's input
//! rounded to 8 bits as the reference rounds it, and GELU as ggml computes it; for the other
//! sizes exact f32 against the exactly dequantized weights (see [`Encoder::load`]). The convolutions' kernels are f32 in the file and
//! stay exact, as the reference keeps them (`conv_1d_f32`). Its flash attention on f32 keys and
//! values is an online softmax of the same scores; this takes it in one pass.

use super::{HParams, ops::Norm, ops::linear};
use crate::{
    Result, Trace,
    cpu::{
        Linear, Mat, Rows, Weight,
        int8::{Dense, int8_unless_told},
        products, tensor, vector,
    },
    gguf::Gguf,
    trace::checkpoint,
};
use rayon::prelude::*;

/// Queries per attention task: a block of scores (64 × 1500 f32) stays in L2.
const QUERY_BLOCK: usize = 64;

pub struct Encoder {
    heads: usize,
    /// The convolutions as linear layers over each output frame's three input frames
    /// (`[channel][tap]`, the reference's im2col order).
    conv1: Dense,
    conv2: Dense,
    /// Whether the products are the int8 ones (and GELU as ggml computes it).
    int8: bool,
    positions: Mat,
    blocks: Vec<Block>,
    final_norm: Norm,
}

struct Block {
    norm_attn: Norm,
    q: Dense,
    k: Dense,
    v: Dense,
    out: Dense,
    norm_ffn: Norm,
    fc1: Dense,
    fc2: Dense,
}

impl Encoder {
    /// The encoder; its products are the int8 ones where that was measured to keep the
    /// transcripts within the lab's gates against the reference (Large v3 Turbo; Small lost
    /// WER under every int8 variant tried), unless `VELLUM_CPU_GEMM` says otherwise.
    pub fn load(file: &Gguf, hp: &HParams) -> Result<Encoder> {
        let int8 = int8_unless_told(hp.d_model == 1280 && hp.enc_layers == 32);
        let blocks = (0..hp.enc_layers)
            .into_par_iter()
            .map(|l| {
                let name = |part: &str| format!("enc.blocks.{l}.{part}");
                Ok(Block {
                    norm_attn: Norm::load(file, &name("norm_attn"))?,
                    q: Dense::load_as(file, &name("attn.q"), int8)?,
                    k: Dense::load_as(file, &name("attn.k"), int8)?,
                    v: Dense::load_as(file, &name("attn.v"), int8)?,
                    out: Dense::load_as(file, &name("attn.out"), int8)?,
                    norm_ffn: Norm::load(file, &name("norm_ffn"))?,
                    fc1: Dense::load_as(file, &name("ffn.fc1"), int8)?,
                    fc2: Dense::load_as(file, &name("ffn.fc2"), int8)?,
                })
            })
            .collect::<Result<_>>()?;
        let pos = tensor(file, "enc.pos_emb.weight")?;
        let rows = pos.dims[1] as usize;
        Ok(Encoder {
            heads: hp.enc_heads,
            conv1: Dense::Exact(conv(file, "enc.conv.0")?),
            conv2: Dense::Exact(conv(file, "enc.conv.1")?),
            int8,
            positions: Mat::from_vec(rows, hp.d_model, file.dequantize(pos)?),
            blocks,
            final_norm: Norm::load(file, "enc.final_norm")?,
        })
    }

    /// Mel features (frames × `num_mels`, an even number of frames) to encoder states
    /// (frames / 2 × `d_model`).
    pub fn forward(&self, mel: &Mat, trace: &mut dyn Trace) -> Mat {
        let mut x = linear(&self.conv1, &im2col(mel, 1));
        super::ops::gelu_rows(&mut x, self.int8);
        checkpoint(trace, "enc.conv1.out", &mut x);
        let mut x = linear(&self.conv2, &im2col(&x, 2));
        super::ops::gelu_rows(&mut x, self.int8);
        checkpoint(trace, "enc.conv2.out", &mut x);
        let mut pos = Mat::from_vec(
            x.rows,
            x.cols,
            self.positions.data[..x.rows * x.cols].to_vec(),
        );
        checkpoint(trace, "enc.pos_emb", &mut pos);
        x.add_assign(&pos);
        checkpoint(trace, "enc.embed.out", &mut x);
        for (l, block) in self.blocks.iter().enumerate() {
            let y = block.norm_attn.apply(&x);
            let attended = self_attention(
                &linear(&block.q, &y),
                &linear(&block.k, &y),
                &linear(&block.v, &y),
                self.heads,
            );
            x.add_assign(&linear(&block.out, &attended));
            let mut h = linear(&block.fc1, &block.norm_ffn.apply(&x));
            super::ops::gelu_rows(&mut h, self.int8);
            x.add_assign(&linear(&block.fc2, &h));
            checkpoint(trace, &format!("enc.block.{l}.out"), &mut x);
        }
        let mut out = self.final_norm.apply(&x);
        checkpoint(trace, "enc.final", &mut out);
        out
    }
}

/// A width-3 convolution's kernel (`[3, in, out]` in ggml's order) and bias as a linear layer
/// over [`im2col`] rows.
fn conv(file: &Gguf, name: &str) -> Result<Linear> {
    let info = tensor(file, &format!("{name}.weight"))?;
    let (taps, inputs, outputs) = (
        info.dims[0] as usize,
        info.dims[1] as usize,
        info.dims[2] as usize,
    );
    Ok(Linear {
        rows: outputs,
        cols: inputs * taps,
        weight: Weight::F32(file.dequantize(info)?),
        bias: Some(vector(file, &format!("{name}.bias"))?),
    })
}

/// Each output frame's receptive field for a width-3, padding-1 convolution of `stride`:
/// input frames `stride·t − 1 ..= stride·t + 1` (zeros outside), channel-major.
fn im2col(x: &Mat, stride: usize) -> Mat {
    let frames = x.rows.div_ceil(stride);
    let channels = x.cols;
    let mut out = Mat::zeros(frames, channels * 3);
    out.data
        .par_chunks_mut(channels * 3)
        .enumerate()
        .for_each(|(t, row)| {
            for tap in 0..3 {
                let Some(src) = (stride * t + tap).checked_sub(1).filter(|&s| s < x.rows) else {
                    continue;
                };
                for (c, &v) in x.row(src).iter().enumerate() {
                    row[c * 3 + tap] = v;
                }
            }
        });
    out
}

/// Bidirectional multi-head attention of every frame over every frame, in one pass. (On the
/// int8 path too: emulating the reference's tiled flash attention measured no closer to it.)
fn self_attention(q: &Mat, k: &Mat, v: &Mat, heads: usize) -> Mat {
    let (frames, d) = (q.rows, q.cols);
    let hd = d / heads;
    let scale = 1. / (hd as f32).sqrt();
    // Per head: queries and keys as rows, values transposed (head_dim rows of frames).
    let split = |m: &Mat, h: usize| -> Vec<f32> {
        let mut out = Vec::with_capacity(frames * hd);
        for t in 0..frames {
            out.extend_from_slice(&m.row(t)[h * hd..(h + 1) * hd]);
        }
        out
    };
    let per_head: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = (0..heads)
        .into_par_iter()
        .map(|h| {
            let vh = split(v, h);
            let mut vt = vec![0f32; frames * hd];
            for t in 0..frames {
                for c in 0..hd {
                    vt[c * frames + t] = vh[t * hd + c];
                }
            }
            (split(q, h), split(k, h), vt)
        })
        .collect();

    let blocks = frames.div_ceil(QUERY_BLOCK);
    let results: Vec<(usize, usize, Vec<f32>)> = (0..heads * blocks)
        .into_par_iter()
        .map(|task| {
            let (h, b) = (task / blocks, task % blocks);
            let (qh, kh, vt) = &per_head[h];
            let rows = b * QUERY_BLOCK..((b + 1) * QUERY_BLOCK).min(frames);
            let queries: Vec<&[f32]> = rows.clone().map(|t| &qh[t * hd..(t + 1) * hd]).collect();
            let mut scores = vec![0f32; queries.len() * frames];
            products(&queries, Rows::F32(kh), hd, &mut scores);
            let mut sums = Vec::with_capacity(queries.len());
            for row in scores.chunks_exact_mut(frames) {
                let max = row.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x * scale));
                let mut sum = 0f64;
                for x in row.iter_mut() {
                    *x = (*x * scale - max).exp();
                    sum += *x as f64;
                }
                sums.push((1. / sum) as f32);
            }
            let weighted: Vec<&[f32]> = scores.chunks_exact(frames).collect();
            let mut out = vec![0f32; queries.len() * hd];
            products(&weighted, Rows::F32(vt), frames, &mut out);
            for (row, inv) in out.chunks_exact_mut(hd).zip(&sums) {
                for x in row {
                    *x *= inv;
                }
            }
            (h, rows.start, out)
        })
        .collect();

    let mut out = Mat::zeros(frames, d);
    for (h, start, block) in results {
        for (i, row) in block.chunks_exact(hd).enumerate() {
            out.row_mut(start + i)[h * hd..(h + 1) * hd].copy_from_slice(row);
        }
    }
    out
}
