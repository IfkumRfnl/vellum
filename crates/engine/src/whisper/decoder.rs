//! Whisper's decoder as transcribe.cpp builds it (`arch/whisper/decoder.cpp` there): token
//! embedding plus learned position, pre-LN blocks of causal self-attention, cross-attention
//! over the encoder's output and a GELU feed-forward, a final layer norm, and the token
//! embedding again as the output head. Keys and values are cached: the encoder's once per
//! window ([`CrossCache`]), the decoded tokens' as they come ([`SelfCache`]), both in f16 as
//! the reference keeps them for a quantized model.

use super::{
    HParams,
    ops::{HeadRows, Norm, attend, linear},
};
use crate::{
    Result,
    cpu::{Linear, Mat, Rows, Weight, expand, gelu, products, tensor, weight},
    gguf::Gguf,
};
use rayon::prelude::*;

/// Output rows per task of the logits head.
const VOCAB_PER_TASK: usize = 256;

pub struct Decoder {
    d_model: usize,
    heads: usize,
    vocab: usize,
    /// The token embedding (vocab × d_model), also the output head.
    embedding: Weight,
    positions: Mat,
    blocks: Vec<Block>,
    final_norm: Norm,
}

struct Block {
    norm_self: Norm,
    self_q: Linear,
    self_k: Linear,
    self_v: Linear,
    self_out: Linear,
    norm_cross: Norm,
    cross_q: Linear,
    cross_k: Linear,
    cross_v: Linear,
    cross_out: Linear,
    norm_ffn: Norm,
    fc1: Linear,
    fc2: Linear,
}

/// The encoder's keys and values for every decoder layer.
pub struct CrossCache {
    layers: Vec<(HeadRows, HeadRows)>,
    frames: usize,
}

/// The decoded tokens' keys and values for every layer, `len` positions filled.
pub struct SelfCache {
    layers: Vec<(HeadRows, HeadRows)>,
    pub len: usize,
}

/// The intermediate states of a pass, as the reference dumps them (prompt pass only).
#[derive(Default)]
pub struct Taps {
    pub token_emb: Option<Mat>,
    pub pos_emb: Option<Mat>,
    pub embed_sum: Option<Mat>,
    pub blocks: Vec<Mat>,
    pub before_head: Option<Mat>,
}

impl Decoder {
    pub fn load(file: &Gguf, hp: &HParams) -> Result<Decoder> {
        let blocks = (0..hp.dec_layers)
            .into_par_iter()
            .map(|l| {
                let name = |part: &str| format!("dec.blocks.{l}.{part}");
                let lin = |part: &str| Linear::load(file, &name(part));
                Ok(Block {
                    norm_self: Norm::load(file, &name("norm_self"))?,
                    self_q: lin("self_attn.q")?,
                    self_k: lin("self_attn.k")?,
                    self_v: lin("self_attn.v")?,
                    self_out: lin("self_attn.out")?,
                    norm_cross: Norm::load(file, &name("norm_cross"))?,
                    cross_q: lin("cross_attn.q")?,
                    cross_k: lin("cross_attn.k")?,
                    cross_v: lin("cross_attn.v")?,
                    cross_out: lin("cross_attn.out")?,
                    norm_ffn: Norm::load(file, &name("norm_ffn"))?,
                    fc1: lin("ffn.fc1")?,
                    fc2: lin("ffn.fc2")?,
                })
            })
            .collect::<Result<_>>()?;
        let (vocab, d_model, embedding) = weight(file, "dec.token_embd.weight")?;
        let pos = tensor(file, "dec.pos_emb.weight")?;
        Ok(Decoder {
            d_model,
            heads: hp.dec_heads,
            vocab,
            embedding,
            positions: Mat::from_vec(pos.dims[1] as usize, d_model, file.dequantize(pos)?),
            blocks,
            final_norm: Norm::load(file, "dec.final_norm")?,
        })
    }

    pub fn max_positions(&self) -> usize {
        self.positions.rows
    }

    /// Every layer's keys and values of the encoder's output. `f16` rounds them as the
    /// reference's cache does; language detection's prefill keeps them in f32.
    pub fn cross_cache(&self, encoded: &Mat, f16: bool) -> CrossCache {
        let hd = self.d_model / self.heads;
        let layers = self
            .blocks
            .iter()
            .map(|b| {
                let mut k = HeadRows::new(self.heads, encoded.rows, hd, f16);
                let mut v = HeadRows::new(self.heads, encoded.rows, hd, f16);
                k.write(0, &linear(&b.cross_k, encoded));
                v.write(0, &linear(&b.cross_v, encoded));
                (k, v)
            })
            .collect();
        CrossCache {
            layers,
            frames: encoded.rows,
        }
    }

    /// An empty cache for `capacity` positions.
    pub fn self_cache(&self, capacity: usize, f16: bool) -> SelfCache {
        let hd = self.d_model / self.heads;
        SelfCache {
            layers: (0..self.blocks.len())
                .map(|_| {
                    (
                        HeadRows::new(self.heads, capacity, hd, f16),
                        HeadRows::new(self.heads, capacity, hd, f16),
                    )
                })
                .collect(),
            len: 0,
        }
    }

    /// `tokens` at the positions after the cache's, to their raw logits (tokens × vocab). Each
    /// token sees itself and every position before it. The cache grows by the tokens; `taps`,
    /// if given, receives the intermediate states.
    pub fn forward(
        &self,
        tokens: &[i32],
        cache: &mut SelfCache,
        cross: &CrossCache,
        mut taps: Option<&mut Taps>,
    ) -> Mat {
        let past = cache.len;
        let n = tokens.len();
        let d = self.d_model;
        let mut token_emb = Mat::zeros(n, d);
        for (i, &id) in tokens.iter().enumerate() {
            self.embedding_row(id as usize, token_emb.row_mut(i));
        }
        let pos_emb = Mat::from_vec(n, d, self.positions.data[past * d..(past + n) * d].to_vec());
        let mut x = token_emb.clone();
        x.add_assign(&pos_emb);
        if let Some(t) = taps.as_deref_mut() {
            t.token_emb = Some(token_emb);
            t.pos_emb = Some(pos_emb);
            t.embed_sum = Some(x.clone());
        }
        for (block, (layer, (cross_k, cross_v))) in self
            .blocks
            .iter()
            .zip(cache.layers.iter_mut().zip(&cross.layers))
        {
            let y = block.norm_self.apply(&x);
            let q = linear(&block.self_q, &y);
            layer.0.write(past, &linear(&block.self_k, &y));
            layer.1.write(past, &linear(&block.self_v, &y));
            let attended = attend(&q, &layer.0, &layer.1, past + n, |i| past + i + 1);
            x.add_assign(&linear(&block.self_out, &attended));

            let q = linear(&block.cross_q, &block.norm_cross.apply(&x));
            let attended = attend(&q, cross_k, cross_v, cross.frames, |_| cross.frames);
            x.add_assign(&linear(&block.cross_out, &attended));

            let mut h = linear(&block.fc1, &block.norm_ffn.apply(&x));
            h.map(gelu);
            x.add_assign(&linear(&block.fc2, &h));
            if let Some(t) = taps.as_deref_mut() {
                t.blocks.push(x.clone());
            }
        }
        cache.len = past + n;
        let x = self.final_norm.apply(&x);
        let logits = self.head(&x);
        if let Some(t) = taps {
            t.before_head = Some(x);
        }
        logits
    }

    /// Row `id` of the token embedding, exactly.
    fn embedding_row(&self, id: usize, out: &mut [f32]) {
        let d = self.d_model;
        match &self.embedding {
            Weight::F32(w) => out.copy_from_slice(&w[id * d..(id + 1) * d]),
            Weight::Q8(blocks) => expand(&blocks[id * d / 32..(id + 1) * d / 32], out),
        }
    }

    /// The tied output head: each row of `x` against every token's embedding.
    fn head(&self, x: &Mat) -> Mat {
        let rounded;
        let x = if super::ops::ggml_rounding() && matches!(self.embedding, Weight::Q8(_)) {
            rounded = super::ops::to_q8(x);
            &rounded
        } else {
            x
        };
        let inputs: Vec<&[f32]> = (0..x.rows).map(|t| x.row(t)).collect();
        let d = self.d_model;
        let tasks = self.vocab.div_ceil(VOCAB_PER_TASK);
        let blocks: Vec<Vec<f32>> = (0..tasks)
            .into_par_iter()
            .map(|task| {
                let range = task * VOCAB_PER_TASK..((task + 1) * VOCAB_PER_TASK).min(self.vocab);
                let mut block = vec![0f32; x.rows * range.len()];
                let rows: Rows = self.embedding.rows(range, d);
                products(&inputs, rows, d, &mut block);
                block
            })
            .collect();
        let mut out = Mat::zeros(x.rows, self.vocab);
        for (task, block) in blocks.iter().enumerate() {
            let start = task * VOCAB_PER_TASK;
            let width = block.len() / x.rows.max(1);
            for t in 0..x.rows {
                out.row_mut(t)[start..start + width]
                    .copy_from_slice(&block[t * width..(t + 1) * width]);
            }
        }
        out
    }
}
