//! What Sortformer adds after the FastConformer: a projection to the transformer's width
//! (`diar.encoder_proj`), a post-LN Transformer encoder (`tf.blocks.<L>.*`: full self-attention
//! without positional encodings, a ReLU feed-forward, each added back and then layer-normed),
//! and the speaker head (`diar.fc1`, `diar.single_spk_head`): ReLU, linear, ReLU, linear,
//! sigmoid, one probability per speaker per frame. As transcribe.cpp builds it
//! (`tf_block` and the diar head in `arch/sortformer/model.cpp`).

use crate::{
    Error, Result,
    cpu::{Mat, dot, layer_norm, relu, sigmoid, vector},
    gguf::Gguf,
    parakeet::encoder::rounding::Layer,
};
use rayon::prelude::*;

/// Every layer norm's epsilon.
const EPS: f32 = 1e-5;

pub struct Transformer {
    heads: usize,
    proj: Layer,
    blocks: Vec<Block>,
    fc1: Layer,
    head: Layer,
}

struct Block {
    q: Layer,
    k: Layer,
    v: Layer,
    out: Layer,
    norm1: Norm,
    ff_in: Layer,
    ff_out: Layer,
    norm2: Norm,
}

struct Norm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Norm {
    fn load(file: &Gguf, name: &str) -> Result<Norm> {
        Ok(Norm {
            weight: vector(file, &format!("{name}.weight"))?,
            bias: vector(file, &format!("{name}.bias"))?,
        })
    }

    fn apply(&self, x: &Mat) -> Mat {
        layer_norm(x, &self.weight, &self.bias, EPS)
    }
}

impl Transformer {
    pub fn load(file: &Gguf, layers: usize, heads: usize) -> Result<Transformer> {
        let blocks = (0..layers)
            .map(|l| {
                let name = |part: &str| format!("tf.blocks.{l}.{part}");
                Ok(Block {
                    q: Layer::load(file, &name("attn.q"))?,
                    k: Layer::load(file, &name("attn.k"))?,
                    v: Layer::load(file, &name("attn.v"))?,
                    out: Layer::load(file, &name("attn.out"))?,
                    norm1: Norm::load(file, &name("norm_1"))?,
                    ff_in: Layer::load(file, &name("ff.in"))?,
                    ff_out: Layer::load(file, &name("ff.out"))?,
                    norm2: Norm::load(file, &name("norm_2"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let proj = Layer::load(file, "diar.encoder_proj")?;
        if heads == 0 || !proj.rows().is_multiple_of(heads) {
            return Err(Error::Gguf(format!(
                "{heads} heads don't fit a transformer of {}",
                proj.rows()
            )));
        }
        Ok(Transformer {
            heads,
            proj,
            blocks,
            fc1: Layer::load(file, "diar.fc1")?,
            head: Layer::load(file, "diar.single_spk_head")?,
        })
    }

    /// The FastConformer's output (frames × its width) projected to the transformer's.
    pub fn project(&self, x: &Mat) -> Mat {
        self.proj.forward(x)
    }

    /// Every transformer block on the projected frames.
    pub fn encode(&self, mut x: Mat) -> Mat {
        for block in &self.blocks {
            x = block.forward(x, self.heads);
        }
        x
    }

    /// The transformer's output to speaker probabilities: frames × speakers.
    pub fn speakers(&self, x: &Mat) -> Mat {
        let mut h = x.clone();
        h.map(relu);
        let mut h = self.fc1.forward(&h);
        h.map(relu);
        let mut p = self.head.forward(&h);
        p.map(sigmoid);
        p
    }
}

impl Block {
    fn forward(&self, x: Mat, heads: usize) -> Mat {
        let mut a = self.attention(&x, heads);
        a.add_assign(&x);
        let x = self.norm1.apply(&a);
        let mut h = self.ff_in.forward(&x);
        h.map(relu);
        let mut f = self.ff_out.forward(&h);
        f.add_assign(&x);
        self.norm2.apply(&f)
    }

    /// Full self-attention. NeMo scales queries and keys each by `d_head^(-1/4)` before
    /// their product (so the scores are divided by √d_head), as the reference does too.
    fn attention(&self, x: &Mat, heads: usize) -> Mat {
        let frames = x.rows;
        let d = self.q.rows();
        let dh = d / heads;
        let inv = 1. / (dh as f32).sqrt().sqrt();
        let scaled = |layer: &Layer| {
            let mut y = by_head(&layer.forward(x), heads);
            for v in &mut y {
                *v *= inv;
            }
            y
        };
        let q = scaled(&self.q);
        let k = scaled(&self.k);
        let v = by_head(&self.v.forward(x), heads);
        let mut context = Mat::zeros(frames, d);
        context.data.par_chunks_mut(d).enumerate().for_each_init(
            || vec![0f32; frames],
            |scores, (i, out)| {
                for h in 0..heads {
                    let qi = &q[(h * frames + i) * dh..][..dh];
                    let keys = &k[h * frames * dh..][..frames * dh];
                    let values = &v[h * frames * dh..][..frames * dh];
                    let mut max = f32::NEG_INFINITY;
                    for (s, key) in scores.iter_mut().zip(keys.chunks_exact(dh)) {
                        *s = dot(qi, key);
                        max = max.max(*s);
                    }
                    let mut sum = 0f64;
                    for s in scores.iter_mut() {
                        *s = (*s - max).exp();
                        sum += *s as f64;
                    }
                    let inv_sum = (1. / sum) as f32;
                    let o = &mut out[h * dh..][..dh];
                    for (s, value) in scores.iter().zip(values.chunks_exact(dh)) {
                        let a = s * inv_sum;
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

/// A frames × (heads · d_head) matrix regrouped head by head: `[head][frame][d_head]`.
fn by_head(x: &Mat, heads: usize) -> Vec<f32> {
    let dh = x.cols / heads;
    let mut out = vec![0f32; x.data.len()];
    for (h, head) in out.chunks_exact_mut(x.rows * dh).enumerate() {
        for (t, row) in head.chunks_exact_mut(dh).enumerate() {
            row.copy_from_slice(&x.row(t)[h * dh..][..dh]);
        }
    }
    out
}
