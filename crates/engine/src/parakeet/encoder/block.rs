//! One conformer block (`enc.blocks.<L>.*`): a half-step feed-forward, relative-position
//! self-attention, the convolution module, another half-step feed-forward, each on a
//! layer-normed copy of the residual stream and added back to it, then a final layer norm.

use super::super::HParams;
use super::attention::SelfAttention;
use super::rounding::{Layer, rounds_like_ggml};
use crate::{
    Error, Result, Trace,
    cpu::{Mat, fma_into, layer_norm, sigmoid, silu, silu_like_ggml, vector},
    gguf::Gguf,
    trace::checkpoint,
};
use rayon::prelude::*;

/// The epsilon of every layer norm and of the batch norm (`kLayerNormEps`, `kBnEps`).
const EPS: f32 = 1e-5;

pub struct Block {
    norm_ff1: Norm,
    ff1: FeedForward,
    norm_attn: Norm,
    attn: SelfAttention,
    norm_conv: Norm,
    conv: ConvModule,
    norm_ff2: Norm,
    ff2: FeedForward,
    norm_out: Norm,
}

impl Block {
    pub fn load(file: &Gguf, name: &str, hparams: &HParams, kv_f16: bool) -> Result<Block> {
        Ok(Block {
            norm_ff1: Norm::load(file, &format!("{name}.norm_ff1"))?,
            ff1: FeedForward::load(file, &format!("{name}.ff1"))?,
            norm_attn: Norm::load(file, &format!("{name}.norm_attn"))?,
            attn: SelfAttention::load(file, &format!("{name}.attn"), hparams.n_heads, kv_f16)?,
            norm_conv: Norm::load(file, &format!("{name}.norm_conv"))?,
            conv: ConvModule::load(file, &format!("{name}.conv"), hparams.conv_kernel)?,
            norm_ff2: Norm::load(file, &format!("{name}.norm_ff2"))?,
            ff2: FeedForward::load(file, &format!("{name}.ff2"))?,
            norm_out: Norm::load(file, &format!("{name}.norm_out"))?,
        })
    }

    /// Block `l` on the residual stream `x` (frames × d_model), with the positional encodings
    /// of every relative offset. Checkpoints the stream after each residual add
    /// (`enc.block.<l>.ff1`, `.attn`, `.conv`, `.ff2`) and the normed output (`.out`).
    pub fn forward(&self, mut x: Mat, pos: &Mat, l: usize, trace: &mut dyn Trace) -> Mat {
        x.add_scaled(&self.ff1.forward(&self.norm_ff1.apply(&x)), 0.5);
        checkpoint(trace, &format!("enc.block.{l}.ff1"), &mut x);
        x.add_assign(&self.attn.forward(&self.norm_attn.apply(&x), pos));
        checkpoint(trace, &format!("enc.block.{l}.attn"), &mut x);
        x.add_assign(&self.conv.forward(&self.norm_conv.apply(&x)));
        checkpoint(trace, &format!("enc.block.{l}.conv"), &mut x);
        x.add_scaled(&self.ff2.forward(&self.norm_ff2.apply(&x)), 0.5);
        checkpoint(trace, &format!("enc.block.{l}.ff2"), &mut x);
        let mut out = self.norm_out.apply(&x);
        checkpoint(trace, &format!("enc.block.{l}.out"), &mut out);
        out
    }
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

/// `linear2(SiLU(linear1(x)))`, d_model → d_ff → d_model.
struct FeedForward {
    linear1: Layer,
    linear2: Layer,
}

impl FeedForward {
    fn load(file: &Gguf, name: &str) -> Result<FeedForward> {
        Ok(FeedForward {
            linear1: Layer::load(file, &format!("{name}.linear1"))?,
            linear2: Layer::load(file, &format!("{name}.linear2"))?,
        })
    }

    fn forward(&self, x: &Mat) -> Mat {
        let mut h = self.linear1.forward(x);
        swish(&mut h.data, h.cols);
        self.linear2.forward(&h)
    }
}

/// The convolution module: pointwise to 2·d_model, GLU, a depthwise convolution over time
/// (centred, zero-padded), batch norm with its running statistics, SiLU, pointwise back.
struct ConvModule {
    pointwise1: Layer,
    /// The depthwise kernel as `[tap][channel]`.
    depthwise: Vec<f32>,
    depthwise_bias: Vec<f32>,
    kernel: usize,
    /// Batch norm folded into a per-channel affine map, as the reference folds it at load:
    /// `scale = γ / √(var + ε)`, `shift = β − mean · scale`.
    bn_scale: Vec<f32>,
    bn_shift: Vec<f32>,
    pointwise2: Layer,
}

impl ConvModule {
    fn load(file: &Gguf, name: &str, kernel: usize) -> Result<ConvModule> {
        let w = vector(file, &format!("{name}.depthwise.weight"))?;
        let depthwise_bias = vector(file, &format!("{name}.depthwise.bias"))?;
        let channels = depthwise_bias.len();
        if kernel.is_multiple_of(2) || w.len() != kernel * channels {
            return Err(Error::Gguf(format!(
                "{name}.depthwise.weight: {} values for a kernel of {kernel}",
                w.len()
            )));
        }
        let depthwise = (0..kernel)
            .flat_map(|k| (0..channels).map(move |c| (k, c)))
            .map(|(k, c)| w[c * kernel + k])
            .collect();
        let gamma = vector(file, &format!("{name}.bn.weight"))?;
        let beta = vector(file, &format!("{name}.bn.bias"))?;
        let mean = vector(file, &format!("{name}.bn.running_mean"))?;
        let var = vector(file, &format!("{name}.bn.running_var"))?;
        let bn_scale: Vec<f32> = gamma
            .iter()
            .zip(&var)
            .map(|(g, v)| g / (v + EPS).sqrt())
            .collect();
        let bn_shift = beta
            .iter()
            .zip(&mean)
            .zip(&bn_scale)
            .map(|((b, m), s)| b - m * s)
            .collect();
        Ok(ConvModule {
            pointwise1: Layer::load(file, &format!("{name}.pointwise1"))?,
            depthwise,
            depthwise_bias,
            kernel,
            bn_scale,
            bn_shift,
            pointwise2: Layer::load(file, &format!("{name}.pointwise2"))?,
        })
    }

    fn forward(&self, x: &Mat) -> Mat {
        let channels = self.depthwise_bias.len();
        let h = self.pointwise1.forward(x);
        // GLU: the first half gated by the sigmoid of the second.
        let mut glu = Mat::zeros(x.rows, channels);
        glu.data
            .par_chunks_mut(channels)
            .zip(h.data.par_chunks(2 * channels))
            .for_each(|(y, h)| {
                let (a, b) = h.split_at(channels);
                for ((y, a), b) in y.iter_mut().zip(a).zip(b) {
                    *y = a * sigmoid(*b);
                }
            });
        let pad = self.kernel / 2;
        let mut conv = Mat::zeros(x.rows, channels);
        conv.data
            .par_chunks_mut(channels)
            .enumerate()
            .for_each(|(t, y)| {
                y.copy_from_slice(&self.depthwise_bias);
                for k in 0..self.kernel {
                    let Some(ti) = (t + k).checked_sub(pad).filter(|&ti| ti < glu.rows) else {
                        continue;
                    };
                    fma_into(y, &self.depthwise[k * channels..][..channels], glu.row(ti));
                }
                for ((y, s), b) in y.iter_mut().zip(&self.bn_scale).zip(&self.bn_shift) {
                    *y = *y * s + b;
                }
                swish_row(y);
            });
        self.pointwise2.forward(&conv)
    }
}

/// SiLU over rows of `width` values in place: as ggml computes it when the products round like
/// ggml's (the int8 path), else `x · σ(x)` exactly.
fn swish(x: &mut [f32], width: usize) {
    if width == 0 {
        return;
    }
    x.par_chunks_mut(width).for_each(swish_row);
}

/// [`swish`] on one row.
fn swish_row(row: &mut [f32]) {
    if rounds_like_ggml() {
        silu_like_ggml(row);
    } else {
        row.iter_mut().for_each(|v| *v = silu(*v));
    }
}
