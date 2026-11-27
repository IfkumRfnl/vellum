//! The 8× subsampling in front of the conformer (NeMo's `DwStridingSubsampling`,
//! `enc.pre_encode.*`): three stride-2 3×3 convolutions over (time, frequency) — a full one
//! from the single mel channel to 256, then two depthwise ones each followed by a pointwise
//! mix — with ReLUs, then every frame's 256 channels × 16 frequencies flattened and projected
//! to `d_model`.
//!
//! Activations are kept channel-last, one row per (time, frequency) point, so the depthwise
//! convolutions vectorize over channels and the pointwise ones are row-by-row matrix products.

use super::super::HParams;
use super::rounding::{Layer, rounds_like_ggml, to_f16};
use crate::{
    Error, Result,
    cpu::{Mat, Rows, axpy, fma_into, products, relu, vector},
    gguf::Gguf,
};
use rayon::prelude::*;

pub struct Subsampling {
    channels: usize,
    /// conv.0: 1 → C, as `[tap][channel]` with tap = 3·Δt + Δf.
    conv0: Vec<f32>,
    conv0_bias: Vec<f32>,
    /// conv.2 and conv.5: depthwise, `[tap][channel]`.
    depthwise: [(Vec<f32>, Vec<f32>); 2],
    /// conv.3 and conv.6: pointwise C → C, `[out][in]`.
    pointwise: [(Vec<f32>, Vec<f32>); 2],
    out: Layer,
}

/// An activation of `channels` per point on a `time × freq` grid, channel-last.
struct Grid {
    time: usize,
    freq: usize,
    data: Mat,
}

impl Subsampling {
    pub fn load(file: &Gguf, hparams: &HParams) -> Result<Subsampling> {
        let c = hparams.subsampling_channels;
        let p = "enc.pre_encode";
        // A 3×3 kernel per channel, stored (ggml [3, 3, 1, C]) as channel, time tap, freq tap.
        let taps = |name: &str| -> Result<Vec<f32>> {
            let w = vector(file, &format!("{p}.{name}.weight"))?;
            if w.len() != 9 * c {
                return Err(Error::Gguf(format!(
                    "{p}.{name}.weight: {} values",
                    w.len()
                )));
            }
            Ok((0..9)
                .flat_map(|tap| (0..c).map(move |ch| (tap, ch)))
                .map(|(tap, ch)| w[ch * 9 + tap])
                .collect())
        };
        let pointwise = |name: &str| -> Result<(Vec<f32>, Vec<f32>)> {
            let w = vector(file, &format!("{p}.{name}.weight"))?;
            if w.len() != c * c {
                return Err(Error::Gguf(format!(
                    "{p}.{name}.weight: {} values",
                    w.len()
                )));
            }
            // ggml rounds the kernel to f16 with its im2col; so do products that round
            // like it, once here.
            let w = if rounds_like_ggml() {
                to_f16(&Mat::from_vec(c, c, w)).data
            } else {
                w
            };
            Ok((w, vector(file, &format!("{p}.{name}.bias"))?))
        };
        Ok(Subsampling {
            channels: c,
            conv0: taps("conv.0")?,
            conv0_bias: vector(file, &format!("{p}.conv.0.bias"))?,
            depthwise: [
                (taps("conv.2")?, vector(file, &format!("{p}.conv.2.bias"))?),
                (taps("conv.5")?, vector(file, &format!("{p}.conv.5.bias"))?),
            ],
            pointwise: [pointwise("conv.3")?, pointwise("conv.6")?],
            out: Layer::load(file, &format!("{p}.out"))?,
        })
    }

    /// Mel features (`num_mels` × frames) to ⌈frames / 8⌉ × `d_model` (each stride-2 stage
    /// keeps ⌈n / 2⌉ of n).
    pub fn forward(&self, mel: &Mat) -> Mat {
        let c = self.channels;
        // conv.0 sees the mel as one channel on a (time, freq) grid.
        let input = Grid {
            time: mel.cols,
            freq: mel.rows,
            data: Mat::from_vec(mel.rows * mel.cols, 1, mel.transpose().data),
        };
        let mut x = conv3x3_stride2(&input, &self.conv0, &self.conv0_bias);
        x.data.map(relu);
        for ((dw, dw_bias), (pw, pw_bias)) in self.depthwise.iter().zip(&self.pointwise) {
            x = conv3x3_stride2(&x, dw, dw_bias);
            x.data = if rounds_like_ggml() {
                pointwise_conv(&to_f16(&x.data), pw, pw_bias)
            } else {
                pointwise_conv(&x.data, pw, pw_bias)
            };
            x.data.map(relu);
        }
        // Each frame's features in the order the projection was trained on: channel-major,
        // frequency-minor.
        let mut flat = Mat::zeros(x.time, c * x.freq);
        flat.data
            .par_chunks_mut(c * x.freq)
            .enumerate()
            .for_each(|(t, row)| {
                for f in 0..x.freq {
                    let point = x.data.row(t * x.freq + f);
                    for ch in 0..c {
                        row[ch * x.freq + f] = point[ch];
                    }
                }
            });
        self.out.forward(&flat)
    }
}

/// A 3×3 convolution with stride 2 and one row/column of zero padding on every side, either
/// full from a single input channel or depthwise (each channel its own kernel); `weights` is
/// `[tap][channel]` with tap = 3·Δt + Δf.
fn conv3x3_stride2(x: &Grid, weights: &[f32], bias: &[f32]) -> Grid {
    let channels = bias.len();
    let single_input = x.data.cols == 1;
    let time = (x.time - 1) / 2 + 1;
    let freq = (x.freq - 1) / 2 + 1;
    let mut out = Mat::zeros(time * freq, channels);
    out.data
        .par_chunks_mut(freq * channels)
        .enumerate()
        .for_each(|(t, rows)| {
            for (f, y) in rows.chunks_exact_mut(channels).enumerate() {
                y.copy_from_slice(bias);
                for dt in 0..3 {
                    let Some(ti) = (2 * t + dt).checked_sub(1).filter(|&ti| ti < x.time) else {
                        continue;
                    };
                    for df in 0..3 {
                        let Some(fi) = (2 * f + df).checked_sub(1).filter(|&fi| fi < x.freq) else {
                            continue;
                        };
                        let w = &weights[(dt * 3 + df) * channels..][..channels];
                        let point = x.data.row(ti * x.freq + fi);
                        if single_input {
                            axpy(y, point[0], w);
                        } else {
                            fma_into(y, w, point);
                        }
                    }
                }
            }
        });
    Grid {
        time,
        freq,
        data: out,
    }
}

/// A 1×1 convolution: every point's channels through an `[out][in]` matrix, plus bias. Each
/// output is exactly [`dot`](crate::cpu::dot) of its point and the weight row ([`products`]
/// tiles the work, not the sums).
fn pointwise_conv(x: &Mat, weight: &[f32], bias: &[f32]) -> Mat {
    let channels = bias.len();
    let mut out = Mat::zeros(x.rows, channels);
    out.data
        .par_chunks_mut(channels * 64)
        .zip(x.data.par_chunks(x.cols * 64))
        .for_each(|(ys, xs)| {
            let points: Vec<&[f32]> = xs.chunks_exact(x.cols).collect();
            products(&points, Rows::F32(weight), x.cols, ys);
            for y in ys.chunks_exact_mut(channels) {
                for (y, b) in y.iter_mut().zip(bias) {
                    *y += b;
                }
            }
        });
    out
}
