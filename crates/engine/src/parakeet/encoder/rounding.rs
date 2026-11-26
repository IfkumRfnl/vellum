//! The encoder's linear layers, with an experiment switch: `VELLUM_GGML_ROUNDING=1` rounds each
//! layer's input the way ggml's CPU matmuls do before multiplying — to Q8_0 blocks (an f16
//! scale per 32 values, `round(x / scale)` as int8) against Q8_0 weights, to f16 against F16
//! weights — and, in the subsampling's pointwise convolutions, both the input and the f32
//! kernel to f16 (`ggml_conv_2d` always builds an f16 im2col). Off (the default), every layer
//! multiplies the exact f32 activations by the exactly dequantized weights. The switch exists to
//! show where our remaining difference from the reference comes from: with it on and the
//! reference's flash attention off (`TRANSCRIBE_NO_FLASH=1`), a block fed the reference's input
//! matches it to float rounding up to the first matmul whose input already differs.

use crate::{
    Result,
    cpu::{Linear, Mat, Q8Block, Weight, tensor},
    gguf::{Dtype, Gguf},
};
use half::f16;
use rayon::prelude::*;
use std::sync::OnceLock;

/// What happens to a layer's input before the product.
#[derive(Clone, Copy, PartialEq)]
enum Rounding {
    Exact,
    Q8,
    F16,
}

/// A linear layer `y = W·x + b` (see [`Linear`]).
pub struct Layer {
    linear: Linear,
    rounding: Rounding,
}

/// Whether `VELLUM_GGML_ROUNDING=1` is set.
pub fn ggml_rounding() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VELLUM_GGML_ROUNDING").is_ok_and(|v| v == "1"))
}

impl Layer {
    pub fn load(file: &Gguf, name: &str) -> Result<Layer> {
        let linear = Linear::load(file, name)?;
        let rounding = match (
            ggml_rounding(),
            tensor(file, &format!("{name}.weight"))?.dtype,
        ) {
            (false, _) => Rounding::Exact,
            (true, Dtype::Q8_0) => Rounding::Q8,
            (true, Dtype::F16) => Rounding::F16,
            (true, _) => Rounding::Exact,
        };
        Ok(Layer { linear, rounding })
    }

    /// Output features.
    pub fn rows(&self) -> usize {
        self.linear.rows
    }

    pub fn forward(&self, x: &Mat) -> Mat {
        match (self.rounding, &self.linear.weight) {
            (Rounding::Q8, Weight::Q8(blocks)) => self.forward_q8(x, blocks),
            (Rounding::F16, _) => self.linear.forward(&to_f16(x)),
            _ => self.linear.forward(x),
        }
    }

    /// ggml's `quantize_row_q8_0` (as its AVX2 build rounds: ties to even) and
    /// `vec_dot_q8_0_q8_0`: per block, the two f16 scales' product times the exact integer dot.
    fn forward_q8(&self, x: &Mat, weight: &[Q8Block]) -> Mat {
        let per_row = self.linear.cols / 32;
        let xq: Vec<(f32, [i8; 32])> = x
            .data
            .par_chunks(32)
            .map(|v| {
                let amax = v.iter().fold(0f32, |m, v| m.max(v.abs()));
                let id = if amax != 0. { 127. / amax } else { 0. };
                let q = std::array::from_fn(|i| (v[i] * id).round_ties_even() as i8);
                (f16::from_f32(amax / 127.).to_f32(), q)
            })
            .collect();
        let mut out = Mat::zeros(x.rows, self.linear.rows);
        out.data
            .par_chunks_mut(self.linear.rows)
            .zip(xq.par_chunks(per_row))
            .for_each(|(y, xb)| {
                for (o, y) in y.iter_mut().enumerate() {
                    let wb = &weight[o * per_row..][..per_row];
                    let mut acc = 0f32;
                    for (w, (dx, qx)) in wb.iter().zip(xb) {
                        let isum: i32 =
                            w.q.iter().zip(qx).map(|(a, b)| *a as i32 * *b as i32).sum();
                        acc = (w.scale * dx).mul_add(isum as f32, acc);
                    }
                    *y = acc + self.linear.bias.as_ref().map_or(0., |b| b[o]);
                }
            });
        out
    }
}

/// `x` with every value rounded to the nearest f16.
pub fn to_f16(x: &Mat) -> Mat {
    let mut x = x.clone();
    x.map(|v| f16::from_f32(v).to_f32());
    x
}
