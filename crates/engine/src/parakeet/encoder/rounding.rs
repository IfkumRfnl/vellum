//! The encoder's linear layers. By default they are the CPU's int8 products
//! ([`crate::cpu::int8::Dense`]): Q8_0 weights times the input quantized to Q8_0 blocks, F16
//! weights times the input rounded to f16, as the reference's CPU path multiplies them, and
//! bit for bit the same sums as the f32 experiment below.
//!
//! `VELLUM_CPU_GEMM=f32` makes them the f32 oracle: every layer multiplies the exact f32
//! activations by the exactly dequantized weights. With it, `VELLUM_GGML_ROUNDING=1` rounds
//! each layer's input the way ggml's CPU matmuls do before multiplying — to Q8_0 blocks (an f16
//! scale per 32 values, `round(x / scale)` as int8) against Q8_0 weights, to f16 against F16
//! weights — in plain scalar code, the int8 kernels' reference. In the subsampling's pointwise
//! convolutions both the input and the f32 kernel are rounded to f16 (`ggml_conv_2d` always
//! builds an f16 im2col) under either the int8 products or the experiment. The experiment
//! exists to show where our remaining difference from the reference comes from: with it on and
//! the reference's flash attention off (`TRANSCRIBE_NO_FLASH=1`), a block fed the reference's
//! input matches it to float rounding up to the first matmul whose input already differs.

use crate::{
    Result,
    cpu::{
        Linear, Mat, Q8Block, Weight,
        int8::{Dense, int8_enabled},
        tensor,
    },
    gguf::{Dtype, Gguf},
};
use half::f16;
use rayon::prelude::*;
use std::sync::OnceLock;

/// What happens to a layer's input before the product.
#[derive(Clone, Copy, PartialEq)]
pub enum Rounding {
    Exact,
    Q8,
    F16,
}

/// A linear layer `y = W·x + b`: the int8 products, or the f32 oracle with its rounding.
pub enum Layer {
    Fast(Dense),
    Oracle { linear: Linear, rounding: Rounding },
}

/// Whether `VELLUM_GGML_ROUNDING=1` is set (it only changes the f32 oracle).
pub fn ggml_rounding() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VELLUM_GGML_ROUNDING").is_ok_and(|v| v == "1"))
}

/// Whether products round their inputs as ggml's CPU matmuls do: the int8 products always
/// do, the f32 oracle under `VELLUM_GGML_ROUNDING=1`.
pub fn rounds_like_ggml() -> bool {
    int8_enabled() || ggml_rounding()
}

impl Layer {
    pub fn load(file: &Gguf, name: &str) -> Result<Layer> {
        if int8_enabled() {
            return Ok(Layer::Fast(Dense::load(file, name)?));
        }
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
        Ok(Layer::Oracle { linear, rounding })
    }

    /// Output features.
    pub fn rows(&self) -> usize {
        match self {
            Layer::Fast(d) => d.rows(),
            Layer::Oracle { linear, .. } => linear.rows,
        }
    }

    pub fn forward(&self, x: &Mat) -> Mat {
        let (linear, rounding) = match self {
            Layer::Fast(d) => return d.forward(x),
            Layer::Oracle { linear, rounding } => (linear, *rounding),
        };
        match (rounding, &linear.weight) {
            (Rounding::Q8, Weight::Q8(blocks)) => forward_q8(linear, x, blocks),
            (Rounding::F16, _) => linear.forward(&to_f16(x)),
            _ => linear.forward(x),
        }
    }
}

/// ggml's `quantize_row_q8_0` (as its AVX2 build rounds: ties to even) and
/// `vec_dot_q8_0_q8_0`: per block, the two f16 scales' product times the exact integer dot.
fn forward_q8(linear: &Linear, x: &Mat, weight: &[Q8Block]) -> Mat {
    let per_row = linear.cols / 32;
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
    let mut out = Mat::zeros(x.rows, linear.rows);
    out.data
        .par_chunks_mut(linear.rows)
        .zip(xq.par_chunks(per_row))
        .for_each(|(y, xb)| {
            for (o, y) in y.iter_mut().enumerate() {
                let wb = &weight[o * per_row..][..per_row];
                let mut acc = 0f32;
                for (w, (dx, qx)) in wb.iter().zip(xb) {
                    let isum: i32 = w.q.iter().zip(qx).map(|(a, b)| *a as i32 * *b as i32).sum();
                    acc = (w.scale * dx).mul_add(isum as f32, acc);
                }
                *y = acc + linear.bias.as_ref().map_or(0., |b| b[o]);
            }
        });
    out
}

/// `x` with every value rounded to the nearest f16.
pub fn to_f16(x: &Mat) -> Mat {
    let mut x = x.clone();
    x.map(|v| f16::from_f32(v).to_f32());
    x
}
