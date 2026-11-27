//! The CPU path: plain row-major f32 matrices and the operations the models are made of. It is
//! the engine's oracle — exact where the reference is exact, accumulating carefully where it
//! isn't — and the fallback where there's no GPU.
//!
//! Weights stay as the GGUF file stores them (Q8_0 blocks, F16, F32) and are read in place:
//! a Q8_0 value is its block's f16 scale times an int8, which f32 represents exactly, so
//! dequantizing loses nothing.

use crate::{
    Error, Result,
    gguf::{Dtype, Gguf, TensorInfo},
};
use rayon::prelude::*;

pub mod flash;
pub mod int8;

/// A row-major matrix: `rows` vectors of `cols` values (time steps × features, usually).
#[derive(Debug, Clone, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Mat {
        Mat {
            rows,
            cols,
            data: vec![0.; rows * cols],
        }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Mat {
        assert_eq!(data.len(), rows * cols, "matrix of {rows}×{cols}");
        Mat { rows, cols, data }
    }

    pub fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// The shape as the trace names it: slowest dimension first.
    pub fn shape(&self) -> [usize; 2] {
        [self.rows, self.cols]
    }

    pub fn transpose(&self) -> Mat {
        let mut out = Mat::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.data[c * self.rows + r] = self.data[r * self.cols + c];
            }
        }
        out
    }

    /// `self += other`, element by element.
    pub fn add_assign(&mut self, other: &Mat) {
        assert_eq!(self.shape(), other.shape());
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += b;
        }
    }

    /// `self += scale · other`, element by element.
    pub fn add_scaled(&mut self, other: &Mat, scale: f32) {
        assert_eq!(self.shape(), other.shape());
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += scale * b;
        }
    }

    pub fn map(&mut self, f: impl Fn(f32) -> f32 + Sync) {
        self.data.par_iter_mut().for_each(|v| *v = f(*v));
    }
}

/// A weight matrix of `rows` output vectors of `cols` inputs each (ggml's `ne[1]` × `ne[0]`),
/// in the form the file keeps it.
pub enum Weight {
    F32(Vec<f32>),
    /// Q8_0: per 32 inputs, an f16 scale and 32 int8 values.
    Q8(Vec<Q8Block>),
}

#[derive(Clone, Copy)]
pub struct Q8Block {
    pub scale: f32,
    pub q: [i8; 32],
}

/// A linear layer's weight and its optional bias, `y = W·x + b`.
pub struct Linear {
    pub rows: usize,
    pub cols: usize,
    pub weight: Weight,
    pub bias: Option<Vec<f32>>,
}

impl Linear {
    /// `name.weight` (and `name.bias` when the file has one). Any tensor whose first
    /// dimension is the input and whose other dimensions multiply to the outputs works,
    /// so pointwise convolutions (`[1, in, out]`) load as linear layers too.
    pub fn load(file: &Gguf, name: &str) -> Result<Linear> {
        let info = tensor(file, &format!("{name}.weight"))?;
        let mut dims: Vec<usize> = info.dims.iter().map(|d| *d as usize).collect();
        // A leading kernel dimension of one (pointwise convolution) isn't an input.
        while dims.len() > 2 && dims[0] == 1 {
            dims.remove(0);
        }
        let cols = dims[0];
        let rows: usize = dims[1..].iter().product();
        let weight = match info.dtype {
            Dtype::Q8_0 if cols.is_multiple_of(32) => {
                let data = file.data(info);
                Weight::Q8(
                    data.as_chunks::<34>()
                        .0
                        .iter()
                        .map(|b| Q8Block {
                            scale: half::f16::from_le_bytes([b[0], b[1]]).to_f32(),
                            q: std::array::from_fn(|i| b[2 + i] as i8),
                        })
                        .collect(),
                )
            }
            _ => Weight::F32(file.dequantize(info)?),
        };
        let bias = match file.tensor(&format!("{name}.bias")) {
            Some(b) => Some(file.dequantize(b)?),
            None => None,
        };
        if let Some(b) = &bias
            && b.len() != rows
        {
            return Err(Error::Gguf(format!(
                "{name}.bias has {} values for {rows} outputs",
                b.len()
            )));
        }
        Ok(Linear {
            rows,
            cols,
            weight,
            bias,
        })
    }

    /// Output row `o` of the weight, as f32 (exact for every stored type).
    pub fn weight_row(&self, o: usize, out: &mut [f32]) {
        match &self.weight {
            Weight::F32(w) => out.copy_from_slice(&w[o * self.cols..(o + 1) * self.cols]),
            Weight::Q8(blocks) => {
                let per_row = self.cols / 32;
                expand(&blocks[o * per_row..(o + 1) * per_row], out);
            }
        }
    }

    /// `x · Wᵀ + b` for every row of `x`, each output exactly `dot(x_t, w_o) + b_o`.
    ///
    /// Blocked for the cache: the outputs are split into panels of [`PANEL`] weight rows, one
    /// task each, whose weights are expanded to f32 once and stay in L2 while every input row
    /// streams past them; inside a panel, tiles of 4 input rows × 3 weight rows share each load
    /// across 12 products (see [`products`]).
    pub fn forward(&self, x: &Mat) -> Mat {
        assert_eq!(x.cols, self.cols, "linear layer of {} inputs", self.cols);
        let mut out = Mat::zeros(x.rows, self.rows);
        if x.rows == 0 {
            return out;
        }
        let inputs: Vec<&[f32]> = (0..x.rows).map(|t| x.row(t)).collect();
        let target = Outputs(out.data.as_mut_ptr());
        let panels = self.rows.div_ceil(PANEL);
        (0..panels).into_par_iter().for_each_init(
            || vec![0f32; PANEL * self.cols],
            |panel, p| {
                let rows = p * PANEL..((p + 1) * PANEL).min(self.rows);
                let n = rows.len();
                let weights: &[f32] = match &self.weight {
                    Weight::F32(w) => &w[rows.start * self.cols..rows.end * self.cols],
                    Weight::Q8(_) => {
                        for (o, row) in rows.clone().zip(panel.chunks_exact_mut(self.cols)) {
                            self.weight_row(o, row);
                        }
                        &panel[..n * self.cols]
                    }
                };
                let mut block = vec![0f32; x.rows * n];
                products(&inputs, Rows::F32(weights), self.cols, &mut block);
                let bias = self.bias.as_deref();
                for (t, ys) in block.chunks_exact(n).enumerate() {
                    for (j, y) in ys.iter().enumerate() {
                        let o = rows.start + j;
                        let v = y + bias.map_or(0., |b| b[o]);
                        // SAFETY: panels cover disjoint output columns, and (t, o) is in bounds.
                        unsafe { target.write(t * self.rows + o, v) };
                    }
                }
            },
        );
        out
    }
}

/// Output rows per task in [`Linear::forward`]: 24 rows (eight tiles wide) of up to 4096 inputs
/// is 384 KiB of f32, which stays in a core's L2 while the inputs stream past.
const PANEL: usize = 24;

/// The output of [`Linear::forward`], written by many tasks at disjoint places.
struct Outputs(*mut f32);
// SAFETY: every task writes its own panel's columns only.
unsafe impl Sync for Outputs {}
unsafe impl Send for Outputs {}

impl Outputs {
    /// # Safety
    /// `at` is inside the output, and no other task writes it.
    unsafe fn write(&self, at: usize, v: f32) {
        // SAFETY: as the caller promises.
        unsafe { self.0.add(at).write(v) }
    }
}

/// A dot product: eight lanes, each summing every eighth product in order, then the lanes
/// summed in a fixed tree, then the tail past the last multiple of eight. The matrix products
/// here ([`products`], [`Linear::forward`]) compute each output exactly this way, so a result
/// never depends on how the work was tiled or batched.
///
/// With AVX2 and FMA (detected at run time) each lane is a fused multiply-add chain; without,
/// a multiply then an add (`mul_add` would be a library call per element there).
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    if simd::available() {
        // SAFETY: the CPU has the features, and the rows are the same length.
        return unsafe { simd::tile::<1, 1>([a], [b]) }[0][0];
    }
    portable_dot(a, b)
}

/// [`dot`] without SIMD intrinsics (the compiler still vectorizes it).
fn portable_dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0f32; 8];
    let (a8, a_rest) = a.as_chunks::<8>();
    let (b8, b_rest) = b.as_chunks::<8>();
    for (x, y) in a8.iter().zip(b8) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut tail = 0f32;
    for (x, y) in a_rest.iter().zip(b_rest) {
        tail += x * y;
    }
    ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7])) + tail
}

/// `y[k] = a · x[k] + y[k]`, each element one fused multiply-add (exactly `a.mul_add(x[k], y[k])`),
/// eight at a time with AVX2 and FMA (detected at run time; `mul_add` is a library call per
/// element otherwise).
pub fn axpy(y: &mut [f32], a: f32, x: &[f32]) {
    assert_eq!(y.len(), x.len());
    #[cfg(target_arch = "x86_64")]
    if simd::available() {
        // SAFETY: the CPU has the features, and the slices are the same length.
        unsafe { simd::axpy(y, a, x) };
        return;
    }
    for (y, x) in y.iter_mut().zip(x) {
        *y = a.mul_add(*x, *y);
    }
}

/// `y[k] = w[k] · x[k] + y[k]`, each element one fused multiply-add (exactly
/// `w[k].mul_add(x[k], y[k])`), as [`axpy`] computes its.
pub fn fma_into(y: &mut [f32], w: &[f32], x: &[f32]) {
    assert!(y.len() == x.len() && y.len() == w.len());
    #[cfg(target_arch = "x86_64")]
    if simd::available() {
        // SAFETY: the CPU has the features, and the slices are the same length.
        unsafe { simd::fma_into(y, w, x) };
        return;
    }
    for ((y, w), x) in y.iter_mut().zip(w).zip(x) {
        *y = w.mul_add(*x, *y);
    }
}

/// `acc[c] += Σ_r rows[r][c] · weights[r]` for each of the `acc.len()` columns of `rows` (one
/// row per weight, `acc.len()` values each), each column one fused multiply-add per row in row
/// order: how ggml's `simd_gemm` accumulates a matrix product (its flash attention's scores and
/// value sums). Eight columns at a time with AVX2 and FMA.
pub fn fma_columns(acc: &mut [f32], rows: &[f32], weights: &[f32]) {
    let width = acc.len();
    assert_eq!(rows.len(), width * weights.len());
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if simd::available() {
        // SAFETY: the CPU has the features, and the lengths were checked.
        done = unsafe { simd::fma_columns(acc, rows, weights) };
    }
    for (r, w) in weights.iter().enumerate() {
        let row = &rows[r * width..][..width];
        for c in done..width {
            acc[c] = row[c].mul_add(*w, acc[c]);
        }
    }
}

/// ggml's `ggml_vec_soft_max_f32`: `x[i] = e^(x[i] − max)` with its vector `expf` eight at a
/// time (the library's for a last few), returning their sum, accumulated in f64 eight by eight
/// as it does.
pub fn softmax_like_ggml(x: &mut [f32], max: f32) -> f64 {
    let mut sum = 0f64;
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if simd::available() {
        // SAFETY: the CPU has the features.
        (done, sum) = unsafe { simd::softmax_like_ggml(x, max) };
    }
    for v in &mut x[done..] {
        *v = (*v - max).exp();
        sum += *v as f64;
    }
    sum
}

/// Layer normalization over each row, with learned scale and shift.
pub fn layer_norm(x: &Mat, weight: &[f32], bias: &[f32], eps: f32) -> Mat {
    let mut out = x.clone();
    out.data.par_chunks_mut(x.cols).for_each(|row| {
        let n = row.len() as f64;
        let mean = row.iter().map(|v| *v as f64).sum::<f64>() / n;
        let var = row.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / n;
        let inv = 1. / (var + eps as f64).sqrt();
        for (i, v) in row.iter_mut().enumerate() {
            *v = ((*v as f64 - mean) * inv) as f32 * weight[i] + bias[i];
        }
    });
    out
}

/// Softmax over each row, in place.
pub fn softmax_rows(x: &mut Mat) {
    x.data.par_chunks_mut(x.cols).for_each(|row| {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if max == f32::NEG_INFINITY {
            row.fill(0.);
            return;
        }
        let mut sum = 0f64;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v as f64;
        }
        let inv = (1. / sum) as f32;
        for v in row.iter_mut() {
            *v *= inv;
        }
    });
}

pub fn relu(v: f32) -> f32 {
    v.max(0.)
}

pub fn sigmoid(v: f32) -> f32 {
    1. / (1. + (-v).exp())
}

pub fn silu(v: f32) -> f32 {
    v * sigmoid(v)
}

/// SiLU over one row in place, as ggml's CPU computes it (`ggml_vec_silu_f32`): `x / (1 + e^−x)`
/// with its own vector `expf` (`ggml_v_expf`, AVX2 and FMA, detected at run time) eight values
/// at a time, and the library's `expf` for the row's last few. Without AVX2 every value takes
/// the scalar form.
pub fn silu_like_ggml(row: &mut [f32]) {
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if simd::available() {
        // SAFETY: the CPU has the features.
        done = unsafe { simd::silu_like_ggml(row) };
    }
    for v in &mut row[done..] {
        *v /= 1. + (-*v).exp();
    }
}

/// GELU as Whisper uses it (the exact, erf-based form).
pub fn gelu(v: f32) -> f32 {
    0.5 * v * (1. + erf(v / std::f32::consts::SQRT_2))
}

/// GELU as ggml's CPU computes `ggml_gelu_erf` (`ggml_vec_gelu_erf_f32`):
/// `0.5 · x · (1 + erff(x · (1/√2)))` in f32, with the C library's `erff`, the function the
/// reference calls.
pub fn gelu_like_ggml(v: f32) -> f32 {
    unsafe extern "C" {
        safe fn erff(x: f32) -> f32;
    }
    0.5 * v * (1. + erff(v * std::f32::consts::FRAC_1_SQRT_2))
}

/// The error function, to about 1e-7 (Abramowitz–Stegun 7.1.26 is too coarse for parity, so
/// this uses a rational approximation evaluated in f64).
pub fn erf(x: f32) -> f32 {
    let x = x as f64;
    // W. J. Cody's rational approximations, as in most libm implementations.
    let t = 1. / (1. + 0.5 * x.abs());
    let y = 1.
        - t * (-x * x - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    (if x >= 0. { y } else { -y }) as f32
}

/// A tensor by name, or an error naming it.
pub fn tensor<'a>(file: &'a Gguf, name: &str) -> Result<&'a TensorInfo> {
    file.tensor(name)
        .ok_or_else(|| Error::Gguf(format!("no tensor {name}")))
}

/// A tensor's values as f32, whatever it's stored as.
pub fn vector(file: &Gguf, name: &str) -> Result<Vec<f32>> {
    file.dequantize(tensor(file, name)?)
}

/// Weight rows as stored: f32, or Q8_0 blocks (read as `scale · q`, which is exact in f32).
#[derive(Clone, Copy)]
pub enum Rows<'a> {
    F32(&'a [f32]),
    Q8(&'a [Q8Block]),
}

impl Weight {
    /// Rows `range` of this weight, `cols` inputs each.
    pub fn rows(&self, range: std::ops::Range<usize>, cols: usize) -> Rows<'_> {
        match self {
            Weight::F32(w) => Rows::F32(&w[range.start * cols..range.end * cols]),
            Weight::Q8(b) => Rows::Q8(&b[range.start * cols / 32..range.end * cols / 32]),
        }
    }
}

/// A tensor as a weight matrix of `rows` outputs × `cols` inputs (ggml dims `[cols, rows…]`),
/// kept as Q8_0 blocks when it's stored so (compact, and exact), else as f32.
pub fn weight(file: &Gguf, name: &str) -> Result<(usize, usize, Weight)> {
    let info = tensor(file, name)?;
    let cols = *info.dims.first().unwrap_or(&0) as usize;
    let rows = info.dims[1..].iter().product::<u64>() as usize;
    let weight = match info.dtype {
        Dtype::Q8_0 if cols.is_multiple_of(32) => Weight::Q8(
            file.data(info)
                .as_chunks::<34>()
                .0
                .iter()
                .map(|b| Q8Block {
                    scale: half::f16::from_le_bytes([b[0], b[1]]).to_f32(),
                    q: std::array::from_fn(|i| b[2 + i] as i8),
                })
                .collect(),
        ),
        _ => Weight::F32(file.dequantize(info)?),
    };
    Ok((rows, cols, weight))
}

/// `out[i·n + j] = dot(xs[i], w_j)` for every input row `xs[i]` and each of the `n` weight rows
/// (`cols` inputs each), on this thread. Exactly [`dot`] for each output, however many rows:
/// tiles of 4 inputs × 3 weights for matrix products, 1 × 8 for matrix–vector ones, so loads are
/// shared and enough independent sums are in flight to keep the multipliers busy.
pub fn products(xs: &[&[f32]], weights: Rows, cols: usize, out: &mut [f32]) {
    let n = match weights {
        Rows::F32(w) => w.len() / cols.max(1),
        Rows::Q8(b) => b.len() * 32 / cols.max(1),
    };
    assert_eq!(out.len(), xs.len() * n, "{} × {n} products", xs.len());
    assert!(xs.iter().all(|x| x.len() == cols), "inputs of {cols}");
    if let Rows::Q8(_) = weights {
        assert!(cols.is_multiple_of(32), "Q8_0 rows of {cols}");
    }
    #[cfg(target_arch = "x86_64")]
    if simd::available() {
        // SAFETY: the CPU has the features, and every length was checked above.
        unsafe { simd::products(xs, weights, cols, n, out) };
        return;
    }
    let mut w = vec![0f32; cols];
    for j in 0..n {
        let row: &[f32] = match weights {
            Rows::F32(all) => &all[j * cols..(j + 1) * cols],
            Rows::Q8(blocks) => {
                expand(&blocks[j * cols / 32..(j + 1) * cols / 32], &mut w);
                &w
            }
        };
        for (i, x) in xs.iter().enumerate() {
            out[i * n + j] = portable_dot(x, row);
        }
    }
}

/// Q8_0 blocks as f32 (`scale · q`, exact), 32 values a block.
pub fn expand(blocks: &[Q8Block], out: &mut [f32]) {
    for (b, chunk) in blocks.iter().zip(out.as_chunks_mut::<32>().0) {
        for (v, q) in chunk.iter_mut().zip(b.q) {
            *v = b.scale * q as f32;
        }
    }
}

/// AVX2 + FMA kernels, chosen at run time (the app ships one binary for every x86-64 CPU).
#[cfg(target_arch = "x86_64")]
// A tile's loops index its inputs, weights and sums in step.
#[allow(clippy::needless_range_loop)]
mod simd {
    use super::{Q8Block, Rows};
    use std::arch::x86_64::*;

    /// Whether this CPU has AVX2 and FMA (the standard library caches the answer).
    #[inline]
    pub fn available() -> bool {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }

    /// The eight lanes summed in the tree `portable_dot` uses for its accumulators:
    /// `((l0 + l4) + (l1 + l5)) + ((l2 + l6) + (l3 + l7))`.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn lane_sum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let pairs = _mm_add_ps(s, _mm_shuffle_ps::<0b10_11_00_01>(s, s));
        _mm_cvtss_f32(_mm_add_ss(pairs, _mm_movehl_ps(pairs, pairs)))
    }

    /// Eight weights of a Q8_0 block, `scale · q` (exact).
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn q8_lanes(block: &Q8Block, part: usize, scale: __m256) -> __m256 {
        let bytes: &[i8; 8] = block.q[8 * part..][..8].try_into().unwrap();
        // SAFETY: the 8 bytes read are `bytes`.
        let q = unsafe { _mm_loadl_epi64(bytes.as_ptr().cast()) };
        _mm256_mul_ps(scale, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q)))
    }

    /// ggml's `ggml_v_expf` (AVX2 and FMA): e^x to about 1.5 ulp, with its handling of large
    /// magnitudes, instruction for instruction.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn ggml_expf(x: __m256) -> __m256 {
        let c = |bits: u32| _mm256_set1_ps(f32::from_bits(bits));
        let r = c(0x4b40_0000);
        let z = _mm256_fmadd_ps(x, c(0x3fb8_aa3b), r);
        let n = _mm256_sub_ps(z, r);
        let b = _mm256_fnmadd_ps(n, c(0x35bf_be8e), _mm256_fnmadd_ps(n, c(0x3f31_7200), x));
        let e = _mm256_slli_epi32::<23>(_mm256_castps_si256(z));
        let k = _mm256_castsi256_ps(_mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1.))));
        let abs_n = _mm256_andnot_ps(_mm256_set1_ps(-0.), n);
        let big = _mm256_castps_si256(_mm256_cmp_ps::<_CMP_GT_OQ>(abs_n, _mm256_set1_ps(126.)));
        let u = _mm256_mul_ps(b, b);
        let j = _mm256_fmadd_ps(
            _mm256_fmadd_ps(
                _mm256_fmadd_ps(c(0x3c07_2010), b, c(0x3d2b_9f17)),
                u,
                _mm256_fmadd_ps(c(0x3e2a_af33), b, c(0x3eff_fedb)),
            ),
            u,
            _mm256_mul_ps(c(0x3f7f_fff6), b),
        );
        if _mm256_movemask_ps(_mm256_castsi256_ps(big)) == 0 {
            return _mm256_fmadd_ps(j, k, k);
        }
        let g = _mm256_and_si256(
            _mm256_castps_si256(_mm256_cmp_ps::<_CMP_LE_OQ>(n, _mm256_setzero_ps())),
            _mm256_set1_epi32(0x8200_0000u32 as i32),
        );
        let s1 = _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f00_0000)));
        let s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
        let huge = _mm256_cmp_ps::<_CMP_GT_OQ>(abs_n, _mm256_set1_ps(192.));
        let big = _mm256_castsi256_ps(big);
        _mm256_or_ps(
            _mm256_and_ps(huge, _mm256_mul_ps(s1, s1)),
            _mm256_andnot_ps(
                huge,
                _mm256_or_ps(
                    _mm256_and_ps(big, _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
                    _mm256_andnot_ps(big, _mm256_fmadd_ps(k, j, k)),
                ),
            ),
        )
    }

    /// [`super::silu_like_ggml`]'s vector part: `x / (1 + e^−x)` for each whole eight of `row`,
    /// as ggml's `ggml_v_silu`; returns how many values it did.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn silu_like_ggml(row: &mut [f32]) -> usize {
        let mut i = 0;
        while i + 8 <= row.len() {
            // SAFETY: values `i..i + 8` are inside the row.
            unsafe {
                let x = _mm256_loadu_ps(row.as_ptr().add(i));
                let e = ggml_expf(_mm256_sub_ps(_mm256_setzero_ps(), x));
                let y = _mm256_div_ps(x, _mm256_add_ps(_mm256_set1_ps(1.), e));
                _mm256_storeu_ps(row.as_mut_ptr().add(i), y);
            }
            i += 8;
        }
        i
    }

    /// [`super::fma_columns`]'s vector part: columns 64 (then 8) at a time, the sums kept in
    /// registers across the rows; returns how many columns it did.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn fma_columns(acc: &mut [f32], rows: &[f32], weights: &[f32]) -> usize {
        let width = acc.len();
        let mut c0 = 0;
        while c0 + 64 <= width {
            let mut a = [_mm256_setzero_ps(); 8];
            for k in 0..8 {
                // SAFETY: columns `c0..c0 + 64` are inside `acc` and every row.
                a[k] = unsafe { _mm256_loadu_ps(acc.as_ptr().add(c0 + 8 * k)) };
            }
            for (r, w) in weights.iter().enumerate() {
                let p = _mm256_set1_ps(*w);
                let row = r * width + c0;
                for k in 0..8 {
                    // SAFETY: as above.
                    let b = unsafe { _mm256_loadu_ps(rows.as_ptr().add(row + 8 * k)) };
                    a[k] = _mm256_fmadd_ps(b, p, a[k]);
                }
            }
            for k in 0..8 {
                // SAFETY: as above.
                unsafe { _mm256_storeu_ps(acc.as_mut_ptr().add(c0 + 8 * k), a[k]) };
            }
            c0 += 64;
        }
        while c0 + 8 <= width {
            // SAFETY: columns `c0..c0 + 8` are inside `acc` and every row.
            let mut a = unsafe { _mm256_loadu_ps(acc.as_ptr().add(c0)) };
            for (r, w) in weights.iter().enumerate() {
                let b = unsafe { _mm256_loadu_ps(rows.as_ptr().add(r * width + c0)) };
                a = _mm256_fmadd_ps(b, _mm256_set1_ps(*w), a);
            }
            unsafe { _mm256_storeu_ps(acc.as_mut_ptr().add(c0), a) };
            c0 += 8;
        }
        c0
    }

    /// [`super::softmax_like_ggml`]'s vector part, as `ggml_vec_soft_max_f32` adds each eight:
    /// `(hi + lo)`, then pairs, then the last two. Returns how many it did and their sum.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn softmax_like_ggml(x: &mut [f32], max: f32) -> (usize, f64) {
        let mut sum = 0f64;
        let mut i = 0;
        let m = _mm256_set1_ps(max);
        while i + 8 <= x.len() {
            // SAFETY: values `i..i + 8` are inside `x`.
            unsafe {
                let val = ggml_expf(_mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), m));
                _mm256_storeu_ps(x.as_mut_ptr().add(i), val);
                let v2 = _mm_add_ps(_mm256_extractf128_ps::<1>(val), _mm256_castps256_ps128(val));
                let v2 = _mm_add_ps(v2, _mm_movehl_ps(v2, v2));
                let v2 = _mm_add_ss(v2, _mm_movehdup_ps(v2));
                sum += _mm_cvtss_f32(v2) as f64;
            }
            i += 8;
        }
        (i, sum)
    }

    /// [`super::axpy`].
    ///
    /// # Safety
    /// The slices are the same length.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn axpy(y: &mut [f32], a: f32, x: &[f32]) {
        let n = y.len();
        let av = _mm256_set1_ps(a);
        let mut i = 0;
        while i + 8 <= n {
            // SAFETY: elements `i..i + 8` are inside both.
            unsafe {
                let r = _mm256_fmadd_ps(
                    av,
                    _mm256_loadu_ps(x.as_ptr().add(i)),
                    _mm256_loadu_ps(y.as_ptr().add(i)),
                );
                _mm256_storeu_ps(y.as_mut_ptr().add(i), r);
            }
            i += 8;
        }
        for k in i..n {
            y[k] = a.mul_add(x[k], y[k]);
        }
    }

    /// [`super::fma_into`].
    ///
    /// # Safety
    /// The slices are the same length.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn fma_into(y: &mut [f32], w: &[f32], x: &[f32]) {
        let n = y.len();
        let mut i = 0;
        while i + 8 <= n {
            // SAFETY: elements `i..i + 8` are inside all three.
            unsafe {
                let r = _mm256_fmadd_ps(
                    _mm256_loadu_ps(w.as_ptr().add(i)),
                    _mm256_loadu_ps(x.as_ptr().add(i)),
                    _mm256_loadu_ps(y.as_ptr().add(i)),
                );
                _mm256_storeu_ps(y.as_mut_ptr().add(i), r);
            }
            i += 8;
        }
        for k in i..n {
            y[k] = w[k].mul_add(x[k], y[k]);
        }
    }

    /// `M` inputs × `N` f32 weight rows of equal length: each sum in one 8-lane FMA chain.
    ///
    /// # Safety
    /// Every row in `xs` and `ws` has the same length.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn tile<const M: usize, const N: usize>(
        xs: [&[f32]; M],
        ws: [&[f32]; N],
    ) -> [[f32; N]; M] {
        let len = xs[0].len();
        let chunks = len / 8;
        let mut acc = [[_mm256_setzero_ps(); N]; M];
        for c in 0..chunks {
            let mut w = [_mm256_setzero_ps(); N];
            for j in 0..N {
                // SAFETY: `8c + 8 <= len` for every row.
                w[j] = unsafe { _mm256_loadu_ps(ws[j].as_ptr().add(8 * c)) };
            }
            for i in 0..M {
                // SAFETY: as above.
                let x = unsafe { _mm256_loadu_ps(xs[i].as_ptr().add(8 * c)) };
                for j in 0..N {
                    acc[i][j] = _mm256_fmadd_ps(x, w[j], acc[i][j]);
                }
            }
        }
        let mut out = [[0f32; N]; M];
        for i in 0..M {
            for j in 0..N {
                let mut tail = 0f32;
                for e in 8 * chunks..len {
                    tail += xs[i][e] * ws[j][e];
                }
                out[i][j] = lane_sum(acc[i][j]) + tail;
            }
        }
        out
    }

    /// [`tile`] with Q8_0 weight rows: the same sums, bit for bit, as on the rows expanded to
    /// f32.
    ///
    /// # Safety
    /// Every input has `32 ·` (blocks per weight row) values, and every weight row as many blocks.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn tile_q8<const M: usize, const N: usize>(
        xs: [&[f32]; M],
        ws: [&[Q8Block]; N],
    ) -> [[f32; N]; M] {
        let mut acc = [[_mm256_setzero_ps(); N]; M];
        for b in 0..ws[0].len() {
            let blocks: [&Q8Block; N] = std::array::from_fn(|j| &ws[j][b]);
            let mut scale = [_mm256_setzero_ps(); N];
            for j in 0..N {
                scale[j] = _mm256_set1_ps(blocks[j].scale);
            }
            for part in 0..4 {
                let mut w = [_mm256_setzero_ps(); N];
                for j in 0..N {
                    w[j] = q8_lanes(blocks[j], part, scale[j]);
                }
                for i in 0..M {
                    // SAFETY: `32b + 8part + 8 <= len`.
                    let x = unsafe { _mm256_loadu_ps(xs[i].as_ptr().add(32 * b + 8 * part)) };
                    for j in 0..N {
                        acc[i][j] = _mm256_fmadd_ps(x, w[j], acc[i][j]);
                    }
                }
            }
        }
        let mut out = [[0f32; N]; M];
        for i in 0..M {
            for j in 0..N {
                out[i][j] = lane_sum(acc[i][j]);
            }
        }
        out
    }

    /// One tile: `M` inputs from `i` × `N` weight rows from `j`, into `out` (`n` per input).
    ///
    /// # Safety
    /// The tile is inside `xs` and the weights, whose lengths [`super::products`] checked.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn put<const M: usize, const N: usize>(
        xs: &[&[f32]],
        weights: Rows,
        cols: usize,
        n: usize,
        (i, j): (usize, usize),
        out: &mut [f32],
    ) {
        let x: [&[f32]; M] = std::array::from_fn(|a| xs[i + a]);
        let sums = match weights {
            Rows::F32(w) => {
                let w: [&[f32]; N] = std::array::from_fn(|b| &w[(j + b) * cols..][..cols]);
                // SAFETY: every row is `cols` long.
                unsafe { tile::<M, N>(x, w) }
            }
            Rows::Q8(q) => {
                let per = cols / 32;
                let w: [&[Q8Block]; N] = std::array::from_fn(|b| &q[(j + b) * per..][..per]);
                // SAFETY: as above.
                unsafe { tile_q8::<M, N>(x, w) }
            }
        };
        for (a, sums) in sums.iter().enumerate() {
            out[(i + a) * n + j..][..N].copy_from_slice(sums);
        }
    }

    /// [`super::products`] on AVX2: 1 × 8 tiles for a single input, 4 inputs × 3 weight rows
    /// otherwise (12 sums, 3 weights and an input: all 16 vector registers, none spilled), and
    /// narrower tiles at the edges.
    ///
    /// # Safety
    /// The lengths were checked by [`super::products`].
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn products(xs: &[&[f32]], weights: Rows, cols: usize, n: usize, out: &mut [f32]) {
        let m = xs.len();
        // SAFETY (every call below): the tile is inside `xs` and the weight rows.
        if m == 1 {
            let mut j = 0;
            while j + 8 <= n {
                unsafe { put::<1, 8>(xs, weights, cols, n, (0, j), out) };
                j += 8;
            }
            while j < n {
                unsafe { put::<1, 1>(xs, weights, cols, n, (0, j), out) };
                j += 1;
            }
            return;
        }
        let mut j = 0;
        while j < n {
            let width = (n - j).min(3);
            let mut i = 0;
            while i < m {
                let height = (m - i).min(4);
                let at = (i, j);
                unsafe {
                    match (height, width) {
                        (4, 3) => put::<4, 3>(xs, weights, cols, n, at, out),
                        (4, 2) => put::<4, 2>(xs, weights, cols, n, at, out),
                        (4, _) => put::<4, 1>(xs, weights, cols, n, at, out),
                        (3, 3) => put::<3, 3>(xs, weights, cols, n, at, out),
                        (3, 2) => put::<3, 2>(xs, weights, cols, n, at, out),
                        (3, _) => put::<3, 1>(xs, weights, cols, n, at, out),
                        (2, 3) => put::<2, 3>(xs, weights, cols, n, at, out),
                        (2, 2) => put::<2, 2>(xs, weights, cols, n, at, out),
                        (2, _) => put::<2, 1>(xs, weights, cols, n, at, out),
                        (_, 3) => put::<1, 3>(xs, weights, cols, n, at, out),
                        (_, 2) => put::<1, 2>(xs, weights, cols, n, at, out),
                        _ => put::<1, 1>(xs, weights, cols, n, at, out),
                    }
                }
                i += height;
            }
            j += width;
        }
    }
}
