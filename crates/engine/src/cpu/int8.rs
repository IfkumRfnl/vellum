//! Integer matrix products for Q8_0 weights: the engine's fast CPU path.
//!
//! The reference's CPU path multiplies a Q8_0 weight matrix by its input quantized the same way:
//! per 32 values an f16 scale `d = max|x| / 127` and the values `round(x · 127 / max|x|)` as
//! int8, ties to even (ggml's `quantize_row_q8_0` as its AVX2 build rounds), then per block of
//! 32 the exact integer dot product, scaled by the product of the two scales and added up in
//! f32 (`ggml_vec_dot_q8_0_q8_0`, llamafile's `tinyBLAS_Q0_AVX`). This does the same
//! arithmetic: the same int8 input, each block's integer sum exact, `(d_w · d_x) · Σ` added by
//! a fused multiply-add, block after block. ggml spreads a block's sum over eight lanes and adds
//! them at the end; the order of those f32 additions is the only difference (about 1e-7).
//!
//! The weights are repacked at load for the CPU's integer dot products: eight output rows
//! interleaved four inputs at a time, so one 32-byte vector holds the same four inputs of eight
//! rows and one `vpdpbusd` adds 32 products into the eight rows' sums, the input's four values
//! broadcast to every row. A block's integer sums for eight rows are then one vector, converted,
//! scaled and accumulated in one instruction each. With AVX-VNNI (or AVX-512 VNNI) the weights
//! are kept offset by 128 as unsigned bytes (the instruction multiplies unsigned by signed) and
//! each input block's `128 · Σx` taken off again, exactly; with plain AVX2 they stay signed and
//! the products go through `maddubs` with the signs moved onto the weights, as ggml does. A
//! portable loop computes the same sums anywhere else.
//!
//! Weights in f16 (Parakeet's pointwise convolutions) are multiplied as the reference does:
//! the input rounded to f16, the products and sums in f32 ([`F16Matrix`]).

use super::{Linear, Mat, Rows, products, tensor};
use crate::{
    Error, Result,
    gguf::{Dtype, Gguf},
};
use half::f16;
use rayon::prelude::*;
use std::sync::OnceLock;

/// Inputs per Q8_0 block.
const BLOCK: usize = 32;
/// Output rows interleaved in a group: one 8-lane vector of sums.
const GROUP: usize = 8;
/// Bytes of one group's block: 8 steps of 8 rows × 4 inputs.
const GROUP_BLOCK: usize = GROUP * BLOCK;
/// Bytes of one Q8_0 block in the file: an f16 scale and 32 int8 values.
const FILE_BLOCK: usize = 34;

/// How the CPU's matrix products are computed: the int8 products, or every layer in f32 against
/// its exactly dequantized weights, the lab's oracle. The int8 products are the default where
/// they have vector kernels (x86-64 with AVX2); elsewhere their portable loop is slower than
/// the f32 path, which stays the default there. `VELLUM_CPU_GEMM=f32` or `=int8` decides.
pub fn int8_enabled() -> bool {
    int8_unless_told(true)
}

/// Whether a part of a model whose default is `default` (int8 or not, as measured for it) takes
/// the int8 products: `VELLUM_CPU_GEMM=f32` or `=int8` overrides every default, and none is
/// int8 without vector kernels.
pub fn int8_unless_told(default: bool) -> bool {
    match gemm_setting() {
        Some(choice) => choice,
        None => default && kernel() != Kernel::Portable,
    }
}

/// `VELLUM_CPU_GEMM`: `Some(true)` for `int8`, `Some(false)` for `f32`, else `None`.
fn gemm_setting() -> Option<bool> {
    static SETTING: OnceLock<Option<bool>> = OnceLock::new();
    *SETTING.get_or_init(|| match std::env::var("VELLUM_CPU_GEMM").as_deref() {
        Ok("f32") => Some(false),
        Ok("int8") => Some(true),
        _ => None,
    })
}

/// A linear layer `y = W·x + b` as the CPU path multiplies it, by [`int8_enabled`] and the
/// weight's type in the file: Q8_0 weights times the input quantized to Q8_0, F16 weights times
/// the input rounded to f16 (both as the reference's CPU path computes them), F32 weights (and
/// every weight under `VELLUM_CPU_GEMM=f32`) in f32 against the exactly dequantized weights.
pub enum Dense {
    Exact(Linear),
    Int8(Q8Matrix, Option<Vec<f32>>),
    Half(F16Matrix, Option<Vec<f32>>),
}

impl Dense {
    /// `name.weight` (and `name.bias` when the file has one), shaped as [`Linear::load`] shapes
    /// it: the first dimension is the input, the others multiply to the outputs.
    pub fn load(file: &Gguf, name: &str) -> Result<Dense> {
        Dense::load_as(file, name, int8_enabled())
    }

    /// [`load`](Self::load), the int8 products only if `int8` (exact otherwise).
    pub fn load_as(file: &Gguf, name: &str, int8: bool) -> Result<Dense> {
        let info = tensor(file, &format!("{name}.weight"))?;
        // As `Linear::load` reads shapes: a leading kernel dimension of one isn't an input.
        let mut dims: Vec<usize> = info.dims.iter().map(|d| *d as usize).collect();
        while dims.len() > 2 && dims[0] == 1 {
            dims.remove(0);
        }
        let cols = dims.first().copied().unwrap_or(0);
        let rows: usize = dims.iter().skip(1).product();
        let int8 = int8
            && match info.dtype {
                Dtype::Q8_0 => cols > 0 && cols.is_multiple_of(BLOCK),
                Dtype::F16 => true,
                _ => false,
            };
        if !int8 {
            return Ok(Dense::Exact(Linear::load(file, name)?));
        }
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
        let data = file.data(info);
        let dense = match info.dtype {
            Dtype::Q8_0 => Dense::Int8(Q8Matrix::from_q8_0(data, rows, cols), bias),
            _ => Dense::Half(F16Matrix::from_f16(data, rows, cols), bias),
        };
        // The layer has its own copy now.
        file.release(info);
        Ok(dense)
    }

    /// Output features.
    pub fn rows(&self) -> usize {
        match self {
            Dense::Exact(l) => l.rows,
            Dense::Int8(w, _) => w.rows,
            Dense::Half(w, _) => w.rows,
        }
    }

    /// Input features.
    pub fn cols(&self) -> usize {
        match self {
            Dense::Exact(l) => l.cols,
            Dense::Int8(w, _) => w.cols,
            Dense::Half(w, _) => w.cols,
        }
    }

    pub fn forward(&self, x: &Mat) -> Mat {
        match self {
            Dense::Exact(l) => l.forward(x),
            Dense::Int8(w, b) => w.forward(x, b.as_deref()),
            Dense::Half(w, b) => w.forward(x, b.as_deref()),
        }
    }
}

/// The integer dot-product instructions this CPU has, best first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// AVX-VNNI (`vpdpbusd`, VEX): Alder Lake and later, Zen 5.
    AvxVnni,
    /// AVX-512 VNNI's 256-bit `vpdpbusd`: Ice Lake, Sapphire Rapids, Zen 4.
    Avx512Vnni,
    /// AVX2 `maddubs`: any x86-64 CPU since Haswell.
    Avx2,
    /// Plain Rust: everything else.
    Portable,
}

impl Kernel {
    /// Whether this kernel takes its weights offset by 128 (unsigned bytes).
    fn unsigned_weights(self) -> bool {
        matches!(self, Kernel::AvxVnni | Kernel::Avx512Vnni)
    }
}

/// The kernel the int8 products use: the best this CPU has, or a slower one when
/// `VELLUM_CPU_INT8=avx2` or `=portable` asks for it (the same sums, bit for bit).
pub fn kernel() -> Kernel {
    static KERNEL: OnceLock<Kernel> = OnceLock::new();
    *KERNEL.get_or_init(|| {
        let best = detect();
        match std::env::var("VELLUM_CPU_INT8").as_deref() {
            Ok("portable") => Kernel::Portable,
            Ok("avx2") if best != Kernel::Portable => Kernel::Avx2,
            _ => best,
        }
    })
}

fn detect() -> Kernel {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::is_x86_feature_detected as has;
        if has!("avx2") && has!("fma") && has!("f16c") {
            if has!("avxvnni") {
                return Kernel::AvxVnni;
            }
            if has!("avx512vnni") && has!("avx512vl") {
                return Kernel::Avx512Vnni;
            }
            return Kernel::Avx2;
        }
    }
    Kernel::Portable
}

/// A Q8_0 weight matrix of `rows` outputs × `cols` inputs, repacked for [`kernel`]: per group of
/// eight rows (the last padded with zeros), per block of 32 inputs, eight steps of the eight
/// rows' next four values; and per group and block the eight rows' f16 scales.
pub struct Q8Matrix {
    pub rows: usize,
    pub cols: usize,
    quants: Vec<u8>,
    scales: Vec<u16>,
    /// Whether `quants` holds `q + 128` (for the VNNI kernels) rather than `q` as a byte.
    unsigned: bool,
}

impl Q8Matrix {
    /// From Q8_0 data as the file stores it: per row, `cols / 32` blocks of an f16 scale and 32
    /// int8 values.
    pub fn from_q8_0(data: &[u8], rows: usize, cols: usize) -> Q8Matrix {
        assert!(cols.is_multiple_of(BLOCK), "Q8_0 rows of {cols}");
        let blocks = cols / BLOCK;
        assert_eq!(data.len(), rows * blocks * FILE_BLOCK, "{rows}×{cols} Q8_0");
        let groups = rows.div_ceil(GROUP);
        let unsigned = kernel().unsigned_weights();
        let offset = if unsigned { 128 } else { 0 };
        let mut quants = vec![0u8; groups * blocks * GROUP_BLOCK];
        let mut scales = vec![0u16; groups * blocks * GROUP];
        quants
            .par_chunks_mut(blocks * GROUP_BLOCK)
            .zip(scales.par_chunks_mut(blocks * GROUP))
            .enumerate()
            .for_each(|(g, (quants, scales))| {
                for r in 0..GROUP {
                    let row = g * GROUP + r;
                    for b in 0..blocks {
                        let block = if row < rows {
                            &data[(row * blocks + b) * FILE_BLOCK..][..FILE_BLOCK]
                        } else {
                            &[0; FILE_BLOCK][..]
                        };
                        scales[b * GROUP + r] = u16::from_le_bytes([block[0], block[1]]);
                        let to = &mut quants[b * GROUP_BLOCK..][..GROUP_BLOCK];
                        for (i, &q) in block[2..].iter().enumerate() {
                            let (step, k) = (i / 4, i % 4);
                            to[step * GROUP * 4 + r * 4 + k] = (q as i8 as i16 + offset) as u8;
                        }
                    }
                }
            });
        Q8Matrix {
            rows,
            cols,
            quants,
            scales,
            unsigned,
        }
    }

    fn blocks(&self) -> usize {
        self.cols / BLOCK
    }

    /// Output row `o` of the weight as f32 (`scale · q`, exact).
    pub fn row(&self, o: usize, out: &mut [f32]) {
        assert_eq!(out.len(), self.cols);
        let blocks = self.blocks();
        let (g, r) = (o / GROUP, o % GROUP);
        let offset = if self.unsigned { 128 } else { 0 };
        for (b, out) in out.as_chunks_mut::<BLOCK>().0.iter_mut().enumerate() {
            let at = g * blocks + b;
            let scale = f16::from_bits(self.scales[at * GROUP + r]).to_f32();
            let quants = &self.quants[at * GROUP_BLOCK..][..GROUP_BLOCK];
            for (i, v) in out.iter_mut().enumerate() {
                let q = quants[(i / 4) * GROUP * 4 + r * 4 + i % 4];
                let q = if self.unsigned {
                    q as i16 - offset
                } else {
                    q as i8 as i16
                };
                *v = scale * q as f32;
            }
        }
    }

    /// `x · Wᵀ + b` with `x` quantized as the reference quantizes it.
    pub fn forward(&self, x: &Mat, bias: Option<&[f32]>) -> Mat {
        self.product(&quantize(x), bias)
    }

    /// `x · Wᵀ + b` for an input already quantized. Each output is the same sum however the
    /// work is split, so a row's result never depends on the other rows.
    pub fn product(&self, x: &Q8Input, bias: Option<&[f32]>) -> Mat {
        assert_eq!(x.cols, self.cols, "a product of {} inputs", self.cols);
        if let Some(b) = bias {
            assert_eq!(b.len(), self.rows);
        }
        let (m, n) = (x.rows, self.rows);
        let mut out = Mat::zeros(m, n);
        if m == 0 || n == 0 {
            return out;
        }
        let groups = n.div_ceil(GROUP);
        let chunk = if m <= 2 * ROW_CHUNK { m } else { ROW_CHUNK };
        let chunks = m.div_ceil(chunk);
        // Panels of four groups (32 rows of weights) unless that leaves too few tasks to share.
        let panel = if groups.div_ceil(4) * chunks >= MIN_TASKS {
            4
        } else {
            2
        };
        let panels = groups.div_ceil(panel);
        let target = Target(out.data.as_mut_ptr());
        let task = |t: usize| {
            let (c, p) = (t / panels, t % panels);
            let rows = c * chunk..((c + 1) * chunk).min(m);
            let groups = p * panel..((p + 1) * panel).min(groups);
            self.tiles(x, rows, groups, bias, &target);
        };
        let tasks = panels * chunks;
        if tasks == 1 || m * n * self.cols < SERIAL_MACS {
            (0..tasks).for_each(task);
        } else {
            (0..tasks).into_par_iter().for_each(task);
        }
        out
    }

    /// Rows `rows` of `x` against groups `groups` of the weight, tile by tile, into `target`.
    fn tiles(
        &self,
        x: &Q8Input,
        rows: std::ops::Range<usize>,
        groups: std::ops::Range<usize>,
        bias: Option<&[f32]>,
        target: &Target,
    ) {
        // A single input row takes four groups at once (it's the weights' stream that counts),
        // more take three rows × two groups (12 sums and 12 accumulators in 16 registers).
        let (tall, wide) = if rows.len() == 1 { (1, 4) } else { (3, 2) };
        let mut i = rows.start;
        while i < rows.end {
            let h = (rows.end - i).min(tall);
            let mut g = groups.start;
            while g < groups.end {
                let w = (groups.end - g).min(wide);
                match (h, w) {
                    (3, 2) => self.tile::<3, 2>(x, i, g, bias, target),
                    (3, _) => self.tile::<3, 1>(x, i, g, bias, target),
                    (2, 2) => self.tile::<2, 2>(x, i, g, bias, target),
                    (2, _) => self.tile::<2, 1>(x, i, g, bias, target),
                    (1, 4) => self.tile::<1, 4>(x, i, g, bias, target),
                    (1, 3) => self.tile::<1, 3>(x, i, g, bias, target),
                    (1, 2) => self.tile::<1, 2>(x, i, g, bias, target),
                    _ => self.tile::<1, 1>(x, i, g, bias, target),
                }
                g += w;
            }
            i += h;
        }
    }

    /// Input rows `i..i + M` against groups `g..g + N`: the sums, plus the bias, into `target`.
    fn tile<const M: usize, const N: usize>(
        &self,
        x: &Q8Input,
        i: usize,
        g: usize,
        bias: Option<&[f32]>,
        target: &Target,
    ) {
        let blocks = self.blocks();
        let inputs = Inputs::<M> {
            quants: std::array::from_fn(|a| &x.quants[(i + a) * self.cols..][..self.cols]),
            scales: std::array::from_fn(|a| &x.scales[(i + a) * blocks..][..blocks]),
            offsets: std::array::from_fn(|a| &x.offsets[(i + a) * blocks..][..blocks]),
        };
        let weights = Weights::<N> {
            quants: std::array::from_fn(|b| {
                &self.quants[(g + b) * blocks * GROUP_BLOCK..][..blocks * GROUP_BLOCK]
            }),
            scales: std::array::from_fn(|b| {
                &self.scales[(g + b) * blocks * GROUP..][..blocks * GROUP]
            }),
        };
        let sums = match (kernel(), self.unsigned) {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: the CPU has the features (`detect`), and every slice spans `blocks`.
            (Kernel::AvxVnni, true) => unsafe { x86::vnni::<M, N>(&inputs, &weights, blocks) },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: as above.
            (Kernel::Avx512Vnni, true) => unsafe {
                x86::vnni512::<M, N>(&inputs, &weights, blocks)
            },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: as above.
            (Kernel::Avx2, false) => unsafe { x86::avx2::<M, N>(&inputs, &weights, blocks) },
            _ => portable::<M, N>(&inputs, &weights, blocks, self.unsigned),
        };
        let n = self.rows;
        for (a, sums) in sums.iter().enumerate() {
            for (b, lanes) in sums.iter().enumerate() {
                let first = (g + b) * GROUP;
                for (r, v) in lanes.iter().enumerate().take(n.saturating_sub(first)) {
                    let o = first + r;
                    let v = v + bias.map_or(0., |bias| bias[o]);
                    // SAFETY: tasks cover disjoint (row, output) ranges, all inside the output.
                    unsafe { target.write((i + a) * n + o, v) };
                }
            }
        }
    }
}

/// Input rows per task when there are many: 48 rows of up to 5120 inputs (240 KB of int8)
/// stay in a core's L2 beside the weights' panel.
const ROW_CHUNK: usize = 48;
/// Tasks a product wants at least, so every thread gets several (cores differ in speed).
const MIN_TASKS: usize = 48;
/// Products smaller than this many multiply-adds run on the calling thread: waking the pool
/// costs more than they take.
const SERIAL_MACS: usize = 1 << 18;

/// The output of a product, written by many tasks at disjoint places.
struct Target(*mut f32);
// SAFETY: every task writes its own outputs only.
unsafe impl Sync for Target {}
unsafe impl Send for Target {}

impl Target {
    /// # Safety
    /// `at` is inside the output, and no other task writes it.
    unsafe fn write(&self, at: usize, v: f32) {
        // SAFETY: as the caller promises.
        unsafe { self.0.add(at).write(v) }
    }
}

/// Rows of an input quantized as ggml's `quantize_row_q8_0` leaves them: per block of 32 values
/// the int8 values, the f16 scale (as f32), and `−128 ·` the values' sum (what the unsigned
/// weights' offset adds to each block's integer sum).
pub struct Q8Input {
    pub rows: usize,
    pub cols: usize,
    quants: Vec<i8>,
    scales: Vec<f32>,
    offsets: Vec<i32>,
}

/// `x` quantized per block of 32 values, exactly as the reference quantizes a Q8_0 product's
/// input: `d = max|x| / 127` rounded to f16, the values `x · (127 / max|x|)` rounded to the
/// nearest integer, ties to even (0 for a block of zeros).
pub fn quantize(x: &Mat) -> Q8Input {
    assert!(x.cols.is_multiple_of(BLOCK), "Q8_0 rows of {}", x.cols);
    let blocks = x.cols / BLOCK;
    let mut quants = vec![0i8; x.rows * x.cols];
    let mut scales = vec![0f32; x.rows * blocks];
    let mut offsets = vec![0i32; x.rows * blocks];
    type Row<'a> = ((&'a [f32], &'a mut [i8]), (&'a mut [f32], &'a mut [i32]));
    let row = |((x, quants), (scales, offsets)): Row| quantize_row(x, quants, scales, offsets);
    if x.rows > 1 && x.data.len() >= 1 << 14 {
        x.data
            .par_chunks(x.cols)
            .zip(quants.par_chunks_mut(x.cols))
            .zip(
                scales
                    .par_chunks_mut(blocks)
                    .zip(offsets.par_chunks_mut(blocks)),
            )
            .for_each(row);
    } else if x.cols > 0 {
        x.data
            .chunks(x.cols)
            .zip(quants.chunks_mut(x.cols))
            .zip(scales.chunks_mut(blocks).zip(offsets.chunks_mut(blocks)))
            .for_each(row);
    }
    Q8Input {
        rows: x.rows,
        cols: x.cols,
        quants,
        scales,
        offsets,
    }
}

fn quantize_row(x: &[f32], quants: &mut [i8], scales: &mut [f32], offsets: &mut [i32]) {
    #[cfg(target_arch = "x86_64")]
    if kernel() != Kernel::Portable {
        // SAFETY: every kernel but the portable one implies AVX2, FMA and F16C (`detect`).
        unsafe { x86::quantize_row(x, quants, scales, offsets) };
        return;
    }
    for (((x, q), scale), offset) in x
        .as_chunks::<BLOCK>()
        .0
        .iter()
        .zip(quants.as_chunks_mut::<BLOCK>().0.iter_mut())
        .zip(scales.iter_mut())
        .zip(offsets.iter_mut())
    {
        let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
        let id = if amax != 0. { 127. / amax } else { 0. };
        let mut sum = 0;
        for (q, v) in q.iter_mut().zip(x) {
            let r = (v * id).round_ties_even() as i32;
            *q = r as i8;
            sum += r;
        }
        *scale = f16::from_f32(amax / 127.).to_f32();
        *offset = -128 * sum;
    }
}

/// `M` quantized input rows for a tile: values, scales and offsets.
struct Inputs<'a, const M: usize> {
    quants: [&'a [i8]; M],
    scales: [&'a [f32]; M],
    offsets: [&'a [i32]; M],
}

/// `N` groups of repacked weights for a tile.
struct Weights<'a, const N: usize> {
    quants: [&'a [u8]; N],
    scales: [&'a [u16]; N],
}

/// The sums of a tile: per input row, per group, the eight rows' outputs.
type Sums<const M: usize, const N: usize> = [[[f32; GROUP]; N]; M];

/// The kernels' sums in plain Rust, bit for bit: each block's integer sum exact, then
/// `acc = fma(d_w · d_x, Σ, acc)`.
// The loops index inputs, weights and sums in step.
#[allow(clippy::needless_range_loop)]
fn portable<const M: usize, const N: usize>(
    x: &Inputs<M>,
    w: &Weights<N>,
    blocks: usize,
    unsigned: bool,
) -> Sums<M, N> {
    let mut acc = [[[0f32; GROUP]; N]; M];
    for b in 0..blocks {
        for a in 0..M {
            let xq = &x.quants[a][b * BLOCK..][..BLOCK];
            for c in 0..N {
                let wq = &w.quants[c][b * GROUP_BLOCK..][..GROUP_BLOCK];
                for r in 0..GROUP {
                    let mut sum = 0i32;
                    for (i, &x) in xq.iter().enumerate() {
                        let q = wq[(i / 4) * GROUP * 4 + r * 4 + i % 4];
                        let q = if unsigned {
                            q as i32 - 128
                        } else {
                            q as i8 as i32
                        };
                        sum += q * x as i32;
                    }
                    let d = f16::from_bits(w.scales[c][b * GROUP + r]).to_f32() * x.scales[a][b];
                    acc[a][c][r] = d.mul_add(sum as f32, acc[a][c][r]);
                }
            }
        }
    }
    acc
}

/// Weights in f16, multiplied as the reference multiplies them: the input rounded to f16, the
/// products and sums in f32 (each output exactly [`super::dot`] of the rounded input and the
/// weight row, as [`super::Linear::forward`] computes it on the same values).
pub struct F16Matrix {
    pub rows: usize,
    pub cols: usize,
    weights: Vec<f16>,
}

/// Output rows per task in [`F16Matrix::forward`], as [`super::Linear::forward`] takes them.
const F16_PANEL: usize = 24;

impl F16Matrix {
    /// From f16 values as the file stores them, row after row.
    pub fn from_f16(data: &[u8], rows: usize, cols: usize) -> F16Matrix {
        assert_eq!(data.len(), rows * cols * 2, "{rows}×{cols} f16");
        F16Matrix {
            rows,
            cols,
            weights: data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f16::from_le_bytes(*b))
                .collect(),
        }
    }

    /// `round_f16(x) · Wᵀ + b`.
    pub fn forward(&self, x: &Mat, bias: Option<&[f32]>) -> Mat {
        assert_eq!(x.cols, self.cols, "a product of {} inputs", self.cols);
        let x = to_f16(x);
        let mut out = Mat::zeros(x.rows, self.rows);
        if x.rows == 0 {
            return out;
        }
        let inputs: Vec<&[f32]> = (0..x.rows).map(|t| x.row(t)).collect();
        let target = Target(out.data.as_mut_ptr());
        let panels = self.rows.div_ceil(F16_PANEL);
        (0..panels).into_par_iter().for_each_init(
            || vec![0f32; F16_PANEL * self.cols],
            |panel, p| {
                let rows = p * F16_PANEL..((p + 1) * F16_PANEL).min(self.rows);
                let n = rows.len();
                let weights = &mut panel[..n * self.cols];
                widen(
                    &self.weights[rows.start * self.cols..rows.end * self.cols],
                    weights,
                );
                let mut block = vec![0f32; x.rows * n];
                products(&inputs, Rows::F32(weights), self.cols, &mut block);
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

/// f16 values as f32 (exact).
fn widen(from: &[f16], to: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if kernel() != Kernel::Portable {
        // SAFETY: every kernel but the portable one implies F16C (`detect`).
        unsafe { x86::widen(from, to) };
        return;
    }
    for (t, f) in to.iter_mut().zip(from) {
        *t = f.to_f32();
    }
}

/// `x` with every value rounded to the nearest f16 (ties to even), as ggml converts a product's
/// input for f16 weights.
pub fn to_f16(x: &Mat) -> Mat {
    let mut x = x.clone();
    x.data.par_chunks_mut(1 << 14).for_each(|chunk| {
        #[cfg(target_arch = "x86_64")]
        if kernel() != Kernel::Portable {
            // SAFETY: every kernel but the portable one implies F16C (`detect`).
            unsafe { x86::round_f16(chunk) };
            return;
        }
        for v in chunk {
            *v = f16::from_f32(*v).to_f32();
        }
    });
    x
}

/// ggml's `ggml_vec_mad_f16` on an f16 accumulator kept as f32 values: `y = f16(x · v + y)`, one
/// fused multiply-add per element, rounded to f16 as it's stored.
pub fn mad_f16(y: &mut [f32], x: &[f32], v: f32) {
    assert_eq!(y.len(), x.len());
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if kernel() != Kernel::Portable {
        // SAFETY: every kernel but the portable one implies FMA and F16C (`detect`).
        done = unsafe { x86::mad_f16(y, x, v) };
    }
    for (y, x) in y[done..].iter_mut().zip(&x[done..]) {
        *y = f16::from_f32(x.mul_add(v, *y)).to_f32();
    }
}

/// ggml's `ggml_vec_scale_f16` on an f16 accumulator kept as f32 values: `y = f16(y · v)`.
pub fn scale_f16(y: &mut [f32], v: f32) {
    for y in y.iter_mut() {
        *y *= v;
    }
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if kernel() != Kernel::Portable {
        // SAFETY: every kernel but the portable one implies F16C (`detect`).
        unsafe { x86::round_f16(y) };
        done = y.len();
    }
    for y in &mut y[done..] {
        *y = f16::from_f32(*y).to_f32();
    }
}

#[cfg(target_arch = "x86_64")]
// A tile's loops index its inputs, weights and sums in step.
#[allow(clippy::needless_range_loop)]
mod x86 {
    use super::{BLOCK, GROUP, GROUP_BLOCK, Inputs, Sums, Weights};
    use half::f16;
    use std::arch::x86_64::*;

    /// The integer kernels' shared shape: per block, eight steps of four inputs, each step's
    /// weights loaded once per group and its inputs broadcast once per row; then the block's
    /// sums of eight rows converted, scaled by `d_w · d_x` and accumulated in one go.
    macro_rules! int8_kernel {
        ($name:ident, $features:literal, |$acc:ident, $w:ident, $x:ident| $step:expr, $start:expr) => {
            /// # Safety
            /// The CPU has the features; every slice of `x` and `w` spans `blocks` blocks.
            #[target_feature(enable = $features)]
            pub unsafe fn $name<const M: usize, const N: usize>(
                x: &Inputs<M>,
                w: &Weights<N>,
                blocks: usize,
            ) -> Sums<M, N> {
                let mut acc = [[_mm256_setzero_ps(); N]; M];
                for b in 0..blocks {
                    let mut sums = [[_mm256_setzero_si256(); N]; M];
                    for a in 0..M {
                        // SAFETY: `b < blocks`.
                        let start: i32 = $start(unsafe { *x.offsets[a].get_unchecked(b) });
                        for c in 0..N {
                            sums[a][c] = _mm256_set1_epi32(start);
                        }
                    }
                    for step in 0..BLOCK / 4 {
                        let mut ws = [_mm256_setzero_si256(); N];
                        for c in 0..N {
                            // SAFETY: the 32 bytes are inside group `c`'s block `b`.
                            ws[c] = unsafe {
                                _mm256_loadu_si256(
                                    w.quants[c].as_ptr().add(b * GROUP_BLOCK + step * 32).cast(),
                                )
                            };
                        }
                        for a in 0..M {
                            // SAFETY: the 4 bytes are inside row `a`'s block `b`.
                            let four = unsafe {
                                x.quants[a]
                                    .as_ptr()
                                    .add(b * BLOCK + step * 4)
                                    .cast::<i32>()
                                    .read_unaligned()
                            };
                            let $x = _mm256_set1_epi32(four);
                            for c in 0..N {
                                let $acc = sums[a][c];
                                let $w = ws[c];
                                sums[a][c] = $step;
                            }
                        }
                    }
                    let mut dw = [_mm256_setzero_ps(); N];
                    for c in 0..N {
                        // SAFETY: the 8 scales are group `c`'s block `b`.
                        dw[c] = unsafe {
                            _mm256_cvtph_ps(_mm_loadu_si128(
                                w.scales[c].as_ptr().add(b * GROUP).cast(),
                            ))
                        };
                    }
                    for a in 0..M {
                        // SAFETY: `b < blocks`.
                        let dx = _mm256_set1_ps(unsafe { *x.scales[a].get_unchecked(b) });
                        for c in 0..N {
                            let d = _mm256_mul_ps(dw[c], dx);
                            acc[a][c] =
                                _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sums[a][c]), acc[a][c]);
                        }
                    }
                }
                let mut out = [[[0f32; GROUP]; N]; M];
                for a in 0..M {
                    for c in 0..N {
                        // SAFETY: 8 floats into an array of 8.
                        unsafe { _mm256_storeu_ps(out[a][c].as_mut_ptr(), acc[a][c]) };
                    }
                }
                out
            }
        };
    }

    // Unsigned weights `q + 128` times signed inputs, from `−128 · Σx`: exactly `Σ q · x`.
    int8_kernel!(
        vnni,
        "avx2,fma,f16c,avxvnni",
        |acc, w, x| _mm256_dpbusd_avx_epi32(acc, w, x),
        |offset| offset
    );
    int8_kernel!(
        vnni512,
        "avx2,fma,f16c,avx512vnni,avx512vl",
        |acc, w, x| _mm256_dpbusd_epi32(acc, w, x),
        |offset| offset
    );
    // Signed weights: `|x| · (w · sign x)` in pairs (at most 2 · 127 · 128, no saturation),
    // the pairs summed to the four-product lanes.
    int8_kernel!(
        avx2,
        "avx2,fma,f16c",
        |acc, w, x| _mm256_add_epi32(
            acc,
            _mm256_madd_epi16(
                _mm256_maddubs_epi16(_mm256_sign_epi8(x, x), _mm256_sign_epi8(w, x)),
                _mm256_set1_epi16(1)
            )
        ),
        |_offset| 0
    );

    /// [`super::quantize_row`] on AVX2: ggml's own instructions (`quantize_row_q8_0`).
    ///
    /// # Safety
    /// The CPU has AVX2, FMA and F16C; `x` and `quants` are equally long, a multiple of 32,
    /// with a scale and an offset per 32.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn quantize_row(
        x: &[f32],
        quants: &mut [i8],
        scales: &mut [f32],
        offsets: &mut [i32],
    ) {
        let blocks = x.len() / BLOCK;
        debug_assert!(quants.len() == x.len() && scales.len() == blocks && offsets.len() == blocks);
        let sign = _mm256_set1_ps(-0.);
        for b in 0..blocks {
            let mut v = [_mm256_setzero_ps(); 4];
            for k in 0..4 {
                // SAFETY: block `b` is inside `x`.
                v[k] = unsafe { _mm256_loadu_ps(x.as_ptr().add(b * BLOCK + 8 * k)) };
            }
            let mut max = _mm256_andnot_ps(sign, v[0]);
            for k in 1..4 {
                max = _mm256_max_ps(max, _mm256_andnot_ps(sign, v[k]));
            }
            let m4 = _mm_max_ps(_mm256_extractf128_ps::<1>(max), _mm256_castps256_ps128(max));
            let m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
            let m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
            let amax = _mm_cvtss_f32(m4);
            let id = if amax != 0. { 127. / amax } else { 0. };
            let mul = _mm256_set1_ps(id);
            let mut q = [_mm256_setzero_si256(); 4];
            for k in 0..4 {
                let scaled = _mm256_mul_ps(v[k], mul);
                q[k] = _mm256_cvtps_epi32(_mm256_round_ps::<
                    { _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC },
                >(scaled));
            }
            let total =
                _mm256_add_epi32(_mm256_add_epi32(q[0], q[1]), _mm256_add_epi32(q[2], q[3]));
            let total = _mm_add_epi32(
                _mm256_castsi256_si128(total),
                _mm256_extracti128_si256::<1>(total),
            );
            let total = _mm_add_epi32(total, _mm_unpackhi_epi64(total, total));
            let total = _mm_add_epi32(total, _mm_shuffle_epi32::<0b10_11_00_01>(total));
            let low = _mm256_packs_epi32(q[0], q[1]);
            let high = _mm256_packs_epi32(q[2], q[3]);
            let bytes = _mm256_permutevar8x32_epi32(
                _mm256_packs_epi16(low, high),
                _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7),
            );
            // SAFETY: block `b` is inside `quants`, `scales` and `offsets`.
            unsafe {
                _mm256_storeu_si256(quants.as_mut_ptr().add(b * BLOCK).cast(), bytes);
                *scales.get_unchecked_mut(b) = f16::from_f32(amax / 127.).to_f32();
                *offsets.get_unchecked_mut(b) = -128 * _mm_cvtsi128_si32(total);
            }
        }
    }

    /// [`super::mad_f16`], eight at a time; returns how many it did.
    ///
    /// # Safety
    /// The CPU has FMA and F16C; the slices are equally long.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn mad_f16(y: &mut [f32], x: &[f32], v: f32) -> usize {
        let vv = _mm256_set1_ps(v);
        let mut i = 0;
        while i + 8 <= y.len() {
            // SAFETY: values `i..i + 8` are inside both.
            unsafe {
                let r = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x.as_ptr().add(i)),
                    vv,
                    _mm256_loadu_ps(y.as_ptr().add(i)),
                );
                let h = _mm256_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(r);
                _mm256_storeu_ps(y.as_mut_ptr().add(i), _mm256_cvtph_ps(h));
            }
            i += 8;
        }
        i
    }

    /// f16 values to f32, eight at a time.
    ///
    /// # Safety
    /// The CPU has F16C; the slices are equally long.
    #[target_feature(enable = "avx2,f16c")]
    pub unsafe fn widen(from: &[f16], to: &mut [f32]) {
        let n = from.len().min(to.len());
        let mut i = 0;
        while i + 8 <= n {
            // SAFETY: elements `i..i + 8` are inside both.
            unsafe {
                let h = _mm_loadu_si128(from.as_ptr().add(i).cast());
                _mm256_storeu_ps(to.as_mut_ptr().add(i), _mm256_cvtph_ps(h));
            }
            i += 8;
        }
        for k in i..n {
            to[k] = from[k].to_f32();
        }
    }

    /// Values rounded to the nearest f16 in place, eight at a time.
    ///
    /// # Safety
    /// The CPU has F16C.
    #[target_feature(enable = "avx2,f16c")]
    pub unsafe fn round_f16(x: &mut [f32]) {
        let (chunks, rest) = x.as_chunks_mut::<8>();
        for c in chunks {
            // SAFETY: 8 values.
            unsafe {
                let v = _mm256_loadu_ps(c.as_ptr());
                let h = _mm256_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(v);
                _mm256_storeu_ps(c.as_mut_ptr(), _mm256_cvtph_ps(h));
            }
        }
        for v in rest {
            *v = f16::from_f32(*v).to_f32();
        }
    }
}
