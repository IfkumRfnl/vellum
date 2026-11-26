//! Int8 matrix multiplies: Q8_0 weights as the file stores them (int8 values with a scale
//! per 32) times activations quantized the same way on the fly. That's what the reference's
//! CPU path computes (its 8-bit activations), and it streams half f16's weight bytes.
//!
//! Two kernels, the same arithmetic (each block's integer sum exact, scaled in f32):
//! [`Kind::Q8`] on int8 tensor cores, four times f16's peak rate (131 against 33 TOPS on an
//! RTX 4060 Laptop GPU, `cargo lab kernels`). WGSL has no int8 cooperative matrices, so it's
//! GLSL (`gemm_q8.comp`, compiled to SPIR-V by `shaders/spirv.sh`, the `.spv` checked in) and
//! runs only on Vulkan through wgpu's SPIR-V passthrough, outside naga's validation. Its
//! epilogue scales each lane's accumulator elements in registers, which needs the driver's
//! arrangement of them: [`available`] checks it, and the whole multiply against the CPU,
//! once per device before anything uses the kernel. [`Kind::Dp4a`] is portable WGSL on
//! packed int8 dot products (`gemm_dp4a.wgsl`), the fallback [`choose`] takes where SPIR-V
//! or int8 cooperative matrices aren't available; on the 4060 it runs at about 0.6× the f16
//! tensor-core kernel, so there it's never chosen.

use super::{
    Gpu, Kernel, Profiler, Tensor,
    gemm::{self, Epilogue, Gemm, Kind, Operands, Variant, Weights},
};
use crate::{Error, Result, cpu::Q8Block};
use std::sync::{Arc, Mutex};

const GEMM_Q8: &[u8] = include_bytes!("shaders/spv/gemm_q8.spv");
pub const QUANTIZE_SOURCE: &str = include_str!("shaders/quantize_q8.wgsl");
pub const DP4A_SOURCE: &str = include_str!("shaders/gemm_dp4a.wgsl");

const PEAK_I8: &[u8] = include_bytes!("shaders/spv/peak_i8.spv");
const PEAK_F16: &[u8] = include_bytes!("shaders/spv/peak_f16.spv");
const PEAK_DP4A: &str = include_str!("shaders/peak_dp4a.wgsl");

/// The adapter's peak multiply rates from synthetic loops with no memory traffic: int8
/// tensor cores (16×16×32, i32 accumulation), f16 tensor cores (16×16×16, f32 accumulation),
/// and the packed int8 dot product (`dot4I8Packed`), in tera-operations per second (a
/// multiply-add is 2). Each is the fastest of several runs.
pub fn peaks(gpu: &Gpu) -> Vec<(&'static str, Result<f64>)> {
    let groups = 24 * 32;
    let mut out = Vec::new();
    let coop = |name, spirv: &[u8], ops_per_mma: f64| -> Result<f64> {
        let iters = 1024;
        let kernel = gpu.kernel_spirv(name, spirv, &[(0, iters)], &[true], 0)?;
        let ops = groups as f64 * 4. * 8. * iters as f64 * ops_per_mma;
        Ok(ops / time(gpu, &kernel, groups)? / 1e9)
    };
    out.push((
        "int8 coop 16×16×32",
        coop("peak_i8", PEAK_I8, 16. * 16. * 32. * 2.),
    ));
    out.push((
        "f16 coop 16×16×16",
        coop("peak_f16", PEAK_F16, 16. * 16. * 16. * 2.),
    ));
    let dp4a = (|| {
        let iters = 4096;
        let groups = 24 * 64;
        let kernel = gpu.kernel("peak_dp4a", PEAK_DP4A, &[("ITERS", iters)], &[true], 0)?;
        let ops = groups as f64 * 256. * 16. * iters as f64 * 8.;
        Ok(ops / time(gpu, &kernel, groups)? / 1e9)
    })();
    out.push(("dot4I8Packed", dp4a));
    out
}

/// The fastest of several dispatches of `groups` workgroups, in milliseconds.
fn time(gpu: &Gpu, kernel: &Arc<Kernel>, groups: u32) -> Result<f64> {
    let out = gpu.zeros_u32("peak", groups as usize * 256);
    let group = gpu.bind(kernel, &[&out]);
    let runs = 8;
    let mut profiler = Profiler::new(gpu, runs);
    if !profiler.enabled() {
        return Err(Error::Unsupported("timestamp queries".into()));
    }
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    for i in 0..runs + 4 {
        // Four untimed first, to bring the clocks up.
        let mut pass = if i < 4 {
            encoder.begin_compute_pass(&Default::default())
        } else {
            profiler.pass(&mut encoder, "peak")
        };
        pass.set_pipeline(&kernel.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    gpu.queue.submit([encoder.finish()]);
    let times = profiler.read(gpu)?;
    gpu.clear_bind_groups();
    Ok(times.iter().map(|(_, t)| *t).fold(f64::INFINITY, f64::min))
}

const COOP_LAYOUT_I32: &[u8] = include_bytes!("shaders/spv/coop_layout_i32.spv");

/// Where each lane's elements of a 16×16 i32 accumulator sit: `(row, col)` per lane and
/// element, as the driver arranges them (from `coop_layout_i32.comp`).
pub fn accumulator_layout(gpu: &Gpu) -> Result<Vec<Vec<(u32, u32)>>> {
    let kernel = gpu.kernel_spirv("coop_layout_i32", COOP_LAYOUT_I32, &[], &[true], 0)?;
    let out = gpu.zeros_f32("coop layout", 32 * 16);
    let group = gpu.bind(&kernel, &[&out]);
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&kernel.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    gpu.queue.submit([encoder.finish()]);
    let words: Vec<u32> = gpu.read_f32(&out)?.iter().map(|v| v.to_bits()).collect();
    gpu.clear_bind_groups();
    Ok(words
        .chunks(16)
        .map(|lane| {
            let len = (lane[15] as usize).min(15);
            lane[..len].iter().map(|&i| (i / 16, i % 16)).collect()
        })
        .collect())
}

/// Whether this device runs [`Kind::Q8`]: Vulkan with SPIR-V passthrough and cooperative
/// matrices on 32-wide subgroups, the accumulator arrangement `gemm_q8.comp` assumes, and a
/// small multiply that matches the CPU on the same quantized values. Measured once per
/// device (`VELLUM_GPU_VERBOSE=1` prints why not).
pub fn available(gpu: &Gpu) -> bool {
    *gpu.q8_checked
        .get_or_init(|| verdict(Kind::Q8, check(gpu, Kind::Q8)))
}

/// Whether this device runs [`Kind::Dp4a`] (subgroups for the quantizer, and a small
/// multiply that matches the CPU). Measured once per device.
pub fn dp4a_available(gpu: &Gpu) -> bool {
    *gpu.dp4a_checked
        .get_or_init(|| verdict(Kind::Dp4a, check(gpu, Kind::Dp4a)))
}

fn verdict(kind: Kind, result: Result<()>) -> bool {
    match result {
        Ok(()) => true,
        Err(e) => {
            if std::env::var_os("VELLUM_GPU_VERBOSE").is_some() {
                eprintln!("{kind:?} multiply unavailable: {e}");
            }
            false
        }
    }
}

fn check(gpu: &Gpu, kind: Kind) -> Result<()> {
    if kind == Kind::Q8 {
        check_q8_layout(gpu)?;
    } else if !gpu.has(wgpu::Features::SUBGROUP) || gpu.info.subgroup_min_size < 8 {
        return Err(Error::Unsupported("subgroups of at least 8".into()));
    }
    let variant = match kind {
        Kind::Q8 => Variant::q8(64, 64, 64, 32, 32),
        _ => Variant::dp4a(64, 64, 32, 4, 4),
    };
    check_multiply(gpu, variant)
}

fn check_q8_layout(gpu: &Gpu) -> Result<()> {
    if !gpu.has_f16_tensor_cores() || gpu.info.subgroup_min_size != 32 {
        return Err(Error::Unsupported(
            "cooperative matrices on 32-wide subgroups".into(),
        ));
    }
    let layout = accumulator_layout(gpu)?;
    for (lane, elements) in layout.iter().enumerate() {
        let lane = lane as u32;
        let expected: Vec<(u32, u32)> = (0..8)
            .map(|e| {
                (
                    lane / 4 + 8 * (e / 4),
                    2 * (lane % 4) + e % 2 + 8 * ((e / 2) % 2),
                )
            })
            .collect();
        if *elements != expected {
            return Err(Error::Unsupported(format!(
                "accumulator layout (lane {lane}: {elements:?})"
            )));
        }
    }
    Ok(())
}

/// A multiply with ragged edges on `variant`, against the CPU on the same quantized values.
fn check_multiply(gpu: &Gpu, variant: Variant) -> Result<()> {
    let (m, n, k) = (77, 70, 128);
    let x = gemm::noise(m * k, 11);
    let w = gemm::noise(n * k, 12);
    let wq = quantize(&w, k);
    let weights = upload(gpu, "q8 check", (n, k), &wq.0, &wq.1, None);
    let gemm = gemm_unchecked(gpu, variant)?;
    let gx = gpu.upload_f32("q8 check x", &x);
    let gy = gpu.zeros_f32("q8 check y", m * n);
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        let op = Operands {
            x: &gx,
            weights: &weights,
            y: &gy,
            m,
        };
        gemm.encode(gpu, &mut pass, &op, Epilogue::default());
    }
    gpu.queue.submit([encoder.finish()]);
    let got = gpu.read_f32(&gy)?;
    gpu.clear_bind_groups();
    let xq = dequantize(&quantize(&x, k));
    let wq = dequantize(&wq);
    let mut worst = 0f32;
    for i in 0..m {
        for j in 0..n {
            let want: f32 = (0..k).map(|t| xq[i * k + t] * wq[j * k + t]).sum();
            worst = worst.max((got[i * n + j] - want).abs() / (1. + want.abs()));
        }
    }
    if worst.is_nan() || worst >= 1e-3 {
        return Err(Error::Other(format!("int8 multiply off by {worst:.2e}")));
    }
    Ok(())
}

/// [`Kind::Q8`]'s parts besides the multiply: the activation quantizer, its output buffers
/// (grown to the largest multiply seen), and the f16 kernel for layers without Q8_0 weights.
pub struct Parts {
    quantize: Arc<Kernel>,
    scratch: Mutex<Option<(Tensor, Tensor)>>,
    pub fallback: Box<Gemm>,
}

/// The [`Kind::Q8`] or [`Kind::Dp4a`] multiply for `variant`, if this device runs it
/// ([`available`], [`dp4a_available`]).
pub fn gemm(gpu: &Gpu, variant: Variant) -> Result<Gemm> {
    let runs = match variant.kind {
        Kind::Q8 => available(gpu),
        _ => dp4a_available(gpu),
    };
    if !runs {
        return Err(Error::Unsupported(format!(
            "the {:?} multiply on {}",
            variant.kind, gpu.info.name
        )));
    }
    gemm_unchecked(gpu, variant)
}

fn gemm_unchecked(gpu: &Gpu, v: Variant) -> Result<Gemm> {
    if v.threads() > gpu.limits.max_compute_invocations_per_workgroup
        || v.shared_bytes() > gpu.limits.max_compute_workgroup_storage_size
        || !v.bk.is_multiple_of(32)
    {
        return Err(Error::Other(format!(
            "{} needs {} threads and {} B of workgroup memory",
            v.name(),
            v.threads(),
            v.shared_bytes()
        )));
    }
    let bindings = [false, false, false, false, false, true];
    let kernel = match v.kind {
        Kind::Q8 => gpu.kernel_spirv(
            "gemm_q8",
            GEMM_Q8,
            &[
                (0, v.bm),
                (1, v.bn),
                (2, v.bk),
                (3, v.tm),
                (4, v.tn),
                (5, v.threads()),
            ],
            &bindings,
            gemm::IMMEDIATES,
        )?,
        _ => gpu.kernel(
            "gemm_dp4a",
            DP4A_SOURCE,
            &[
                ("BM", v.bm),
                ("BN", v.bn),
                ("BK", v.bk),
                ("TM", v.tm),
                ("TN", v.tn),
            ],
            &bindings,
            gemm::IMMEDIATES,
        )?,
    };
    let quantize = gpu.kernel("quantize_q8", QUANTIZE_SOURCE, &[], &[false, true, true], 4)?;
    let fallback = Gemm::new(gpu, f16_fallback(gpu))?;
    Ok(Gemm::from_q8(
        v,
        kernel,
        Parts {
            quantize,
            scratch: Mutex::new(None),
            fallback: Box::new(fallback),
        },
    ))
}

/// The f16 multiply for layers that aren't Q8_0 (the tile shape Parakeet measures best).
fn f16_fallback(gpu: &Gpu) -> Variant {
    let (kind, bm, bn, bk, tm, tn) = if gpu.has_f16_tensor_cores() {
        (Kind::Coop, 64, 128, 64, 32, 64)
    } else {
        (Kind::Simt, 128, 64, 16, 8, 4)
    };
    Variant {
        kind,
        bm,
        bn,
        bk,
        tm,
        tn,
    }
}

impl Parts {
    /// Records the quantization of `x` and the multiply, one dispatch each.
    pub(super) fn encode(
        &self,
        gpu: &Gpu,
        pass: &mut wgpu::ComputePass,
        kernel: &Kernel,
        variant: Variant,
        op: &Operands,
        epilogue: Epilogue,
    ) {
        let Operands { x, weights, y, m } = *op;
        let (n, k) = (weights.n, weights.k);
        assert!(
            k.is_multiple_of(variant.bk as usize),
            "K = {k} with {}",
            variant.name()
        );
        assert!(
            x.dtype == super::Dtype::F32 && x.len >= m * k,
            "x is {m}×{k} f32"
        );
        assert!(
            y.dtype == super::Dtype::F32 && y.len >= m * n,
            "y is {m}×{n} f32"
        );
        let blocks = m * k / 32;
        let mut scratch = self.scratch.lock().unwrap();
        if scratch.as_ref().is_none_or(|(_, s)| s.len < blocks) {
            *scratch = Some((
                gpu.zeros_u32("q8 activations", blocks * 8),
                gpu.zeros_f32("q8 activation scales", blocks),
            ));
        }
        let (values, scales) = scratch.as_ref().unwrap();
        let group = gpu.bind(&self.quantize, &[x, values, scales]);
        pass.set_pipeline(&self.quantize.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.set_immediates(0, &(blocks as u32).to_le_bytes());
        let groups = (blocks * 8).div_ceil(256) as u32;
        let (gx, gy) = if groups <= 65535 {
            (groups, 1)
        } else {
            (32768, groups.div_ceil(32768))
        };
        pass.dispatch_workgroups(gx, gy, 1);

        let weight_scales = weights.scales.as_ref().expect("Q8_0 weights");
        let bias = weights.bias.as_ref().unwrap_or(weight_scales);
        let group = gpu.bind(
            kernel,
            &[values, scales, &weights.weight, weight_scales, bias, y],
        );
        pass.set_pipeline(&kernel.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.set_immediates(0, &gemm::immediates(m, weights, epilogue));
        pass.dispatch_workgroups(
            (n as u32).div_ceil(variant.bn),
            (m as u32).div_ceil(variant.bm),
            1,
        );
    }
}

/// Q8_0 blocks as the kernel takes them: the int8 values four to a word in order, and the
/// scales.
pub fn pack(blocks: &[Q8Block]) -> (Vec<u32>, Vec<f32>) {
    let mut values = Vec::with_capacity(blocks.len() * 8);
    for b in blocks {
        for q in b.q.as_chunks::<4>().0 {
            values.push(u32::from_le_bytes(q.map(|v| v as u8)));
        }
    }
    (values, blocks.iter().map(|b| b.scale).collect())
}

/// Rows of `k` values quantized as `quantize_q8.wgsl` does it: per 32, d = max|x| / 127 and
/// round(x / d) half away from zero. Words and scales as [`pack`] gives them.
pub fn quantize(x: &[f32], k: usize) -> (Vec<u32>, Vec<f32>) {
    assert!(k.is_multiple_of(32) && x.len().is_multiple_of(k));
    let mut values = Vec::with_capacity(x.len() / 4);
    let mut scales = Vec::with_capacity(x.len() / 32);
    for block in x.as_chunks::<32>().0 {
        let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.;
        let inv = if d > 0. { 1. / d } else { 0. };
        for four in block.as_chunks::<4>().0 {
            let q = four.map(|v| {
                let s = v * inv;
                (s.signum() * (s.abs() + 0.5).floor()) as i8 as u8
            });
            values.push(u32::from_le_bytes(q));
        }
        scales.push(d);
    }
    (values, scales)
}

/// The values packed words and scales stand for: `scale · q`.
pub fn dequantize((values, scales): &(Vec<u32>, Vec<f32>)) -> Vec<f32> {
    values
        .iter()
        .enumerate()
        .flat_map(|(i, w)| {
            let d = scales[i / 8];
            w.to_le_bytes().map(|q| q as i8 as f32 * d)
        })
        .collect()
}

/// A Q8_0 layer on the GPU: `n` rows of `k` weights as [`pack`] gives them.
pub fn upload(
    gpu: &Gpu,
    label: &str,
    (n, k): (usize, usize),
    values: &[u32],
    scales: &[f32],
    bias: Option<&[f32]>,
) -> Weights {
    assert!(values.len() * 4 == n * k && scales.len() * 32 == n * k);
    Weights {
        n,
        k,
        weight: gpu.upload_u32(label, values),
        scales: Some(gpu.upload_f32(label, scales)),
        bias: bias.map(|b| gpu.upload_f32(label, b)),
        inexact: 0,
    }
}

/// The tile shape that measures best on an RTX 4060 Laptop GPU (`cargo lab kernels`).
pub const DEFAULT: Variant = Variant::q8(128, 64, 64, 64, 32);

/// The packed int8 kernel's (`cargo lab kernels --kernels dp4a`).
pub const DP4A_DEFAULT: Variant = Variant::dp4a(128, 64, 32, 8, 4);

/// Rows, columns, values four to a word and scales: a Q8_0 weight as [`pack`] gives it.
pub type Q8Weight = (usize, usize, Vec<u32>, Vec<f32>);

/// A layer's weight as the file's Q8_0, read straight from its blocks (no conversion): rows,
/// columns, values four to a word and scales as [`pack`] gives them. `None` when the weight
/// isn't Q8_0 or its rows aren't whole blocks.
pub fn load_weight(file: &crate::gguf::Gguf, name: &str) -> Result<Option<Q8Weight>> {
    use rayon::prelude::*;
    let info = crate::cpu::tensor(file, &format!("{name}.weight"))?;
    // As `Linear::load` reads shapes: a leading kernel dimension of one isn't an input.
    let mut dims: Vec<usize> = info.dims.iter().map(|d| *d as usize).collect();
    while dims.len() > 2 && dims[0] == 1 {
        dims.remove(0);
    }
    let cols = dims[0];
    let rows: usize = dims[1..].iter().product();
    if info.dtype != crate::gguf::Dtype::Q8_0 || !cols.is_multiple_of(32) {
        return Ok(None);
    }
    let blocks = file.data(info).as_chunks::<34>().0;
    if blocks.len() * 32 != rows * cols {
        return Err(Error::Gguf(format!(
            "{name}.weight: {} blocks",
            blocks.len()
        )));
    }
    let mut values = vec![0u32; rows * cols / 4];
    values
        .par_chunks_mut(8 * 4096)
        .zip(blocks.par_chunks(4096))
        .for_each(|(out, blocks)| {
            for (b, out) in blocks.iter().zip(out.as_chunks_mut::<8>().0) {
                for (o, q) in out.iter_mut().zip(b[2..].as_chunks::<4>().0) {
                    *o = u32::from_le_bytes(*q);
                }
            }
        });
    let scales = blocks
        .iter()
        .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect();
    Ok(Some((rows, cols, values, scales)))
}

/// The multiply a model's encoder uses on this adapter. `f16` is its f16 kernel; with
/// `int8` (the model's transcripts hold up under 8-bit activations, `GPU.md`) an int8 one
/// instead: [`Kind::Q8`] (int8 tensor cores, SPIR-V on Vulkan) where [`available`], else
/// [`Kind::Dp4a`] (WGSL, any backend) where it measures faster than `f16` on a large
/// multiply. Decided once per device. `VELLUM_GEMM=f16`, `=q8` or `=dp4a` overrides the choice for
/// every model (where that kernel runs), and `VELLUM_Q8_TILE=BM,BN,BK,WM,WN` sets the int8
/// tensor-core kernel's tiles (kernel work).
pub fn choose(gpu: &Gpu, f16: Variant, int8: bool) -> Variant {
    let q8 = tile().unwrap_or(DEFAULT);
    match std::env::var("VELLUM_GEMM").as_deref() {
        Ok("f16") => return f16,
        Ok("q8") if available(gpu) => return q8,
        Ok("dp4a") if dp4a_available(gpu) => return DP4A_DEFAULT,
        _ => {}
    }
    if !int8 {
        return f16;
    }
    *gpu.int8_choice.get_or_init(|| {
        // Int8 tensor cores run at four times f16's rate wherever they exist, so they're
        // taken unmeasured (measuring costs a load about 0.2 s); packed dot products only
        // where they beat the f16 kernel (not on an RTX 4060, about 0.6×).
        if available(gpu) {
            return q8;
        }
        if dp4a_available(gpu)
            && measure(gpu, DP4A_DEFAULT, f16).is_ok_and(|(int8, f16_ms)| int8 < f16_ms)
        {
            return DP4A_DEFAULT;
        }
        f16
    })
}

/// `VELLUM_Q8_TILE`, if set and well formed.
fn tile() -> Option<Variant> {
    let v: Vec<u32> = std::env::var("VELLUM_Q8_TILE")
        .ok()?
        .split(',')
        .map(|t| t.trim().parse().ok())
        .collect::<Option<_>>()?;
    match v[..] {
        [bm, bn, bk, wm, wn] => Some(Variant::q8(bm, bn, bk, wm, wn)),
        _ => None,
    }
}

/// Milliseconds of one 1400×4096×1024 multiply (Parakeet's feed-forward at 4 windows, about a
/// Turbo window's) on the int8 kernel `int8`, quantization included, and on the f16 one: the
/// fastest of several calls each, alternating so both see the same clocks.
fn measure(gpu: &Gpu, int8: Variant, f16: Variant) -> Result<(f64, f64)> {
    let (m, n, k) = (1400, 1024, 4096);
    let x = gpu.upload_f32("measure x", &gemm::noise(m * k, 1));
    let y = gpu.zeros_f32("measure y", m * n);
    let w = gemm::noise(n * k, 2);
    let wq = quantize(&w, k);
    let q8_weights = upload(gpu, "measure w", (n, k), &wq.0, &wq.1, None);
    let f16_weights = Weights {
        n,
        k,
        weight: gpu.upload_f16(
            "measure w",
            &w.iter()
                .map(|v| half::f16::from_f32(*v))
                .collect::<Vec<_>>(),
        ),
        scales: None,
        bias: None,
        inexact: 0,
    };
    let q8 = Gemm::new(gpu, int8)?;
    let f16 = Gemm::new(gpu, f16)?;
    let op = |weights| Operands {
        x: &x,
        weights,
        y: &y,
        m,
    };
    f16.warm_up(
        gpu,
        &op(&f16_weights),
        std::time::Duration::from_millis(100),
    )?;
    let (mut a, mut b) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..3 {
        a = a.min(
            q8.time(gpu, &op(&q8_weights), Epilogue::default(), 1, 4)?
                .min,
        );
        b = b.min(
            f16.time(gpu, &op(&f16_weights), Epilogue::default(), 1, 4)?
                .min,
        );
    }
    gpu.clear_bind_groups();
    Ok((a, b))
}
