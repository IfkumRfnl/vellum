//! The matrix multiply every linear layer runs on: `Y[M,N] = X[M,K] · W[N,K]ᵀ + b` with f32
//! activations, f16 weights, f32 accumulation and output, and a fused epilogue (bias, ReLU or
//! SiLU, a scale, and accumulation into Y for residual adds).
//!
//! Two kernels: [`Kind::Simt`], portable (register tiles over workgroup memory, plain f32
//! arithmetic), and [`Kind::Coop`], tensor cores through WGSL cooperative matrices
//! (`coop_mat16x16`, f16 × f16 → f32), where [`Gpu::has_f16_tensor_cores`]. Shapes are
//! immediates, never compiled in: one pipeline per tile variant, chosen per adapter by
//! [`Gemm::autotune`].

use super::{Dtype, Gpu, Kernel, Profiler, Tensor};
use crate::{
    Error, Result,
    cpu::{Linear, Weight},
};
use half::f16;
use std::{sync::Arc, time::Duration};

pub const SIMT_SOURCE: &str = include_str!("shaders/gemm_simt.wgsl");
pub const COOP_SOURCE: &str = include_str!("shaders/gemm_coop.wgsl");

/// Bytes of immediate data both kernels take (`Params` in the shaders).
const IMMEDIATES: u32 = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    Simt,
    Coop,
}

/// A kernel and its tile shape: the output tile (`bm`×`bn`), the K slice staged per step
/// (`bk`), and per thread (SIMT: `tm`×`tn` outputs) or per subgroup (cooperative: `tm`×`tn`
/// outputs in 16×16 accumulators).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Variant {
    pub kind: Kind,
    pub bm: u32,
    pub bn: u32,
    pub bk: u32,
    pub tm: u32,
    pub tn: u32,
}

impl Variant {
    const fn simt(bm: u32, bn: u32, bk: u32, tm: u32, tn: u32) -> Variant {
        Variant {
            kind: Kind::Simt,
            bm,
            bn,
            bk,
            tm,
            tn,
        }
    }

    const fn coop(bm: u32, bn: u32, bk: u32, wm: u32, wn: u32) -> Variant {
        Variant {
            kind: Kind::Coop,
            bm,
            bn,
            bk,
            tm: wm,
            tn: wn,
        }
    }

    /// The candidates autotuning measures.
    pub fn candidates(kind: Kind) -> &'static [Variant] {
        const SIMT: &[Variant] = &[
            Variant::simt(64, 64, 16, 4, 4),
            Variant::simt(64, 64, 32, 4, 4),
            Variant::simt(128, 64, 16, 8, 4),
            Variant::simt(64, 128, 16, 4, 8),
            Variant::simt(128, 128, 16, 8, 8),
            Variant::simt(128, 128, 32, 8, 8),
        ];
        const COOP: &[Variant] = &[
            Variant::coop(64, 64, 32, 32, 32),
            Variant::coop(64, 64, 64, 32, 32),
            Variant::coop(128, 64, 32, 64, 32),
            Variant::coop(64, 128, 32, 32, 64),
            Variant::coop(128, 128, 32, 64, 64),
            Variant::coop(128, 128, 32, 32, 64),
        ];
        match kind {
            Kind::Simt => SIMT,
            Kind::Coop => COOP,
        }
    }

    pub fn threads(&self) -> u32 {
        match self.kind {
            Kind::Simt => (self.bm / self.tm) * (self.bn / self.tn),
            Kind::Coop => (self.bm / self.tm) * (self.bn / self.tn) * 32,
        }
    }

    /// Workgroup memory the kernel declares.
    pub fn shared_bytes(&self) -> u32 {
        match self.kind {
            Kind::Simt => (self.bm + self.bn) * (self.bk + 4) * 4,
            Kind::Coop => (self.bm + self.bn) * (self.bk + 8) * 2 + self.threads() / 32 * 1024,
        }
    }

    /// "simt 128×128×16 t8×8".
    pub fn name(&self) -> String {
        let (kind, per) = match self.kind {
            Kind::Simt => ("simt", "t"),
            Kind::Coop => ("coop", "sg"),
        };
        format!(
            "{kind} {}×{}×{} {per}{}×{}",
            self.bm, self.bn, self.bk, self.tm, self.tn
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    None,
    Relu,
    Silu,
}

/// What happens to `x·Wᵀ` before it's stored: `v = scale · act(x·Wᵀ + b)`, then `Y = v`, or
/// `Y += v` when `accumulate` (a residual add, in place on the residual stream).
#[derive(Debug, Clone, Copy)]
pub struct Epilogue {
    pub activation: Activation,
    pub scale: f32,
    pub accumulate: bool,
}

impl Default for Epilogue {
    fn default() -> Epilogue {
        Epilogue {
            activation: Activation::None,
            scale: 1.,
            accumulate: false,
        }
    }
}

/// A linear layer on the GPU: `n` output rows of `k` f16 weights, and an optional f32 bias.
pub struct Weights {
    pub n: usize,
    pub k: usize,
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    /// How many weights f16 couldn't hold exactly (a Q8_0 value is an f16 scale times an int8,
    /// which can need up to 18 significant bits); each is off by at most half an f16 ulp.
    pub inexact: usize,
}

impl Weights {
    /// Uploads a CPU layer's weight (as f16) and bias.
    pub fn upload(gpu: &Gpu, label: &str, linear: &Linear) -> Weights {
        let (w, inexact) = to_f16(linear);
        Weights {
            n: linear.rows,
            k: linear.cols,
            weight: gpu.upload_f16(label, &w),
            bias: linear.bias.as_ref().map(|b| gpu.upload_f32(label, b)),
            inexact,
        }
    }
}

/// A layer's weight as row-major f16, and how many values rounded.
pub fn to_f16(linear: &Linear) -> (Vec<f16>, usize) {
    let mut inexact = 0;
    let mut round = |v: f32| {
        let h = f16::from_f32(v);
        inexact += usize::from(h.to_f32() != v);
        h
    };
    let w = match &linear.weight {
        Weight::F32(w) => w.iter().map(|v| round(*v)).collect(),
        Weight::Q8(blocks) => blocks
            .iter()
            .flat_map(|b| b.q.map(|q| b.scale * q as f32))
            .map(&mut round)
            .collect(),
    };
    (w, inexact)
}

/// One compiled GEMM variant.
pub struct Gemm {
    pub variant: Variant,
    kernel: Arc<Kernel>,
}

impl Gemm {
    pub fn new(gpu: &Gpu, variant: Variant) -> Result<Gemm> {
        let v = variant;
        if v.threads() > gpu.limits.max_compute_invocations_per_workgroup
            || v.shared_bytes() > gpu.limits.max_compute_workgroup_storage_size
        {
            return Err(Error::Other(format!(
                "{} needs {} threads and {} B of workgroup memory",
                v.name(),
                v.threads(),
                v.shared_bytes()
            )));
        }
        let coop_source;
        let (name, source, consts) = match v.kind {
            Kind::Simt => (
                "gemm_simt",
                SIMT_SOURCE,
                [
                    ("BM", v.bm),
                    ("BN", v.bn),
                    ("BK", v.bk),
                    ("TM", v.tm),
                    ("TN", v.tn),
                ],
            ),
            Kind::Coop => {
                if !gpu.has_f16_tensor_cores() {
                    return Err(Error::Unsupported(format!(
                        "cooperative matrices on {}",
                        gpu.info.name
                    )));
                }
                coop_source = coop_shader(v);
                (
                    "gemm_coop",
                    coop_source.as_str(),
                    [
                        ("BM", v.bm),
                        ("BN", v.bn),
                        ("BK", v.bk),
                        ("WM", v.tm),
                        ("WN", v.tn),
                    ],
                )
            }
        };
        let kernel = gpu.kernel(
            name,
            source,
            &consts,
            &[false, false, false, true],
            IMMEDIATES,
        )?;
        Ok(Gemm { variant, kernel })
    }

    /// Records `y = epilogue(x · Wᵀ + b)` into `pass`. `k` must be a multiple of the
    /// variant's `bk` (all of Parakeet's are: 640, 1024, 4096).
    pub fn encode(
        &self,
        gpu: &Gpu,
        pass: &mut wgpu::ComputePass,
        op: &Operands,
        epilogue: Epilogue,
    ) {
        let Operands { x, weights, y, m } = *op;
        let (n, k) = (weights.n, weights.k);
        assert!(
            k.is_multiple_of(self.variant.bk as usize),
            "K = {k} with {}",
            self.variant.name()
        );
        assert!(x.dtype == Dtype::F32 && x.len >= m * k, "x is {m}×{k} f32");
        assert!(y.dtype == Dtype::F32 && y.len >= m * n, "y is {m}×{n} f32");
        let bias = weights.bias.as_ref().unwrap_or(&weights.weight);
        let group = gpu.bind(&self.kernel, &[x, &weights.weight, bias, y]);
        let flags = u32::from(weights.bias.is_some())
            | match epilogue.activation {
                Activation::None => 0,
                Activation::Relu => 1 << 1,
                Activation::Silu => 2 << 1,
            }
            | (u32::from(epilogue.accumulate) << 3);
        let mut immediates = Vec::with_capacity(IMMEDIATES as usize);
        for v in [
            m as u32,
            n as u32,
            k as u32,
            flags,
            epilogue.scale.to_bits(),
        ] {
            immediates.extend_from_slice(&v.to_le_bytes());
        }
        pass.set_pipeline(&self.kernel.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.set_immediates(0, &immediates);
        pass.dispatch_workgroups(
            (n as u32).div_ceil(self.variant.bn),
            (m as u32).div_ceil(self.variant.bm),
            1,
        );
    }

    /// The fastest candidate of `kind` on this adapter, measured on Parakeet's two biggest
    /// shapes (feed-forward up and down at a 4-window batch) and a one-window 1024×1024.
    pub fn autotune(gpu: &Gpu, kind: Kind) -> Result<Tuned> {
        let shapes = [(1400, 1024, 4096), (1400, 4096, 1024), (350, 1024, 1024)];
        let (max_m, max_kn) = (1400, 4096 * 1024);
        let x = gpu.upload_f32("autotune x", &noise(max_m * 4096, 1));
        let y = gpu.zeros_f32("autotune y", max_m * 4096);
        let w: Vec<f16> = noise(max_kn, 2)
            .iter()
            .map(|v| f16::from_f32(v * 0.03))
            .collect();
        let weights: Vec<Weights> = shapes
            .iter()
            .map(|&(_, n, k)| Weights {
                n,
                k,
                weight: gpu.upload_f16("autotune w", &w[..n * k]),
                bias: None,
                inexact: 0,
            })
            .collect();
        let ops: Vec<Operands> = shapes
            .iter()
            .zip(&weights)
            .map(|(&(m, _, _), weights)| Operands {
                x: &x,
                weights,
                y: &y,
                m,
            })
            .collect();
        let mut candidates = Vec::new();
        let mut best: Option<(Gemm, f64)> = None;
        // Calls per shape: ten, or fewer on a slow adapter, so tuning stays a few seconds.
        let mut runs = 0;
        for &variant in Variant::candidates(kind) {
            let time = Gemm::new(gpu, variant).and_then(|gemm| {
                if runs == 0 {
                    gemm.warm_up(gpu, &ops[0], Duration::from_millis(300))?;
                    let once = gemm.time(gpu, &ops[0], Epilogue::default(), 0, 1)?;
                    runs = (40. / once.min).clamp(2., 10.) as u32;
                }
                let mut total = 0.;
                for op in &ops {
                    // The fastest of several calls: other work on the GPU only ever adds time.
                    total += gemm.time(gpu, op, Epilogue::default(), 1, runs)?.min;
                    // Already slower than the best: the rest of its time doesn't matter.
                    if best.as_ref().is_some_and(|(_, t)| total > *t) {
                        break;
                    }
                }
                if best.as_ref().is_none_or(|(_, t)| total < *t) {
                    best = Some((gemm, total));
                }
                Ok(total)
            });
            candidates.push((variant, time));
        }
        gpu.clear_bind_groups();
        let (gemm, _) = best.ok_or_else(|| {
            let why = candidates
                .iter()
                .find_map(|(_, r)| r.as_ref().err())
                .map_or(String::new(), |e| format!(": {e}"));
            Error::Other(format!(
                "no {kind:?} GEMM variant runs on {}{why}",
                gpu.info.name
            ))
        })?;
        Ok(Tuned { gemm, candidates })
    }

    /// Runs this multiply back to back for `duration`, so the GPU is at its working clocks
    /// before anything is timed (a laptop GPU idles low and takes a while to ramp up).
    pub fn warm_up(&self, gpu: &Gpu, op: &Operands, duration: Duration) -> Result<()> {
        let started = std::time::Instant::now();
        while started.elapsed() < duration {
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            for _ in 0..10 {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                self.encode(gpu, &mut pass, op, Epilogue::default());
            }
            gpu.queue.submit([encoder.finish()]);
            gpu.wait()?;
        }
        Ok(())
    }

    /// GPU milliseconds of one call, over `runs` timed passes after `warmup` untimed ones.
    /// Every call is its own pass, timestamped at both ends.
    pub fn time(
        &self,
        gpu: &Gpu,
        op: &Operands,
        epilogue: Epilogue,
        warmup: u32,
        runs: u32,
    ) -> Result<Timing> {
        let mut profiler = Profiler::new(gpu, runs);
        if !profiler.enabled() {
            return Err(Error::Unsupported(format!(
                "timestamp queries on {}",
                gpu.info.name
            )));
        }
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        for _ in 0..warmup {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            self.encode(gpu, &mut pass, op, epilogue);
        }
        for _ in 0..runs {
            let mut pass = profiler.pass(&mut encoder, "gemm");
            self.encode(gpu, &mut pass, op, epilogue);
        }
        gpu.queue.submit([encoder.finish()]);
        let mut times: Vec<f64> = profiler.read(gpu)?.into_iter().map(|(_, t)| t).collect();
        times.sort_by(f64::total_cmp);
        Ok(Timing {
            median: times[times.len() / 2],
            min: times[0],
        })
    }
}

/// What one multiply reads and writes: `m` rows of `x` (`m`×`k` f32) times the layer's
/// weights into `y` (`m`×`n` f32; also read when the epilogue accumulates).
#[derive(Clone, Copy)]
pub struct Operands<'a> {
    pub x: &'a Tensor,
    pub weights: &'a Weights,
    pub y: &'a Tensor,
    pub m: usize,
}

/// What autotuning chose, and every candidate's milliseconds over the tuning shapes (the
/// sum of each shape's fastest call, or why it can't run). A candidate stops as soon as it's
/// behind the best so far, so a loser's figure can be a partial sum (still above the winner).
pub struct Tuned {
    pub gemm: Gemm,
    pub candidates: Vec<(Variant, Result<f64>)>,
}

/// GPU milliseconds per call over a set of timed calls.
#[derive(Debug, Clone, Copy)]
pub struct Timing {
    pub median: f64,
    /// The fastest call: what the kernel does when nothing else shares the GPU.
    pub min: f64,
}

/// One accumulator's epilogue: the 16×16 block of Y at (ROW0, COL0), through this subgroup's
/// patch of workgroup memory (every subgroup runs it equally often: it has barriers).
const FINISH: &str = "    coopStoreT(ACC, &cs[sg * 256u], 16u);
    workgroupBarrier();
    for (var e = 0u; e < 8u; e++) {
        let i = lane + e * 32u;
        let row = ROW0 + i / 16u;
        let col = COL0 + i % 16u;
        if row < p.m && col < p.n {
            var v = cs[sg * 256u + i];
            if has_bias {
                v += bias[col];
            }
            v = activate(v, kind) * p.scale;
            let at = row * p.n + col;
            if accumulate {
                v += y[at];
            }
            y[at] = v;
        }
    }
    workgroupBarrier();
";

/// The cooperative kernel for one variant: `COOP_SOURCE` with its accumulators declared, its
/// multiply unrolled and its epilogue called, for the variant's `(WM/16)×(WN/16)` accumulators
/// per subgroup and `BK/16` steps per slice.
fn coop_shader(v: Variant) -> String {
    let (fm, fn_, steps) = (v.tm / 16, v.tn / 16, v.bk / 16);
    let tiles = || (0..fm).flat_map(move |i| (0..fn_).map(move |j| (i, j)));
    let mut accumulators = String::new();
    let mut multiply = String::new();
    let mut epilogue = String::new();
    for (i, j) in tiles() {
        accumulators.push_str(&format!("    var acc{i}_{j} = coop_mat16x16<f32, C>();\n"));
        epilogue.push_str(
            &FINISH
                .replace("ACC", &format!("acc{i}_{j}"))
                .replace("ROW0", &format!("m0 + wm0 + {}u", i * 16))
                .replace("COL0", &format!("n0 + wn0 + {}u", j * 16)),
        );
    }
    for step in 0..steps {
        let kk = step * 16;
        for i in 0..fm {
            multiply.push_str(&format!(
                "        let a{step}_{i} = coopLoadT<coop_mat16x16<f16, A>>(&xs[(wm0 + {}u) * LD + {kk}u], LD);\n",
                i * 16
            ));
        }
        for j in 0..fn_ {
            multiply.push_str(&format!(
                "        let b{step}_{j} = coopLoad<coop_mat16x16<f16, B>>(&ws[(wn0 + {}u) * LD + {kk}u], LD);\n",
                j * 16
            ));
        }
        for (i, j) in tiles() {
            multiply.push_str(&format!(
                "        acc{i}_{j} = coopMultiplyAdd(a{step}_{i}, b{step}_{j}, acc{i}_{j});\n"
            ));
        }
    }
    COOP_SOURCE
        .replace("    //@ACCUMULATORS\n", &accumulators)
        .replace("        //@MULTIPLY\n", &multiply)
        .replace("    //@EPILOGUE\n", &epilogue)
}

/// Deterministic values in [-1, 1) (a xorshift stream), for measurements that only need
/// realistic magnitudes.
pub fn noise(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 23) as f32 - 1.
        })
        .collect()
}
