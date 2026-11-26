//! `cargo lab kernels`: the GPU kernels on Parakeet's real shapes, against the CPU, timed.
//!
//! Every linear layer of the encoder (and the joint network's encoder projection) at one
//! window (M = 350 frames) and batches of 4 and 8 (1400, 2800), with the real weights from the
//! GGUF and synthetic activations at the scale each layer sees. Each kernel's output is checked
//! against `cpu::Linear::forward` with the exactly dequantized weights — relative L2 error and
//! max |Δ| — next to the floor its rounding can't go under (weights to f16 for both kernels,
//! activations to f16 too for the cooperative one), then timed with GPU timestamps (median of
//! `--runs` calls after warm-up). The kernel dev loop: one command, seconds.

use crate::{models, paths, report::Report, system};
use anyhow::{Context, Result};
use std::{
    fmt::Write,
    time::{Duration, Instant},
};
use vellum_engine::{
    cpu::{Linear, Mat, Weight, relu, silu},
    gguf::Gguf,
    gpu::{
        self, Gpu, GpuOptions,
        gemm::{Activation, Epilogue, Gemm, Kind, Operands, Tuned, Variant, Weights, noise},
    },
};

pub struct Args {
    /// `simt`, `coop` (default both, where the adapter has them).
    pub kernels: Vec<String>,
    /// Rows per call (frames in a batch).
    pub rows: Vec<usize>,
    /// Only shapes whose label contains this.
    pub filter: Option<String>,
    pub runs: u32,
    /// Passes over every measurement; each keeps its best median.
    pub rounds: u32,
    /// An adapter index or name part (overrides `VELLUM_GPU`).
    pub adapter: Option<String>,
    /// `vulkan`, `dx12`, `metal` (overrides `VELLUM_GPU_BACKEND`).
    pub backend: Option<String>,
}

/// What the activations entering a layer look like.
#[derive(Clone, Copy)]
enum Input {
    /// Layer-normed: zero mean, unit RMS.
    Normed,
    /// After SiLU (feed-forward hidden, the convolution module's output).
    Silu,
    /// After ReLU (the subsampling's output).
    Relu,
}

struct Shape {
    label: &'static str,
    /// The layers multiplied as one (q, k and v are fused into one 3072-row multiply).
    tensors: &'static [&'static str],
    input: Input,
    epilogue: Epilogue,
}

const fn epilogue(activation: Activation, scale: f32, accumulate: bool) -> Epilogue {
    Epilogue {
        activation,
        scale,
        accumulate,
    }
}

const SHAPES: &[Shape] = &[
    Shape {
        label: "ff.linear1 + SiLU",
        tensors: &["enc.blocks.0.ff1.linear1"],
        input: Input::Normed,
        epilogue: epilogue(Activation::Silu, 1., false),
    },
    Shape {
        label: "ff.linear2, ½ + residual",
        tensors: &["enc.blocks.0.ff1.linear2"],
        input: Input::Silu,
        epilogue: epilogue(Activation::None, 0.5, true),
    },
    Shape {
        label: "attn.qkv (fused)",
        tensors: &[
            "enc.blocks.0.attn.linear_q",
            "enc.blocks.0.attn.linear_k",
            "enc.blocks.0.attn.linear_v",
        ],
        input: Input::Normed,
        epilogue: epilogue(Activation::None, 1., false),
    },
    Shape {
        label: "attn.out + residual",
        tensors: &["enc.blocks.0.attn.linear_out"],
        input: Input::Normed,
        epilogue: epilogue(Activation::None, 1., true),
    },
    Shape {
        label: "attn.pos",
        tensors: &["enc.blocks.0.attn.linear_pos"],
        input: Input::Normed,
        epilogue: epilogue(Activation::None, 1., false),
    },
    Shape {
        label: "conv.pointwise1",
        tensors: &["enc.blocks.0.conv.pointwise1"],
        input: Input::Normed,
        epilogue: epilogue(Activation::None, 1., false),
    },
    Shape {
        label: "conv.pointwise2 + residual",
        tensors: &["enc.blocks.0.conv.pointwise2"],
        input: Input::Silu,
        epilogue: epilogue(Activation::None, 1., true),
    },
    Shape {
        label: "pre_encode.out",
        tensors: &["enc.pre_encode.out"],
        input: Input::Relu,
        epilogue: epilogue(Activation::None, 1., false),
    },
    Shape {
        label: "joint.enc",
        tensors: &["joint.enc"],
        input: Input::Normed,
        epilogue: epilogue(Activation::None, 1., false),
    },
];

/// One measurement: a shape at some M on one kernel.
#[derive(serde::Serialize)]
struct Row {
    shape: String,
    m: usize,
    n: usize,
    k: usize,
    kernel: String,
    rel_l2: f64,
    max_abs: f64,
    /// The error the kernel's rounding alone gives (CPU, same rounding, exact arithmetic).
    floor: f64,
    /// Whether the layer adds into its output (which it then reads as well as writes).
    accumulate: bool,
    /// Median GPU milliseconds per call (the best of the rounds).
    ms: f64,
    /// The fastest call seen.
    min_ms: f64,
    tflops: f64,
    gbps: f64,
    passed: bool,
}

pub fn kernels(args: &Args) -> Result<bool> {
    let mut options = GpuOptions::from_env();
    if args.adapter.is_some() {
        options.adapter = args.adapter.clone();
    }
    if let Some(b) = &args.backend {
        options.backends = Some(gpu::wgpu::Backends::from_comma_list(b));
    }
    options.cache_dir = Some(paths::root()?.join("gpu-cache"));
    let machine = system::machine();
    let mut report = Report::new("kernels", serde_json::to_value(&machine)?);

    let mut adapters = String::new();
    for (i, a) in gpu::adapters().iter().enumerate() {
        writeln!(
            adapters,
            "{i}: {} ({:?}, {:?}, {})",
            a.name, a.backend, a.device_type, a.driver_info
        )?;
    }
    println!("adapters (VELLUM_GPU=<index or name>):\n{adapters}");
    let started = Instant::now();
    let gpu = Gpu::new(&options)?;
    let device_ms = started.elapsed().as_secs_f64() * 1e3;
    let description = gpu.describe();
    println!("{description}\ndevice in {device_ms:.0} ms");
    report.section(
        "Device",
        format!(
            "```\n{adapters}\n{description}\ndevice in {device_ms:.0} ms, pipeline cache {}\n",
            gpu.pipeline_cache_path()
                .map_or("off".into(), |p| p.display().to_string())
        ),
    );

    let wanted = |k: &str| args.kernels.is_empty() || args.kernels.iter().any(|w| w == k);
    let mut kinds = Vec::new();
    if wanted("simt") {
        kinds.push(Kind::Simt);
    }
    if wanted("coop") {
        if gpu.has_f16_tensor_cores() {
            kinds.push(Kind::Coop);
        } else {
            println!("no f16 16×16×16 cooperative matrices on this adapter: SIMT only");
        }
    }

    // Every candidate compiled up front, to see what the pipeline cache saves (a second run
    // should take a fraction of the first).
    let started = Instant::now();
    let mut compiled = 0;
    for &kind in &kinds {
        for &variant in Variant::candidates(kind) {
            compiled += usize::from(Gemm::new(&gpu, variant).is_ok());
        }
    }
    let compile = format!(
        "{compiled} kernels compiled in {:.0} ms",
        started.elapsed().as_secs_f64() * 1e3
    );
    println!("{compile}");

    // Autotune: each kind's tile variants on three representative shapes.
    let mut gemms = Vec::new();
    let mut tune = String::from("| kernel | variant | ms (3 shapes) |\n| --- | --- | ---: |\n");
    for &kind in &kinds {
        let started = Instant::now();
        let Tuned { gemm, candidates } = Gemm::autotune(&gpu, kind)?;
        for (variant, time) in &candidates {
            let cell = match time {
                Ok(ms) => format!("{ms:.3}"),
                Err(e) => format!("— ({})", first_line(&e.to_string())),
            };
            let mark = if *variant == gemm.variant { " ←" } else { "" };
            writeln!(tune, "| {kind:?} | {}{mark} | {cell} |", variant.name())?;
        }
        println!(
            "{kind:?}: {} (autotuned in {:.1} s)",
            gemm.variant.name(),
            started.elapsed().as_secs_f64()
        );
        gemms.push(gemm);
    }
    gpu.save_pipeline_cache()?;
    print!("{tune}");
    if let Some((_, device)) = report.sections.iter_mut().find(|(t, _)| t == "Device") {
        device.push_str(&format!(
            "{compile}
```"
        ));
    }
    report.section("Autotune", tune);

    let path = models::get("parakeet")?.ensure()?;
    let file = Gguf::open(&path)?;
    let rows_list = if args.rows.is_empty() {
        vec![350, 1400, 2800]
    } else {
        args.rows.clone()
    };
    let mut rows = Vec::new();
    let mut weights_note = String::new();
    // Correctness first, for every shape, M and kernel; then the timing, apart from the CPU's
    // work: on a laptop the CPU reference draws power the GPU would otherwise clock up with.
    let mut layers = Vec::new();
    for shape in SHAPES {
        if args
            .filter
            .as_deref()
            .is_some_and(|f| !shape.label.contains(f))
        {
            continue;
        }
        let linear = fused(&file, shape.tensors)?;
        let weights = Weights::upload(&gpu, shape.label, &linear);
        let (n, k) = (linear.rows, linear.cols);
        writeln!(
            weights_note,
            "- {} ({}×{}): {} of {} weights ({:.2}%) not exact in f16",
            shape.label,
            k,
            n,
            weights.inexact,
            n * k,
            100. * weights.inexact as f64 / (n * k) as f64
        )?;
        let w16 = rounded(&linear);
        for &m in &rows_list {
            let x = input(shape.input, m, k, m as u64 ^ k as u64);
            let reference = apply(&linear.forward(&x), shape.epilogue);
            // The rounding floors: f16 weights, and f16 weights with f16 activations.
            let floor_w = rel_l2(
                &apply(&w16.forward(&x), shape.epilogue).data,
                &reference.data,
            );
            let floor_wx = rel_l2(
                &apply(&w16.forward(&f16_mat(&x)), shape.epilogue).data,
                &reference.data,
            );
            let residual: Vec<f32> = noise(m * n, 7).iter().map(|v| v * 4.).collect();
            let gx = gpu.upload_f32("x", &x.data);
            for gemm in &gemms {
                let gy = if shape.epilogue.accumulate {
                    gpu.upload_f32("y", &residual)
                } else {
                    gpu.zeros_f32("y", m * n)
                };
                let mut encoder = gpu.device.create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    let op = Operands {
                        x: &gx,
                        weights: &weights,
                        y: &gy,
                        m,
                    };
                    gemm.encode(&gpu, &mut pass, &op, shape.epilogue);
                }
                gpu.queue.submit([encoder.finish()]);
                let mut out = gpu.read_f32(&gy)?;
                if shape.epilogue.accumulate {
                    // Compare what the multiply added, not the residual it was added to.
                    for (o, r) in out.iter_mut().zip(&residual) {
                        *o -= r;
                    }
                }
                let err = rel_l2(&out, &reference.data);
                let max_abs = out
                    .iter()
                    .zip(&reference.data)
                    .map(|(a, b)| (a - b).abs() as f64)
                    .fold(0., f64::max);
                let floor = match gemm.variant.kind {
                    Kind::Simt => floor_w,
                    Kind::Coop => floor_wx,
                };
                rows.push(Row {
                    shape: shape.label.into(),
                    m,
                    n,
                    k,
                    kernel: gemm.variant.name(),
                    rel_l2: err,
                    max_abs,
                    floor,
                    accumulate: shape.epilogue.accumulate,
                    ms: f64::NAN,
                    min_ms: f64::NAN,
                    tflops: f64::NAN,
                    gbps: f64::NAN,
                    // Within the rounding's own error, with room for f32 summation order.
                    passed: err.is_finite() && err <= 1.5 * floor + 1e-5,
                });
            }
            gpu.clear_bind_groups();
        }
        layers.push((shape, weights));
    }

    // Timing: `rounds` passes over every measurement, keeping each one's best median, so a
    // burst of someone else's work on the GPU (another lab run, the app) spoils one round at
    // most. The fastest single call is kept too.
    let mut warm = false;
    for _ in 0..args.rounds.max(1) {
        let mut timed = rows.iter_mut();
        for (shape, weights) in &layers {
            let (n, k) = (weights.n, weights.k);
            for &m in &rows_list {
                let x = input(shape.input, m, k, m as u64 ^ k as u64);
                let gx = gpu.upload_f32("x", &x.data);
                let gy = gpu.zeros_f32("y", m * n);
                let op = Operands {
                    x: &gx,
                    weights,
                    y: &gy,
                    m,
                };
                for gemm in &gemms {
                    if !warm {
                        gemm.warm_up(&gpu, &op, Duration::from_millis(500))?;
                        warm = true;
                    }
                    let row = timed.next().context("a row per measurement")?;
                    let t = gemm.time(&gpu, &op, shape.epilogue, 5, args.runs)?;
                    if row.ms.is_nan() || t.median < row.ms {
                        row.ms = t.median;
                    }
                    if row.min_ms.is_nan() || t.min < row.min_ms {
                        row.min_ms = t.min;
                    }
                }
                gpu.clear_bind_groups();
            }
        }
    }
    for row in &mut rows {
        let (m, n, k) = (row.m, row.n, row.k);
        let flops = 2. * (m * n * k) as f64;
        let y_bytes = m * n * 4 * (1 + usize::from(row.accumulate));
        let bytes = (n * k * 2 + m * k * 4 + y_bytes) as f64;
        row.tflops = flops / (row.ms * 1e-3) / 1e12;
        row.gbps = bytes / (row.ms * 1e-3) / 1e9;
        println!(
            "{:<28} {:>5} {:>5}×{:<5} {:<24} rel L2 {:.2e} (floor {:.2e}) max|Δ| {:.2e} {:>7.3} ms (min {:>7.3}) {:>6.2} TFLOP/s {:>4.0} GB/s{}",
            row.shape,
            m,
            k,
            n,
            row.kernel,
            row.rel_l2,
            row.floor,
            row.max_abs,
            row.ms,
            row.min_ms,
            row.tflops,
            row.gbps,
            if row.passed { "" } else { "  ✗" }
        );
    }
    gpu.save_pipeline_cache()?;

    let mut table = String::from(
        "| shape | M | K×N | kernel | rel L2 | floor | max abs Δ | ms | min ms | TFLOP/s | GB/s |\n\
         | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
    );
    for r in &rows {
        writeln!(
            table,
            "| {} | {} | {}×{} | {} | {:.2e}{} | {:.2e} | {:.2e} | {:.3} | {:.3} | {:.2} | {:.0} |",
            r.shape,
            r.m,
            r.k,
            r.n,
            r.kernel,
            r.rel_l2,
            if r.passed { "" } else { " ✗" },
            r.floor,
            r.max_abs,
            r.ms,
            r.min_ms,
            r.tflops,
            r.gbps
        )?;
    }
    report.section("Kernels", table);
    report.section(
        "Notes",
        format!(
            "Error is against `cpu::Linear::forward` with the exactly dequantized weights. The \
             floor is the same product on the CPU with the kernel's rounding (weights to f16; \
             for the cooperative kernel activations to f16 as well); a kernel passes within 1.5× \
             its floor + 1e-5. Accumulating layers are checked on what they added to a random \
             residual. Times are GPU timestamps around one call: the median of {} calls after 5 \
             warm-up calls, the best of {} rounds over all measurements (TFLOP/s and GB/s \
             from it), and the fastest call seen. GB/s counts weights, activations and output \
             once (output twice when accumulating).\n\n{weights_note}",
            args.runs, args.rounds
        ),
    );
    report.passed = rows.iter().all(|r| r.passed);
    report.results = serde_json::json!({
        "adapter": gpu.name(),
        "variants": gemms.iter().map(|g| g.variant.name()).collect::<Vec<_>>(),
        "rows": rows,
    });
    println!("{}", report.save()?.display());
    Ok(report.passed)
}

/// Several layers stacked into one (their rows in order), like the fused q, k, v.
fn fused(file: &Gguf, names: &[&str]) -> Result<Linear> {
    let layers: Vec<Linear> = names
        .iter()
        .map(|n| Linear::load(file, n).with_context(|| format!("loading {n}")))
        .collect::<Result<_>>()?;
    if layers.len() == 1 {
        return Ok(layers.into_iter().next().unwrap());
    }
    let cols = layers[0].cols;
    let rows = layers.iter().map(|l| l.rows).sum();
    let mut blocks = Vec::new();
    let mut bias = Vec::new();
    for l in &layers {
        let Weight::Q8(b) = &l.weight else {
            anyhow::bail!("fusing works on Q8_0 layers");
        };
        blocks.extend_from_slice(b);
        bias.extend_from_slice(l.bias.as_deref().context("fused layers have biases")?);
    }
    Ok(Linear {
        rows,
        cols,
        weight: Weight::Q8(blocks),
        bias: Some(bias),
    })
}

/// The layer with its weights rounded to f16 (as the GPU holds them).
fn rounded(linear: &Linear) -> Linear {
    let (w, _) = gpu::gemm::to_f16(linear);
    Linear {
        rows: linear.rows,
        cols: linear.cols,
        weight: Weight::F32(w.iter().map(|v| v.to_f32()).collect()),
        bias: linear.bias.clone(),
    }
}

fn f16_mat(x: &Mat) -> Mat {
    let mut out = x.clone();
    out.map(|v| gpu::f16::from_f32(v).to_f32());
    out
}

/// `m` rows of activations at the scale a layer sees.
fn input(kind: Input, m: usize, k: usize, seed: u64) -> Mat {
    // Uniform noise scaled to unit RMS.
    let mut x = Mat::from_vec(m, k, noise(m * k, seed));
    let s = 3f32.sqrt();
    match kind {
        Input::Normed => x.map(|v| v * s),
        Input::Silu => x.map(|v| silu(2. * s * v)),
        Input::Relu => x.map(|v| relu(s * v)),
    }
    x
}

/// The CPU's epilogue on `x·Wᵀ + b` (as `forward` returns it), without the residual.
fn apply(y: &Mat, epilogue: Epilogue) -> Mat {
    let mut out = y.clone();
    let scale = epilogue.scale;
    match epilogue.activation {
        Activation::None => out.map(|v| v * scale),
        Activation::Relu => out.map(|v| relu(v) * scale),
        Activation::Silu => out.map(|v| silu(v) * scale),
        Activation::Gelu => out.map(|v| vellum_engine::cpu::gelu(v) * scale),
    }
    out
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut n) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += ((x - y) as f64).powi(2);
        n += (*y as f64).powi(2);
    }
    (d / n.max(f64::MIN_POSITIVE)).sqrt()
}

fn first_line(s: &str) -> &str {
    s.lines().find(|l| !l.trim().is_empty()).unwrap_or(s)
}
