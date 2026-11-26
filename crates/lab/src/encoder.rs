//! `cargo lab encoder`: Parakeet's GPU encoder against the CPU oracle, checkpoint by checkpoint,
//! then timed by batch size. The GPU encoder's dev loop: one command, seconds.
//!
//! For each clip (default: Parakeet's probe clips), the engine's own mel features go through
//! both encoders; every checkpoint the oracle records (`enc.pre_encode.out`, each block's
//! `ff1`, `attn`, `conv`, `ff2` and `out`, and the output as `enc.final`) is compared — relative
//! L2, max |Δ|, cosine — with the calibrated tolerance (twice the reference's own GPU-against-CPU
//! spread, as `probe` allows; a block's sub-layers without their own calibration take the
//! block's). Then a batch of `--batch` windows is encoded together and each window's result
//! checked against encoding it alone (it must not depend on its batch). Last, batches of
//! 1, 2, 4, 8 windows of `--seconds` of audio are timed: wall time per call, GPU time per
//! batch, and the GPU time per kernel family (each dispatch timed on its own).

use crate::{
    check,
    corpus::{Clip, Corpus, Purpose, Tier},
    models, paths,
    report::Report,
    score::{
        table::Table,
        tensors::{TensorDiff, compare},
    },
    system,
};
use anyhow::{Context, Result};
use std::{
    collections::BTreeMap,
    fmt::Write,
    sync::Arc,
    time::{Duration, Instant},
};
use vellum_engine::{
    Trace,
    cpu::Mat,
    gguf::Gguf,
    gpu::{
        self, Gpu, GpuOptions,
        gemm::{Gemm, Kind, Variant},
    },
    mel::Mel,
    parakeet::{
        HParams,
        encoder::Encoder,
        gpu_encoder::{GpuEncoder, GpuEncoderOptions},
    },
};

pub struct Args {
    /// Clip ids (default: Parakeet's probe clips).
    pub clips: Vec<String>,
    /// Windows encoded together for the batch check.
    pub batch: usize,
    /// `simt` or `coop` (default: tensor cores where the adapter has them).
    pub kernel: Option<String>,
    /// Pick the multiply's tile shape by measuring (a few seconds).
    pub autotune: bool,
    /// Seconds of audio per timed window.
    pub seconds: f64,
    /// Timed calls per batch size.
    pub runs: u32,
    pub adapter: Option<String>,
    pub backend: Option<String>,
}

/// Every checkpoint of a run, in order.
#[derive(Default)]
struct Recorded(Vec<(String, Vec<usize>, Vec<f32>)>);

impl Trace for Recorded {
    fn wants(&self, _name: &str) -> bool {
        true
    }

    fn tensor(&mut self, name: &str, shape: &[usize], data: &[f32]) {
        self.0.push((name.into(), shape.to_vec(), data.to_vec()));
    }
}

struct Checked {
    name: String,
    diff: TensorDiff,
    spread: Option<f64>,
    tolerance: f64,
    within: bool,
}

pub fn encoder(args: &Args) -> Result<bool> {
    let mut options = GpuOptions::from_env();
    if args.adapter.is_some() {
        options.adapter = args.adapter.clone();
    }
    if let Some(b) = &args.backend {
        options.backends = Some(gpu::wgpu::Backends::from_comma_list(b));
    }
    options.cache_dir = Some(paths::root()?.join("gpu-cache"));
    let machine = system::machine();
    let mut report = Report::new("encoder", serde_json::to_value(&machine)?);
    let started = Instant::now();
    let gpu = Arc::new(Gpu::new(&options)?);
    println!("{} (device in {:.0} ms)", gpu.name(), ms(started));

    let path = models::get("parakeet")?.ensure()?;
    let file = Gguf::open(&path)?;
    let hparams = HParams::from_gguf(&file)?;
    let mel = Mel::from_gguf(&file)?;
    let started = Instant::now();
    let cpu = Encoder::load(&file, &hparams)?;
    let cpu_load = ms(started);

    let kind = match args.kernel.as_deref() {
        None => None,
        Some("simt") => Some(Kind::Simt),
        Some("coop") => Some(Kind::Coop),
        Some(other) => anyhow::bail!("--kernel {other}: simt or coop"),
    };
    let mut tuning = String::new();
    let variant = match (kind, args.autotune) {
        (kind, true) => {
            let kind = kind.unwrap_or(if gpu.has_f16_tensor_cores() {
                Kind::Coop
            } else {
                Kind::Simt
            });
            let started = Instant::now();
            let tuned = Gemm::autotune(&gpu, kind)?;
            tuning = format!(", autotuned in {:.1} s", started.elapsed().as_secs_f64());
            Some(tuned.gemm.variant)
        }
        (Some(Kind::Simt), false) => Some(Variant {
            kind: Kind::Simt,
            bm: 128,
            bn: 64,
            bk: 16,
            tm: 8,
            tn: 4,
        }),
        (Some(Kind::Coop), false) => Some(Variant {
            kind: Kind::Coop,
            bm: 64,
            bn: 128,
            bk: 32,
            tm: 32,
            tn: 64,
        }),
        (None, false) => None,
    };
    let started = Instant::now();
    let encoder_options = GpuEncoderOptions {
        gemm: variant,
        max_batch: args.batch.max(8),
        ..GpuEncoderOptions::default()
    };
    let encoder = GpuEncoder::load_with(&gpu, &file, &hparams, &encoder_options)?;
    let gpu_load = ms(started);
    gpu.save_pipeline_cache()?;
    let setup = format!(
        "{}\nmultiply: {}{tuning}\nCPU oracle loaded in {cpu_load:.0} ms, GPU encoder in {gpu_load:.0} ms (weights to f16, positional table, kernels)",
        gpu.name(),
        encoder.gemm_variant().name(),
    );
    println!("{setup}");
    report.section("Setup", format!("```\n{setup}\n```"));

    // The clips and their mel features, as the encoder sees them.
    let corpus = Corpus::load(&paths::corpus()?)?;
    let model = models::get("parakeet")?;
    let clips: Vec<&Clip> = if args.clips.is_empty() {
        let runs_on = check::clips(&corpus, model, Tier::Long);
        corpus
            .select(Tier::Quick, Purpose::Probe)
            .into_iter()
            .filter(|c| runs_on.iter().any(|r| r.id == c.id) && c.seconds <= 28.)
            .collect()
    } else {
        args.clips
            .iter()
            .map(|id| corpus.get(id).with_context(|| format!("no clip {id}")))
            .collect::<Result<_>>()?
    };
    anyhow::ensure!(!clips.is_empty(), "no probe clips; run `cargo lab fetch`");
    let max_samples = (encoder.max_mel_frames() - 1) * 160;
    let mut pcms = Vec::new();
    for clip in &clips {
        let mut pcm = vellum_harness::audio::read(&corpus.audio(clip))?;
        pcm.truncate(max_samples);
        pcms.push(pcm);
    }
    let features = |pcm: &[f32]| -> Mat {
        let full = mel.compute(pcm);
        let valid = mel.valid_frames(pcm.len()).min(full.cols);
        let mut m = Mat::zeros(full.rows, valid);
        for r in 0..full.rows {
            m.row_mut(r).copy_from_slice(&full.row(r)[..valid]);
        }
        m
    };
    let mels: Vec<Mat> = pcms.iter().map(|p| features(p)).collect();

    // Tolerances: twice the reference's spread, as `probe` allows.
    let spreads: BTreeMap<String, f64> =
        std::fs::read(paths::root()?.join("tolerances").join("parakeet.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
    if spreads.is_empty() {
        eprintln!("! no calibration (`cargo lab probe --calibrate`): allowing 1e-2");
    }
    let spread = |name: &str| -> Option<f64> {
        spreads.get(name).copied().or_else(|| {
            let (block, _) = name.rsplit_once('.')?;
            spreads.get(&format!("{block}.out")).copied()
        })
    };

    // Checkpoints, clip by clip.
    let mut passed = true;
    let mut results = Vec::new();
    let mut alone = Vec::new();
    for ((clip, mel), pcm) in clips.iter().zip(&mels).zip(&pcms) {
        let mut cpu_trace = Recorded::default();
        let started = Instant::now();
        let cpu_out = cpu.forward(mel, &mut cpu_trace);
        let cpu_ms = ms(started);
        let mut gpu_trace = Recorded::default();
        let gpu_out = encoder
            .encode_traced(std::slice::from_ref(mel), &mut gpu_trace)?
            .remove(0);
        let ours: BTreeMap<&str, &Vec<f32>> = gpu_trace
            .0
            .iter()
            .map(|(n, _, d)| (n.as_str(), d))
            .collect();
        let mut checked = Vec::new();
        let mut check_one = |name: &str, reference: &[f32], candidate: &[f32]| {
            let diff = compare(reference, candidate);
            let spread = spread(name);
            let tolerance = spread.map_or(1e-2, |s| (2. * s).max(1e-5));
            let within = diff.within(tolerance);
            checked.push(Checked {
                name: name.into(),
                diff,
                spread,
                tolerance,
                within,
            });
        };
        for (name, _, reference) in &cpu_trace.0 {
            if let Some(candidate) = ours.get(name.as_str()) {
                check_one(name, reference, candidate);
            }
        }
        check_one("enc.final", &cpu_out.data, &gpu_out.data);
        let missing: Vec<&str> = cpu_trace
            .0
            .iter()
            .map(|(n, _, _)| n.as_str())
            .filter(|n| *n != "enc.pos_emb" && !ours.contains_key(n))
            .collect();
        let clip_passed = missing.is_empty() && checked.iter().all(|c| c.within);
        passed &= clip_passed;

        let worst = checked
            .iter()
            .filter(|c| c.spread.is_some())
            .max_by(|a, b| {
                (a.diff.rel_l2 / a.spread.unwrap()).total_cmp(&(b.diff.rel_l2 / b.spread.unwrap()))
            });
        let mut table = Table::new([
            "checkpoint",
            "rel L2",
            "max |Δ|",
            "cosine",
            "ref spread",
            "÷ spread",
            "",
        ]);
        let shown = |n: &str| {
            !n.starts_with("enc.block.")
                || n.ends_with(".out")
                || n.starts_with("enc.block.0.")
                || n.starts_with(&format!("enc.block.{}.", hparams.n_layers - 1))
        };
        for c in checked.iter().filter(|c| shown(&c.name)) {
            table.row([
                c.name.clone(),
                format!("{:.2e}", c.diff.rel_l2),
                format!("{:.2e}", c.diff.max_abs),
                format!("{:.7}", c.diff.cosine),
                c.spread.map_or("—".into(), |s| format!("{s:.2e}")),
                c.spread
                    .map_or("—".into(), |s| format!("{:.2}", c.diff.rel_l2 / s)),
                if c.within { "".into() } else { "✗".into() },
            ]);
        }
        let mut body = format!(
            "{:.1} s, {} mel frames → {} encoder frames. CPU oracle {cpu_ms:.0} ms. {} checkpoints compared{}; all sub-layers of every block are gated, the table shows each block's output and the first and last blocks' sub-layers.{}\n\n{}",
            pcm.len() as f64 / 16000.,
            mel.cols,
            cpu_out.rows,
            checked.len(),
            if missing.is_empty() {
                String::new()
            } else {
                format!(", **missing on the GPU: {}**", missing.join(", "))
            },
            worst.map_or(String::new(), |w| format!(
                " Worst against the reference's spread: {} at {:.2}× (rel L2 {:.2e}).",
                w.name,
                w.diff.rel_l2 / w.spread.unwrap(),
                w.diff.rel_l2
            )),
            table.markdown()
        );
        let failed: Vec<String> = checked
            .iter()
            .filter(|c| !c.within)
            .map(|c| format!("{} ({:.2e} > {:.2e})", c.name, c.diff.rel_l2, c.tolerance))
            .collect();
        if !failed.is_empty() {
            write!(body, "\n\nOut of tolerance: {}", failed.join(", "))?;
        }
        println!(
            "{}: {} checkpoints, final rel L2 {:.2e}{}",
            clip.id,
            checked.len(),
            checked.last().map_or(f64::NAN, |c| c.diff.rel_l2),
            if clip_passed { "" } else { "  ✗" }
        );
        report.section(format!("Checkpoints: {}", clip.id), body);
        results.push(serde_json::json!({
            "clip": clip.id,
            "passed": clip_passed,
            "checkpoints": checked.iter().map(|c| serde_json::json!({
                "name": c.name, "rel_l2": c.diff.rel_l2, "max_abs": c.diff.max_abs,
                "cosine": c.diff.cosine, "spread": c.spread, "within": c.within,
            })).collect::<Vec<_>>(),
        }));
        alone.push(gpu_out);
    }

    // A batch: each window as it comes out alone.
    let batch: Vec<Mat> = (0..args.batch.max(1))
        .map(|i| mels[i % mels.len()].clone())
        .collect();
    let together = encoder.encode(&batch)?;
    let mut table = Table::new(["window", "frames", "max |Δ| vs alone", "bit-exact"]);
    let mut batch_ok = true;
    for (i, out) in together.iter().enumerate() {
        let single = &alone[i % alone.len()];
        let max = single
            .data
            .iter()
            .zip(&out.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let exact = single.data == out.data;
        batch_ok &= max <= 1e-5 * single.data.iter().fold(0f32, |m, v| m.max(v.abs()));
        table.row([
            format!("{i} ({})", clips[i % clips.len()].id),
            out.rows.to_string(),
            format!("{max:.2e}"),
            if exact { "yes".into() } else { "no".into() },
        ]);
    }
    passed &= batch_ok;
    println!(
        "batch of {}: {}",
        batch.len(),
        if batch_ok {
            "each window as alone"
        } else {
            "✗ windows depend on their batch"
        }
    );
    report.section(
        format!("Batch of {}", batch.len()),
        format!(
            "Every window encoded in one batch against the same window encoded alone.\n\n{}",
            table.markdown()
        ),
    );

    // Timing: batches of windows of `seconds` of audio (the first clip, repeated).
    let samples = ((args.seconds * 16000.) as usize).min(max_samples);
    let window_pcm: Vec<f32> = pcms[0].iter().copied().cycle().take(samples).collect();
    let window = features(&window_pcm);
    let frames = encoder
        .encode(std::slice::from_ref(&window))?
        .remove(0)
        .rows;
    let flops = Flops::new(&hparams, window.cols, frames);
    let mut timing = Table::new([
        "batch",
        "wall ms / batch",
        "wall ms / window",
        "record + submit ms",
        "GPU ms / batch (median)",
        "GPU ms / batch",
        "GPU ms / window",
        "× real time",
        "TFLOP/s",
    ]);
    let mut breakdown = Table::new([
        "batch",
        "family",
        "ms / window",
        "share",
        "TFLOP/s",
        "dispatches",
    ]);
    let mut labels_table = String::new();
    let mut timings = Vec::new();
    // Warm the GPU up to its working clocks first.
    let warm = Instant::now();
    while warm.elapsed() < Duration::from_millis(1500) {
        encoder.encode(std::slice::from_ref(&window))?;
    }
    for b in [1usize, 2, 4, 8]
        .into_iter()
        .filter(|b| *b <= encoder.max_batch())
    {
        let windows: Vec<Mat> = (0..b).map(|_| window.clone()).collect();
        encoder.encode(&windows)?;
        let mut wall = Vec::new();
        let mut record = Vec::new();
        let mut gpu_ms = Vec::new();
        for _ in 0..args.runs.max(1) {
            let started = Instant::now();
            encoder.encode(&windows)?;
            wall.push(ms(started));
            // The CPU's part: recording and submitting a batch, without waiting for it.
            let started = Instant::now();
            let pending = encoder.submit(&windows)?;
            record.push(ms(started));
            encoder.finish(pending)?;
            let (_, times) = encoder.encode_timed(&windows, false)?;
            gpu_ms.push(times.iter().map(|(_, t)| t).sum::<f64>());
        }
        let wall = median(&mut wall);
        let record = median(&mut record);
        let gpu_median = median(&mut gpu_ms);
        // Other work on the GPU (other lab runs, the app) only ever adds time: the fastest
        // batch is the encoder's own.
        let gpu_t = gpu_ms[0];
        // Per dispatch: each dispatch's fastest of five runs (the same sequence every run;
        // other work on the GPU only ever adds time), summed by label.
        let mut best: Vec<(String, f64)> = Vec::new();
        for _ in 0..5 {
            let (_, times) = encoder.encode_timed(&windows, true)?;
            if best.len() != times.len() {
                best = times;
            } else {
                for (b, (_, t)) in best.iter_mut().zip(times) {
                    b.1 = b.1.min(t);
                }
            }
        }
        let mut by_label: BTreeMap<String, (f64, usize)> = BTreeMap::new();
        for (label, t) in best {
            let e = by_label.entry(label).or_default();
            e.0 += t;
            e.1 += 1;
        }
        let total: f64 = by_label.values().map(|(t, _)| t).sum();
        let mut families: BTreeMap<&str, (f64, usize)> = BTreeMap::new();
        for (label, (t, n)) in &by_label {
            let family = label.split('/').next().unwrap_or(label);
            let e = families.entry(family).or_default();
            e.0 += t;
            e.1 += n;
        }
        let seconds = window_pcm.len() as f64 / 16000.;
        timing.row([
            b.to_string(),
            format!("{wall:.2}"),
            format!("{:.2}", wall / b as f64),
            format!("{record:.2}"),
            format!("{gpu_median:.2}"),
            format!("{gpu_t:.2}"),
            format!("{:.2}", gpu_t / b as f64),
            format!("{:.0}", seconds * b as f64 / (wall / 1e3)),
            format!("{:.2}", flops.total() * b as f64 / (gpu_t * 1e-3) / 1e12),
        ]);
        for (family, (t, n)) in &families {
            let f = match *family {
                "gemm" => Some(flops.gemm),
                "attn" => Some(flops.attention),
                _ => None,
            };
            breakdown.row([
                b.to_string(),
                family.to_string(),
                format!("{:.2}", t / b as f64),
                format!("{:.0}%", 100. * t / total),
                f.map_or("—".into(), |f| {
                    format!("{:.2}", f * b as f64 / (t * 1e-3) / 1e12)
                }),
                n.to_string(),
            ]);
        }
        if b == encoder.max_batch().min(8) {
            let mut t = Table::new(["dispatch", "ms / batch", "calls", "share"]);
            let mut sorted: Vec<_> = by_label.iter().collect();
            sorted.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
            for (label, (ms, n)) in sorted {
                t.row([
                    label.clone(),
                    format!("{ms:.2}"),
                    n.to_string(),
                    format!("{:.1}%", 100. * ms / total),
                ]);
            }
            labels_table = format!(
                "Batch of {b}, each dispatch timed alone:\n\n{}",
                t.markdown()
            );
        }
        println!(
            "B={b}: wall {wall:.1} ms ({:.2} ms/window), GPU {gpu_t:.1} ms ({:.2} ms/window)",
            wall / b as f64,
            gpu_t / b as f64
        );
        timings.push(serde_json::json!({
            "batch": b, "wall_ms": wall, "gpu_ms": gpu_t,
            "families": families.iter().map(|(f, (t, n))| (f.to_string(), serde_json::json!({"ms": t, "dispatches": n}))).collect::<BTreeMap<_, _>>(),
        }));
    }
    gpu.save_pipeline_cache()?;
    report.section(
        "Timing",
        format!(
            "Windows of {:.1} s ({} mel frames, {} encoder frames; {:.1} GFLOP each: {:.1} in matrix multiplies, {:.1} in attention). Wall: `encode` from mel upload to the last readback, median of {} calls; record + submit: the CPU's part of it (mel upload, every dispatch recorded, submitted; median). GPU: timestamps around each submit's pass, the median and the fastest batch (the GPU is shared with whatever else runs, which only adds time; per-window figures use the fastest). TFLOP/s over all the encoder's FLOPs.\n\n{}",
            window_pcm.len() as f64 / 16000.,
            window.cols,
            frames,
            flops.total() / 1e9,
            flops.gemm / 1e9,
            flops.attention / 1e9,
            args.runs,
            timing.markdown()
        ),
    );
    report.section(
        "By kernel family",
        format!(
            "Each dispatch in its own timed pass (so the sum is a little above the batch's own GPU time), each dispatch's fastest of five runs, summed. gemm: every linear layer (TFLOP/s over their FLOPs); attn: scores, softmax and weighted values (TFLOP/s over the score and value products); norm: layer norms; conv: GLU + depthwise + batch norm + SiLU; pre_encode: the subsampling's 3×3 convolutions.\n\n{}\n\n{labels_table}",
            breakdown.markdown()
        ),
    );
    report.passed = passed;
    report.results = serde_json::json!({
        "adapter": gpu.name(),
        "gemm": encoder.gemm_variant().name(),
        "clips": results,
        "batch_ok": batch_ok,
        "timing": timings,
    });
    println!("{}", report.save()?.display());
    Ok(passed)
}

fn ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1e3
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// The encoder's arithmetic for one window, in FLOPs (a multiply-add is two).
struct Flops {
    gemm: f64,
    attention: f64,
}

impl Flops {
    fn new(h: &HParams, mel_frames: usize, frames: usize) -> Flops {
        let (d, ff, t) = (h.d_model as f64, h.d_ff as f64, frames as f64);
        let c = h.subsampling_channels as f64;
        // Per block: two feed-forwards (up and down), q/k/v, out, pointwise 1 (to 2d) and 2.
        let per_frame = 2. * 2. * d * ff + 3. * d * d + d * d + 2. * d * d + d * d;
        let t2 = mel_frames.div_ceil(4) as f64;
        let pre = 2. * (t2 * 32. + t * 16.) * c * c + 2. * t * (c * 16.) * d;
        Flops {
            gemm: 2. * t * per_frame * h.n_layers as f64 + pre,
            // Content and position scores, and the weighted values: three T×T×d products.
            attention: h.n_layers as f64 * 3. * 2. * t * t * d,
        }
    }

    fn total(&self) -> f64 {
        self.gemm + self.attention
    }
}
