//! `cargo lab whisper --sustain <seconds>`: Whisper's GPU encoder alone, batch after batch on
//! the same windows, for a sustained run: each batch's GPU time (timestamps around its
//! submits) next to what NVML says the GPU was doing then (clocks, memory-controller load,
//! power, PCIe link, throttle reasons). For finding why the encoder sometimes runs many times
//! slower under sustained load ("slow spells"): only the GPU model is loaded, nothing else
//! runs, and the per-second lines show when a spell starts and what changes with it.
//!
//! `VELLUM_SUSTAIN` picks what runs: `timed` (default: each batch with timestamps around its
//! submits), `plain` (the pipeline's own `encode`, no timestamps or readbacks; wall time to
//! the GPU idle), or `gemm` (one of the encoder's multiplies, the last block's fc2, 32 times
//! per submit on the batch's rows: the multiply alone on the same buffers).

use crate::{
    corpus::{Clip, Corpus, Purpose, Tier},
    models, paths,
    report::Report,
    score::table::Table,
    system::{self, Sample, Sampler},
};
use anyhow::{Context, Result};
use std::{
    fmt::Write,
    sync::Arc,
    time::{Duration, Instant},
};
use vellum_engine::{
    cpu::Mat,
    gguf::Gguf,
    gpu::{Gpu, GpuOptions},
    whisper::{Whisper, gpu::GpuOptions as WhisperGpuOptions},
};

pub struct Args {
    pub model: String,
    pub seconds: f64,
    /// Windows per batch (0: the encoder's own batch size).
    pub batch: usize,
    pub adapter: Option<String>,
}

/// One encoder batch: when it ended, its GPU and wall milliseconds, and its submits' GPU
/// milliseconds.
struct Batch {
    t: f64,
    gpu_ms: f64,
    wall_ms: f64,
    submits: Vec<f64>,
}

pub fn sustain(args: &Args) -> Result<bool> {
    let mut options = GpuOptions::from_env();
    if args.adapter.is_some() {
        options.adapter = args.adapter.clone();
    }
    options.cache_dir = Some(paths::root()?.join("gpu-cache"));
    let machine = system::machine();
    let mut report = Report::new(
        &format!("sustain-{}", args.model),
        serde_json::to_value(&machine)?,
    );
    let device = Arc::new(Gpu::new(&options)?);
    let model = models::get(&args.model)?;
    let file = Gguf::open(&model.ensure()?)?;
    let batch = if args.batch == 0 { 8 } else { args.batch };
    let ours = Whisper::load_with(
        &file,
        Some(device.clone()),
        &WhisperGpuOptions {
            max_batch: batch.max(8),
            ..WhisperGpuOptions::default()
        },
    )?;
    let g = ours.gpu_model().context("the GPU model")?;
    let batch = if args.batch == 0 {
        g.encoder.max_batch()
    } else {
        args.batch.min(g.encoder.max_batch())
    };

    let corpus = Corpus::load(&paths::corpus()?)?;
    let clip: &Clip = corpus
        .select(Tier::Quick, Purpose::Probe)
        .into_iter()
        .find(|c| c.seconds <= 28.)
        .context("no probe clip; run `cargo lab fetch`")?;
    let mut pcm = vellum_harness::audio::read(&corpus.audio(clip))?;
    pcm.truncate(30 * 16000);
    let chunk: Mat = ours.chunk_features(&pcm);
    let mels: Vec<&Mat> = vec![&chunk; batch];
    let setup = format!(
        "{}\n{}: encoder batches of {batch} ({}), {:.0} s, back to back",
        device.name(),
        model.id,
        clip.id,
        args.seconds
    );
    println!("{setup}");

    let mode = std::env::var("VELLUM_SUSTAIN").unwrap_or_else(|_| "timed".into());
    println!("mode: {mode}");
    // `VELLUM_SUSTAIN_BALLAST=<MiB>`: that much more device memory held (zeroed, in 256 MiB
    // buffers) and never touched again, to see whether the footprint matters.
    let ballast_mib: usize = std::env::var("VELLUM_SUSTAIN_BALLAST")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let _ballast: Vec<_> = (0..ballast_mib.div_ceil(256))
        .map(|_| device.zeros_u32("ballast", 64 << 20))
        .collect();
    if ballast_mib > 0 {
        let mut encoder = device.device.create_command_encoder(&Default::default());
        for b in &_ballast {
            encoder.clear_buffer(&b.buffer, 0, None);
        }
        device.queue.submit([encoder.finish()]);
        device.wait()?;
        println!("ballast: {} MiB", _ballast.len() * 256);
    }
    let windows: Vec<(&Mat, usize)> = (0..batch).map(|i| (&chunk, i)).collect();
    let mut profiler = vellum_engine::gpu::Profiler::new(&device, 64);
    // `dispatches` mode: every batch's dispatches timed one by one, kept by position.
    let mut per_dispatch: Vec<Vec<(String, f64)>> = Vec::new();
    let sampler = Sampler::start(Duration::from_millis(100));
    let t0 = Instant::now();
    let mut batches = Vec::new();
    while t0.elapsed().as_secs_f64() < args.seconds {
        let started = Instant::now();
        let submits: Vec<f64> = match mode.as_str() {
            "plain" => {
                g.encode(&windows)?;
                device.wait()?;
                vec![started.elapsed().as_secs_f64() * 1e3]
            }
            "gemm" => {
                let mut encoder = device.device.create_command_encoder(&Default::default());
                {
                    let mut pass = profiler.pass(&mut encoder, "gemm");
                    for _ in 0..32 {
                        g.encoder.record_last_fc2(&mut pass, batch);
                    }
                }
                device.queue.submit([encoder.finish()]);
                profiler
                    .read(&device)?
                    .into_iter()
                    .map(|(_, t)| t)
                    .collect()
            }
            "dispatches" => {
                let times = g.encoder.encode_timed(&mels, true)?;
                let total = times.iter().map(|(_, t)| t).sum();
                per_dispatch.push(times);
                vec![total]
            }
            _ => g
                .encoder
                .encode_timed(&mels, false)?
                .into_iter()
                .map(|(_, t)| t)
                .collect(),
        };
        batches.push(Batch {
            t: t0.elapsed().as_secs_f64(),
            gpu_ms: submits.iter().sum(),
            wall_ms: started.elapsed().as_secs_f64() * 1e3,
            submits,
        });
    }
    let trace = sampler.map(Sampler::stop);
    let samples: &[Sample] = trace.as_ref().map_or(&[], |t| &t.samples);

    // A batch is slow at more than twice the fastest one.
    let fastest = batches
        .iter()
        .map(|b| b.gpu_ms)
        .fold(f64::INFINITY, f64::min);
    let slow = |b: &Batch| b.gpu_ms > 2. * fastest;
    let mut table = Table::new([
        "t (s)",
        "batches",
        "slow",
        "GPU ms / batch (median, max)",
        "slowest submit ms",
        "SM MHz",
        "memory MHz",
        "memory controller %",
        "GPU %",
        "W",
        "PCIe",
        "reasons",
    ]);
    let step = 5.;
    let mut t = 0.;
    while t < args.seconds {
        let in_step: Vec<&Batch> = batches
            .iter()
            .filter(|b| b.t >= t && b.t < t + step)
            .collect();
        let near: Vec<&Sample> = samples
            .iter()
            .filter(|s| (s.t_ms as f64 / 1e3) >= t && (s.t_ms as f64 / 1e3) < t + step)
            .collect();
        let mut ms: Vec<f64> = in_step.iter().map(|b| b.gpu_ms).collect();
        ms.sort_by(f64::total_cmp);
        let median = ms.get(ms.len() / 2).copied().unwrap_or(f64::NAN);
        let max = ms.last().copied().unwrap_or(f64::NAN);
        let submit_max = in_step
            .iter()
            .flat_map(|b| b.submits.iter().copied())
            .fold(0., f64::max);
        let mean = |f: &dyn Fn(&Sample) -> Option<u32>| -> String {
            let v: Vec<f64> = near.iter().filter_map(|s| f(s)).map(f64::from).collect();
            if v.is_empty() {
                "—".into()
            } else {
                format!("{:.0}", v.iter().sum::<f64>() / v.len() as f64)
            }
        };
        let pcie = near
            .last()
            .and_then(|s| Some(format!("gen{} x{}", s.pcie_gen?, s.pcie_width?)))
            .unwrap_or_else(|| "—".into());
        let reasons = near
            .iter()
            .filter_map(|s| s.reasons)
            .fold(0u64, |a, r| a | r);
        table.row([
            format!("{t:.0}"),
            in_step.len().to_string(),
            in_step.iter().filter(|b| slow(b)).count().to_string(),
            format!("{median:.1}, {max:.1}"),
            format!("{submit_max:.1}"),
            mean(&|s| s.sm_mhz),
            mean(&|s| s.memory_mhz),
            mean(&|s| s.memory_utilization),
            mean(&|s| s.utilization),
            mean(&|s| s.power_mw.map(|p| p / 1000)),
            pcie,
            format!("{reasons:#x}"),
        ]);
        t += step;
    }
    let n_slow = batches.iter().filter(|b| slow(b)).count();
    let first_slow = batches.iter().find(|b| slow(b)).map(|b| b.t);
    let pattern: String = batches
        .iter()
        .map(|b| if slow(b) { 'S' } else { '.' })
        .collect();
    let mut gpu_ms: Vec<f64> = batches.iter().map(|b| b.gpu_ms).collect();
    gpu_ms.sort_by(f64::total_cmp);
    let total_gpu: f64 = gpu_ms.iter().sum();
    let wall: f64 = batches.iter().map(|b| b.wall_ms).sum();
    let summary = format!(
        "{} batches: fastest {fastest:.1} ms, median {:.1} ms, mean {:.1} ms; {n_slow} slow (> 2× the fastest), the first at {}; GPU busy {:.0}% of the wall time",
        batches.len(),
        gpu_ms[gpu_ms.len() / 2],
        total_gpu / batches.len() as f64,
        first_slow.map_or("—".into(), |t| format!("{t:.1} s")),
        100. * total_gpu / wall
    );
    println!("{}", table.markdown());
    println!("{summary}");
    if !per_dispatch.is_empty() {
        // Per family: mean ms per batch in fast and in slow batches; and where in the batch
        // (by dispatch index) the slow batches first depart from the fast ones.
        let slow_flags: Vec<bool> = batches.iter().map(&slow).collect();
        let mut families: std::collections::BTreeMap<String, (f64, f64)> = Default::default();
        let (n_fast, n_slow) = (
            slow_flags.iter().filter(|s| !**s).count().max(1) as f64,
            slow_flags.iter().filter(|s| **s).count().max(1) as f64,
        );
        for (times, &is_slow) in per_dispatch.iter().zip(&slow_flags) {
            for (label, t) in times {
                let family = label.split('/').next().unwrap_or(label).to_owned();
                let e = families.entry(family).or_default();
                if is_slow {
                    e.1 += t / n_slow;
                } else {
                    e.0 += t / n_fast;
                }
            }
        }
        let mut t = Table::new(["family", "fast ms / batch", "slow ms / batch", "ratio"]);
        for (family, (fast, slow)) in &families {
            t.row([
                family.clone(),
                format!("{fast:.1}"),
                format!("{slow:.1}"),
                format!("{:.1}", slow / fast),
            ]);
        }
        println!("{}", t.markdown());
        // The fastest time of each dispatch position, and per slow batch the first position
        // over 3× it, and how many over 3×.
        let len = per_dispatch.iter().map(Vec::len).min().unwrap_or(0);
        let best: Vec<f64> = (0..len)
            .map(|i| {
                per_dispatch
                    .iter()
                    .map(|d| d[i].1)
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();
        for (k, (times, &is_slow)) in per_dispatch.iter().zip(&slow_flags).enumerate() {
            let over: Vec<usize> = (0..len)
                .filter(|&i| times[i].1 > 3. * best[i] + 0.05)
                .collect();
            if is_slow || k % 5 == 0 {
                println!(
                    "batch {k} ({}): {} of {len} dispatches over 3× their best, first #{} {}",
                    if is_slow { "slow" } else { "fast" },
                    over.len(),
                    over.first().map_or("—".into(), |i| i.to_string()),
                    over.first().map_or("", |&i| times[i].0.as_str()),
                );
            }
        }
    }
    let mut pattern_lines = String::new();
    for line in pattern.as_bytes().chunks(100) {
        writeln!(pattern_lines, "{}", String::from_utf8_lossy(line))?;
    }
    println!("{pattern_lines}");
    report.section("Setup", format!("```\n{setup}\n```"));
    report.section(
        "Per 5 seconds",
        format!(
            "GPU time per batch from timestamps around each submit's pass; NVML sampled every 100 ms (means over the 5 s; reasons OR-ed).\n\n{}\n\n{summary}",
            table.markdown()
        ),
    );
    report.section(
        "Batches in order",
        format!("`.` fast, `S` slow:\n\n```\n{pattern_lines}```"),
    );
    report.passed = true;
    report.results = serde_json::json!({
        "adapter": device.name(),
        "model": model.id,
        "batch": batch,
        "batches": batches.iter().map(|b| serde_json::json!({"t": b.t, "gpu_ms": b.gpu_ms, "wall_ms": b.wall_ms, "submits": b.submits})).collect::<Vec<_>>(),
        "gpu": trace,
    });
    println!("{}", report.save()?.display());
    Ok(true)
}
