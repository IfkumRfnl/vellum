//! `cargo lab whisper`: Whisper's GPU path against the CPU oracle, checkpoint by checkpoint and
//! decision by decision, then timed by batch size. The GPU path's dev loop.
//!
//! For each clip (default: the model's probe clips), the first 30 s pass's mel features go
//! through both encoders: every encoder checkpoint both record (`enc.conv1.out`,
//! `enc.embed.out`, `enc.block.<L>.out`, `enc.final`) is compared with the calibrated tolerance
//! (twice the reference's own GPU-against-CPU spread, as `probe` allows). Then the decoders,
//! both on the oracle's encoder states: the raw logits after every token of the oracle's
//! greedy decoding (teacher-forced: the prompt's rows as `dec.logits_raw`, every step's as
//! `dec.logits.steps`), and each decoder's own greedy decisions, compared token by token, a
//! divergence reported with both margins (a near-tie can go either way); last, end to end, the
//! GPU decoder on the GPU encoder's states. A batch of windows is encoded together and checked
//! against each alone. Last, the encoder per batch of 1, 2, 4, 8 windows and a decoder step per
//! batch of 1, 2, 4, 8 windows are timed with GPU timestamps, by kernel family, and the cost of
//! reading a step back.

use crate::{
    check,
    corpus::{Clip, Corpus, Purpose, Tier},
    models, paths,
    report::Report,
    score::{
        table::Table,
        tensors::{TensorDiff, compare, compare_logits},
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
    gpu::{self, Gpu, GpuOptions},
    whisper::{
        Whisper,
        gpu::{GpuOptions as WhisperGpuOptions, decoder::Mode},
    },
};

pub struct Args {
    /// `small` or `turbo`.
    pub model: String,
    /// Clip ids (default: the model's probe clips).
    pub clips: Vec<String>,
    /// Windows encoded together for the batch check.
    pub batch: usize,
    /// Timed calls per batch size.
    pub runs: u32,
    pub adapter: Option<String>,
    pub backend: Option<String>,
    /// Run the clips whole through both and say why their texts differ (see [`explain`]).
    pub explain: bool,
    /// A long clip run through the pipeline against one window at a time.
    pub pipeline: String,
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

/// A token and its margin.
type Side = (i32, f32);

/// How two decoders' greedy decisions compare.
struct Decisions {
    tokens: (usize, usize),
    /// The first step they differ at, with both sides' token and margin there.
    diverged: Option<(usize, Side, Side)>,
    /// The smallest margin of either side over the steps they agree on.
    min_margin: f32,
}

fn decisions(a: &[(i32, f32, f32)], b: &[(i32, f32, f32)]) -> Decisions {
    let mut min_margin = f32::INFINITY;
    let mut diverged = None;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if x.0 != y.0 {
            diverged = Some((i, (x.0, x.2), (y.0, y.2)));
            break;
        }
        min_margin = min_margin.min(x.2).min(y.2);
    }
    if diverged.is_none() && a.len() != b.len() {
        let i = a.len().min(b.len());
        let at = |s: &[(i32, f32, f32)]| s.get(i).map_or((-1, f32::NAN), |c| (c.0, c.2));
        diverged = Some((i, at(a), at(b)));
    }
    Decisions {
        tokens: (a.len(), b.len()),
        diverged,
        min_margin,
    }
}

fn describe(d: &Decisions) -> String {
    match d.diverged {
        None => format!(
            "identical ({} choices; smallest margin {:.3})",
            d.tokens.0, d.min_margin
        ),
        Some((i, (ta, ma), (tb, mb))) => format!(
            "differ at step {i} of {}/{}: oracle {ta} (margin {ma:.3}), GPU {tb} (margin {mb:.3})",
            d.tokens.0, d.tokens.1
        ),
    }
}

pub fn whisper(args: &Args) -> Result<bool> {
    if args.explain {
        return explain(args);
    }
    let mut options = GpuOptions::from_env();
    if args.adapter.is_some() {
        options.adapter = args.adapter.clone();
    }
    if let Some(b) = &args.backend {
        options.backends = Some(gpu::wgpu::Backends::from_comma_list(b));
    }
    options.cache_dir = Some(paths::root()?.join("gpu-cache"));
    let machine = system::machine();
    let mut report = Report::new(
        &format!("whisper-{}", args.model),
        serde_json::to_value(&machine)?,
    );
    let started = Instant::now();
    let device = Arc::new(Gpu::new(&options)?);
    println!("{} (device in {:.0} ms)", device.name(), ms(started));

    let model = models::get(&args.model)?;
    let path = model.ensure()?;
    let file = Gguf::open(&path)?;
    let started = Instant::now();
    let cpu = Whisper::load(&file)?;
    let cpu_load = ms(started);
    let started = Instant::now();
    let max_batch = args.batch.clamp(1, 8).max(8);
    let gpu_options = WhisperGpuOptions {
        max_batch,
        ..WhisperGpuOptions::default()
    };
    let ours = Whisper::load_with(&file, Some(device.clone()), &gpu_options)?;
    let gpu_load = ms(started);
    device.save_pipeline_cache()?;
    let g = ours.gpu_model().context("the GPU model")?;
    let hp = ours.hparams().clone();
    let setup = format!(
        "{}\n{} ({} encoder and {} decoder layers, d_model {})\nCPU oracle loaded in {cpu_load:.0} ms, GPU model in {gpu_load:.0} ms",
        device.name(),
        model.id,
        hp.enc_layers,
        hp.dec_layers,
        hp.d_model
    );
    println!("{setup}");
    report.section("Setup", format!("```\n{setup}\n```"));

    let corpus = Corpus::load(&paths::corpus()?)?;
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
    let mut chunks = Vec::new();
    for clip in &clips {
        let mut pcm = vellum_harness::audio::read(&corpus.audio(clip))?;
        pcm.truncate(30 * 16000);
        chunks.push(cpu.chunk_features(&pcm));
    }

    let spreads: BTreeMap<String, f64> = std::fs::read(
        paths::root()?
            .join("tolerances")
            .join(format!("{}.json", model.id)),
    )
    .ok()
    .and_then(|b| serde_json::from_slice(&b).ok())
    .unwrap_or_default();
    if spreads.is_empty() {
        eprintln!("! no calibration (`cargo lab probe --calibrate`): allowing 1e-2");
    }

    let mut passed = true;
    let mut results = Vec::new();
    let mut summary = Table::new([
        "clip",
        "enc.final rel L2",
        "÷ spread",
        "prompt logits rel L2",
        "step logits rel L2 (max)",
        "raw top-1 agree",
        "decoder alone",
        "end to end",
    ]);
    let mut encoded_cpu = Vec::new();
    // Per checkpoint, in order: the worst relative L2 over the clips, and its spread.
    let mut worst: Vec<(String, f64, Option<f64>)> = Vec::new();
    for (clip, chunk) in clips.iter().zip(&chunks) {
        let language = if clip.language.is_empty() {
            "en"
        } else {
            clip.language.as_str()
        };
        let mut cpu_trace = Recorded::default();
        let started = Instant::now();
        let cpu_enc = cpu.encode_checked(chunk, &mut cpu_trace)?;
        let cpu_ms = ms(started);
        let mut gpu_trace = Recorded::default();
        let gpu_enc = ours.encode_checked(chunk, &mut gpu_trace)?;
        let theirs: BTreeMap<&str, &Vec<f32>> = gpu_trace
            .0
            .iter()
            .map(|(n, _, d)| (n.as_str(), d))
            .collect();
        let mut checked = Vec::new();
        let mut check_one = |name: &str, reference: &[f32], candidate: &[f32]| {
            let diff = compare(reference, candidate);
            let spread = spreads.get(name).copied();
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
            if let Some(candidate) = theirs.get(name.as_str()) {
                check_one(name, reference, candidate);
            }
        }

        // The decoders on the oracle's encoder states.
        let prompt = cpu.prompt_for(Some(language))?;
        let oracle = cpu.greedy_choices(&cpu_enc, &prompt)?;
        let mut tokens = prompt.clone();
        tokens.extend(oracle.iter().map(|c| c.0).filter(|&t| t != hp.eos));
        let cpu_logits = cpu.teacher_forced(&cpu_enc, &tokens)?;
        let gpu_logits = ours.teacher_forced(&cpu_enc, &tokens)?;
        let p = prompt.len();
        let v = hp.vocab;
        check_one(
            "dec.logits_raw",
            &cpu_logits.data[..p * v],
            &gpu_logits.data[..p * v],
        );
        let rows: Vec<f64> = (0..tokens.len())
            .map(|r| compare(cpu_logits.row(r), gpu_logits.row(r)).rel_l2)
            .collect();
        let worst_row = rows.iter().copied().fold(0., f64::max);
        let logit_diff = compare_logits(&cpu_logits.data, &gpu_logits.data, tokens.len(), v);
        let alone = ours.greedy_choices(&cpu_enc, &prompt)?;
        let decoder_alone = decisions(&oracle, &alone);
        let e2e = ours.greedy_choices(&gpu_enc, &prompt)?;
        let end_to_end = decisions(&oracle, &e2e);

        let clip_passed = checked.iter().all(|c| c.within);
        passed &= clip_passed;
        let failed: Vec<String> = checked
            .iter()
            .filter(|c| !c.within)
            .map(|c| format!("{} ({:.2e} > {:.2e})", c.name, c.diff.rel_l2, c.tolerance))
            .collect();
        let mut table = Table::new([
            "checkpoint",
            "rel L2",
            "max |Δ|",
            "cosine",
            "ref spread",
            "÷ spread",
            "",
        ]);
        for c in &checked {
            match worst.iter_mut().find(|w| w.0 == c.name) {
                Some(w) => w.1 = w.1.max(c.diff.rel_l2),
                None => worst.push((c.name.clone(), c.diff.rel_l2, c.spread)),
            }
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
        let final_check = checked.iter().find(|c| c.name == "enc.final");
        let prompt_check = checked.iter().find(|c| c.name == "dec.logits_raw");
        summary.row([
            clip.id.clone(),
            final_check.map_or("—".into(), |c| format!("{:.2e}", c.diff.rel_l2)),
            final_check
                .and_then(|c| c.spread.map(|s| format!("{:.2}", c.diff.rel_l2 / s)))
                .unwrap_or("—".into()),
            prompt_check.map_or("—".into(), |c| format!("{:.2e}", c.diff.rel_l2)),
            format!("{worst_row:.2e}"),
            format!("{:.1}%", 100. * logit_diff.top1_agreement),
            describe(&decoder_alone),
            describe(&end_to_end),
        ]);
        let body = format!(
            "{} ({language}). CPU oracle encoder {cpu_ms:.0} ms. Decoder: {} tokens teacher-forced (the prompt and the oracle's greedy choices); raw logits' top-1 agree on {:.1}% of rows, mean KL {:.2e}.\n\nDecoder alone (both on the oracle's encoder states): {}.\n\nEnd to end (the GPU decoder on the GPU encoder's states): {}.\n\n{}",
            clip.id,
            tokens.len(),
            100. * logit_diff.top1_agreement,
            logit_diff.mean_kl,
            describe(&decoder_alone),
            describe(&end_to_end),
            table.markdown()
        ) + &if failed.is_empty() {
            String::new()
        } else {
            format!(
                "

Out of tolerance: {}",
                failed.join(", ")
            )
        };
        println!(
            "{}: enc.final {:.2e}, prompt logits {:.2e}, steps ≤ {worst_row:.2e}; decoder alone {}; end to end {}{}",
            clip.id,
            final_check.map_or(f64::NAN, |c| c.diff.rel_l2),
            prompt_check.map_or(f64::NAN, |c| c.diff.rel_l2),
            describe(&decoder_alone),
            describe(&end_to_end),
            if clip_passed { "" } else { "  ✗" }
        );
        report.section(format!("Checkpoints: {}", clip.id), body);
        results.push(serde_json::json!({
            "clip": clip.id,
            "passed": clip_passed,
            "checkpoints": checked.iter().map(|c| serde_json::json!({
                "name": c.name, "rel_l2": c.diff.rel_l2, "max_abs": c.diff.max_abs,
                "spread": c.spread, "within": c.within,
            })).collect::<Vec<_>>(),
            "decoder_alone": describe(&decoder_alone),
            "end_to_end": describe(&end_to_end),
        }));
        encoded_cpu.push(cpu_enc);
    }
    report.section("Summary", summary.markdown());
    let mut table = Table::new(["checkpoint", "worst rel L2", "ref spread", "÷ spread"]);
    for (name, rel, spread) in &worst {
        table.row([
            name.clone(),
            format!("{rel:.2e}"),
            spread.map_or("—".into(), |s| format!("{s:.2e}")),
            spread.map_or("—".into(), |s| format!("{:.3}", rel / s)),
        ]);
    }
    report.section(
        format!("Worst over {} clips", clips.len()),
        format!(
            "Each checkpoint's largest relative L2 against the CPU oracle over the clips, with the reference's own GPU-against-CPU spread (the gate allows twice it).\n\n{}",
            table.markdown()
        ),
    );

    // The pipeline: a long recording's windows decoded in lockstep (`run_each`) against the
    // same windows one at a time (`run`, as the harness calls it), text for text.
    if let Some(clip) = corpus.get(&args.pipeline) {
        use vellum_harness::windows;
        let pcm = vellum_harness::audio::read(&corpus.audio(clip))?;
        let size = windows::window_samples(28_000, ours.max_audio_ms());
        let ranges: Vec<_> = windows::windows(&pcm, size)
            .into_iter()
            .filter(|r| !windows::silent(&pcm[r.clone()]))
            .collect();
        let slices: Vec<&[f32]> = ranges.iter().map(|r| &pcm[r.clone()]).collect();
        // Warm up, then each way twice, keeping the faster.
        let _ = ours.run(
            slices[0],
            &vellum_engine::RunOptions::default(),
            &mut vellum_engine::NoTrace,
        )?;
        let mut alone = Vec::new();
        let mut alone_ms = f64::INFINITY;
        for _ in 0..2 {
            let started = Instant::now();
            alone = run_clip(&ours, &pcm)?
                .into_iter()
                .map(|(t, _)| t)
                .collect::<Vec<_>>();
            alone_ms = alone_ms.min(ms(started));
        }
        let mut together = Vec::new();
        let mut together_ms = f64::INFINITY;
        for _ in 0..2 {
            let mut texts = Vec::new();
            let started = Instant::now();
            ours.run_each(
                &slices,
                &vellum_engine::RunOptions::default(),
                &mut vellum_engine::NoTrace,
                &|| false,
                &mut |_, t| {
                    texts.push(t.text);
                    Ok(())
                },
            )?;
            together_ms = together_ms.min(ms(started));
            together = texts;
        }
        let same = alone.iter().zip(&together).filter(|(a, b)| a == b).count();
        let ok = same == alone.len() && alone.len() == together.len();
        passed &= ok;
        let line = format!(
            "{}: {} windows ({:.0} s); one at a time {alone_ms:.0} ms ({:.0}× real time), in lockstep {together_ms:.0} ms ({:.0}× real time, {:.2}× faster); {same}/{} windows the same text{}",
            clip.id,
            slices.len(),
            pcm.len() as f64 / 16000.,
            pcm.len() as f64 / 16. / alone_ms,
            pcm.len() as f64 / 16. / together_ms,
            alone_ms / together_ms,
            alone.len(),
            if ok { "" } else { "  ✗" }
        );
        println!("pipeline {line}");
        report.section("Pipeline", line);
    }

    // A batch: each window as it comes out alone.
    let n = args.batch.clamp(1, g.encoder.max_batch());
    let batch: Vec<&Mat> = (0..n).map(|i| &chunks[i % chunks.len()]).collect();
    let together = g.encoder.encode(&batch, None)?;
    let mut batch_ok = true;
    let mut table = Table::new(["window", "max |Δ| vs alone", "bit-exact"]);
    for (i, out) in together.iter().enumerate() {
        let single = g.encoder.encode(&[batch[i]], None)?.remove(0);
        let max = single
            .data
            .iter()
            .zip(&out.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        batch_ok &= single.data == out.data;
        table.row([
            format!("{i} ({})", clips[i % clips.len()].id),
            format!("{max:.2e}"),
            if single.data == out.data { "yes" } else { "no" }.to_string(),
        ]);
    }
    passed &= batch_ok;
    println!(
        "encoder batch of {n}: {}",
        if batch_ok {
            "each window as alone"
        } else {
            "✗ windows depend on their batch"
        }
    );
    report.section(format!("Encoder batch of {n}"), table.markdown());

    // Timing: the encoder by batch size.
    let flops = encoder_flops(&hp);
    let warm = Instant::now();
    while warm.elapsed() < Duration::from_millis(1500) {
        g.encode(&[(&chunks[0], 0)])?;
        device.wait()?;
    }
    let mut timing = Table::new([
        "batch",
        "wall ms / batch",
        "GPU ms / batch",
        "GPU ms / window",
        "TFLOP/s",
    ]);
    let mut breakdown = Table::new(["batch", "family", "ms / window", "share", "dispatches"]);
    let mut timings = Vec::new();
    for b in [1usize, 2, 4, 8]
        .into_iter()
        .filter(|b| *b <= g.encoder.max_batch())
    {
        let windows: Vec<(&Mat, usize)> = (0..b).map(|i| (&chunks[i % chunks.len()], i)).collect();
        let mels: Vec<&Mat> = windows.iter().map(|(m, _)| *m).collect();
        let mut wall = Vec::new();
        let mut gpu_ms = Vec::new();
        for _ in 0..args.runs.max(1) {
            let started = Instant::now();
            g.encode(&windows)?;
            device.wait()?;
            wall.push(ms(started));
            let times = g.encoder.encode_timed(&mels, false)?;
            gpu_ms.push(times.iter().map(|(_, t)| t).sum::<f64>());
        }
        let wall = median(&mut wall);
        gpu_ms.sort_by(f64::total_cmp);
        let gpu_t = gpu_ms[0];
        let families = families(|| g.encoder.encode_timed(&mels, true))?;
        let total: f64 = families.values().map(|(t, _)| t).sum();
        for (family, (t, count)) in &families {
            breakdown.row([
                b.to_string(),
                family.clone(),
                format!("{:.2}", t / b as f64),
                format!("{:.0}%", 100. * t / total),
                count.to_string(),
            ]);
        }
        timing.row([
            b.to_string(),
            format!("{wall:.1}"),
            format!("{gpu_t:.1}"),
            format!("{:.1}", gpu_t / b as f64),
            format!("{:.1}", flops * b as f64 / (gpu_t * 1e-3) / 1e12),
        ]);
        println!(
            "encoder B={b}: wall {wall:.1} ms, GPU {gpu_t:.1} ms ({:.1} ms/window)",
            gpu_t / b as f64
        );
        timings.push(
            serde_json::json!({"stage": "encoder", "batch": b, "wall_ms": wall, "gpu_ms": gpu_t}),
        );
    }
    report.section(
        "Encoder timing",
        format!(
            "One 30 s window is {:.0} GFLOP. Wall: `encode` (mel upload, record, submit) to the GPU idle, median of {}; GPU: timestamps around each submit's pass, the fastest batch. Then each dispatch timed alone, the fastest of three, summed by family.\n\n{}\n\n{}",
            flops / 1e9,
            args.runs,
            timing.markdown(),
            breakdown.markdown()
        ),
    );

    // Timing: a decoder step by batch size, every slot decoding (host mode never stops).
    let prompt = cpu.prompt_for(Some("en"))?;
    for (slot, enc) in (0..g.decoder.slots()).zip(encoded_cpu.iter().cycle()) {
        g.decoder.load_cross(enc, slot)?;
    }
    let mut steps = Table::new([
        "batch",
        "GPU ms / step",
        "GPU ms / window-step",
        "wall ms / step, 16 per submit",
        "wall ms / step, read back each",
    ]);
    let mut step_breakdown = Table::new(["batch", "family", "ms / step", "share", "dispatches"]);
    let mut step_labels = String::new();
    for b in [1usize, 2, 4, 8]
        .into_iter()
        .filter(|b| *b <= g.decoder.slots())
    {
        let start_all = || {
            for slot in 0..b {
                g.decoder.start(slot, &prompt, Mode::Host, usize::MAX);
            }
        };
        let steps_each = 32;
        let mut gpu_ms = Vec::new();
        for _ in 0..args.runs.max(1) {
            start_all();
            let times = g.decoder.step_timed(b, steps_each, false)?;
            gpu_ms.push(times.iter().map(|(_, t)| t).sum::<f64>() / steps_each as f64);
        }
        gpu_ms.sort_by(f64::total_cmp);
        let gpu_step = gpu_ms[0];
        start_all();
        let started = Instant::now();
        let mut pending = std::collections::VecDeque::new();
        for _ in 0..2 {
            pending.push_back(g.decoder.submit(b, 16, &[])?);
        }
        for _ in 0..4 {
            g.decoder.finish(pending.pop_front().unwrap())?;
            pending.push_back(g.decoder.submit(b, 16, &[])?);
        }
        for p in pending {
            g.decoder.finish(p)?;
        }
        let chunked = ms(started) / (6. * 16.);
        start_all();
        let started = Instant::now();
        for _ in 0..32 {
            g.decoder.finish(g.decoder.submit(b, 1, &[])?)?;
        }
        let each = ms(started) / 32.;
        steps.row([
            b.to_string(),
            format!("{gpu_step:.3}"),
            format!("{:.3}", gpu_step / b as f64),
            format!("{chunked:.3}"),
            format!("{each:.3}"),
        ]);
        println!(
            "decoder B={b}: GPU {gpu_step:.3} ms/step, wall {chunked:.3} ms/step chunked, {each:.3} ms/step read back each"
        );
        start_all();
        let labels = by_label(|| g.decoder.step_timed(b, 8, true))?;
        if b == 1 || b == g.decoder.slots() {
            let mut t = Table::new(["dispatch", "ms / step", "per step"]);
            let mut sorted: Vec<_> = labels.iter().collect();
            sorted.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
            for (label, (ms, n)) in sorted {
                t.row([
                    label.clone(),
                    format!("{:.3}", ms / 8.),
                    (n / 8).to_string(),
                ]);
            }
            write!(
                step_labels,
                "

Batch of {b}, each dispatch timed alone:

{}",
                t.markdown()
            )?;
        }
        start_all();
        let families = families(|| g.decoder.step_timed(b, 8, true))?;
        let total: f64 = families.values().map(|(t, _)| t).sum();
        for (family, (t, count)) in &families {
            step_breakdown.row([
                b.to_string(),
                family.clone(),
                format!("{:.3}", t / 8.),
                format!("{:.0}%", 100. * t / total),
                (count / 8).to_string(),
            ]);
        }
        timings.push(serde_json::json!({"stage": "decoder step", "batch": b, "gpu_ms": gpu_step, "wall_chunked_ms": chunked, "wall_each_ms": each}));
    }
    for slot in 0..g.decoder.slots() {
        g.decoder.stop(slot);
    }
    // What a step costs once every slot is done (a run submitted ahead of the end).
    let idle = g.decoder.step_timed(1, 32, false)?;
    let idle_ms = idle.iter().map(|(_, t)| t).sum::<f64>() / 32.;
    println!("decoder idle step: GPU {idle_ms:.3} ms");
    report.section(
        "Decoder step timing",
        format!(
            "Steps from position 0 with every slot decoding (positions 3–35). GPU: timestamps around the submits of 32 steps, the fastest run, per step. Wall: steps submitted 16 at a time two runs in flight, the state read back after each run (the greedy path); and one step per submit read back each time (the sampling path, plus its logits row). Then each dispatch timed alone over 8 steps, by family. A step once every slot is done (every kernel returns at once): {idle_ms:.3} ms.\n\n{}\n\n{}",
            steps.markdown(),
            step_breakdown.markdown()
        ) + &step_labels,
    );
    device.save_pipeline_cache()?;
    report.passed = passed;
    report.results = serde_json::json!({
        "adapter": device.name(),
        "model": model.id,
        "clips": results,
        "batch_ok": batch_ok,
        "timing": timings,
    });
    let mut out = String::new();
    writeln!(out, "{}", report.save()?.display())?;
    print!("{out}");
    Ok(passed)
}

/// Each dispatch's fastest of three runs, summed by label and by family (the label before
/// `/`): GPU milliseconds and dispatches.
fn families(
    run: impl FnMut() -> vellum_engine::Result<Vec<(String, f64)>>,
) -> Result<BTreeMap<String, (f64, usize)>> {
    Ok(by_label(run)?
        .into_iter()
        .fold(BTreeMap::new(), |mut families, (label, (t, n))| {
            let family = label.split('/').next().unwrap_or(&label).to_owned();
            let e: &mut (f64, usize) = families.entry(family).or_default();
            e.0 += t;
            e.1 += n;
            families
        }))
}

/// Each dispatch's fastest of three runs, summed by label.
fn by_label(
    mut run: impl FnMut() -> vellum_engine::Result<Vec<(String, f64)>>,
) -> Result<BTreeMap<String, (f64, usize)>> {
    let mut best: Vec<(String, f64)> = Vec::new();
    for _ in 0..3 {
        let times = run()?;
        if best.len() != times.len() {
            best = times;
        } else {
            for (b, (_, t)) in best.iter_mut().zip(times) {
                b.1 = b.1.min(t);
            }
        }
    }
    let mut labels: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    for (label, t) in best {
        let e = labels.entry(label).or_default();
        e.0 += t;
        e.1 += 1;
    }
    Ok(labels)
}

/// The encoder's FLOPs for one 30 s window (a multiply-add is two).
fn encoder_flops(hp: &vellum_engine::whisper::HParams) -> f64 {
    let (d, t, l) = (hp.d_model as f64, 1500., hp.enc_layers as f64);
    let ff = 4. * d;
    let mels = if hp.d_model == 1280 { 128. } else { 80. };
    let conv = 2. * 3000. * 3. * mels * d + 2. * t * 3. * d * d;
    let linear = 2. * t * (4. * d * d + 2. * d * ff) * l;
    let attention = l * 2. * 2. * t * t * d;
    conv + linear + attention
}

fn ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1e3
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Every event of a run, in order.
#[derive(Default)]
struct Events(Vec<(String, serde_json::Value)>);

impl Trace for Events {
    fn event(&mut self, kind: &str, fields: serde_json::Value) {
        self.0.push((kind.to_owned(), fields));
    }
}

/// One clip through an engine as the harness runs it without a language: its windows in
/// order, each detecting until one has text; each window's text and events.
fn run_clip(model: &Whisper, pcm: &[f32]) -> Result<Vec<(String, Events)>> {
    use vellum_harness::windows;
    let size = windows::window_samples(28_000, model.max_audio_ms());
    let mut language: Option<String> = None;
    let mut out = Vec::new();
    for range in windows::windows(pcm, size) {
        let pcm = &pcm[range];
        if windows::silent(pcm) {
            continue;
        }
        let mut events = Events::default();
        let options = vellum_engine::RunOptions {
            language: language.clone(),
        };
        let t = model.run(pcm, &options, &mut events)?;
        if language.is_none() && !t.text.trim().is_empty() {
            language = t.language.clone();
        }
        out.push((t.text, events));
    }
    Ok(out)
}

/// Why two runs of a window part: the first event where they differ, classified.
fn first_difference(a: &Events, b: &Events) -> String {
    let f = |v: &serde_json::Value, k: &str| v[k].as_f64().unwrap_or(f64::NAN);
    for (x, y) in a.0.iter().zip(&b.0) {
        if x == y {
            continue;
        }
        let (kind, ex, ey) = (&x.0, &x.1, &y.1);
        if kind != &y.0 {
            return format!("events differ: {kind} / {}", y.0);
        }
        return match kind.as_str() {
            "whisper.language" if ex["language"] != ey["language"] => format!(
                "language near-tie: oracle {} (margin {:.3}), GPU {} (margin {:.3})",
                ex["language"],
                f(ex, "margin"),
                ey["language"],
                f(ey, "margin")
            ),
            "whisper.language" => continue,
            "whisper.step" if ex["token"] != ey["token"] => {
                let tier = ex["tier"].as_u64().unwrap_or(0);
                let what = if tier == 0 {
                    "greedy near-tie".to_owned()
                } else {
                    format!("sampled at T={:.1}", tier as f64 * 0.2)
                };
                format!(
                    "{what} at seek {} step {}: oracle {} (margin {:.3}), GPU {} (margin {:.3})",
                    ex["seek"],
                    ex["step"],
                    ex["token"],
                    f(ex, "top2_margin"),
                    ey["token"],
                    f(ey, "top2_margin")
                )
            }
            "whisper.step" => continue,
            "whisper.pass"
                if ex["temperature"] == ey["temperature"] && ex["tokens"] == ey["tokens"] =>
            {
                let (ax, ay) = (f(ex, "avg_logprob"), f(ey, "avg_logprob"));
                let (cx, cy) = (f(ex, "compression_ratio"), f(ey, "compression_ratio"));
                let (nx, ny) = (f(ex, "no_speech_prob"), f(ey, "no_speech_prob"));
                if ex["no_speech_skip"] != ey["no_speech_skip"] {
                    format!(
                        "no-speech near-tie at seek {}: p {nx:.4} / {ny:.4} (threshold 0.6), avg log-prob {ax:.4} / {ay:.4}",
                        ex["seek"]
                    )
                } else if (ax >= -1.) != (ay >= -1.) {
                    format!(
                        "log-prob threshold near-tie at seek {}: {ax:.4} / {ay:.4} (threshold -1)",
                        ex["seek"]
                    )
                } else if (cx <= 2.4) != (cy <= 2.4) {
                    format!(
                        "compression threshold near-tie at seek {}: {cx:.4} / {cy:.4}",
                        ex["seek"]
                    )
                } else {
                    continue;
                }
            }
            _ => format!("{kind}: {ex} / {ey}"),
        };
    }
    if a.0.len() != b.0.len() {
        return format!("one run has more events ({} / {})", a.0.len(), b.0.len());
    }
    "no event differs".into()
}

/// `--explain`: each clip through the CPU oracle and the GPU path, the windows whose text
/// differs and why.
pub fn explain(args: &Args) -> Result<bool> {
    let mut options = GpuOptions::from_env();
    options.cache_dir = Some(paths::root()?.join("gpu-cache"));
    let device = Arc::new(Gpu::new(&options)?);
    let model = models::get(&args.model)?;
    let file = Gguf::open(&model.ensure()?)?;
    let cpu = Whisper::load(&file)?;
    let ours = Whisper::load_with(&file, Some(device), &WhisperGpuOptions::default())?;
    let corpus = Corpus::load(&paths::corpus()?)?;
    let mut table = Table::new(["clip", "windows", "differ", "why"]);
    let mut same = 0;
    for id in &args.clips {
        let clip = corpus.get(id).with_context(|| format!("no clip {id}"))?;
        let pcm = vellum_harness::audio::read(&corpus.audio(clip))?;
        let a = run_clip(&cpu, &pcm)?;
        let b = run_clip(&ours, &pcm)?;
        let mut why = Vec::new();
        let passes = |e: &Events| e.0.iter().filter(|(k, _)| k == "whisper.pass").count();
        println!(
            "{id}: passes per window, oracle {:?}, GPU {:?}",
            a.iter().map(|(_, e)| passes(e)).collect::<Vec<_>>(),
            b.iter().map(|(_, e)| passes(e)).collect::<Vec<_>>()
        );
        for (w, ((ta, ea), (tb, eb))) in a.iter().zip(&b).enumerate() {
            if ta != tb {
                why.push(format!("window {w}: {}", first_difference(ea, eb)));
                let passes = |e: &Events| -> Vec<String> {
                    e.0.iter()
                        .filter(|(k, _)| k == "whisper.pass")
                        .map(|(_, v)| {
                            format!(
                                "seek {} T={} logprob {:.3} compression {:.2} no-speech {:.3} tokens {}",
                                v["seek"], v["temperature"], v["avg_logprob"].as_f64().unwrap_or(0.),
                                v["compression_ratio"].as_f64().unwrap_or(0.),
                                v["no_speech_prob"].as_f64().unwrap_or(0.), v["tokens"]
                            )
                        })
                        .collect()
                };
                println!(
                    "  oracle passes: {:?}
  GPU passes:    {:?}
  oracle: {ta}
  GPU:    {tb}",
                    passes(ea),
                    passes(eb)
                );
            }
        }
        if a.len() != b.len() {
            why.push(format!("{} / {} windows", a.len(), b.len()));
        }
        same += usize::from(why.is_empty());
        println!(
            "{id}: {}",
            if why.is_empty() {
                "GPU = oracle".to_owned()
            } else {
                why.join("; ")
            }
        );
        table.row([
            id.clone(),
            a.len().to_string(),
            why.len().to_string(),
            why.join("; "),
        ]);
    }
    println!(
        "{same}/{} clips: the GPU's text is the oracle's",
        args.clips.len()
    );
    let mut report = Report::new(
        &format!("whisper-explain-{}", model.id),
        serde_json::to_value(system::machine())?,
    );
    report.section(
        "GPU against the CPU oracle",
        format!(
            "{same}/{} clips give the oracle's text. A window that differs is classified by the first decision that does: a greedy step (both margins), a sampled step (the fallback temperature), the language, or an attempt judged across a threshold.\n\n{}",
            args.clips.len(),
            table.markdown()
        ),
    );
    report.passed = true;
    println!("{}", report.save()?.display());
    Ok(true)
}
