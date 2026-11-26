//! `cargo lab bench`: two engines raced live on the speed clips, in alternating order (ABBA), each
//! job a fresh process that loads its model once. Speedups come from paired runs, so a laptop
//! warming up or a background task slows both sides alike instead of flattering one.

use crate::{
    check,
    corpus::{Clip, Corpus, Purpose, Tier},
    jobs::{self, Plan},
    models::{self, Kind, Model},
    paths,
    report::Report,
    run::{self, Candidate},
    score::{self, stats},
    system,
};
use anyhow::Result;
use serde::Serialize;
use std::{collections::BTreeMap, time::Duration};
use vellum_harness::Windowing;

pub struct Args {
    pub tier: Tier,
    pub models: Vec<String>,
    pub a: Candidate,
    pub b: Candidate,
    /// Jobs per side; pairs per clip.
    pub blocks: u32,
    pub env: Vec<(String, String)>,
    /// Wait for the GPU to cool below this between jobs (°C).
    pub cool: Option<u32>,
}

/// One side's job in one block.
#[derive(Debug, Clone, Serialize)]
struct JobRun {
    side: char,
    block: u32,
    error: Option<String>,
    ready_ms: f64,
    load_ms: f64,
    peak_rss: u64,
    /// Per clip: the first run (kernel compilation, cold caches) and the measured warm run.
    first_ms: BTreeMap<String, f64>,
    warm_ms: BTreeMap<String, f64>,
    gpu: Option<system::GpuSummary>,
    /// Its GPU ran under the clocks the others got (see `mark_slowed`): its pairs don't count.
    slowed: bool,
}

/// Whether a job's GPU ran slower than it can. The driver's reasons don't say: this laptop flags
/// "software thermal slowdown" with every power cap, at full boost clocks, while a GPU still
/// ramping up from idle shows no reason at all. So the clocks decide — compared only among the
/// same engine's jobs, since an engine that keeps the GPU busier holds it at its power cap and a
/// lower clock by design. A job whose median clock is more than 5% under its own side's best, or
/// that saw a hardware slowdown, doesn't count.
fn mark_slowed(runs: &mut [JobRun]) {
    for side in ['A', 'B'] {
        let best = runs
            .iter()
            .filter(|r| r.side == side)
            .filter_map(|r| r.gpu.as_ref()?.sm_mhz_median)
            .max()
            .unwrap_or(0);
        for run in runs.iter_mut().filter(|r| r.side == side) {
            let Some(gpu) = &run.gpu else { continue };
            let slow_clock = gpu
                .sm_mhz_median
                .is_some_and(|mhz| (mhz as f64) < 0.95 * best as f64);
            let hardware = gpu.throttled.iter().any(|r| r.starts_with("hardware"));
            run.slowed = slow_clock || hardware;
        }
    }
}

fn speed_clips<'a>(corpus: &'a Corpus, model: &Model, tier: Tier) -> Vec<&'a Clip> {
    let purpose = match model.kind {
        Kind::Speech => Purpose::Asr,
        Kind::Speakers => Purpose::Diarization,
    };
    let runs_on = check::clips(corpus, model, tier);
    corpus
        .select(tier, Purpose::Speed)
        .into_iter()
        .filter(|c| c.purposes.contains(&purpose) && runs_on.iter().any(|r| r.id == c.id))
        .collect()
}

pub fn bench(args: &Args) -> Result<bool> {
    let corpus = Corpus::load(&paths::corpus()?)?;
    let machine = system::machine();
    let warnings = system::conditions();
    for w in &warnings {
        eprintln!("! {w}");
    }
    let plan = Plan {
        windowing: Windowing::default(),
        env: args.env.clone(),
        machine: machine.fingerprint.clone(),
    };
    let mut report = Report::new(
        &format!("bench {:?}", args.tier).to_lowercase(),
        serde_json::to_value(&machine)?,
    );
    if !warnings.is_empty() {
        report.section(
            "Conditions",
            warnings
                .iter()
                .map(|w| format!("- {w}\n"))
                .collect::<String>(),
        );
    }
    let work = paths::reports().join("runs").join(&report.id);
    let models: Vec<&Model> = if args.models.is_empty() {
        models::MODELS.iter().collect()
    } else {
        args.models
            .iter()
            .map(|m| models::get(m))
            .collect::<Result<_>>()?
    };
    let same = args.a == args.b;
    let mut overview = score::table::Table::new([
        "model",
        "clips",
        "pairs",
        "A",
        "B",
        "speedup B/A",
        "95% interval",
        "p",
        "first run A",
        "first run B",
    ]);
    let mut results = serde_json::Map::new();
    for model in models {
        let clips = speed_clips(&corpus, model, args.tier);
        if clips.is_empty() {
            eprintln!("· {}: no speed clips in the {:?} tier", model.id, args.tier);
            continue;
        }
        let programs = [run::runner(args.a)?, run::runner(args.b)?];
        let sides = [(args.a, 'A'), (args.b, 'B')];
        let mut runs: Vec<JobRun> = Vec::new();
        for block in 0..args.blocks {
            // ABBA: each side goes first as often as second.
            let order = if block % 2 == 0 { [0, 1] } else { [1, 0] };
            for i in order {
                let (candidate, side) = sides[i];
                if let Some(limit) = args.cool {
                    system::wait_for_cool(limit, Duration::from_secs(120));
                }
                eprintln!(
                    "· {} block {}/{} {side} ({candidate})",
                    model.id,
                    block + 1,
                    args.blocks
                );
                let job = jobs::job(
                    &plan,
                    model,
                    &corpus,
                    &clips,
                    candidate,
                    &work.join(format!("{}-{block}-{side}", model.id)),
                    1,
                    1,
                )?;
                let outcome = run::run(&programs[i], &job, true)?;
                let r = &outcome.report;
                runs.push(JobRun {
                    side,
                    block,
                    error: r.error.clone(),
                    ready_ms: r.ready_ms,
                    load_ms: r.load_ms,
                    peak_rss: r.peak_rss,
                    first_ms: r
                        .clips
                        .iter()
                        .filter_map(|c| c.warmup_ms.first().map(|ms| (c.id.clone(), *ms)))
                        .collect(),
                    warm_ms: r
                        .clips
                        .iter()
                        .filter(|c| c.error.is_none())
                        .filter_map(|c| c.runs.first().map(|t| (c.id.clone(), t.wall_ms)))
                        .collect(),
                    slowed: false,
                    gpu: outcome.gpu,
                });
                if let Some(error) = &r.error {
                    eprintln!("  {error}");
                    if error.contains("supported yet") {
                        break;
                    }
                }
            }
        }
        mark_slowed(&mut runs);
        let summary = summarize(model, &clips, &runs, same);
        overview.row(summary.row.clone());
        report.section(format!("{} ({})", model.name, model.id), summary.body);
        results.insert(
            model.id.into(),
            serde_json::json!({ "speed": summary.speed, "runs": runs }),
        );
    }
    report.sections.insert(
        0,
        (
            "Summary".into(),
            format!(
                "A = {}, B = {}; speedup = A's time / B's time (above 1: B is faster).{}\n\n{}",
                args.a,
                args.b,
                if same {
                    " Both sides are the same engine: the spread is this machine's noise."
                } else {
                    ""
                },
                overview.markdown()
            ),
        ),
    );
    report.results = serde_json::Value::Object(results);
    let path = report.save()?;
    println!("{}", overview.markdown());
    println!("{}", path.display());
    Ok(true)
}

struct Summary {
    row: Vec<String>,
    body: String,
    speed: Option<stats::PairedSpeed>,
}

fn ms(v: f64) -> String {
    if v >= 10_000. {
        format!("{:.1} s", v / 1000.)
    } else {
        format!("{v:.0} ms")
    }
}

fn summarize(model: &Model, clips: &[&Clip], runs: &[JobRun], same: bool) -> Summary {
    let side = |s: char| {
        runs.iter()
            .filter(move |r| r.side == s && r.error.is_none())
    };
    // Pairs: the same clip in the same block on both sides, neither held back.
    let mut pairs: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut table = score::table::Table::new([
        "clip",
        "audio",
        "A",
        "B",
        "A × real time",
        "B × real time",
        "speedup",
    ]);
    for clip in clips {
        let mut clip_pairs = Vec::new();
        for a in side('A').filter(|r| !r.slowed) {
            if let (Some(ta), Some(tb)) = (
                a.warm_ms.get(&clip.id),
                side('B')
                    .filter(|b| b.block == a.block && !b.slowed)
                    .find_map(|b| b.warm_ms.get(&clip.id)),
            ) {
                clip_pairs.push((*ta, *tb));
            }
        }
        if clip_pairs.is_empty() {
            continue;
        }
        let a = stats::median(&clip_pairs.iter().map(|p| p.0).collect::<Vec<_>>());
        let b = stats::median(&clip_pairs.iter().map(|p| p.1).collect::<Vec<_>>());
        let ratio = stats::median(&clip_pairs.iter().map(|p| p.0 / p.1).collect::<Vec<_>>());
        table.row(vec![
            clip.id.clone(),
            format!("{:.1} s", clip.seconds),
            ms(a),
            ms(b),
            format!("{:.0}×", clip.seconds * 1000. / a),
            format!("{:.0}×", clip.seconds * 1000. / b),
            format!("{ratio:.3}"),
        ]);
        pairs.push(clip_pairs);
    }
    let speed = (!pairs.is_empty()).then(|| stats::paired(&pairs));
    let first = |s: char| {
        let v: Vec<f64> = side(s)
            .filter_map(|r| r.first_ms.values().next().copied())
            .collect();
        (!v.is_empty()).then(|| stats::median(&v))
    };
    let cold = |s: char, f: fn(&JobRun) -> f64| {
        let v: Vec<f64> = side(s).map(f).collect();
        (!v.is_empty()).then(|| stats::median(&v))
    };
    let mut body = String::new();
    let failed: Vec<&JobRun> = runs.iter().filter(|r| r.error.is_some()).collect();
    for r in failed.iter().take(4) {
        body.push_str(&format!(
            "- {} block {}: {}\n",
            r.side,
            r.block + 1,
            r.error.as_deref().unwrap_or("")
        ));
    }
    let slowed = runs.iter().filter(|r| r.slowed).count();
    if slowed > 0 {
        body.push_str(&format!(
            "- {slowed} jobs ran with the GPU clocked more than 5% under the same engine's best job, or \
             hardware-slowed; their pairs don't count.\n"
        ));
    }
    body.push_str(&format!(
        "\nWarm runs, per clip (medians over pairs):\n\n{}",
        table.markdown()
    ));
    let mut cold_table = score::table::Table::new(["", "A", "B"]);
    let opt = |v: Option<f64>| v.map_or("–".into(), ms);
    cold_table.row(vec![
        "process start to model ready".into(),
        opt(cold('A', |r| r.ready_ms)),
        opt(cold('B', |r| r.ready_ms)),
    ]);
    cold_table.row(vec![
        "model load".into(),
        opt(cold('A', |r| r.load_ms)),
        opt(cold('B', |r| r.load_ms)),
    ]);
    cold_table.row(vec![
        "first run (compiles kernels)".into(),
        opt(first('A')),
        opt(first('B')),
    ]);
    cold_table.row(vec![
        "peak memory".into(),
        cold('A', |r| r.peak_rss as f64).map_or("–".into(), |b| format!("{:.0} MB", b / 1e6)),
        cold('B', |r| r.peak_rss as f64).map_or("–".into(), |b| format!("{:.0} MB", b / 1e6)),
    ]);
    let gpu_mem = |s: char| {
        let v: Vec<f64> = side(s)
            .filter_map(|r| r.gpu.as_ref()?.memory_used_peak_mb.map(f64::from))
            .collect();
        (!v.is_empty()).then(|| format!("{:.0} MB", stats::median(&v)))
    };
    cold_table.row(vec![
        "GPU memory in use, peak".into(),
        gpu_mem('A').unwrap_or("–".into()),
        gpu_mem('B').unwrap_or("–".into()),
    ]);
    body.push_str(&format!(
        "\nCold costs (medians over jobs):\n\n{}",
        cold_table.markdown()
    ));
    let total = |s: char| -> Option<f64> {
        let per_clip: Vec<f64> = pairs
            .iter()
            .map(|p| {
                stats::median(
                    &p.iter()
                        .map(|x| if s == 'A' { x.0 } else { x.1 })
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        (!per_clip.is_empty()).then(|| per_clip.iter().sum())
    };
    let row = vec![
        model.id.into(),
        clips.len().to_string(),
        pairs.iter().map(Vec::len).sum::<usize>().to_string(),
        total('A').map_or("–".into(), ms),
        total('B').map_or("–".into(), ms),
        speed
            .as_ref()
            .map_or("–".into(), |s| format!("{:.3}", s.median_speedup)),
        speed
            .as_ref()
            .map_or("–".into(), |s| format!("{:.3}–{:.3}", s.ci95.0, s.ci95.1)),
        speed
            .as_ref()
            .map_or("–".into(), |s| format!("{:.3}", s.wilcoxon_p)),
        opt(first('A')),
        opt(first('B')),
    ];
    if same && let Some(s) = &speed {
        body.push_str(&format!(
            "\nThe same engine on both sides: speedups within ±{:.1}% are noise on this machine.\n",
            100. * stats::noise_floor(&pairs)
        ));
        let _ = s;
    }
    Summary { row, body, speed }
}
