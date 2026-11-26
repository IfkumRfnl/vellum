//! `cargo lab probe`: one window through both engines with every checkpoint dumped, lined up
//! tensor by tensor, to find the first stage where they part. The reference's dumps are kept.

use crate::{
    check,
    corpus::{Clip, Corpus, Purpose, Tier},
    goldens,
    jobs::{self, Plan},
    models, paths, reference,
    report::Report,
    run::{self, Candidate, Engine},
    score, system,
};
use anyhow::{Context, Result, bail};
use std::{fs, path::PathBuf};
use vellum_harness::{DumpRequest, InjectRequest, Windowing};

pub struct Args {
    pub model: String,
    pub clip: Option<String>,
    pub engine: Candidate,
    pub against: Candidate,
    /// Block indices to dump sub-layer activations of.
    pub sub_blocks: Vec<u32>,
    /// Checkpoints to feed from the reference into the engine.
    pub inject: Vec<String>,
    /// Relative error past which a checkpoint counts as parted; by default twice the spread
    /// the reference shows between its own CPU and GPU (see `calibrate`), else 1e-3.
    pub tolerance: Option<f64>,
    pub env: Vec<(String, String)>,
    /// Measure that spread instead of probing.
    pub calibrate: bool,
}

/// How far the reference's GPU path strays from its CPU path, per checkpoint: the largest
/// relative L2 error over a model's probe clips. Our engine's GPU path is held to a small
/// multiple of it against an exact CPU path; parity with ggml beyond that isn't the goal.
type Spreads = std::collections::BTreeMap<String, f64>;

fn spreads_path(model: &models::Model) -> Result<PathBuf> {
    Ok(paths::root()?
        .join("tolerances")
        .join(format!("{}.json", model.id)))
}

fn calibrate(
    args: &Args,
    corpus: &Corpus,
    model: &models::Model,
    plan: &Plan,
    report: &mut Report,
    work: &std::path::Path,
) -> Result<bool> {
    let runs_on = check::clips(corpus, model, Tier::Long);
    let clips: Vec<&Clip> = corpus
        .select(Tier::Quick, Purpose::Probe)
        .into_iter()
        .filter(|c| runs_on.iter().any(|r| r.id == c.id) && c.seconds <= 28.)
        .collect();
    anyhow::ensure!(!clips.is_empty(), "no probe clips for {}", model.id);
    let cpu: Candidate = "ref-cpu".parse()?;
    let gpu: Candidate = "ref-gpu".parse()?;
    let mut spreads = Spreads::new();
    for clip in &clips {
        let a = dumps(plan, model, corpus, clip, cpu, &args.sub_blocks, None, work)?;
        let b = dumps(plan, model, corpus, clip, gpu, &args.sub_blocks, None, work)?;
        for c in score::tensors::compare_dirs(&a, &b, f64::INFINITY)?.checkpoints {
            let s = spreads.entry(c.name).or_insert(0.);
            *s = s.max(c.diff.rel_l2);
        }
    }
    let path = spreads_path(model)?;
    fs::create_dir_all(path.parent().unwrap())?;
    fs::write(&path, serde_json::to_vec_pretty(&spreads)?)?;
    let mut table = score::table::Table::new(["checkpoint", "GPU against CPU, rel L2"]);
    for (name, spread) in &spreads {
        table.row(vec![name.clone(), format!("{spread:.2e}")]);
    }
    let body = format!(
        "The reference on {} probe clips, its GPU against its CPU: the largest relative error per checkpoint. Probes allow twice this.

{}",
        clips.len(),
        table.markdown()
    );
    println!(
        "{body}
{}",
        path.display()
    );
    report.section(format!("Calibration: {}", model.id), body);
    report.results = serde_json::to_value(&spreads)?;
    println!("{}", report.save()?.display());
    Ok(true)
}

fn probe_clip<'a>(corpus: &'a Corpus, model: &models::Model, id: Option<&str>) -> Result<&'a Clip> {
    if let Some(id) = id {
        return corpus
            .get(id)
            .with_context(|| format!("no clip {id} in the corpus"));
    }
    let runs_on = check::clips(corpus, model, Tier::Long);
    let mut probes: Vec<&Clip> = corpus
        .select(Tier::Long, Purpose::Probe)
        .into_iter()
        .filter(|c| runs_on.iter().any(|r| r.id == c.id))
        .collect();
    probes.sort_by(|a, b| a.tier.cmp(&b.tier).then(a.id.cmp(&b.id)));
    probes.first().copied().with_context(|| {
        format!(
            "no probe clip for {}; run `cargo lab fetch --tier smoke`",
            model.id
        )
    })
}

pub fn probe(args: &Args) -> Result<bool> {
    let corpus = Corpus::load(&paths::corpus()?)?;
    let model = models::get(&args.model)?;
    let clip = probe_clip(&corpus, model, args.clip.as_deref())?;
    if clip.seconds > 28. {
        eprintln!(
            "! {} is {:.0} s, more than one window: the dumps show its last window",
            clip.id, clip.seconds
        );
    }
    let machine = system::machine();
    let plan = Plan {
        windowing: Windowing::default(),
        env: args.env.clone(),
        machine: machine.fingerprint.clone(),
    };
    let mut report = Report::new(
        &format!("probe {}", model.id),
        serde_json::to_value(&machine)?,
    );
    let work = paths::reports().join("runs").join(&report.id);
    if args.calibrate {
        return calibrate(args, &corpus, model, &plan, &mut report, &work);
    }
    let reference_dir = dumps(
        &plan,
        model,
        &corpus,
        clip,
        args.against,
        &args.sub_blocks,
        None,
        &work,
    )?;
    let candidate_dir = dumps(
        &plan,
        model,
        &corpus,
        clip,
        args.engine,
        &args.sub_blocks,
        (!args.inject.is_empty()).then(|| InjectRequest {
            dir: reference_dir.clone(),
            names: args.inject.clone(),
        }),
        &work,
    )?;
    let spreads: Option<Spreads> = fs::read(spreads_path(model)?)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok());
    if args.tolerance.is_none() && spreads.is_none() {
        eprintln!(
            "! no calibration for {}: using 1e-3 (run `cargo lab probe --calibrate --model {}`)",
            model.id, model.id
        );
    }
    let tolerance = |sidecar: &vellum_harness::dump::Sidecar| match (args.tolerance, &spreads) {
        (Some(t), _) => t,
        (None, Some(spreads)) => spreads
            .get(&sidecar.name)
            .map_or(1e-3, |s| (2. * s).max(1e-5)),
        (None, None) => 1e-3,
    };
    let comparison = score::tensors::compare_dirs(&reference_dir, &candidate_dir, tolerance)?;
    let markdown = comparison.markdown();
    println!(
        "{} on {} ({:.1} s): {} against {}\n\n{markdown}",
        model.id, clip.id, clip.seconds, args.engine, args.against
    );
    report.passed = comparison.first_parted.is_none() && comparison.missing.is_empty();
    report.section(
        format!("{} against {} on {}", args.engine, args.against, clip.id),
        markdown,
    );
    report.results = serde_json::to_value(&comparison)?;
    println!("{}", report.save()?.display());
    Ok(report.passed)
}

/// A dump of `candidate` on `clip`: the reference's kept ones are reused; others are made fresh.
#[allow(clippy::too_many_arguments)]
fn dumps(
    plan: &Plan,
    model: &models::Model,
    corpus: &Corpus,
    clip: &Clip,
    candidate: Candidate,
    sub_blocks: &[u32],
    inject: Option<InjectRequest>,
    work: &std::path::Path,
) -> Result<PathBuf> {
    let keep = candidate.engine == Engine::Reference;
    let dir = if keep {
        let reference = reference::ensure()?;
        let mut key = goldens::Key::new(
            candidate,
            &reference,
            model,
            &plan.machine,
            plan.windowing,
            &plan.env,
        );
        key.env
            .push(("sub_blocks".into(), format!("{sub_blocks:?}")));
        paths::dumps()?
            .join(model.id)
            .join(candidate.to_string())
            .join(key.id())
            .join(clip.id.replace(['/', '\\', ':'], "__"))
    } else {
        work.join(format!("dump-{candidate}"))
    };
    let done = dir.join(".done");
    if keep && done.is_file() {
        return Ok(dir);
    }
    let _ = fs::remove_dir_all(&dir);
    let mut job = jobs::job(
        plan,
        model,
        corpus,
        &[clip],
        candidate,
        &work.join(format!("probe-{candidate}")),
        0,
        1,
    )?;
    job.dump = Some(DumpRequest {
        dir: dir.clone(),
        all_blocks: true,
        sub_blocks: sub_blocks.to_vec(),
    });
    job.inject = inject;
    eprintln!("· {candidate} {}: dumping {}", model.id, clip.id);
    let outcome = run::run(&run::runner(candidate)?, &job, false)?;
    if let Some(error) = outcome
        .report
        .error
        .as_ref()
        .or_else(|| outcome.report.clips.first().and_then(|c| c.error.as_ref()))
    {
        bail!("{candidate} failed: {error}\n{}", outcome.log_tail(12));
    }
    if keep {
        fs::write(&done, "")?;
    }
    Ok(dir)
}
