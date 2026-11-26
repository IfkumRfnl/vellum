//! Jobs for a model over some clips, and the reference results they're compared with.

use crate::{
    corpus::{Clip, Corpus},
    goldens,
    models::{Kind, Model},
    reference,
    run::{self, Candidate, Engine},
};
use anyhow::Result;
use std::path::Path;
use vellum_harness::{ClipJob, ClipReport, EngineInfo, Job, SCHEMA, Task, Windowing};

/// Settings shared by the jobs of one lab run.
#[derive(Clone)]
pub struct Plan {
    pub windowing: Windowing,
    /// The reference's environment knobs (`TRANSCRIBE_NO_FLASH=1`…); part of the golden key.
    pub env: Vec<(String, String)>,
    /// `system::Machine::fingerprint`.
    pub machine: String,
}

#[allow(clippy::too_many_arguments)]
pub fn job(
    plan: &Plan,
    model: &Model,
    corpus: &Corpus,
    clips: &[&Clip],
    candidate: Candidate,
    out_dir: &Path,
    warmup: u32,
    repeats: u32,
) -> Result<Job> {
    let path = model.ensure()?;
    Ok(Job {
        schema: SCHEMA,
        task: match model.kind {
            Kind::Speech => Task::Transcribe {
                model: path,
                model_id: model.app_id.into(),
                // As the app asks: English for Parakeet, Whisper detects.
                language: model.language.map(str::to_owned),
                windowing: plan.windowing,
            },
            Kind::Speakers => Task::Diarize { model: path },
        },
        device: candidate.device,
        clips: clips
            .iter()
            .map(|c| ClipJob {
                id: c.id.clone(),
                audio: corpus.audio(c),
            })
            .collect(),
        warmup,
        repeats,
        dump: None,
        inject: None,
        env: match candidate.engine {
            Engine::Reference => plan.env.clone(),
            Engine::Vellum => Vec::new(),
        },
        out_dir: out_dir.to_owned(),
    })
}

/// The reference's results for `clips`, kept from before or made now (one job for all that are
/// missing). In clip order; a clip the reference failed on carries its error.
pub fn goldens(
    plan: &Plan,
    model: &Model,
    corpus: &Corpus,
    clips: &[&Clip],
    candidate: Candidate,
    work: &Path,
) -> Result<(Vec<ClipReport>, Option<EngineInfo>)> {
    let reference = reference::ensure()?;
    let key = goldens::Key::new(
        candidate,
        &reference,
        model,
        &plan.machine,
        plan.windowing,
        &plan.env,
    );
    let mut found = Vec::with_capacity(clips.len());
    let mut missing = Vec::new();
    for clip in clips {
        let kept = goldens::load(&key, clip)?;
        if kept.is_none() {
            missing.push(*clip);
        }
        found.push(kept);
    }
    let mut engine = goldens::engine(&key)?;
    if !missing.is_empty() {
        let hours: f64 = missing.iter().map(|c| c.seconds).sum::<f64>() / 3600.;
        eprintln!(
            "· {candidate} {}: reference results for {} clips ({hours:.2} h of audio)",
            model.id,
            missing.len()
        );
        // Twice when it's cheap, to catch a reference that isn't bit-stable.
        let repeats = if hours < 0.25 { 2 } else { 1 };
        // In jobs of about an hour of audio, each kept as it finishes, so a run that's
        // interrupted loses at most one.
        let mut chunks: Vec<Vec<&Clip>> = vec![Vec::new()];
        let mut seconds = 0.;
        for clip in &missing {
            if seconds >= 3600. {
                chunks.push(Vec::new());
                seconds = 0.;
            }
            seconds += clip.seconds;
            chunks.last_mut().unwrap().push(clip);
        }
        for chunk in &chunks {
            let job = job(
                plan,
                model,
                corpus,
                chunk,
                candidate,
                &work.join(format!("golden-{candidate}-{}", model.id)),
                0,
                repeats,
            )?;
            let outcome = run::run(&reference.exe, &job, false)?;
            if let Some(error) = &outcome.report.error {
                eprintln!(
                    "  the reference failed: {error}
{}",
                    outcome.log_tail(12)
                );
            }
            goldens::save(&key, &outcome.report.engine, &outcome.report.clips)?;
            if !outcome.report.engine.name.is_empty() {
                engine = Some(outcome.report.engine.clone());
            }
            for clip in outcome.report.clips {
                if let Some(i) = clips.iter().position(|c| c.id == clip.id) {
                    found[i] = Some(clip);
                }
            }
        }
    }
    let reports = clips
        .iter()
        .zip(found)
        .map(|(clip, report)| {
            report.unwrap_or_else(|| ClipReport {
                id: clip.id.clone(),
                error: Some("the reference produced no result".into()),
                ..ClipReport::default()
            })
        })
        .collect();
    Ok((reports, engine))
}
