//! `cargo lab check`: the loop. Our engine (or any candidate) on a tier's clips, against the
//! reference's kept results and against the truth, with gates that say whether it's as good.

use crate::{
    corpus::{Clip, Corpus, Purpose, Tier},
    jobs::{self, Plan},
    models::{self, Kind, Model},
    paths,
    report::Report,
    run::{self, Candidate},
    score::{self, Mode},
    system,
};
use anyhow::Result;
use serde::Serialize;
use std::collections::BTreeMap;
use vellum_harness::{ClipReport, Output, Windowing};

pub struct Args {
    pub tier: Tier,
    pub models: Vec<String>,
    pub engine: Candidate,
    pub against: Candidate,
    /// The reference's settings (`TRANSCRIBE_NO_FLASH=1`, …).
    pub env: Vec<(String, String)>,
    /// Count a model the engine doesn't support yet as a failure.
    pub strict: bool,
}

/// How much worse than the reference the candidate may be before a gate fails.
const WER_MARGIN: f64 = 0.002;
const DER_MARGIN: f64 = 0.005;
const SPEAKER_MARGIN: f64 = 0.005;

/// One clip's results.
#[derive(Debug, Clone, Serialize)]
struct ClipScore {
    id: String,
    set: String,
    seconds: f64,
    /// Why there's no comparison: the candidate or the reference failed, or isn't there yet.
    problem: Option<String>,
    /// The reference against the truth, whether or not the candidate ran: the target.
    reference: Option<Truthful>,
    parity: Option<score::TranscriptParity>,
    speaker_parity: Option<score::DiarizationParity>,
    /// Against the truth, reference then candidate.
    wer: Option<(score::WerScore, score::WerScore)>,
    der: Option<(score::DerScore, score::DerScore)>,
    /// Words heard in a clip without speech, reference then candidate.
    invented: Option<(usize, usize)>,
    /// The reference's transcript labelled with each engine's speaker turns as the app labels
    /// it, against a script naming who said each line: reference then candidate.
    speakers: Option<(score::SpeakerReport, score::SpeakerReport)>,
}

/// A model's results over a set of clips, summed so long clips weigh what they should.
#[derive(Debug, Clone, Default, Serialize)]
struct Totals {
    clips: usize,
    seconds: f64,
    failed: usize,
    /// The reference against the truth on every clip it ran: the target, shown even while the
    /// candidate can't run. The fields after these count only clips both engines ran.
    target_words: usize,
    target_errors: usize,
    target_der_seconds: f64,
    target_der_error: f64,
    target_speaker_words: usize,
    target_speaker_correct: usize,
    /// Words in the truth, and the errors each engine made against it.
    truth_words: usize,
    ref_errors: usize,
    cand_errors: usize,
    /// Words the reference produced, and the candidate's differences from them.
    parity_words: usize,
    parity_errors: usize,
    identical: usize,
    compared: usize,
    /// Speaker error against the truth, in scored seconds.
    der_seconds: f64,
    ref_der_seconds: f64,
    cand_der_seconds: f64,
    invented: (usize, usize),
    /// Words aligned with a speaker-named script, and those labelled right by each engine.
    speaker_words: usize,
    ref_speaker_correct: usize,
    cand_speaker_correct: usize,
}

impl Totals {
    fn add(&mut self, s: &ClipScore) {
        self.clips += 1;
        self.seconds += s.seconds;
        if let Some(r) = &s.reference {
            if let Some(w) = &r.wer {
                self.target_words += w.reference_words;
                self.target_errors += w.errors();
            }
            if let Some(d) = &r.der {
                self.target_der_seconds += d.scored_seconds;
                self.target_der_error += d.error_seconds();
            }
            if let Some(sp) = &r.speakers {
                self.target_speaker_words += sp.words.aligned;
                self.target_speaker_correct += sp.words.correct;
            }
        }
        if let Some(problem) = &s.problem {
            // A model the engine doesn't support yet hasn't failed; it's pending.
            if !problem.contains("supported yet") {
                self.failed += 1;
            }
            return;
        }
        if let Some(p) = &s.speaker_parity {
            self.compared += 1;
            self.identical += p.identical as usize;
        }
        if let Some(p) = &s.parity {
            self.compared += 1;
            self.identical += p.identical_text as usize;
            self.parity_words += p.wer.reference_words;
            self.parity_errors += p.wer.errors();
        }
        if let Some((r, c)) = &s.wer {
            self.truth_words += r.reference_words;
            self.ref_errors += r.errors();
            self.cand_errors += c.errors();
        }
        if let Some((r, c)) = &s.der {
            self.der_seconds += r.scored_seconds;
            self.ref_der_seconds += r.error_seconds();
            self.cand_der_seconds += c.error_seconds();
        }
        if let Some((r, c)) = s.invented {
            self.invented.0 += r;
            self.invented.1 += c;
        }
        if let Some((r, c)) = &s.speakers {
            // Both label the same transcript, so they align the same words.
            self.speaker_words += r.words.aligned;
            self.ref_speaker_correct += r.words.correct;
            self.cand_speaker_correct += c.words.correct;
        }
    }

    fn ref_speakers(&self) -> Option<f64> {
        Self::rate(self.ref_speaker_correct, self.speaker_words)
    }

    fn target_wer(&self) -> Option<f64> {
        Self::rate(self.target_errors, self.target_words)
    }

    fn target_der(&self) -> Option<f64> {
        (self.target_der_seconds > 0.).then(|| self.target_der_error / self.target_der_seconds)
    }

    fn target_speakers(&self) -> Option<f64> {
        Self::rate(self.target_speaker_correct, self.target_speaker_words)
    }

    fn cand_speakers(&self) -> Option<f64> {
        Self::rate(self.cand_speaker_correct, self.speaker_words)
    }

    fn rate(errors: usize, words: usize) -> Option<f64> {
        (words > 0).then(|| errors as f64 / words as f64)
    }

    fn ref_wer(&self) -> Option<f64> {
        Self::rate(self.ref_errors, self.truth_words)
    }

    fn cand_wer(&self) -> Option<f64> {
        Self::rate(self.cand_errors, self.truth_words)
    }

    fn parity_wer(&self) -> Option<f64> {
        Self::rate(self.parity_errors, self.parity_words)
    }

    fn ref_der(&self) -> Option<f64> {
        (self.der_seconds > 0.).then(|| self.ref_der_seconds / self.der_seconds)
    }

    fn cand_der(&self) -> Option<f64> {
        (self.der_seconds > 0.).then(|| self.cand_der_seconds / self.der_seconds)
    }
}

/// The clips a model runs on in `tier`.
pub fn clips<'a>(corpus: &'a Corpus, model: &Model, tier: Tier) -> Vec<&'a Clip> {
    match model.kind {
        Kind::Speech => {
            let mut clips: Vec<&Clip> = corpus
                .select(tier, Purpose::Asr)
                .into_iter()
                .filter(|c| model.speaks(&c.language))
                .collect();
            // Edge cases without speech go to every speech model; with speech, to those that
            // speak it.
            for clip in corpus.select(tier, Purpose::Robustness) {
                if (clip.language.is_empty() || model.speaks(&clip.language))
                    && !clips.iter().any(|c| c.id == clip.id)
                {
                    clips.push(clip);
                }
            }
            clips.sort_by(|a, b| a.id.cmp(&b.id));
            clips
        }
        Kind::Speakers => corpus.select(tier, Purpose::Diarization),
    }
}

pub fn check(args: &Args) -> Result<bool> {
    let corpus = Corpus::load(&paths::corpus()?)?;
    let machine = system::machine();
    let plan = Plan {
        windowing: Windowing::default(),
        env: args.env.clone(),
        machine: machine.fingerprint.clone(),
    };
    let mut report = Report::new(
        &format!("check {:?}", args.tier).to_lowercase(),
        serde_json::to_value(&machine)?,
    );
    let work = paths::reports().join("runs").join(&report.id);
    let models: Vec<&Model> = if args.models.is_empty() {
        models::MODELS.iter().collect()
    } else {
        args.models
            .iter()
            .map(|m| models::get(m))
            .collect::<Result<_>>()?
    };
    let mut overview = score::table::Table::new([
        "model",
        "clips",
        "audio",
        "ref WER",
        "our WER",
        "Δ",
        "parity WER",
        "identical",
        "DER ref",
        "DER ours",
        "speakers ref",
        "speakers ours",
        "status",
    ]);
    let mut results = serde_json::Map::new();
    results.insert("engine".into(), args.engine.to_string().into());
    results.insert("against".into(), args.against.to_string().into());
    let history = crate::report::history(&report.command)?;
    for model in models {
        let clips = clips(&corpus, model, args.tier);
        if clips.is_empty() {
            eprintln!(
                "· {}: no clips in the {:?} tier; run `cargo lab fetch --tier {:?}`",
                model.id, args.tier, args.tier
            );
            continue;
        }
        let (golden, _) = jobs::goldens(&plan, model, &corpus, &clips, args.against, &work)?;
        eprintln!("· {} {}: {} clips", args.engine, model.id, clips.len());
        let program = run::runner(args.engine)?;
        let job = jobs::job(
            &plan,
            model,
            &corpus,
            &clips,
            args.engine,
            &work.join(format!("{}-{}", args.engine, model.id)),
            0,
            1,
        )?;
        let outcome = run::run(&program, &job, false)?;
        let pending = outcome
            .report
            .error
            .as_deref()
            .filter(|e| e.contains("supported yet"))
            .map(str::to_owned);
        if let (None, Some(error)) = (&pending, &outcome.report.error) {
            eprintln!(
                "  {} failed: {error}\n{}",
                args.engine,
                outcome.log_tail(12)
            );
        }
        let transcripts = if model.kind == Kind::Speakers {
            speaker_transcripts(&plan, &corpus, &clips, args.against, &work)?
        } else {
            BTreeMap::new()
        };
        let scores: Vec<ClipScore> = clips
            .iter()
            .zip(&golden)
            .map(|(clip, golden)| {
                let candidate = outcome.report.clips.iter().find(|c| c.id == clip.id);
                score_clip(
                    clip,
                    golden,
                    candidate,
                    outcome.report.error.as_deref(),
                    &transcripts,
                )
            })
            .collect();
        let mut by_set: BTreeMap<String, Totals> = BTreeMap::new();
        let mut all = Totals::default();
        for s in &scores {
            by_set.entry(s.set.clone()).or_default().add(s);
            all.add(s);
        }
        let (status, passed) = gates(&all, pending.is_some(), args.strict);
        report.passed &= passed;
        overview.row(row(model.id, &all, &status));
        let mut sets = score::table::Table::new([
            "set",
            "clips",
            "audio",
            "ref WER",
            "our WER",
            "Δ",
            "parity WER",
            "identical",
            "DER ref",
            "DER ours",
            "speakers ref",
            "speakers ours",
            "failed",
        ]);
        for (set, t) in &by_set {
            let failed = t.failed.to_string();
            sets.row(row(set, t, &failed));
        }
        let mut body = format!(
            "{} against {}, {} clips, {:.1} min of audio. {}\n\n{}",
            if outcome.report.engine.name.is_empty() {
                args.engine.to_string()
            } else {
                outcome.report.engine.name.clone()
            },
            args.against,
            all.clips,
            all.seconds / 60.,
            status,
            sets.markdown()
        );
        if let Some(line) = previous(&history, args, model.id, &all) {
            eprintln!("  {line}");
            body.push_str(&format!(
                "
{line}
"
            ));
        }
        body.push_str(&worst(&scores));
        report.section(format!("{} ({})", model.name, model.id), body);
        results.insert(
            model.id.into(),
            serde_json::json!({ "status": status, "passed": passed, "totals": all, "sets": by_set }),
        );
        std::fs::write(
            work.join(format!("{}-scores.json", model.id)),
            serde_json::to_vec_pretty(&scores)?,
        )?;
    }
    report
        .sections
        .insert(0, ("Summary".into(), overview.markdown()));
    report.results = serde_json::Value::Object(results);
    let path = report.save()?;
    println!("{}", overview.markdown());
    println!(
        "{} · {}",
        if report.passed { "passed" } else { "FAILED" },
        path.display()
    );
    Ok(report.passed)
}

/// `cargo lab goldens`: the reference's results for a tier, made now so later checks don't wait.
pub fn goldens(
    tier: Tier,
    models: &[String],
    against: Candidate,
    env: Vec<(String, String)>,
) -> Result<()> {
    let corpus = Corpus::load(&paths::corpus()?)?;
    let plan = Plan {
        windowing: Windowing::default(),
        env,
        machine: system::machine().fingerprint,
    };
    let work = paths::scratch()?.join("goldens");
    let models: Vec<&Model> = if models.is_empty() {
        models::MODELS.iter().collect()
    } else {
        models
            .iter()
            .map(|m| models::get(m))
            .collect::<Result<_>>()?
    };
    for model in models {
        let clips = clips(&corpus, model, tier);
        let (reports, _) = jobs::goldens(&plan, model, &corpus, &clips, against, &work)?;
        let failed = reports.iter().filter(|r| r.error.is_some()).count();
        let unstable = reports.iter().filter(|r| !r.deterministic).count();
        println!(
            "{}: {} clips, {failed} failed, {unstable} not bit-stable across repeats",
            model.id,
            reports.len()
        );
    }
    let _ = std::fs::remove_dir_all(&work);
    Ok(())
}

fn percent(v: Option<f64>) -> String {
    v.map_or("–".into(), |v| format!("{:.2}%", 100. * v))
}

fn row(name: &str, t: &Totals, status: &str) -> Vec<String> {
    let delta = match (t.ref_wer(), t.cand_wer()) {
        (Some(r), Some(c)) if t.compared > 0 => format!("{:+.2}", 100. * (c - r)),
        _ => "–".into(),
    };
    vec![
        name.into(),
        t.clips.to_string(),
        format!("{:.1} min", t.seconds / 60.),
        percent(t.target_wer()),
        if t.compared > 0 {
            percent(t.cand_wer())
        } else {
            "–".into()
        },
        delta,
        percent(t.parity_wer()),
        if t.compared > 0 {
            format!("{}/{}", t.identical, t.compared)
        } else {
            "–".into()
        },
        percent(t.target_der()),
        if t.compared > 0 {
            percent(t.cand_der())
        } else {
            "–".into()
        },
        percent(t.target_speakers()),
        if t.compared > 0 {
            percent(t.cand_speakers())
        } else {
            "–".into()
        },
        status.into(),
    ]
}

/// How this run compares with the last one of the same check on the same engines, so a change
/// that makes things worse shows even while the gates hold.
fn previous(
    history: &[serde_json::Value],
    args: &Args,
    model: &str,
    now: &Totals,
) -> Option<String> {
    let last = history.iter().rev().find(|h| {
        let r = &h["results"];
        r["engine"] == args.engine.to_string()
            && r["against"] == args.against.to_string()
            && r[model]["totals"].is_object()
    })?;
    let t = &last["results"][model]["totals"];
    let n = |k: &str| t[k].as_f64().unwrap_or(0.);
    let mut parts = Vec::new();
    if n("truth_words") > 0. && now.truth_words > 0 && n("compared") > 0. {
        let (was, is) = (
            n("cand_errors") / n("truth_words"),
            now.cand_wer().unwrap_or(0.),
        );
        parts.push(format!("our WER {:.2}% → {:.2}%", 100. * was, 100. * is));
    }
    if n("parity_words") > 0. && now.parity_words > 0 {
        let (was, is) = (
            n("parity_errors") / n("parity_words"),
            now.parity_wer().unwrap_or(0.),
        );
        parts.push(format!("parity WER {:.2}% → {:.2}%", 100. * was, 100. * is));
    }
    if n("der_seconds") > 0. && now.der_seconds > 0. {
        let (was, is) = (
            n("cand_der_seconds") / n("der_seconds"),
            now.cand_der().unwrap_or(0.),
        );
        parts.push(format!("our DER {:.2}% → {:.2}%", 100. * was, 100. * is));
    }
    (!parts.is_empty()).then(|| {
        format!(
            "Since {} ({}): {}.",
            last["id"].as_str().unwrap_or("the last run"),
            last["commit"].as_str().unwrap_or(""),
            parts.join(", ")
        )
    })
}

/// Whether a model's results hold: no failures where the reference worked, and no worse than
/// the reference against the truth, within a small margin.
fn gates(t: &Totals, pending: bool, strict: bool) -> (String, bool) {
    if pending {
        return ("not implemented yet".into(), !strict);
    }
    let mut failed = Vec::new();
    if t.failed > 0 {
        failed.push(format!("{} clips failed", t.failed));
    }
    if let (Some(r), Some(c)) = (t.ref_wer(), t.cand_wer())
        && c > r + WER_MARGIN
    {
        failed.push(format!("WER {:.2}% > {:.2}%", 100. * c, 100. * r));
    }
    if let (Some(r), Some(c)) = (t.ref_der(), t.cand_der())
        && c > r + DER_MARGIN
    {
        failed.push(format!("DER {:.2}% > {:.2}%", 100. * c, 100. * r));
    }
    if let (Some(r), Some(c)) = (t.ref_speakers(), t.cand_speakers())
        && c < r - SPEAKER_MARGIN
    {
        failed.push(format!("speakers {:.2}% < {:.2}%", 100. * c, 100. * r));
    }
    if t.invented.1 > t.invented.0 {
        failed.push(format!(
            "{} words from no speech (reference {})",
            t.invented.1, t.invented.0
        ));
    }
    if failed.is_empty() {
        ("ok".into(), true)
    } else {
        (failed.join("; "), false)
    }
}

/// The reference's Parakeet transcripts of diarization clips with speech in English, to label
/// with each engine's speaker turns as the app would.
fn speaker_transcripts(
    plan: &Plan,
    corpus: &Corpus,
    clips: &[&Clip],
    against: Candidate,
    work: &std::path::Path,
) -> Result<BTreeMap<String, vellum_harness::Transcript>> {
    let parakeet = models::get("parakeet")?;
    let wanted: Vec<&Clip> = clips
        .iter()
        .copied()
        .filter(|c| c.truth.lines.is_some() && parakeet.speaks(&c.language))
        .collect();
    if wanted.is_empty() {
        return Ok(BTreeMap::new());
    }
    let (reports, _) = jobs::goldens(plan, parakeet, corpus, &wanted, against, work)?;
    Ok(wanted
        .iter()
        .zip(reports)
        .filter_map(|(clip, r)| match r.output {
            Some(Output::Transcript(t)) => Some((clip.id.clone(), t)),
            _ => None,
        })
        .collect())
}

/// One engine's output against the clip's truth: word errors, speaker errors, and speaker labels
/// (the reference's transcript labelled with these turns, as the app would).
#[derive(Debug, Clone, Default, Serialize)]
struct Truthful {
    wer: Option<score::WerScore>,
    der: Option<score::DerScore>,
    speakers: Option<score::SpeakerReport>,
    /// Words heard in a clip without speech.
    invented: Option<usize>,
}

fn against_truth(
    clip: &Clip,
    output: &Output,
    transcripts: &BTreeMap<String, vellum_harness::Transcript>,
) -> Truthful {
    let lang = clip.language.as_str();
    let mut t = Truthful::default();
    match output {
        Output::Transcript(x) => {
            if let Some(truth) = &clip.truth.text {
                let truth = score::text::words(truth, Mode::Asr, lang);
                t.wer = Some(score::wer::score(
                    &truth,
                    &score::text::words(&x.text, Mode::Asr, lang),
                ));
            }
            if lang.is_empty() {
                t.invented = Some(x.words.len());
            }
        }
        Output::Diarization(x) => {
            if let Some(turns) = &clip.truth.turns {
                t.der = Some(score::der::against_truth(turns, x));
            }
            if let (Some(lines), Some(transcript)) = (&clip.truth.lines, transcripts.get(&clip.id))
            {
                let labels = score::speakers::label(transcript, x);
                t.speakers = Some(score::speakers::score(
                    lines,
                    transcript,
                    &labels,
                    Mode::Asr,
                    lang,
                ));
            }
        }
    }
    t
}

fn score_clip(
    clip: &Clip,
    golden: &ClipReport,
    candidate: Option<&ClipReport>,
    job_error: Option<&str>,
    transcripts: &BTreeMap<String, vellum_harness::Transcript>,
) -> ClipScore {
    let mut s = ClipScore {
        id: clip.id.clone(),
        set: clip.set.clone(),
        seconds: clip.seconds,
        problem: None,
        reference: None,
        parity: None,
        speaker_parity: None,
        wer: None,
        der: None,
        invented: None,
        speakers: None,
    };
    let Some(r) = golden.output.as_ref().filter(|_| golden.error.is_none()) else {
        s.problem = Some(format!(
            "reference: {}",
            golden.error.as_deref().unwrap_or("no result")
        ));
        return s;
    };
    let reference = against_truth(clip, r, transcripts);
    s.reference = Some(reference.clone());
    let c = match candidate {
        Some(c) if c.error.is_none() => c.output.as_ref(),
        Some(c) => {
            s.problem = c.error.clone();
            return s;
        }
        None => {
            s.problem = Some(job_error.unwrap_or("no result").to_owned());
            return s;
        }
    };
    let Some(c) = c else {
        s.problem = Some("no output".into());
        return s;
    };
    match (r, c) {
        (Output::Transcript(r), Output::Transcript(c)) => {
            s.parity = Some(score::parity::transcripts(r, c));
        }
        (Output::Diarization(r), Output::Diarization(c)) => {
            s.speaker_parity = Some(score::parity::diarizations(r, c));
        }
        _ => {
            s.problem = Some("the engines returned different kinds of output".into());
            return s;
        }
    }
    let candidate = against_truth(clip, c, transcripts);
    s.wer = reference.wer.zip(candidate.wer);
    s.der = reference.der.zip(candidate.der);
    s.speakers = reference.speakers.zip(candidate.speakers);
    s.invented = reference.invented.zip(candidate.invented);
    s
}

/// The clips where the engines differ most, to replay with `cargo lab probe --clip`.
fn worst(scores: &[ClipScore]) -> String {
    let mut differing: Vec<&ClipScore> = scores
        .iter()
        .filter(|s| s.parity.as_ref().is_some_and(|p| !p.identical_text))
        .collect();
    differing.sort_by(|a, b| {
        let w = |s: &ClipScore| s.parity.as_ref().map_or(0., |p| p.wer.wer);
        w(b).total_cmp(&w(a))
    });
    let failed: Vec<&ClipScore> = scores.iter().filter(|s| s.problem.is_some()).collect();
    let mut out = String::new();
    if !differing.is_empty() {
        let mut t = score::table::Table::new(["clip", "parity WER", "first difference"]);
        for s in differing.iter().take(8) {
            let p = s.parity.as_ref().unwrap();
            t.row(vec![
                s.id.clone(),
                percent(Some(p.wer.wer)),
                p.first_difference
                    .as_ref()
                    .map_or("–".into(), |d| d.to_string()),
            ]);
        }
        out.push_str(&format!(
            "\nWhere the engines part most ({} clips differ):\n\n{}",
            differing.len(),
            t.markdown()
        ));
    }
    if !failed.is_empty() {
        out.push_str(&format!("\n{} clips failed:\n\n", failed.len()));
        for s in failed.iter().take(8) {
            out.push_str(&format!(
                "- {}: {}\n",
                s.id,
                s.problem.as_deref().unwrap_or("")
            ));
        }
    }
    out
}
