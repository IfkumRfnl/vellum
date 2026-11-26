use crate::{
    ClipReport, Diarization, EngineInfo, Job, JobReport, Output, RunTiming, SCHEMA, Task, Token,
    Transcript, Turn, WindowInfo, Word, audio, sys, windows,
};
use anyhow::{Context, Result, ensure};
use std::{
    fs,
    panic::{AssertUnwindSafe, catch_unwind},
    path::Path,
    process::ExitCode,
    time::Instant,
};

/// What one window gave. Times are milliseconds from the start of the window.
#[derive(Debug, Clone, Default)]
pub struct WindowResult {
    pub text: String,
    /// The language the model detected, if it detects one.
    pub language: Option<String>,
    pub words: Vec<(i64, i64, String)>,
    pub tokens: Vec<Token>,
    pub mel_ms: f64,
    pub encode_ms: f64,
    pub decode_ms: f64,
    /// The window's own wall time, when it ran alone (0 inside a batch).
    pub wall_ms: f64,
}

/// One engine's model, loaded, as the runner drives it.
pub trait Engine {
    fn info(&self) -> EngineInfo;

    /// The longest window the model takes, in milliseconds; 0 for no limit.
    fn max_audio_ms(&self) -> u32 {
        0
    }

    fn transcribe(&mut self, _pcm: &[f32], _language: Option<&str>) -> Result<WindowResult> {
        anyhow::bail!("this model doesn't transcribe")
    }

    /// Every window of a recording at once. An engine can batch or pipeline them (encode the
    /// next while decoding this one). Without a `language`, the app's rule holds: each window
    /// is detected until one has words, and that window's language is used for the rest. By
    /// default the windows run one by one.
    fn transcribe_all(
        &mut self,
        windows: &[&[f32]],
        language: Option<&str>,
    ) -> Result<Vec<WindowResult>> {
        let mut language = language.map(str::to_owned);
        let mut out = Vec::with_capacity(windows.len());
        for (i, pcm) in windows.iter().enumerate() {
            let start = Instant::now();
            let mut result = self
                .transcribe(pcm, language.as_deref())
                .with_context(|| format!("window {i}"))?;
            result.wall_ms = ms(start);
            if language.is_none() && !result.text.trim().is_empty() {
                language = result.language.clone();
            }
            out.push(result);
        }
        Ok(out)
    }

    fn diarize(&mut self, _pcm: &[f32]) -> Result<Vec<Turn>> {
        anyhow::bail!("this model doesn't label speakers")
    }

    /// Anything else measured over the job, reported as `JobReport::profile`.
    fn profile(&mut self) -> Option<serde_json::Value> {
        None
    }
}

/// Runs the job in `job_path` and writes its report beside the outputs. `load` opens the
/// job's model (and, for a dump job, arranges the dumps). `started` is when the process began.
pub fn run_job(
    job_path: &Path,
    started: Instant,
    load: impl FnOnce(&Job) -> Result<Box<dyn Engine>>,
) -> ExitCode {
    let job: Job = match fs::read(job_path)
        .context("reading the job")
        .and_then(|bytes| serde_json::from_slice(&bytes).context("parsing the job"))
    {
        Ok(job) => job,
        Err(error) => {
            eprintln!("{error:#}");
            return ExitCode::FAILURE;
        }
    };
    let mut report = JobReport {
        schema: SCHEMA,
        ..JobReport::default()
    };
    let result = drive(&job, started, load, &mut report);
    if let Err(error) = &result {
        report.error = Some(format!("{error:#}"));
    }
    report.wall_ms = ms(started);
    report.peak_rss = sys::peak_rss();
    if let Err(error) = save(&job, &report) {
        eprintln!("{error:#}");
        return ExitCode::FAILURE;
    }
    if result.is_ok() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(2)
    }
}

fn save(job: &Job, report: &JobReport) -> Result<()> {
    fs::create_dir_all(&job.out_dir)?;
    let path = job.out_dir.join("report.json");
    let partial = path.with_extension("json.partial");
    fs::write(&partial, serde_json::to_vec_pretty(report)?)?;
    fs::rename(&partial, &path)?;
    Ok(())
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1000.
}

fn drive(
    job: &Job,
    started: Instant,
    load: impl FnOnce(&Job) -> Result<Box<dyn Engine>>,
    report: &mut JobReport,
) -> Result<()> {
    ensure!(
        job.schema == SCHEMA,
        "job schema {} isn't {SCHEMA}",
        job.schema
    );
    ensure!(job.repeats >= 1, "a job needs at least one measured run");
    if job.dump.is_some() {
        ensure!(
            job.clips.len() == 1 && job.repeats == 1 && job.warmup == 0,
            "a dump job runs one clip once"
        );
    }
    let loading = Instant::now();
    let mut engine = catch_unwind(AssertUnwindSafe(|| load(job)))
        .map_err(panic_message)
        .context("loading the model")??;
    report.load_ms = ms(loading);
    report.ready_ms = ms(started);
    report.engine = engine.info();
    for clip in &job.clips {
        let outcome = catch_unwind(AssertUnwindSafe(|| run_clip(job, engine.as_mut(), clip)))
            .map_err(panic_message)
            .and_then(|r| r);
        report
            .clips
            .push(outcome.unwrap_or_else(|error| ClipReport {
                id: clip.id.clone(),
                error: Some(format!("{error:#}")),
                ..ClipReport::default()
            }));
        report.peak_rss = sys::peak_rss();
        save(job, report)?;
    }
    report.profile = engine.profile();
    Ok(())
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> anyhow::Error {
    let text = payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".into());
    anyhow::anyhow!("panicked: {text}")
}

fn run_clip(job: &Job, engine: &mut dyn Engine, clip: &crate::ClipJob) -> Result<ClipReport> {
    let pcm = audio::read(&clip.audio)?;
    let mut report = ClipReport {
        id: clip.id.clone(),
        seconds: pcm.len() as f64 / audio::RATE as f64,
        pcm_sha256: audio::pcm_sha256(&pcm),
        deterministic: true,
        ..ClipReport::default()
    };
    for _ in 0..job.warmup {
        let start = Instant::now();
        run_once(&job.task, engine, &pcm)?;
        report.warmup_ms.push(ms(start));
    }
    for _ in 0..job.repeats {
        let (output, timing) = run_once(&job.task, engine, &pcm)?;
        report.runs.push(timing);
        match &report.output {
            None => report.output = Some(output),
            Some(first) => report.deterministic &= *first == output,
        }
    }
    Ok(report)
}

fn run_once(task: &Task, engine: &mut dyn Engine, pcm: &[f32]) -> Result<(Output, RunTiming)> {
    let mut timing = RunTiming::default();
    let start = Instant::now();
    let output = match task {
        Task::Transcribe {
            language,
            windowing,
            ..
        } => {
            let window = windows::window_samples(windowing.max_ms, engine.max_audio_ms());
            let ranges = windows::windows(pcm, window);
            let mut language = language.clone();
            let mut out = Transcript::default();
            let mut infos: Vec<WindowInfo> = ranges
                .iter()
                .map(|r| WindowInfo {
                    start_sample: r.start as u64,
                    end_sample: r.end as u64,
                    silent: windows::silent(&pcm[r.clone()]),
                    language: language.clone(),
                    detected: None,
                })
                .collect();
            let speaking: Vec<usize> = (0..ranges.len()).filter(|i| !infos[*i].silent).collect();
            let mut results: Vec<Option<WindowResult>> = vec![None; ranges.len()];
            // Every window at once; the engine applies the language rule (see
            // `Engine::transcribe_all`), and the reports say which language each window ran in.
            let batch: Vec<&[f32]> = speaking.iter().map(|i| &pcm[ranges[*i].clone()]).collect();
            let got = engine.transcribe_all(&batch, language.as_deref())?;
            anyhow::ensure!(
                got.len() == batch.len(),
                "{} results for {} windows",
                got.len(),
                batch.len()
            );
            for (i, result) in speaking.iter().copied().zip(got) {
                infos[i].language = language.clone();
                if language.is_none() && !result.text.trim().is_empty() {
                    language = result.language.clone();
                }
                results[i] = Some(result);
            }
            for (index, (range, (mut info, result))) in ranges
                .iter()
                .zip(infos.into_iter().zip(results))
                .enumerate()
            {
                let index = index as u32;
                let Some(result) = result else {
                    out.windows.push(info);
                    timing.windows_ms.push(0.);
                    continue;
                };
                timing.windows_ms.push(result.wall_ms);
                timing.mel_ms += result.mel_ms;
                timing.encode_ms += result.encode_ms;
                timing.decode_ms += result.decode_ms;
                info.detected = result.language.clone();
                let samples = range.end - range.start;
                let offset = (range.start as f64 * 1000. / audio::RATE as f64).round() as i64;
                let before = out.words.len();
                for (t0, t1, text) in result.words {
                    if !text.trim().is_empty() {
                        out.words.push(Word {
                            t0_ms: offset + t0,
                            t1_ms: offset + t1,
                            text,
                            window: index,
                        });
                    }
                }
                if out.words.len() == before && !result.text.trim().is_empty() {
                    let length = (samples as f64 * 1000. / audio::RATE as f64) as i64;
                    out.words.push(Word {
                        t0_ms: offset,
                        t1_ms: offset + length,
                        text: result.text.trim().to_owned(),
                        window: index,
                    });
                }
                out.tokens.extend(result.tokens.into_iter().map(|t| Token {
                    t0_ms: offset + t.t0_ms,
                    t1_ms: offset + t.t1_ms,
                    window: index,
                    ..t
                }));
                if !out.text.is_empty() && !result.text.trim().is_empty() {
                    out.text.push(' ');
                }
                out.text.push_str(result.text.trim());
                out.windows.push(info);
            }
            out.language = language;
            Output::Transcript(out)
        }
        Task::Diarize { .. } => {
            let turns = engine.diarize(pcm)?;
            timing.windows_ms.push(ms(start));
            Output::Diarization(Diarization { turns })
        }
    };
    timing.wall_ms = ms(start);
    Ok((output, timing))
}
