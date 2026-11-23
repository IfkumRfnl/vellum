use super::{Cue, models};
use crate::{
    job::Progress,
    pipeline::{Stage, Update},
};
use anyhow::{Context, Result, ensure};
use sherpa_onnx::{
    OfflineSpeakerDiarization, OfflineSpeakerDiarizationConfig, OfflineSpeakerDiarizationSegment,
};
use std::{collections::HashMap, path::Path, sync::OnceLock};
use transcribe_cpp::{Backend, Error, Model, ModelOptions, RunOptions, Session, TimestampKind};

const RATE: usize = 16000;

fn open_session(path: &Path, backend: Backend) -> transcribe_cpp::Result<Session> {
    Model::load_with(
        path,
        &ModelOptions {
            backend,
            ..Default::default()
        },
    )?
    .session()
}

fn device_error(error: &Error) -> bool {
    matches!(error, Error::Backend(_) | Error::OutOfMemory(_))
}

fn cpu_session(path: &Path, error: Error, progress: &Progress) -> Result<Session> {
    progress.note("Switching to the CPU…")?;
    progress.update(Update::Device("CPU".into()))?;
    open_session(path, Backend::Cpu).with_context(|| format!("CPU fallback failed after {error}"))
}

// Look for the quietest 40 ms near the end of a full window.
fn boundary(pcm: &[f32]) -> usize {
    let frame = RATE / 25;
    if pcm.len() < 4 * RATE {
        return pcm.len();
    }
    let start = pcm.len() - 3 * RATE;
    let (quietest, _) = pcm[start..]
        .chunks_exact(frame)
        .enumerate()
        .map(|(i, values)| (i, values.iter().map(|v| v * v).sum::<f32>()))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();
    start + quietest * frame + frame / 2
}

pub fn transcribe(
    wav: &Path,
    cache: &Path,
    model: &models::SpeechModel,
    progress: &Progress,
) -> Result<(Vec<Cue>, String, String)> {
    progress.stage(Stage::Model)?;
    let path = model.artifact.get(cache, progress)?;
    progress.update(Update::Estimated)?;
    // Registration mutates the native process-wide device registry. Finish it
    // before any job opens a model, including when a cancelled job is unwinding.
    static BACKENDS: OnceLock<Result<(), String>> = OnceLock::new();
    BACKENDS
        .get_or_init(|| transcribe_cpp::init_backends_default().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| anyhow::anyhow!("Could not initialize speech devices: {e}"))?;
    let mut session = match open_session(&path, Backend::Auto) {
        Err(error) if device_error(&error) => cpu_session(&path, error, progress)?,
        result => result?,
    };
    let device = session.model().device()?;
    let mut on_cpu = device.kind == "cpu";
    progress.update(Update::Device(if on_cpu {
        "CPU".to_owned()
    } else {
        let mut kind = device.kind.clone();
        if let Some(first) = kind.get_mut(..1) {
            first.make_ascii_uppercase();
        }
        format!("{kind} · {}", device.description)
    }))?;
    session.set_cancel_token(progress.token());
    let max_ms = session.limits()?.effective_max_audio_ms;
    let window = if max_ms > 0 {
        max_ms.min(28_000)
    } else {
        28_000
    } as usize
        * RATE
        / 1000;
    let mut options = RunOptions {
        language: (model.id == "parakeet_unified_en").then(|| "en".into()),
        ..Default::default()
    };
    let mut reader = hound::WavReader::open(wav)?;
    let total = reader.duration() as usize;
    let mut samples = reader.samples::<i16>();
    let mut pcm = Vec::with_capacity(window);
    let mut offset = 0;
    let mut cues = Vec::new();
    progress.stage(Stage::Transcribe)?;
    while offset < total {
        for sample in samples.by_ref().take(window - pcm.len()) {
            pcm.push(sample? as f32 / 32768.);
        }
        ensure!(!pcm.is_empty(), "The audio ended unexpectedly");
        let count = if offset + pcm.len() < total {
            boundary(&pcm)
        } else {
            pcm.len()
        };
        progress.fraction(offset as f32 / total as f32)?;
        if pcm[..count].iter().any(|s| *s != 0.) {
            let result = match session.run(&pcm[..count], &options) {
                Err(error) if !on_cpu && device_error(&error) => {
                    drop(session);
                    session = cpu_session(&path, error, progress)?;
                    session.set_cancel_token(progress.token());
                    on_cpu = true;
                    session.run(&pcm[..count], &options)?
                }
                result => result?,
            };
            if options.language.is_none() && !result.text.trim().is_empty() {
                options.language = result.language;
            }
            let start = offset as f64 / RATE as f64;
            let prior = cues.len();
            let mut add = |from: i64, to: i64, text: String| {
                if !text.trim().is_empty() {
                    cues.push(Cue::new(
                        start + from as f64 / 1000.,
                        start + to as f64 / 1000.,
                        text,
                    ));
                }
            };
            if matches!(
                result.timestamp_kind,
                TimestampKind::Word | TimestampKind::Token
            ) && !result.words.is_empty()
            {
                for w in result.words {
                    add(w.t0_ms, w.t1_ms, w.text);
                }
            } else {
                for s in result.segments {
                    add(s.t0_ms, s.t1_ms, s.text);
                }
            }
            if cues.len() == prior && !result.text.trim().is_empty() {
                cues.push(Cue::new(
                    start,
                    start + count as f64 / RATE as f64,
                    result.text,
                ));
            }
            let heard = super::join_words(cues[prior..].iter().map(|c| c.text.as_str()));
            if !heard.is_empty() {
                progress.update(Update::Heard {
                    words: crate::document::word_count(&heard),
                    text: heard,
                    until: (offset + count) as f64 / RATE as f64,
                })?;
            }
        }
        offset += count;
        progress.fraction(offset as f32 / total as f32)?;
        pcm.drain(..count);
    }
    Ok((
        cues,
        model.name.into(),
        options.language.unwrap_or_default(),
    ))
}

pub fn label_speakers(
    wav: &Path,
    cache: &Path,
    cues: &mut [Cue],
    progress: &Progress,
) -> Result<()> {
    progress.stage(Stage::Speakers)?;
    #[cfg(windows)]
    let _library = speaker_library()?;
    let mut config = OfflineSpeakerDiarizationConfig::default();
    // Let clustering infer the number of voices. TitaNet's embeddings can
    // drift between short turns from the same person (especially around music).
    config.clustering.threshold = 0.7;
    config.segmentation.pyannote.model = Some(
        models::SEGMENTATION
            .get(cache, progress)?
            .to_string_lossy()
            .into_owned(),
    );
    config.segmentation.num_threads = 2;
    config.embedding.model = Some(
        models::EMBEDDING
            .get(cache, progress)?
            .to_string_lossy()
            .into_owned(),
    );
    config.embedding.num_threads = 2;
    let diarizer =
        OfflineSpeakerDiarization::create(&config).context("Could not load the speaker models")?;
    let mut audio = hound::WavReader::open(wav)?;
    ensure!(
        audio.spec().sample_rate as i32 == diarizer.sample_rate(),
        "Unsupported speaker analysis sample rate"
    );
    ensure!(
        audio.duration() <= i32::MAX as u32,
        "Video too long for speaker analysis"
    );
    progress.update(Update::Estimated)?;
    let samples = audio
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / 32768.))
        .collect::<Result<Vec<_>, _>>()?;
    let result = diarizer
        .process(&samples)
        .context("Speaker analysis failed")?;
    progress.check()?;
    assign_speakers(cues, &result.sort_by_start_time());
    Ok(())
}

#[cfg(windows)]
fn speaker_library() -> Result<libloading::os::windows::Library> {
    use libloading::os::windows::{
        LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, Library,
    };
    let exe = std::env::current_exe()?;
    let dir = exe.parent().context("Missing application directory")?;
    // Cargo places its test harness in deps and the native libraries in the profile directory.
    #[cfg(test)]
    let dir = dir.parent().context("Missing Cargo profile directory")?;
    ensure!(
        dir.join("onnxruntime.dll").is_file(),
        "Missing Vellum’s ONNX Runtime library"
    );
    // SAFETY: Load the package's native library by absolute path. Resolve its matching
    // ONNX Runtime beside it before Windows can substitute its older system copy.
    unsafe {
        Library::load_with_flags(
            dir.join("sherpa-onnx-c-api.dll"),
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
        )
    }
    .context("Could not load Vellum’s speaker library")
}

fn assign_speakers(cues: &mut [Cue], turns: &[OfflineSpeakerDiarizationSegment]) {
    let mut names = HashMap::new();
    let mut first = 0;
    for cue in cues {
        while first < turns.len() && f64::from(turns[first].end) <= cue.start {
            first += 1;
        }
        let mut best = 0.;
        let mut speaker = None;
        for turn in turns[first..]
            .iter()
            .take_while(|t| f64::from(t.start) < cue.end)
        {
            let overlap = cue.end.min(turn.end.into()) - cue.start.max(turn.start.into());
            if overlap > best {
                best = overlap;
                speaker = Some(turn.speaker);
            }
        }
        if let Some(speaker) = speaker {
            let next = names.len() + 1;
            let number = names.entry(speaker).or_insert(next);
            cue.speaker = Some(format!("Speaker {number}"));
        }
    }
}
