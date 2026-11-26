use super::{Cue, ends_sentence, models};
use crate::{
    job::Progress,
    pipeline::{Stage, Update},
};
use anyhow::{Context, Result, ensure};
use std::{collections::HashMap, ops::Range, path::Path, sync::OnceLock};
use transcribe_cpp::{
    Backend, DiarizeExtension, DiarizeOptions, Error, Model, ModelOptions, RunOptions, Session,
    SortformerDiarizeOptions, SortformerPreset, SpeakerSegment, TimestampKind,
};
use vellum_engine::{
    Device, Diarizer, Error as EngineError, NoTrace, Options as EngineOptions,
    RunOptions as EngineRunOptions, Transcriber,
};

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

/// Registers the GPUs and CPU the speech models run on, once.
fn speech_devices() -> Result<()> {
    // Registration mutates the native process-wide device registry. Finish it
    // before any job opens a model, including when a cancelled job is unwinding.
    static BACKENDS: OnceLock<Result<(), String>> = OnceLock::new();
    BACKENDS
        .get_or_init(|| transcribe_cpp::init_backends_default().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| anyhow::anyhow!("Could not initialize speech devices: {e}"))?;
    Ok(())
}

/// Whether Vellum's own engine (`crates/engine`) runs what it can: Parakeet, which it
/// transcribes as transcribe.cpp does about three times as fast, and the speaker labels
/// (Sortformer, the same turns about three times as fast). `VELLUM_ENGINE=cpp` switches both
/// back to transcribe.cpp, to compare the two.
fn own_engine() -> bool {
    !std::env::var("VELLUM_ENGINE").is_ok_and(|v| v.eq_ignore_ascii_case("cpp"))
}

/// Whether a speech model runs on Vellum's engine. The Whisper models stay on transcribe.cpp
/// until their port lands.
fn own_speech_engine(model: &models::SpeechModel) -> bool {
    model.id == "parakeet_unified_en" && own_engine()
}

/// How Vellum's engine loads a model on `device`: with the GPU's compiled kernels kept in
/// Vellum's own cache folder (`%LOCALAPPDATA%\Vellum\cache\gpu` on Windows), so a model loads
/// without the driver's compiles after the first time. They are specific to the machine's GPU
/// and driver, so they live in the local cache rather than with the settings, which roam.
fn engine_options(device: Device) -> EngineOptions {
    EngineOptions {
        device,
        cache_dir: directories::ProjectDirs::from("", "", "Vellum")
            .map(|dirs| dirs.cache_dir().join("gpu")),
    }
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

/// The windows a recording is read in, in order, covering it exactly: `window` samples at
/// most, each but the last ending at a [`boundary`].
fn windows(pcm: &[f32], window: usize) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut offset = 0;
    while offset < pcm.len() {
        let end = (offset + window).min(pcm.len());
        let count = if end < pcm.len() {
            boundary(&pcm[offset..end])
        } else {
            end - offset
        };
        out.push(offset..offset + count);
        offset += count;
    }
    out
}

/// The window length in samples: 28 s, or the model's limit when that's shorter (0: none).
fn window_samples(max_ms: i64) -> usize {
    let ms = if max_ms > 0 {
        max_ms.min(28_000)
    } else {
        28_000
    };
    ms as usize * RATE / 1000
}

/// One window's recognized pieces (`(t0_ms, t1_ms, text)` from the window's start) as cues, or
/// its whole text as one cue when it has no timed pieces, and what was heard to the progress
/// screen. The window is `count` samples from `offset` into the recording.
fn add_window(
    cues: &mut Vec<Cue>,
    offset: usize,
    count: usize,
    pieces: impl IntoIterator<Item = (i64, i64, String)>,
    text: String,
    progress: &Progress,
) -> Result<()> {
    let start = offset as f64 / RATE as f64;
    let prior = cues.len();
    for (from, to, piece) in pieces {
        if !piece.trim().is_empty() {
            cues.push(Cue::new(
                start + from as f64 / 1000.,
                start + to as f64 / 1000.,
                piece,
            ));
        }
    }
    if cues.len() == prior && !text.trim().is_empty() {
        cues.push(Cue::new(start, start + count as f64 / RATE as f64, text));
    }
    let heard = super::join_words(cues[prior..].iter().map(|c| c.text.as_str()));
    if !heard.is_empty() {
        progress.update(Update::Heard {
            words: crate::document::word_count(&heard),
            text: heard,
            until: (offset + count) as f64 / RATE as f64,
        })?;
    }
    Ok(())
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
    if own_speech_engine(model) {
        return transcribe_with_engine(wav, &path, model, progress);
    }
    speech_devices()?;
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
    let window = window_samples(session.limits()?.effective_max_audio_ms);
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
            let pieces: Vec<(i64, i64, String)> = if matches!(
                result.timestamp_kind,
                TimestampKind::Word | TimestampKind::Token
            ) && !result.words.is_empty()
            {
                let words = result.words.into_iter();
                words.map(|w| (w.t0_ms, w.t1_ms, w.text)).collect()
            } else {
                let segments = result.segments.into_iter();
                segments.map(|s| (s.t0_ms, s.t1_ms, s.text)).collect()
            };
            add_window(&mut cues, offset, count, pieces, result.text, progress)?;
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

/// Loads Parakeet on Vellum's engine: on the GPU when there's a good one, else (or when the GPU
/// can't take the model) on the CPU. Names the device on the progress screen as the
/// transcribe.cpp path does ("Vulkan · NVIDIA GeForce RTX 4060 Laptop GPU", "CPU").
fn load_engine(path: &Path, progress: &Progress) -> Result<Transcriber> {
    let load = |device| Transcriber::load(path, &engine_options(device));
    let transcriber = match load(Device::Auto) {
        // A file the engine can't read fails the same way on the CPU.
        Err(error @ (EngineError::Io(_) | EngineError::Gguf(_) | EngineError::Unsupported(_))) => {
            return Err(error).context("Could not load the speech model");
        }
        Err(error) => {
            progress.note("Switching to the CPU…")?;
            load(Device::Cpu).with_context(|| format!("CPU fallback failed after {error}"))?
        }
        Ok(transcriber) => transcriber,
    };
    progress.update(Update::Device(engine_device(transcriber.gpu())))?;
    Ok(transcriber)
}

/// The device a model runs on as the progress screen names it: the graphics API and the GPU,
/// or "CPU".
fn engine_device(gpu: Option<&vellum_engine::gpu::Gpu>) -> String {
    use vellum_engine::gpu::wgpu::Backend;
    let Some(gpu) = gpu else {
        return "CPU".into();
    };
    let api = match gpu.info.backend {
        Backend::Vulkan => "Vulkan".to_owned(),
        Backend::Metal => "Metal".to_owned(),
        Backend::Dx12 => "DirectX 12".to_owned(),
        Backend::Gl => "OpenGL".to_owned(),
        other => format!("{other:?}"),
    };
    format!("{api} · {}", gpu.info.name)
}

/// Parakeet on Vellum's engine: the whole recording, cut into the same windows as the
/// transcribe.cpp path (exactly-silent ones skipped), handed to the engine at once so it works
/// on several windows together. Each window's words reach the progress screen as soon as it
/// and those before it are done. A GPU that fails mid-way leaves the windows still to do to
/// the CPU.
fn transcribe_with_engine(
    wav: &Path,
    path: &Path,
    model: &models::SpeechModel,
    progress: &Progress,
) -> Result<(Vec<Cue>, String, String)> {
    let mut transcriber = load_engine(path, progress)?;
    let window = window_samples(transcriber.max_audio_ms().into());
    let mut reader = hound::WavReader::open(wav)?;
    ensure!(
        reader.spec().sample_rate as usize == RATE,
        "Unsupported speech sample rate"
    );
    let pcm = reader
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / 32768.))
        .collect::<Result<Vec<_>, _>>()?;
    let total = pcm.len().max(1);
    let speaking: Vec<Range<usize>> = windows(&pcm, window)
        .into_iter()
        .filter(|r| pcm[r.clone()].iter().any(|s| *s != 0.))
        .collect();
    // Parakeet Unified is English only: there's nothing to detect.
    let language = "en";
    let options = EngineRunOptions {
        language: Some(language.into()),
    };
    let token = progress.token();
    let stop = || token.is_cancelled();
    let mut cues = Vec::new();
    // Windows of `speaking` whose words are in `cues`.
    let mut done = 0;
    progress.stage(Stage::Transcribe)?;
    while done < speaking.len() {
        let rest: Vec<&[f32]> = speaking[done..].iter().map(|r| &pcm[r.clone()]).collect();
        // An error of the app's own (a cancelled job) stops the engine, and is what's reported.
        let mut failed = None;
        let ran = transcriber.run_each(&rest, &options, &mut NoTrace, &stop, &mut |_, result| {
            let range = &speaking[done];
            let words = result.words.into_iter().map(|w| (w.t0_ms, w.t1_ms, w.text));
            let added = add_window(
                &mut cues,
                range.start,
                range.len(),
                words,
                result.text,
                progress,
            )
            .and_then(|()| progress.fraction(range.end as f32 / total as f32));
            done += 1;
            added.map_err(|error| {
                failed = Some(error);
                EngineError::Cancelled
            })
        });
        match (ran, failed) {
            (Ok(()), _) => {}
            (Err(_), Some(error)) => return Err(error),
            (Err(EngineError::Cancelled), None) => {
                progress.check()?;
                anyhow::bail!("Cancelled");
            }
            (Err(error), None) if transcriber.gpu().is_some() => {
                progress.note("Switching to the CPU…")?;
                progress.update(Update::Device("CPU".into()))?;
                drop(transcriber);
                transcriber = Transcriber::load(path, &engine_options(Device::Cpu))
                    .with_context(|| format!("CPU fallback failed after {error}"))?;
            }
            (Err(error), None) => return Err(error).context("Speech recognition failed"),
        }
    }
    progress.fraction(1.)?;
    Ok((cues, model.name.into(), language.into()))
}

pub fn label_speakers(
    wav: &Path,
    cache: &Path,
    cues: &mut [Cue],
    progress: &Progress,
) -> Result<()> {
    progress.stage(Stage::Speakers)?;
    let path = models::SPEAKERS.get(cache, progress)?;
    progress.update(Update::Estimated)?;
    let mut audio = hound::WavReader::open(wav)?;
    ensure!(
        audio.spec().sample_rate as usize == RATE,
        "Unsupported speaker analysis sample rate"
    );
    let samples = audio
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / 32768.))
        .collect::<Result<Vec<_>, _>>()?;
    let segments = if own_engine() {
        diarize_with_engine(&path, &samples, progress)?
    } else {
        diarize_with_cpp(&path, &samples, progress)?
    };
    assign_speakers(cues, &turns(segments));
    Ok(())
}

/// Speaker segments from transcribe.cpp's Sortformer, on the GPU when it can, else the CPU.
fn diarize_with_cpp(
    path: &Path,
    samples: &[f32],
    progress: &Progress,
) -> Result<Vec<SpeakerSegment>> {
    speech_devices()?;
    let diarize = |backend| -> transcribe_cpp::Result<Vec<SpeakerSegment>> {
        let model = Model::load_with(
            path,
            &ModelOptions {
                backend,
                ..Default::default()
            },
        )?;
        let mut session = model.diarize_session()?;
        session.set_cancel_token(progress.token());
        // The offline operating point: half-minute windows, the most accurate and, on a GPU,
        // ten times faster than the default real-time one (two hours in 21 s against 207 s).
        let preset = Some(SortformerPreset::VeryHighLatency);
        session.run(
            samples,
            &DiarizeOptions {
                family: Some(DiarizeExtension::Sortformer(SortformerDiarizeOptions {
                    preset,
                })),
            },
        )
    };
    Ok(match diarize(Backend::Auto) {
        Err(error) if device_error(&error) => {
            progress.note("Switching to the CPU…")?;
            diarize(Backend::Cpu).with_context(|| format!("CPU fallback failed after {error}"))?
        }
        result => result.context("Speaker analysis failed")?,
    })
}

/// Speaker segments from Sortformer on Vellum's engine, which streams the recording in
/// half-minute chunks as transcribe.cpp's offline preset does: on the GPU when there's a good
/// one, else (or when the GPU fails, at load or mid-way) on the CPU. The device goes to the
/// progress screen; a cancelled job stops it between chunks.
fn diarize_with_engine(
    path: &Path,
    samples: &[f32],
    progress: &Progress,
) -> Result<Vec<SpeakerSegment>> {
    let load = |device| Diarizer::load(path, &engine_options(device));
    let mut diarizer = match load(Device::Auto) {
        // A file the engine can't read fails the same way on the CPU.
        Err(error @ (EngineError::Io(_) | EngineError::Gguf(_) | EngineError::Unsupported(_))) => {
            return Err(error).context("Could not load the speaker model");
        }
        Err(error) => {
            progress.note("Switching to the CPU…")?;
            load(Device::Cpu).with_context(|| format!("CPU fallback failed after {error}"))?
        }
        Ok(diarizer) => diarizer,
    };
    progress.update(Update::Device(engine_device(diarizer.gpu())))?;
    let token = progress.token();
    let stop = || token.is_cancelled();
    let mut ran = diarizer.run_until(samples, &mut NoTrace, &stop);
    if let Err(error) = &ran
        && !matches!(error, EngineError::Cancelled)
        && diarizer.gpu().is_some()
    {
        // The stream's state is on the GPU: the CPU starts the recording over.
        progress.note("Switching to the CPU…")?;
        progress.update(Update::Device("CPU".into()))?;
        drop(diarizer);
        let mut cpu =
            load(Device::Cpu).with_context(|| format!("CPU fallback failed after {error}"))?;
        ran = cpu.run_until(samples, &mut NoTrace, &stop);
    }
    let speaker_turns = match ran {
        Err(EngineError::Cancelled) => {
            progress.check()?;
            anyhow::bail!("Cancelled");
        }
        result => result.context("Speaker analysis failed")?,
    };
    Ok(speaker_turns
        .into_iter()
        .map(|t| SpeakerSegment {
            t0_ms: t.t0_ms,
            t1_ms: t.t1_ms,
            speaker_id: t.speaker,
            p: t.p,
        })
        .collect())
}

/// The least a voice is heard in the whole video to count as a speaker. The model also
/// returns laughs, crosstalk and second-long fragments as voices of their own.
const SPEAKER_SECONDS: f64 = 5.0;

struct Turn {
    start: f64,
    end: f64,
    speaker: i32,
}

/// The model's speaker segments as turns sorted by start time, without voices heard too
/// briefly to be a speaker.
fn turns(segments: Vec<SpeakerSegment>) -> Vec<Turn> {
    let mut talk: HashMap<i32, f64> = HashMap::new();
    let mut turns: Vec<Turn> = segments
        .into_iter()
        .map(|s| Turn {
            start: s.t0_ms as f64 / 1000.,
            end: s.t1_ms as f64 / 1000.,
            speaker: s.speaker_id,
        })
        .inspect(|t| *talk.entry(t.speaker).or_default() += t.end - t.start)
        .collect();
    turns.retain(|t| talk[&t.speaker] >= SPEAKER_SECONDS);
    turns.sort_by(|a, b| a.start.total_cmp(&b.start));
    turns
}

fn assign_speakers(cues: &mut [Cue], turns: &[Turn]) {
    let mut first = 0;
    let mut previous = None;
    let mut speakers: Vec<Option<i32>> = cues
        .iter()
        .map(|cue| {
            while first < turns.len() && turns[first].end <= cue.start {
                first += 1;
            }
            // A recognized word can carry the pause after it in its timing: "ship?" lasted
            // until the answer four seconds later, and went to whoever gave it.
            let words = crate::document::word_count(&cue.text).max(1) as f64;
            let end = cue.end.min(cue.start + SECONDS_PER_WORD * words);
            let mut heard: Vec<(i32, f64)> = Vec::new();
            for turn in turns[first..].iter().take_while(|t| t.start < end) {
                let overlap = end.min(turn.end) - cue.start.max(turn.start);
                if overlap > 0. {
                    match heard.iter_mut().find(|(s, _)| *s == turn.speaker) {
                        Some((_, seconds)) => *seconds += overlap,
                        None => heard.push((turn.speaker, overlap)),
                    }
                }
            }
            // Someone talking over a word, with a "yeah" or a laugh, doesn't take it from
            // whoever was speaking.
            let speaker = heard
                .iter()
                .find(|(s, seconds)| Some(*s) == previous && *seconds >= (end - cue.start) / 2.)
                .or_else(|| heard.iter().max_by(|a, b| a.1.total_cmp(&b.1)))
                .map(|(s, _)| *s);
            previous = speaker.or(previous);
            speaker
        })
        .collect();
    fill_gaps(cues, &mut speakers);
    settle_turns(cues, &mut speakers);
    let mut names = HashMap::new();
    for (cue, speaker) in cues.iter_mut().zip(speakers) {
        if let Some(speaker) = speaker {
            let next = names.len() + 1;
            let number = names.entry(speaker).or_insert(next);
            cue.speaker = Some(format!("Speaker {number}"));
        }
    }
}

/// The longest pause after or before a labelled word whose speaker an unlabelled word takes.
const GAP_SECONDS: f64 = 2.0;

/// Gives a word that falls between two turns, where no voice was heard, the speaker of the word
/// just before it, or else just after it.
fn fill_gaps(cues: &[Cue], speakers: &mut [Option<i32>]) {
    let mut last: Option<(i32, f64)> = None;
    for (cue, speaker) in cues.iter().zip(speakers.iter_mut()) {
        match *speaker {
            Some(s) => last = Some((s, cue.end)),
            None => {
                if let Some((s, end)) = last.filter(|(_, end)| cue.start - end <= GAP_SECONDS) {
                    *speaker = Some(s);
                    last = Some((s, end.max(cue.end)));
                }
            }
        }
    }
    let mut next: Option<(i32, f64)> = None;
    for (cue, speaker) in cues.iter().zip(speakers.iter_mut()).rev() {
        match *speaker {
            Some(s) => next = Some((s, cue.start)),
            None => {
                if let Some((s, start)) = next.filter(|(_, start)| start - cue.end <= GAP_SECONDS) {
                    *speaker = Some(s);
                    next = Some((s, start.min(cue.start)));
                }
            }
        }
    }
}

/// The most words, and seconds between their starts, moved to put a change of speaker at a
/// sentence end.
const SETTLE_WORDS: usize = 3;
const SETTLE_SECONDS: f64 = 1.5;
/// The longest a recognized word is taken to be spoken for.
const SECONDS_PER_WORD: f64 = 1.0;

/// Moves each change of speaker that falls mid-sentence to the nearest sentence end a few words
/// away. Voice turns and word timings each drift by a fraction of a second, which otherwise
/// hands a question's last word to whoever answers it ("So what do you" / "ship? We…").
fn settle_turns(cues: &[Cue], speakers: &mut [Option<i32>]) {
    let mut i = 1;
    while i < cues.len() {
        let (before, after) = (speakers[i - 1], speakers[i]);
        if before == after
            || before.is_none()
            || after.is_none()
            || ends_sentence(&cues[i - 1].text)
        {
            i += 1;
            continue;
        }
        // The answerer's first words, up to the end of the asker's sentence…
        let forward = (i..cues.len().min(i + SETTLE_WORDS))
            .take_while(|&j| {
                speakers[j] == after && cues[j].start - cues[i].start <= SETTLE_SECONDS
            })
            .find(|&j| ends_sentence(&cues[j].text))
            .map(|j| (cues[j].start - cues[i].start, j));
        // …or the asker's last words, after the end of their previous sentence.
        let backward = (i.saturating_sub(SETTLE_WORDS + 1)..i - 1)
            .rev()
            .take_while(|&j| {
                speakers[j + 1] == before && cues[i - 1].start - cues[j + 1].start <= SETTLE_SECONDS
            })
            .find(|&j| ends_sentence(&cues[j].text))
            .map(|j| (cues[i - 1].start - cues[j + 1].start, j));
        match (forward, backward) {
            (Some((ahead, j)), behind) if behind.is_none_or(|(behind, _)| ahead <= behind) => {
                speakers[i..=j].fill(before);
                i = j + 1;
            }
            (_, Some((_, j))) => {
                speakers[j + 1..i].fill(after);
                i += 1;
            }
            _ => i += 1,
        }
    }
}
