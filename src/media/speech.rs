use super::{Cue, ends_sentence, models};
use crate::{
    job::Progress,
    pipeline::{Stage, Update},
};
use anyhow::{Context, Result, ensure};
use std::{collections::HashMap, ops::Range, path::Path};
use vellum_engine::{
    Device, Diarizer, Error as EngineError, NoTrace, Options as EngineOptions, RunOptions,
    SpeakerTurn, Transcriber,
};

const RATE: usize = 16000;

/// The device the app asks the engine for: the best GPU, or else the CPU, unless
/// `VELLUM_DEVICE=cpu` asks for the CPU alone.
fn requested_device() -> Device {
    if std::env::var("VELLUM_DEVICE").is_ok_and(|v| v.eq_ignore_ascii_case("cpu")) {
        Device::Cpu
    } else {
        Device::Auto
    }
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

/// The whole recording as samples from -1 to 1, in a buffer of exactly its length: a
/// 2½-hour recording takes 560 MB, where a growing one would ask for 1 GB on top of the
/// 512 MB it outgrew.
fn read_audio(wav: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(wav)?;
    ensure!(
        reader.spec().sample_rate as usize == RATE,
        "Unsupported audio sample rate"
    );
    let mut pcm = Vec::with_capacity(reader.duration() as usize);
    for sample in reader.samples::<i16>() {
        pcm.push(sample? as f32 / 32768.);
    }
    Ok(pcm)
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

/// A recording's cues, the model's name and the language it was transcribed in, on Vellum's
/// engine: the whole recording, cut into windows (exactly-silent ones skipped), handed to the
/// engine at once so it works on several windows together. Each window's words reach the
/// progress screen as soon as it and those before it are done. A GPU that fails mid-way leaves
/// the windows still to do to the CPU.
///
/// Parakeet Unified is English only. Whisper detects the language until a window has words,
/// then keeps it for the rest of the recording (the engine applies that rule itself; the app
/// follows it to name the language and to carry it over to the CPU).
pub fn transcribe(
    wav: &Path,
    cache: &Path,
    model: &models::SpeechModel,
    progress: &Progress,
) -> Result<(Vec<Cue>, String, String)> {
    progress.stage(Stage::Model)?;
    let path = model.artifact.get(cache, progress)?;
    progress.update(Update::Estimated)?;
    let mut transcriber = load_engine(&path, progress)?;
    let window = window_samples(transcriber.max_audio_ms().into());
    let pcm = read_audio(wav)?;
    let total = pcm.len().max(1);
    let speaking: Vec<Range<usize>> = windows(&pcm, window)
        .into_iter()
        .filter(|r| pcm[r.clone()].iter().any(|s| *s != 0.))
        .collect();
    let mut language: Option<String> = (model.id == "parakeet_unified_en").then(|| "en".into());
    let token = progress.token();
    let stop = || token.is_cancelled();
    let mut cues = Vec::new();
    // Windows of `speaking` whose words are in `cues`.
    let mut done = 0;
    progress.stage(Stage::Transcribe)?;
    while done < speaking.len() {
        let rest: Vec<&[f32]> = speaking[done..].iter().map(|r| &pcm[r.clone()]).collect();
        let options = RunOptions {
            language: language.clone(),
        };
        // An error of the app's own (a cancelled job) stops the engine, and is what's reported.
        let mut failed = None;
        let ran = transcriber.run_each(&rest, &options, &mut NoTrace, &stop, &mut |_, result| {
            let range = &speaking[done];
            if language.is_none() && !result.text.trim().is_empty() {
                language = result.language.clone();
            }
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
                transcriber = Transcriber::load(&path, &engine_options(Device::Cpu))
                    .with_context(|| format!("CPU fallback failed after {error}"))?;
            }
            (Err(error), None) => return Err(error).context("Speech recognition failed"),
        }
    }
    progress.fraction(1.)?;
    Ok((cues, model.name.into(), language.unwrap_or_default()))
}

/// Loads a speech model on Vellum's engine: on the GPU when there's a good one, else (or when
/// the GPU can't take the model) on the CPU. Names the device on the progress screen
/// ("Vulkan · NVIDIA GeForce RTX 4060 Laptop GPU", "CPU").
fn load_engine(path: &Path, progress: &Progress) -> Result<Transcriber> {
    let load = |device| Transcriber::load(path, &engine_options(device));
    let transcriber = match load(requested_device()) {
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

pub fn label_speakers(
    wav: &Path,
    cache: &Path,
    cues: &mut [Cue],
    progress: &Progress,
) -> Result<()> {
    progress.stage(Stage::Speakers)?;
    let path = models::SPEAKERS.get(cache, progress)?;
    progress.update(Update::Estimated)?;
    let samples = read_audio(wav)?;
    let segments = diarize(&path, &samples, progress)?;
    assign_speakers(cues, &turns(segments));
    Ok(())
}

/// Who spoke when, from Streaming Sortformer on Vellum's engine, which streams the recording in
/// half-minute chunks (the offline operating point: the most accurate, and on a GPU ten times
/// faster than the real-time one): on the GPU when there's a good one, else (or when the GPU
/// fails, at load or mid-way) on the CPU. The device goes to the progress screen; a cancelled
/// job stops it between chunks.
fn diarize(path: &Path, samples: &[f32], progress: &Progress) -> Result<Vec<SpeakerTurn>> {
    let load = |device| Diarizer::load(path, &engine_options(device));
    let mut diarizer = match load(requested_device()) {
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
    match ran {
        Err(EngineError::Cancelled) => {
            progress.check()?;
            anyhow::bail!("Cancelled");
        }
        result => result.context("Speaker analysis failed"),
    }
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
fn turns(segments: Vec<SpeakerTurn>) -> Vec<Turn> {
    let mut talk: HashMap<i32, f64> = HashMap::new();
    let mut turns: Vec<Turn> = segments
        .into_iter()
        .map(|s| Turn {
            start: s.t0_ms as f64 / 1000.,
            end: s.t1_ms as f64 / 1000.,
            speaker: s.speaker,
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
