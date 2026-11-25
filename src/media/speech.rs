use super::{Cue, ends_sentence, models};
use crate::{
    job::Progress,
    pipeline::{Stage, Update},
};
use anyhow::{Context, Result, ensure};
use std::{collections::HashMap, path::Path, sync::OnceLock};
use transcribe_cpp::{
    Backend, DiarizeExtension, DiarizeOptions, Error, Model, ModelOptions, RunOptions, Session,
    SortformerDiarizeOptions, SortformerPreset, SpeakerSegment, TimestampKind,
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
    let path = models::SPEAKERS.get(cache, progress)?;
    progress.update(Update::Estimated)?;
    speech_devices()?;
    let mut audio = hound::WavReader::open(wav)?;
    ensure!(
        audio.spec().sample_rate as usize == RATE,
        "Unsupported speaker analysis sample rate"
    );
    let samples = audio
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / 32768.))
        .collect::<Result<Vec<_>, _>>()?;
    let diarize = |backend| -> transcribe_cpp::Result<Vec<SpeakerSegment>> {
        let model = Model::load_with(
            &path,
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
            &samples,
            &DiarizeOptions {
                family: Some(DiarizeExtension::Sortformer(SortformerDiarizeOptions {
                    preset,
                })),
            },
        )
    };
    let segments = match diarize(Backend::Auto) {
        Err(error) if device_error(&error) => {
            progress.note("Switching to the CPU…")?;
            diarize(Backend::Cpu).with_context(|| format!("CPU fallback failed after {error}"))?
        }
        result => result.context("Speaker analysis failed")?,
    };
    assign_speakers(cues, &turns(segments));
    Ok(())
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
