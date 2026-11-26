//! Speaker labels on words: given as the app gives them, and scored as the live tests score
//! them against speaker-named lines (BBC scripts, podcast transcripts).

use super::{
    der::assign_max,
    text::{Mode, words},
    wer::{Alignment, WerScore, align},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use vellum_harness::{Diarization, Transcript};

// ---------------------------------------------------------------------------------------------
// Attribution: src/media/speech.rs, kept line for line so the two can later be shared. A `Cue`
// is one recognized word; the app's comes from transcribe.cpp's words, the lab's from a
// protocol `Transcript`'s.

struct Cue {
    start: f64,
    end: f64,
    text: String,
    speaker: Option<String>,
}

/// Words as a reader counts them: whitespace-separated runs, and each CJK character
/// (src/document.rs).
pub fn word_count(text: &str) -> usize {
    let cjk = |c: char| matches!(c, '\u{3040}'..='\u{30ff}' | '\u{3400}'..='\u{9fff}');
    text.split_whitespace()
        .map(|w| {
            let ideographs = w.chars().filter(|c| cjk(*c)).count();
            if ideographs > 0 {
                ideographs + usize::from(w.chars().any(|c| !cjk(c) && c.is_alphanumeric()))
            } else {
                1
            }
        })
        .sum()
}

/// src/media/mod.rs.
pub fn ends_sentence(text: &str) -> bool {
    text.trim_end()
        .trim_end_matches(['”', '"', '’', ')'])
        .ends_with(['.', '!', '?', '…', '。', '！', '？'])
}

/// The speaker the app would show for each of `transcript`'s words, from `diarization`'s
/// turns: "Speaker 1", "Speaker 2"… in order of first appearance, or `None`. Words with no
/// text (which the app drops) get `None`.
pub fn label(transcript: &Transcript, diarization: &Diarization) -> Vec<Option<String>> {
    let kept: Vec<usize> = (0..transcript.words.len())
        .filter(|&i| !transcript.words[i].text.trim().is_empty())
        .collect();
    let mut cues: Vec<Cue> = kept
        .iter()
        .map(|&i| {
            let w = &transcript.words[i];
            Cue {
                start: w.t0_ms as f64 / 1000.,
                end: w.t1_ms as f64 / 1000.,
                text: w.text.clone(),
                speaker: None,
            }
        })
        .collect();
    assign_speakers(&mut cues, &turns(diarization));
    let mut out = vec![None; transcript.words.len()];
    for (i, cue) in kept.into_iter().zip(cues) {
        out[i] = cue.speaker;
    }
    out
}

/// The least a voice is heard in the whole video to count as a speaker. The model also
/// returns laughs, crosstalk and second-long fragments as voices of their own.
pub const SPEAKER_SECONDS: f64 = 5.0;

struct Turn {
    start: f64,
    end: f64,
    speaker: i32,
}

/// The model's speaker segments as turns sorted by start time, without voices heard too
/// briefly to be a speaker.
fn turns(diarization: &Diarization) -> Vec<Turn> {
    let mut talk: HashMap<i32, f64> = HashMap::new();
    let mut turns: Vec<Turn> = diarization
        .turns
        .iter()
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
            let words = word_count(&cue.text).max(1) as f64;
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
pub const GAP_SECONDS: f64 = 2.0;

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
pub const SETTLE_WORDS: usize = 3;
pub const SETTLE_SECONDS: f64 = 1.5;
/// The longest a recognized word is taken to be spoken for.
pub const SECONDS_PER_WORD: f64 = 1.0;

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

// ---------------------------------------------------------------------------------------------
// Scoring: tests/app/live/score.rs.

/// A transcript's words as compared, each with the speaker it's labelled with. Consecutive
/// words with the same label are normalized together, as the app's paragraphs are, so
/// "twenty five" is still "25" in `Asr` mode.
pub fn labelled(
    transcript: &Transcript,
    labels: &[Option<String>],
    mode: Mode,
    language: &str,
) -> (Vec<String>, Vec<Option<String>>) {
    let mut all = Vec::new();
    let mut speakers = Vec::new();
    let mut run = String::new();
    let mut flush = |run: &mut String, label: Option<&Option<String>>| {
        if let Some(label) = label {
            let w = words(run, mode, language);
            speakers.extend(std::iter::repeat_n(label.clone(), w.len()));
            all.extend(w);
        }
        run.clear();
    };
    let mut current: Option<&Option<String>> = None;
    for (word, label) in transcript.words.iter().zip(labels) {
        if current.is_some_and(|c| c != label) {
            flush(&mut run, current);
        }
        current = Some(label);
        run.push_str(word.text.trim());
        run.push(' ');
    }
    flush(&mut run, current);
    (all, speakers)
}

/// A reference's words, each with the name of whoever said it. `lines` are (name, text).
pub fn named(lines: &[(String, String)], mode: Mode, language: &str) -> (Vec<String>, Vec<String>) {
    let mut all = Vec::new();
    let mut names = Vec::new();
    for (name, text) in lines {
        let w = words(text, mode, language);
        names.extend(std::iter::repeat_n(name.clone(), w.len()));
        all.extend(w);
    }
    (all, names)
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Person {
    pub name: String,
    /// The label paired with them, if any.
    pub label: Option<String>,
    /// Their aligned words carrying that label.
    pub correct: usize,
    /// Their words the alignment paired with a recognized word.
    pub aligned: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpeakerScore {
    /// Aligned words whose label is their speaker's, under the best one-to-one pairing of labels
    /// and names: splitting a voice or merging two costs the smaller side's words.
    pub correct: usize,
    pub aligned: usize,
    pub accuracy: f64,
    pub people: Vec<Person>,
    /// Distinct labels in the transcript.
    pub labels: usize,
}

impl SpeakerScore {
    pub fn recall(&self, name: &str) -> f64 {
        self.people
            .iter()
            .find(|p| p.name == name)
            .map_or(0.0, |p| p.correct as f64 / p.aligned.max(1) as f64)
    }

    pub fn summary(&self) -> String {
        let people = self
            .people
            .iter()
            .map(|p| {
                format!(
                    "{} {:.0}% of {} as {}",
                    p.name,
                    100.0 * p.correct as f64 / p.aligned.max(1) as f64,
                    p.aligned,
                    p.label.as_deref().unwrap_or("nobody")
                )
            })
            .collect::<Vec<_>>()
            .join(" · ");
        format!(
            "speakers {:.1}% of {} aligned words | {} labels for {} voices | {people}",
            100.0 * self.accuracy,
            self.aligned,
            self.labels,
            self.people.len()
        )
    }
}

/// Distinct values in order of first appearance.
fn distinct<'a>(values: impl Iterator<Item = &'a String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut seen: HashMap<&String, ()> = HashMap::new();
    for v in values {
        if seen.insert(v, ()).is_none() {
            out.push(v.clone());
        }
    }
    out
}

/// Who-spoke-what accuracy over the words `alignment` pairs up (reference = `names`' words,
/// hypothesis = `labels`' words).
pub fn speakers(
    alignment: &Alignment,
    names: &[String],
    labels: &[Option<String>],
) -> SpeakerScore {
    let people = distinct(names.iter());
    let found = distinct(labels.iter().flatten());
    let person_of: HashMap<&String, usize> = people.iter().zip(0..).collect();
    let label_of: HashMap<&String, usize> = found.iter().zip(0..).collect();
    let mut counts = vec![vec![0i64; people.len()]; found.len()];
    let mut totals = vec![0; people.len()];
    for &(r, h) in &alignment.matches {
        let person = person_of[&names[r]];
        totals[person] += 1;
        if let Some(label) = &labels[h] {
            counts[label_of[label]][person] += 1;
        }
    }
    // The best pairing of labels with people.
    let pairing = assign_max(&counts);
    let correct = pairing
        .iter()
        .enumerate()
        .filter_map(|(l, p)| p.map(|p| counts[l][p] as usize))
        .sum();
    let people = people
        .into_iter()
        .enumerate()
        .map(|(person, name)| {
            let label = pairing.iter().position(|p| *p == Some(person));
            Person {
                name,
                label: label.map(|l| found[l].clone()),
                correct: label.map_or(0, |l| counts[l][person] as usize),
                aligned: totals[person],
            }
        })
        .collect();
    let aligned = alignment.matches.len();
    SpeakerScore {
        correct,
        aligned,
        accuracy: correct as f64 / aligned.max(1) as f64,
        people,
        labels: found.len(),
    }
}

/// Speaker-attributed WER, cpWER-style: each person's words against the words of the label
/// `score` paired with them, concatenated in order; unpaired people's words are deletions and
/// unpaired labels' (and unlabelled) words insertions. Unlike `SpeakerScore::accuracy`, which
/// counts only words the alignment found, dropping hard words doesn't improve it.
pub fn cpwer(
    reference: &[String],
    names: &[String],
    hypothesis: &[String],
    labels: &[Option<String>],
    score: &SpeakerScore,
) -> WerScore {
    let mut by_label: HashMap<Option<&str>, Vec<&String>> = HashMap::new();
    for (word, label) in hypothesis.iter().zip(labels) {
        by_label.entry(label.as_deref()).or_default().push(word);
    }
    let mut by_name: HashMap<&str, Vec<&String>> = HashMap::new();
    for (word, name) in reference.iter().zip(names) {
        by_name.entry(name).or_default().push(word);
    }
    let mut parts = Vec::new();
    for person in &score.people {
        let said = by_name.remove(person.name.as_str()).unwrap_or_default();
        let heard = person
            .label
            .as_deref()
            .and_then(|l| by_label.remove(&Some(l)))
            .unwrap_or_default();
        parts.push(align(&said, &heard).score(said.len()));
    }
    // Anyone left over: insertions.
    for (_, heard) in by_label {
        parts.push(WerScore {
            insertions: heard.len(),
            ..WerScore::default()
        });
    }
    WerScore::total(&parts)
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpeakerReport {
    pub words: SpeakerScore,
    /// Speaker-attributed WER.
    pub cpwer: WerScore,
    /// WER ignoring speakers, for comparison with `cpwer`.
    pub wer: WerScore,
}

/// A labelled transcript against speaker-named lines: `labels` as from `label`.
pub fn score(
    lines: &[(String, String)],
    transcript: &Transcript,
    labels: &[Option<String>],
    mode: Mode,
    language: &str,
) -> SpeakerReport {
    let (said, names) = named(lines, mode, language);
    let (heard, found) = labelled(transcript, labels, mode, language);
    let alignment = align(&said, &heard);
    let words = speakers(&alignment, &names, &found);
    let cpwer = cpwer(&said, &names, &heard, &found, &words);
    SpeakerReport {
        cpwer,
        wer: alignment.score(said.len()),
        words,
    }
}
