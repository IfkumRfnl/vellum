//! One engine against another on the same clip: how far apart their outputs are, and where
//! they first part.

use super::{
    der::{self, DerOptions, DerScore},
    stats::quantile,
    text::{Mode, words},
    wer::{WerScore, align, lcs},
};
use serde::{Deserialize, Serialize};
use vellum_harness::{Diarization, Transcript};

/// Where two outputs first differ.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Divergence {
    /// Position in the sequence compared (tokens, or normalized words).
    pub index: usize,
    /// The window and time of the reference's item there, else the candidate's; `None` when
    /// neither has one.
    pub window: Option<u32>,
    pub t_ms: Option<i64>,
    /// What each side has there; `None` past its end.
    pub reference: Option<String>,
    pub candidate: Option<String>,
}

impl std::fmt::Display for Divergence {
    /// "#12 in window 3 at 41.20 s: 'the' / 'a'".
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.index)?;
        if let Some(w) = self.window {
            write!(f, " in window {w}")?;
        }
        if let Some(t) = self.t_ms {
            write!(f, " at {:.2} s", t as f64 / 1000.)?;
        }
        let side = |s: &Option<String>| {
            s.as_deref()
                .map_or("(end)".to_owned(), |s| format!("'{s}'"))
        };
        write!(f, ": {} / {}", side(&self.reference), side(&self.candidate))
    }
}

/// Word timestamps against the reference's, over words both recognized identically.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimingError {
    pub matched: usize,
    /// Matched words over the reference's words.
    pub matched_fraction: f64,
    pub t0_mean_ms: f64,
    pub t0_p95_ms: f64,
    pub t1_mean_ms: f64,
    pub t1_p95_ms: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WindowAgreement {
    pub windows: usize,
    /// Windows with the same words (and token ids, when both have them).
    pub identical: usize,
    pub differing: Vec<u32>,
    /// Whether both cut the recording into the same windows.
    pub boundaries_match: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TranscriptParity {
    pub identical_text: bool,
    /// The candidate's text against the reference's, `Plain` words.
    pub wer: WerScore,
    /// Longest common subsequence of token ids over the longer sequence; `None` when either
    /// side has no token ids.
    pub token_agreement: Option<f64>,
    pub first_token: Option<Divergence>,
    pub first_word: Option<Divergence>,
    /// Where to look first: the first token that differs when both have token ids, else the
    /// first word.
    pub first_difference: Option<Divergence>,
    /// `None` when either side has no timed words.
    pub timing: Option<TimingError>,
    pub windows: WindowAgreement,
}

/// A transcript's timed words as `Plain` words, each with the index of the word it came from.
fn timed_words(t: &Transcript, language: &str) -> (Vec<String>, Vec<usize>) {
    let mut all = Vec::new();
    let mut from = Vec::new();
    for (i, w) in t.words.iter().enumerate() {
        for word in words(&w.text, Mode::Plain, language) {
            all.push(word);
            from.push(i);
        }
    }
    (all, from)
}

/// `candidate` against `reference`, both transcripts of the same clip.
pub fn transcripts(reference: &Transcript, candidate: &Transcript) -> TranscriptParity {
    let language = reference
        .language
        .as_deref()
        .or(candidate.language.as_deref())
        .unwrap_or("");
    let (ref_text, cand_text) = (
        words(&reference.text, Mode::Plain, language),
        words(&candidate.text, Mode::Plain, language),
    );
    let wer = align(&ref_text, &cand_text).score(ref_text.len());

    // Tokens.
    let has_tokens = !reference.tokens.is_empty() && !candidate.tokens.is_empty();
    let (ref_ids, cand_ids): (Vec<i32>, Vec<i32>) = (
        reference.tokens.iter().map(|t| t.id).collect(),
        candidate.tokens.iter().map(|t| t.id).collect(),
    );
    let token_agreement = has_tokens
        .then(|| lcs(&ref_ids, &cand_ids) as f64 / ref_ids.len().max(cand_ids.len()) as f64);
    let first_token = has_tokens
        .then(|| first_difference(&ref_ids, &cand_ids))
        .flatten()
        .map(|index| {
            let (r, c) = (reference.tokens.get(index), candidate.tokens.get(index));
            let at = r.or(c).unwrap();
            Divergence {
                index,
                window: Some(at.window),
                t_ms: Some(at.t0_ms),
                reference: r.map(|t| format!("{} {:?}", t.id, t.text)),
                candidate: c.map(|t| format!("{} {:?}", t.id, t.text)),
            }
        });

    // Words, timed when both have timed words, else from the text.
    let timed = !reference.words.is_empty() && !candidate.words.is_empty();
    let (rw, rfrom) = timed_words(reference, language);
    let (cw, cfrom) = timed_words(candidate, language);
    let first_word = if timed {
        first_difference(&rw, &cw).map(|index| {
            let source = rfrom
                .get(index)
                .map(|&i| &reference.words[i])
                .or_else(|| cfrom.get(index).map(|&i| &candidate.words[i]));
            Divergence {
                index,
                window: source.map(|w| w.window),
                t_ms: source.map(|w| w.t0_ms),
                reference: rw.get(index).cloned(),
                candidate: cw.get(index).cloned(),
            }
        })
    } else {
        first_difference(&ref_text, &cand_text).map(|index| Divergence {
            index,
            reference: ref_text.get(index).cloned(),
            candidate: cand_text.get(index).cloned(),
            ..Divergence::default()
        })
    };
    let timing = timed.then(|| {
        let mut pairs: Vec<(usize, usize)> = align(&rw, &cw)
            .matches
            .iter()
            .map(|&(r, c)| (rfrom[r], cfrom[c]))
            .collect();
        pairs.dedup_by_key(|p| p.0);
        let (d0, d1): (Vec<f64>, Vec<f64>) = pairs
            .iter()
            .map(|&(r, c)| {
                let (r, c) = (&reference.words[r], &candidate.words[c]);
                (
                    (r.t0_ms - c.t0_ms).abs() as f64,
                    (r.t1_ms - c.t1_ms).abs() as f64,
                )
            })
            .unzip();
        let mut sources = rfrom.clone();
        sources.dedup();
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
        TimingError {
            matched: pairs.len(),
            matched_fraction: pairs.len() as f64 / sources.len().max(1) as f64,
            t0_mean_ms: mean(&d0),
            t0_p95_ms: if d0.is_empty() {
                0.
            } else {
                quantile(&d0, 0.95)
            },
            t1_mean_ms: mean(&d1),
            t1_p95_ms: if d1.is_empty() {
                0.
            } else {
                quantile(&d1, 0.95)
            },
        }
    });

    let first_difference = if has_tokens {
        first_token.clone()
    } else {
        first_word.clone()
    };
    TranscriptParity {
        first_difference,
        identical_text: reference.text.trim() == candidate.text.trim(),
        wer,
        token_agreement,
        first_token,
        first_word,
        timing,
        windows: windows(reference, candidate, language, has_tokens),
    }
}

fn first_difference<T: PartialEq>(a: &[T], b: &[T]) -> Option<usize> {
    let common = a.iter().zip(b).take_while(|(x, y)| x == y).count();
    (common < a.len().max(b.len())).then_some(common)
}

fn windows(
    reference: &Transcript,
    candidate: &Transcript,
    language: &str,
    tokens: bool,
) -> WindowAgreement {
    let last = |t: &Transcript| {
        let words = t.words.iter().map(|w| w.window + 1).max().unwrap_or(0);
        let tokens = t.tokens.iter().map(|w| w.window + 1).max().unwrap_or(0);
        t.windows.len().max(words as usize).max(tokens as usize)
    };
    let count = last(reference).max(last(candidate));
    // Each window's words and token ids, in one pass per side.
    let split = |t: &Transcript| {
        let mut w: Vec<Vec<String>> = vec![Vec::new(); count];
        let mut k: Vec<Vec<i32>> = vec![Vec::new(); count];
        for word in &t.words {
            w[word.window as usize].extend(words(&word.text, Mode::Plain, language));
        }
        for token in &t.tokens {
            k[token.window as usize].push(token.id);
        }
        (w, k)
    };
    let (rw, rk) = split(reference);
    let (cw, ck) = split(candidate);
    let differing: Vec<u32> = (0..count)
        .filter(|&i| rw[i] != cw[i] || (tokens && rk[i] != ck[i]))
        .map(|i| i as u32)
        .collect();
    let boundaries_match = reference.windows.len() == candidate.windows.len()
        && reference
            .windows
            .iter()
            .zip(&candidate.windows)
            .all(|(a, b)| a.start_sample == b.start_sample && a.end_sample == b.end_sample);
    WindowAgreement {
        windows: count,
        identical: count - differing.len(),
        differing,
        boundaries_match,
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiarizationParity {
    pub identical: bool,
    /// The candidate's turns scored against the reference's, no collar.
    pub der: DerScore,
    /// Share of 10 ms frames with the same speakers, under the DER's pairing.
    pub frame_agreement: f64,
}

/// `candidate`'s speaker turns against `reference`'s, both of the same clip.
pub fn diarizations(reference: &Diarization, candidate: &Diarization) -> DiarizationParity {
    let (r, c) = (
        der::from_turns(&reference.turns),
        der::from_turns(&candidate.turns),
    );
    DiarizationParity {
        identical: reference.turns.len() == candidate.turns.len()
            && reference
                .turns
                .iter()
                .zip(&candidate.turns)
                .all(|(a, b)| a.t0_ms == b.t0_ms && a.t1_ms == b.t1_ms && a.speaker == b.speaker),
        der: der::der(&r, &c, &DerOptions::default()),
        frame_agreement: der::frame_agreement(&r, &c),
    }
}
