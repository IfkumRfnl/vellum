//! How close a transcript is to a reference: its words, and who it says spoke them.

use crate::Document;
use std::collections::HashMap;

/// Words as compared: lowercase letters and digits, punctuation dropped.
pub fn words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(|s| {
            let word: String = s
                .chars()
                .filter(|c| c.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect();
            (!word.is_empty()).then_some(word)
        })
        .collect()
}

/// A transcript's words, each with the speaker it's labelled with.
pub fn labelled(doc: &Document) -> (Vec<String>, Vec<Option<String>>) {
    let mut all = Vec::new();
    let mut speakers = Vec::new();
    for p in &doc.paragraphs {
        let w = words(&p.text);
        speakers.extend(std::iter::repeat_n(p.speaker.clone(), w.len()));
        all.extend(w);
    }
    (all, speakers)
}

/// A reference's words, each with the name of whoever said it.
pub fn named(turns: &[(String, String)]) -> (Vec<String>, Vec<String>) {
    let mut all = Vec::new();
    let mut names = Vec::new();
    for (name, text) in turns {
        let w = words(text);
        names.extend(std::iter::repeat_n(name.clone(), w.len()));
        all.extend(w);
    }
    (all, names)
}

pub struct Alignment {
    /// Substitutions, insertions and deletions turning the reference into the hypothesis.
    pub errors: usize,
    /// Identical words paired up: (reference index, hypothesis index).
    pub matches: Vec<(usize, usize)>,
}

impl Alignment {
    pub fn wer(&self, reference: usize) -> f64 {
        self.errors as f64 / reference.max(1) as f64
    }
}

/// How far an alignment may stray from the diagonal, in words. Two transcripts of the same
/// audio drift apart by intros, ads and skipped passages, not by thousands of words.
const BAND: usize = 2500;

/// The minimum-edit alignment of `hypothesis` to `reference`, within a band around the
/// diagonal, so a two-hour podcast (20,000 words each) takes seconds and ~100 MB, not 400M cells.
pub fn align(reference: &[String], hypothesis: &[String]) -> Alignment {
    let (n, m) = (reference.len(), hypothesis.len());
    let span = |i: usize| {
        let center = (i * m).checked_div(n).unwrap_or(m);
        (center.saturating_sub(BAND), (center + BAND).min(m))
    };
    const DIAGONAL: u8 = 0;
    const UP: u8 = 1;
    const LEFT: u8 = 2;
    let mut moves: Vec<Vec<u8>> = Vec::with_capacity(n + 1);
    let (lo, hi) = span(0);
    let mut previous: Vec<u32> = (lo..=hi).map(|j| j as u32).collect();
    let mut previous_lo = lo;
    moves.push(vec![LEFT; hi - lo + 1]);
    for i in 1..=n {
        let (lo, hi) = span(i);
        let at = |row: &[u32], row_lo: usize, j: usize| {
            j.checked_sub(row_lo)
                .and_then(|k| row.get(k))
                .copied()
                .unwrap_or(u32::MAX / 2)
        };
        let mut current = Vec::with_capacity(hi - lo + 1);
        let mut row_moves = Vec::with_capacity(hi - lo + 1);
        for j in lo..=hi {
            let up = at(&previous, previous_lo, j) + 1;
            let (mut best, mut how) = (up, UP);
            if j > 0 {
                let diagonal = at(&previous, previous_lo, j - 1)
                    + u32::from(reference[i - 1] != hypothesis[j - 1]);
                if diagonal <= best {
                    (best, how) = (diagonal, DIAGONAL);
                }
                if j > lo {
                    let left = current[j - 1 - lo] + 1;
                    if left < best {
                        (best, how) = (left, LEFT);
                    }
                }
            }
            current.push(best);
            row_moves.push(how);
        }
        moves.push(row_moves);
        previous = current;
        previous_lo = lo;
    }
    let errors = previous[m - previous_lo] as usize;
    let mut matches = Vec::new();
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        let how = if i == 0 {
            LEFT
        } else {
            moves[i][j - span(i).0]
        };
        match how {
            DIAGONAL => {
                if reference[i - 1] == hypothesis[j - 1] {
                    matches.push((i - 1, j - 1));
                }
                i -= 1;
                j -= 1;
            }
            UP => i -= 1,
            _ => j -= 1,
        }
    }
    matches.reverse();
    Alignment { errors, matches }
}

pub struct SpeakerScore {
    /// Aligned words whose label is their speaker's, under the best one-to-one pairing of labels
    /// and names: splitting a voice or merging two costs the smaller side's words.
    pub correct: usize,
    pub aligned: usize,
    /// Each reference speaker: their label, words labelled correctly, words aligned.
    pub people: Vec<(String, Option<String>, usize, usize)>,
    pub labels: usize,
}

impl SpeakerScore {
    pub fn accuracy(&self) -> f64 {
        self.correct as f64 / self.aligned.max(1) as f64
    }

    pub fn recall(&self, name: &str) -> f64 {
        self.people
            .iter()
            .find(|p| p.0 == name)
            .map_or(0.0, |p| p.2 as f64 / p.3.max(1) as f64)
    }

    pub fn summary(&self) -> String {
        let people = self
            .people
            .iter()
            .map(|(name, label, correct, total)| {
                format!(
                    "{name} {:.0}% of {total} as {}",
                    100.0 * *correct as f64 / (*total).max(1) as f64,
                    label.as_deref().unwrap_or("nobody")
                )
            })
            .collect::<Vec<_>>()
            .join(" · ");
        format!(
            "speakers {:.1}% of {} aligned words | {} labels for {} voices | {people}",
            100.0 * self.accuracy(),
            self.aligned,
            self.labels,
            self.people.len()
        )
    }
}

/// Who-spoke-what accuracy over the words `alignment` pairs up.
pub fn speakers(
    alignment: &Alignment,
    names: &[String],
    labels: &[Option<String>],
) -> SpeakerScore {
    let mut people: Vec<String> = Vec::new();
    for name in names {
        if !people.contains(name) {
            people.push(name.clone());
        }
    }
    let mut found: Vec<String> = Vec::new();
    for label in labels.iter().flatten() {
        if !found.contains(label) {
            found.push(label.clone());
        }
    }
    let mut counts: HashMap<(usize, usize), usize> = HashMap::new();
    let mut totals = vec![0; people.len()];
    for &(r, h) in &alignment.matches {
        let person = people.iter().position(|p| *p == names[r]).unwrap();
        totals[person] += 1;
        if let Some(label) = &labels[h] {
            let label = found.iter().position(|l| l == label).unwrap();
            *counts.entry((label, person)).or_default() += 1;
        }
    }
    // The best pairing of labels with people, trying every one: a video has a handful of each.
    fn best(
        label: usize,
        labels: usize,
        taken: &mut Vec<bool>,
        counts: &HashMap<(usize, usize), usize>,
    ) -> (usize, Vec<Option<usize>>) {
        if label == labels {
            return (0, Vec::new());
        }
        let (mut top, mut tail) = best(label + 1, labels, taken, counts);
        tail.insert(0, None);
        for person in 0..taken.len() {
            if taken[person] {
                continue;
            }
            taken[person] = true;
            let (score, mut rest) = best(label + 1, labels, taken, counts);
            taken[person] = false;
            let score = score + counts.get(&(label, person)).copied().unwrap_or(0);
            if score > top {
                rest.insert(0, Some(person));
                (top, tail) = (score, rest);
            }
        }
        (top, tail)
    }
    let (correct, pairing) = best(0, found.len(), &mut vec![false; people.len()], &counts);
    let people = people
        .into_iter()
        .enumerate()
        .map(|(person, name)| {
            let label = pairing.iter().position(|p| *p == Some(person));
            let right = label.map_or(0, |l| counts.get(&(l, person)).copied().unwrap_or(0));
            (name, label.map(|l| found[l].clone()), right, totals[person])
        })
        .collect();
    SpeakerScore {
        correct,
        aligned: alignment.matches.len(),
        people,
        labels: found.len(),
    }
}
