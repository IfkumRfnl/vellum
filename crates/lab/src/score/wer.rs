//! Word alignment and word error rate.

use serde::{Deserialize, Serialize};

/// The minimum-edit alignment of a hypothesis to a reference.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Alignment {
    pub substitutions: usize,
    /// Hypothesis words with no reference word.
    pub insertions: usize,
    /// Reference words with no hypothesis word.
    pub deletions: usize,
    /// Identical items paired up: (reference index, hypothesis index), in order.
    pub matches: Vec<(usize, usize)>,
}

impl Alignment {
    pub fn errors(&self) -> usize {
        self.substitutions + self.insertions + self.deletions
    }

    pub fn score(&self, reference_words: usize) -> WerScore {
        WerScore {
            reference_words,
            substitutions: self.substitutions,
            insertions: self.insertions,
            deletions: self.deletions,
            wer: self.errors() as f64 / reference_words.max(1) as f64,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WerScore {
    pub reference_words: usize,
    pub substitutions: usize,
    pub insertions: usize,
    pub deletions: usize,
    /// Errors over reference words; a hypothesis against an empty reference scores its
    /// insertions over one word.
    pub wer: f64,
}

impl WerScore {
    pub fn errors(&self) -> usize {
        self.substitutions + self.insertions + self.deletions
    }

    /// Several scores as one: errors and words summed, not rates averaged.
    pub fn total<'a>(scores: impl IntoIterator<Item = &'a WerScore>) -> WerScore {
        let mut t = WerScore::default();
        for s in scores {
            t.reference_words += s.reference_words;
            t.substitutions += s.substitutions;
            t.insertions += s.insertions;
            t.deletions += s.deletions;
        }
        t.wer =
            (t.substitutions + t.insertions + t.deletions) as f64 / t.reference_words.max(1) as f64;
        t
    }
}

/// The word error rate of `hypothesis` against `reference`.
pub fn score<T: PartialEq>(reference: &[T], hypothesis: &[T]) -> WerScore {
    align(reference, hypothesis).score(reference.len())
}

/// How far an alignment may stray from the diagonal, in words. Two transcripts of the same
/// audio drift apart by intros, ads and skipped passages, not by thousands of words.
const BAND: usize = 2500;

/// The minimum-edit alignment of `hypothesis` to `reference`. Exact while the hypothesis is
/// within `BAND` words; beyond, within a band around the diagonal, so a two-hour podcast
/// (20,000 words each) takes a second and ~100 MB, not 400M cells. A shared beginning and end
/// are matched first (which never costs optimality), so near-identical transcripts are instant.
pub fn align<T: PartialEq>(reference: &[T], hypothesis: &[T]) -> Alignment {
    let prefix = reference
        .iter()
        .zip(hypothesis)
        .take_while(|(a, b)| a == b)
        .count();
    let suffix = reference[prefix..]
        .iter()
        .rev()
        .zip(hypothesis[prefix..].iter().rev())
        .take_while(|(a, b)| a == b)
        .count();
    let middle = banded(
        &reference[prefix..reference.len() - suffix],
        &hypothesis[prefix..hypothesis.len() - suffix],
    );
    let mut matches: Vec<(usize, usize)> = (0..prefix).map(|i| (i, i)).collect();
    matches.extend(
        middle
            .matches
            .iter()
            .map(|&(r, h)| (r + prefix, h + prefix)),
    );
    let (r_end, h_end) = (reference.len() - suffix, hypothesis.len() - suffix);
    matches.extend((0..suffix).map(|k| (r_end + k, h_end + k)));
    Alignment { matches, ..middle }
}

/// `align` without the shortcut: the banded dynamic programme, as in tests/app/live/score.rs.
fn banded<T: PartialEq>(reference: &[T], hypothesis: &[T]) -> Alignment {
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
    let mut alignment = Alignment::default();
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
                    alignment.matches.push((i - 1, j - 1));
                } else {
                    alignment.substitutions += 1;
                }
                i -= 1;
                j -= 1;
            }
            UP => {
                alignment.deletions += 1;
                i -= 1;
            }
            _ => {
                alignment.insertions += 1;
                j -= 1;
            }
        }
    }
    alignment.matches.reverse();
    alignment
}

/// The length of the longest common subsequence, within the same band as `align`.
pub fn lcs<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    let prefix = a.iter().zip(b).take_while(|(x, y)| x == y).count();
    let (a, b) = (&a[prefix..], &b[prefix..]);
    let suffix = a
        .iter()
        .rev()
        .zip(b.iter().rev())
        .take_while(|(x, y)| x == y)
        .count();
    let (a, b) = (&a[..a.len() - suffix], &b[..b.len() - suffix]);
    let (n, m) = (a.len(), b.len());
    let span = |i: usize| {
        let center = (i * m).checked_div(n).unwrap_or(m);
        (center.saturating_sub(BAND), (center + BAND).min(m))
    };
    // Row i holds L[i][j] for j in its span; cells outside a span count as 0, a lower bound.
    let (lo, hi) = span(0);
    let mut previous = vec![0u32; hi - lo + 1];
    let mut previous_lo = lo;
    for i in 1..=n {
        let (lo, hi) = span(i);
        let at = |row: &[u32], row_lo: usize, j: usize| {
            j.checked_sub(row_lo)
                .and_then(|k| row.get(k))
                .copied()
                .unwrap_or(0)
        };
        let mut current = Vec::with_capacity(hi - lo + 1);
        for j in lo..=hi {
            let value = if j == 0 {
                0
            } else if a[i - 1] == b[j - 1] {
                at(&previous, previous_lo, j - 1) + 1
            } else {
                let left = if j > lo { current[j - 1 - lo] } else { 0 };
                left.max(at(&previous, previous_lo, j))
            };
            current.push(value);
        }
        previous = current;
        previous_lo = lo;
    }
    prefix + suffix + previous.last().copied().unwrap_or(0) as usize
}
