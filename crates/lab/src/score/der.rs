//! Diarization error rate: who spoke when, against who really spoke when.
//!
//! Frame-based at 10 ms, as md-eval and pyannote score it: at every frame the reference has
//! some speakers and the hypothesis some; speech one side has and the other doesn't is missed
//! or a false alarm, and speech both have under different (mapped) speakers is confusion.
//! Overlapped speech counts once per speaker. The hypothesis's speakers are paired with the
//! reference's one to one so that the most speech agrees (the Hungarian algorithm).

use crate::corpus::TruthTurn;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use vellum_harness::{Diarization, Turn};

/// Seconds per frame.
pub const FRAME: f64 = 0.01;

/// One speaker talking, in seconds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    pub t0: f64,
    pub t1: f64,
    pub speaker: String,
}

/// Ground-truth turns as segments.
pub fn from_truth(turns: &[TruthTurn]) -> Vec<Segment> {
    turns
        .iter()
        .map(|t| Segment {
            t0: t.t0,
            t1: t.t1,
            speaker: t.speaker.clone(),
        })
        .collect()
}

/// An engine's turns as segments, speakers named by their number.
pub fn from_turns(turns: &[Turn]) -> Vec<Segment> {
    turns
        .iter()
        .map(|t| Segment {
            t0: t.t0_ms as f64 / 1000.,
            t1: t.t1_ms as f64 / 1000.,
            speaker: t.speaker.to_string(),
        })
        .collect()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DerOptions {
    /// Seconds left unscored on each side of every point where a reference speaker starts or
    /// stops talking (md-eval's `-c`: 0.25 is the usual forgiveness collar, VoxConverse's; 0
    /// scores every frame, DIHARD's).
    pub collar: f64,
    /// Only frames in [start, end) seconds are scored (a UEM); `None` scores everything.
    pub region: Option<(f64, f64)>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DerScore {
    /// Reference speech scored, in speaker-seconds (overlap counts once per speaker).
    pub scored_seconds: f64,
    /// Speaker-seconds the hypothesis missed, labelled as nobody's when someone was speaking.
    pub missed: f64,
    pub false_alarm: f64,
    /// Speaker-seconds both sides heard but gave to different people.
    pub confusion: f64,
    /// (missed + false alarm + confusion) / scored seconds.
    pub der: f64,
    pub missed_rate: f64,
    pub false_alarm_rate: f64,
    pub confusion_rate: f64,
    /// Each reference speaker and the hypothesis speaker paired with them, with the seconds
    /// they agree on; unpaired speakers on either side are listed with `None`.
    pub mapping: Vec<(Option<String>, Option<String>, f64)>,
    pub reference_speakers: usize,
    pub hypothesis_speakers: usize,
}

impl DerScore {
    /// Missed, false-alarm and confused speaker-seconds together.
    pub fn error_seconds(&self) -> f64 {
        self.missed + self.false_alarm + self.confusion
    }
}

/// The collar ground truth is scored with: md-eval's customary quarter second, which
/// VoxConverse and most published Sortformer numbers use.
pub const TRUTH_COLLAR: f64 = 0.25;

/// An engine's speaker turns against a clip's ground truth, with `TRUTH_COLLAR`.
pub fn against_truth(truth: &[TruthTurn], hypothesis: &Diarization) -> DerScore {
    der(
        &from_truth(truth),
        &from_turns(&hypothesis.turns),
        &DerOptions {
            collar: TRUTH_COLLAR,
            region: None,
        },
    )
}

/// Frames as half-open ranges, per speaker, merged so a speaker overlapping themself counts once.
struct Speakers {
    names: Vec<String>,
    ranges: Vec<Vec<(usize, usize)>>,
}

fn frame(seconds: f64) -> usize {
    (seconds / FRAME).round().max(0.) as usize
}

fn speakers(segments: &[Segment]) -> Speakers {
    let mut by: BTreeMap<&str, Vec<(usize, usize)>> = BTreeMap::new();
    for s in segments {
        let (a, b) = (frame(s.t0), frame(s.t1));
        if b > a {
            by.entry(&s.speaker).or_default().push((a, b));
        }
    }
    let mut names = Vec::new();
    let mut ranges = Vec::new();
    for (name, mut r) in by {
        r.sort_unstable();
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(r.len());
        for (a, b) in r {
            match merged.last_mut() {
                Some(last) if a <= last.1 => last.1 = last.1.max(b),
                _ => merged.push((a, b)),
            }
        }
        names.push(name.to_owned());
        ranges.push(merged);
    }
    Speakers { names, ranges }
}

/// Speakers at each frame.
fn counts(speakers: &Speakers, frames: usize) -> Vec<u16> {
    let mut diff = vec![0i32; frames + 1];
    for ranges in &speakers.ranges {
        for &(a, b) in ranges {
            let (a, b) = (a.min(frames), b.min(frames));
            diff[a] += 1;
            diff[b] -= 1;
        }
    }
    let mut out = Vec::with_capacity(frames);
    let mut running = 0;
    for d in &diff[..frames] {
        running += d;
        out.push(running as u16);
    }
    out
}

/// Scored frames in [a, b), from a prefix sum of the scored mask.
fn scored_in(prefix: &[u32], a: usize, b: usize) -> u64 {
    let last = prefix.len() - 1;
    let (a, b) = (a.min(last), b.min(last));
    if b <= a {
        0
    } else {
        (prefix[b] - prefix[a]) as u64
    }
}

/// Scored frames both speakers talk in.
fn together(x: &[(usize, usize)], y: &[(usize, usize)], prefix: &[u32]) -> u64 {
    let (mut i, mut j, mut total) = (0, 0, 0);
    while i < x.len() && j < y.len() {
        let (a, b) = (x[i].0.max(y[j].0), x[i].1.min(y[j].1));
        if a < b {
            total += scored_in(prefix, a, b);
        }
        if x[i].1 < y[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    total
}

/// The diarization error rate of `hypothesis` against `reference`.
pub fn der(reference: &[Segment], hypothesis: &[Segment], options: &DerOptions) -> DerScore {
    let r = speakers(reference);
    let h = speakers(hypothesis);
    let end = |s: &Speakers| {
        s.ranges
            .iter()
            .filter_map(|r| r.last().map(|x| x.1))
            .max()
            .unwrap_or(0)
    };
    let mut frames = end(&r).max(end(&h));
    let mut scored = vec![true; frames];
    if let Some((start, stop)) = options.region {
        let (a, b) = (frame(start).min(frames), frame(stop).min(frames));
        scored[..a].fill(false);
        scored[b.max(a)..].fill(false);
        frames = frames.min(b.max(a));
        scored.truncate(frames);
    }
    if options.collar > 0. {
        let c = (options.collar / FRAME).round() as usize;
        for ranges in &r.ranges {
            for &(a, b) in ranges {
                for edge in [a, b] {
                    let (lo, hi) = (edge.saturating_sub(c).min(frames), (edge + c).min(frames));
                    scored[lo..hi].fill(false);
                }
            }
        }
    }
    let mut prefix = Vec::with_capacity(frames + 1);
    prefix.push(0u32);
    for s in &scored {
        prefix.push(prefix.last().unwrap() + u32::from(*s));
    }
    let (nr, nh) = (counts(&r, frames), counts(&h, frames));
    let (mut speech, mut missed, mut false_alarm, mut both) = (0u64, 0u64, 0u64, 0u64);
    for f in 0..frames {
        if !scored[f] {
            continue;
        }
        let (a, b) = (nr[f] as u64, nh[f] as u64);
        speech += a;
        missed += a.saturating_sub(b);
        false_alarm += b.saturating_sub(a);
        both += a.min(b);
    }
    let overlap: Vec<Vec<i64>> = r
        .ranges
        .iter()
        .map(|x| {
            h.ranges
                .iter()
                .map(|y| together(x, y, &prefix) as i64)
                .collect()
        })
        .collect();
    let pairing = assign_max(&overlap);
    let correct: u64 = pairing
        .iter()
        .enumerate()
        .filter_map(|(i, j)| j.map(|j| overlap[i][j] as u64))
        .sum();
    let confusion = both - correct;
    let mut mapping = Vec::new();
    for (i, j) in pairing.iter().enumerate() {
        mapping.push((
            Some(r.names[i].clone()),
            j.map(|j| h.names[j].clone()),
            j.map_or(0., |j| overlap[i][j] as f64 * FRAME),
        ));
    }
    for (j, name) in h.names.iter().enumerate() {
        if !pairing.contains(&Some(j)) {
            mapping.push((None, Some(name.clone()), 0.));
        }
    }
    let rate = |x: u64| x as f64 / speech.max(1) as f64;
    DerScore {
        scored_seconds: speech as f64 * FRAME,
        missed: missed as f64 * FRAME,
        false_alarm: false_alarm as f64 * FRAME,
        confusion: confusion as f64 * FRAME,
        der: rate(missed + false_alarm + confusion),
        missed_rate: rate(missed),
        false_alarm_rate: rate(false_alarm),
        confusion_rate: rate(confusion),
        mapping,
        reference_speakers: r.names.len(),
        hypothesis_speakers: h.names.len(),
    }
}

/// Per-frame agreement of two diarizations: the share of frames, up to the later one's end,
/// where both have the same set of speakers once `hypothesis` speakers are renamed by the
/// best pairing (silence agreeing with silence counts).
pub fn frame_agreement(reference: &[Segment], hypothesis: &[Segment]) -> f64 {
    let score = der(reference, hypothesis, &DerOptions::default());
    let rename: BTreeMap<&str, &str> = score
        .mapping
        .iter()
        .filter_map(|(r, h, _)| Some((h.as_deref()?, r.as_deref()?)))
        .collect();
    let renamed: Vec<Segment> = hypothesis
        .iter()
        .map(|s| Segment {
            speaker: rename
                .get(s.speaker.as_str())
                .map_or_else(|| format!("\u{0}{}", s.speaker), |r| r.to_string()),
            ..s.clone()
        })
        .collect();
    let (r, h) = (speakers(reference), speakers(&renamed));
    let frames = [&r, &h]
        .iter()
        .flat_map(|s| s.ranges.iter().filter_map(|x| x.last().map(|y| y.1)))
        .max()
        .unwrap_or(0);
    if frames == 0 {
        return 1.;
    }
    // Each frame's speakers as a set of bits, over the union of names.
    let mut names: Vec<&String> = r.names.iter().chain(&h.names).collect();
    names.sort();
    names.dedup();
    let bits = |s: &Speakers| {
        let mut v = vec![0u128; frames];
        for (name, ranges) in s.names.iter().zip(&s.ranges) {
            let bit = names.binary_search(&name).unwrap().min(127);
            for &(a, b) in ranges {
                for x in &mut v[a..b] {
                    *x |= 1 << bit;
                }
            }
        }
        v
    };
    let (a, b) = (bits(&r), bits(&h));
    a.iter().zip(&b).filter(|(x, y)| x == y).count() as f64 / frames as f64
}

/// The one-to-one pairing of rows with columns that maximizes the sum of `weight`: for each
/// row, its column, or `None` if there are more rows than columns or pairing it gains nothing.
/// The Hungarian algorithm, O(n³) in the larger side.
pub fn assign_max(weight: &[Vec<i64>]) -> Vec<Option<usize>> {
    let rows = weight.len();
    let cols = weight.first().map_or(0, Vec::len);
    let n = rows.max(cols);
    if n == 0 {
        return Vec::new();
    }
    let top = weight.iter().flatten().copied().max().unwrap_or(0).max(0);
    // Minimize top − weight on a square matrix padded with zeros' cost `top`.
    let cost = |i: usize, j: usize| -> i64 {
        if i < rows && j < cols {
            top - weight[i][j]
        } else {
            top
        }
    };
    // e-maxx: 1-based, u/v potentials, p[j] the row matched to column j.
    let inf = i64::MAX / 4;
    let (mut u, mut v) = (vec![0i64; n + 1], vec![0i64; n + 1]);
    let (mut p, mut way) = (vec![0usize; n + 1], vec![0usize; n + 1]);
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0;
        let mut minv = vec![inf; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0;
            for j in 1..=n {
                if !used[j] {
                    let cur = cost(i0 - 1, j - 1) - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut out = vec![None; rows];
    for (j, &row) in p.iter().enumerate().skip(1).take(n) {
        let (i, col) = (row - 1, j - 1);
        if i < rows && col < cols && weight[i][col] > 0 {
            out[i] = Some(col);
        }
    }
    out
}
