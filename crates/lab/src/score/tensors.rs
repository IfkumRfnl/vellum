//! How close two engines' tensors are, checkpoint by checkpoint, from their dumps.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::Path};
use vellum_harness::dump::{self, Sidecar};

/// Below this reference norm, relative error is taken against this instead, so an all-zero
/// tensor doesn't divide by zero.
const NORM_FLOOR: f64 = 1e-12;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TensorDiff {
    /// Elements compared: the shorter side's count.
    pub elements: usize,
    pub reference_len: usize,
    pub candidate_len: usize,
    /// NaNs and infinities on each side.
    pub reference_nonfinite: usize,
    pub candidate_nonfinite: usize,
    /// Positions where one side is finite and the other not, or both nonfinite but different
    /// (NaN against infinity, +∞ against −∞). Excluded from the statistics below.
    pub nonfinite_mismatches: usize,
    pub max_abs: f64,
    pub p99_abs: f64,
    pub p999_abs: f64,
    pub mean_abs: f64,
    /// ‖candidate − reference‖ / max(‖reference‖, floor).
    pub rel_l2: f64,
    /// 1 when both are zero; 0 when only one is.
    pub cosine: f64,
    /// 10·log10(‖reference‖² / ‖Δ‖²); `None` when identical.
    pub snr_db: Option<f64>,
    /// The element with the largest |Δ|, and its values on each side.
    pub worst: Option<(usize, f32, f32)>,
}

impl TensorDiff {
    /// Whether it's within `tolerance` (relative L2) with no new NaNs or infinities and the same
    /// length.
    pub fn within(&self, tolerance: f64) -> bool {
        self.rel_l2 <= tolerance
            && self.nonfinite_mismatches == 0
            && self.reference_len == self.candidate_len
    }
}

/// `candidate` against `reference`, element by element.
pub fn compare(reference: &[f32], candidate: &[f32]) -> TensorDiff {
    let n = reference.len().min(candidate.len());
    let mut d = TensorDiff {
        elements: n,
        reference_len: reference.len(),
        candidate_len: candidate.len(),
        reference_nonfinite: reference.iter().filter(|v| !v.is_finite()).count(),
        candidate_nonfinite: candidate.iter().filter(|v| !v.is_finite()).count(),
        ..TensorDiff::default()
    };
    let (mut rr, mut cc, mut rc, mut dd, mut sum_abs) = (0f64, 0f64, 0f64, 0f64, 0f64);
    let mut abs: Vec<f32> = Vec::with_capacity(n);
    let mut worst: Option<(usize, f64)> = None;
    for (i, (&r, &c)) in reference[..n].iter().zip(&candidate[..n]).enumerate() {
        if !(r.is_finite() && c.is_finite()) {
            let same = (r.is_nan() && c.is_nan()) || r == c;
            d.nonfinite_mismatches += usize::from(!same);
            continue;
        }
        let (r, c) = (r as f64, c as f64);
        let delta = c - r;
        rr += r * r;
        cc += c * c;
        rc += r * c;
        dd += delta * delta;
        let a = delta.abs();
        sum_abs += a;
        abs.push(a as f32);
        if worst.is_none_or(|(_, w)| a > w) {
            worst = Some((i, a));
        }
    }
    d.max_abs = worst.map_or(0., |w| w.1);
    d.worst = worst.map(|(i, _)| (i, reference[i], candidate[i]));
    d.mean_abs = sum_abs / abs.len().max(1) as f64;
    d.p99_abs = select(&mut abs, 0.99);
    d.p999_abs = select(&mut abs, 0.999);
    d.rel_l2 = dd.sqrt() / rr.sqrt().max(NORM_FLOOR);
    d.cosine = match (rr > 0., cc > 0.) {
        (true, true) => rc / (rr.sqrt() * cc.sqrt()),
        (false, false) => 1.,
        _ => 0.,
    };
    d.snr_db = (dd > 0.).then(|| 10. * (rr.max(f64::MIN_POSITIVE) / dd).log10());
    d
}

/// The `q` quantile (nearest rank) of `values`, reordering them.
fn select(values: &mut [f32], q: f64) -> f64 {
    if values.is_empty() {
        return 0.;
    }
    let k = ((values.len() as f64 * q).ceil() as usize).clamp(1, values.len()) - 1;
    let (_, v, _) = values.select_nth_unstable_by(k, f32::total_cmp);
    *v as f64
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LogitRow {
    pub row: usize,
    pub reference_top1: usize,
    pub candidate_top1: usize,
    /// The reference's top-1 logit minus its top-2: near zero is a near-tie either engine
    /// could break either way; large is a real disagreement.
    pub reference_margin: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LogitDiff {
    pub rows: usize,
    /// Rows with a NaN on either side, left out of everything below.
    pub nonfinite_rows: usize,
    /// Share of rows whose argmax agrees.
    pub top1_agreement: f64,
    /// Mean KL(reference ‖ candidate) of the softmaxes, in nats. A candidate probability of
    /// zero where the reference's isn't counts as e⁻¹⁰⁰, to stay finite.
    pub mean_kl: f64,
    pub max_kl: f64,
    pub disagreeing: usize,
    /// The largest reference margin among disagreeing rows.
    pub max_disagreeing_margin: f64,
    /// The disagreeing rows, the first `LOGIT_ROWS`.
    pub disagreements: Vec<LogitRow>,
}

/// The most disagreeing rows listed in a `LogitDiff`.
pub const LOGIT_ROWS: usize = 64;

fn top2(row: &[f32]) -> (usize, f64) {
    let (mut best, mut second) = ((0usize, f64::NEG_INFINITY), f64::NEG_INFINITY);
    for (i, &v) in row.iter().enumerate() {
        let v = v as f64;
        if v > best.1 {
            second = best.1;
            best = (i, v);
        } else if v > second {
            second = v;
        }
    }
    let margin = if second.is_finite() {
        best.1 - second
    } else {
        f64::MAX
    };
    (best.0, margin)
}

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let max = row
        .iter()
        .map(|&v| v as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return vec![f64::NEG_INFINITY; row.len()];
    }
    let lse = max
        + row
            .iter()
            .map(|&v| (v as f64 - max).exp())
            .sum::<f64>()
            .ln();
    row.iter().map(|&v| v as f64 - lse).collect()
}

/// Rows of logits (decoder steps × vocabulary, frames × classes) compared as distributions.
pub fn compare_logits(reference: &[f32], candidate: &[f32], rows: usize, cols: usize) -> LogitDiff {
    let mut d = LogitDiff {
        rows,
        ..LogitDiff::default()
    };
    let (mut agree, mut kl_sum, mut counted) = (0usize, 0f64, 0usize);
    for row in 0..rows {
        let range = row * cols..(row + 1) * cols;
        let (Some(r), Some(c)) = (reference.get(range.clone()), candidate.get(range)) else {
            break;
        };
        if r.iter().chain(c).any(|v| v.is_nan()) {
            d.nonfinite_rows += 1;
            continue;
        }
        counted += 1;
        let ((rt, margin), (ct, _)) = (top2(r), top2(c));
        if rt == ct {
            agree += 1;
        } else {
            d.disagreeing += 1;
            d.max_disagreeing_margin = d.max_disagreeing_margin.max(margin);
            if d.disagreements.len() < LOGIT_ROWS {
                d.disagreements.push(LogitRow {
                    row,
                    reference_top1: rt,
                    candidate_top1: ct,
                    reference_margin: margin,
                });
            }
        }
        let (lp, lq) = (log_softmax(r), log_softmax(c));
        let kl: f64 = lp
            .iter()
            .zip(&lq)
            .filter(|(p, _)| p.is_finite())
            .map(|(p, q)| p.exp() * (p - q.max(-100.)))
            .sum::<f64>()
            .max(0.);
        kl_sum += kl;
        d.max_kl = d.max_kl.max(kl);
    }
    d.top1_agreement = agree as f64 / counted.max(1) as f64;
    d.mean_kl = kl_sum / counted.max(1) as f64;
    d
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Checkpoint {
    /// The sidecar's name.
    pub name: String,
    pub stage: Option<String>,
    pub shape: Vec<usize>,
    pub tolerance: f64,
    pub diff: TensorDiff,
    /// For tensors named like logits: the last dimension as classes, the rest as rows.
    pub logits: Option<LogitDiff>,
    pub within: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DumpComparison {
    /// Every checkpoint both dumped with the same shape, in the reference's write order.
    pub checkpoints: Vec<Checkpoint>,
    /// In the reference only.
    pub missing: Vec<String>,
    /// In the candidate only.
    pub extra: Vec<String>,
    /// Same name, different shapes: (name, reference shape, candidate shape).
    pub shape_mismatches: Vec<(String, Vec<usize>, Vec<usize>)>,
    /// The first checkpoint, in compute order, outside its tolerance: where the engines part.
    pub first_parted: Option<String>,
}

impl DumpComparison {
    /// Every checkpoint as a table row, then what didn't pair up.
    pub fn markdown(&self) -> String {
        let mut t = super::table::Table::new([
            "checkpoint",
            "shape",
            "rel L2",
            "tolerance",
            "max |Δ|",
            "p99.9 |Δ|",
            "cosine",
            "SNR dB",
            "top-1",
            "",
        ]);
        for c in &self.checkpoints {
            let d = &c.diff;
            t.row([
                c.name.clone(),
                format!("{:?}", c.shape),
                format!("{:.2e}", d.rel_l2),
                format!("{:.0e}", c.tolerance),
                format!("{:.2e}", d.max_abs),
                format!("{:.2e}", d.p999_abs),
                format!("{:.6}", d.cosine),
                d.snr_db.map_or("∞".into(), |s| format!("{s:.1}")),
                c.logits.as_ref().map_or(String::new(), |l| {
                    format!("{:.2}%", 100. * l.top1_agreement)
                }),
                match (c.within, d.nonfinite_mismatches) {
                    (true, _) => "ok".into(),
                    (false, 0) => "PARTS".into(),
                    (false, n) => format!("PARTS ({n} NaN/∞)"),
                },
            ]);
        }
        let mut out = t.markdown();
        if let Some(name) = &self.first_parted {
            out.push_str(&format!(
                "
First outside tolerance: `{name}`.
"
            ));
        }
        let list = |names: &[String]| {
            names
                .iter()
                .map(|n| format!("`{n}`"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        if !self.missing.is_empty() {
            out.push_str(&format!(
                "
Only the reference dumped: {}.
",
                list(&self.missing)
            ));
        }
        if !self.extra.is_empty() {
            out.push_str(&format!(
                "
Only the candidate dumped: {}.
",
                list(&self.extra)
            ));
        }
        for (name, r, c) in &self.shape_mismatches {
            out.push_str(&format!(
                "
Shapes differ for `{name}`: {r:?} against {c:?}.
"
            ));
        }
        out
    }
}

/// A checkpoint's allowed relative L2 error: one number for all, or a function of the sidecar
/// (per-stage tolerances calibrated from reference CPU-against-GPU spreads).
pub trait Tolerance {
    fn of(&self, sidecar: &Sidecar) -> f64;
}

impl Tolerance for f64 {
    fn of(&self, _: &Sidecar) -> f64 {
        *self
    }
}

impl<F: Fn(&Sidecar) -> f64> Tolerance for F {
    fn of(&self, sidecar: &Sidecar) -> f64 {
        self(sidecar)
    }
}

/// Checkpoints holding rows of scores over a vocabulary: Whisper's logits, Parakeet's joint
/// network output (log-probabilities over tokens and blank).
fn is_logits(name: &str) -> bool {
    let name = name.to_lowercase();
    name.contains("logit") || name.contains("joint")
}

/// The candidate's dump against the reference's, tensors paired by sidecar name (the n-th of
/// a name with the n-th), in the reference's write order. `tolerance` gives each checkpoint's
/// allowed relative L2 error.
pub fn compare_dirs(
    reference_dir: &Path,
    candidate_dir: &Path,
    tolerance: impl Tolerance,
) -> Result<DumpComparison> {
    let reference = dump::list(reference_dir)?;
    let candidate = dump::list(candidate_dir)?;
    let mut pool: HashMap<&str, Vec<&dump::Dumped>> = HashMap::new();
    for c in candidate.iter().rev() {
        pool.entry(&c.sidecar.name).or_default().push(c);
    }
    let mut out = DumpComparison::default();
    for r in &reference {
        let name = &r.sidecar.name;
        let Some(c) = pool.get_mut(name.as_str()).and_then(Vec::pop) else {
            out.missing.push(name.clone());
            continue;
        };
        if r.sidecar.shape != c.sidecar.shape {
            out.shape_mismatches.push((
                name.clone(),
                r.sidecar.shape.clone(),
                c.sidecar.shape.clone(),
            ));
            continue;
        }
        let a = r.load().with_context(|| format!("reference {name}"))?;
        let b = c.load().with_context(|| format!("candidate {name}"))?;
        let diff = compare(&a, &b);
        let logits = (is_logits(name) && !r.sidecar.shape.is_empty()).then(|| {
            let cols = *r.sidecar.shape.last().unwrap();
            compare_logits(&a, &b, a.len() / cols.max(1), cols)
        });
        let tolerance = tolerance.of(&r.sidecar);
        let within = diff.within(tolerance);
        if !within && out.first_parted.is_none() {
            out.first_parted = Some(name.clone());
        }
        out.checkpoints.push(Checkpoint {
            name: name.clone(),
            stage: r.sidecar.stage.clone(),
            shape: r.sidecar.shape.clone(),
            tolerance,
            diff,
            logits,
            within,
        });
    }
    let mut extra: Vec<&dump::Dumped> = pool.into_values().flatten().collect();
    extra.sort_by(|a, b| a.written.cmp(&b.written).then(a.key.cmp(&b.key)));
    out.extra = extra.into_iter().map(|d| d.sidecar.name.clone()).collect();
    Ok(out)
}
