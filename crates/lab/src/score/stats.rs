//! Robust statistics, and whether one engine is really faster than another.
//!
//! Speed is measured in pairs: the reference and the candidate run the same clip back to back,
//! several times, so each pair shares the machine's state (clocks, temperature, caches). A pair
//! gives a log ratio ln(reference / candidate), positive when the candidate is faster. Runs of
//! one clip are correlated, so confidence comes from resampling whole clips, not runs.

use serde::{Deserialize, Serialize};

/// The `q` quantile (0..=1) of `values`, linearly interpolated between order statistics (R's
/// type 7, NumPy's default). NaNs are ignored; NaN when nothing is left.
pub fn quantile(values: &[f64], q: f64) -> f64 {
    let mut v: Vec<f64> = values.iter().copied().filter(|x| !x.is_nan()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    sorted_quantile(&v, q)
}

fn sorted_quantile(v: &[f64], q: f64) -> f64 {
    let h = (v.len() - 1) as f64 * q.clamp(0., 1.);
    let (lo, hi) = (h.floor() as usize, h.ceil() as usize);
    v[lo] + (h - lo as f64) * (v[hi] - v[lo])
}

/// NaN when `values` is empty.
pub fn median(values: &[f64]) -> f64 {
    quantile(values, 0.5)
}

/// The median absolute deviation from the median (unscaled: ×1.4826 estimates a normal σ).
pub fn mad(values: &[f64]) -> f64 {
    let m = median(values);
    median(&values.iter().map(|x| (x - m).abs()).collect::<Vec<_>>())
}

/// A small, fixed-seed generator (SplitMix64), so every report resamples identically.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in 0..n.
    pub fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }
}

/// Bootstrap resamples behind `PairedSpeed::ci95`.
pub const BOOTSTRAP: usize = 2000;
const SEED: u64 = 0x5eed_1ab0_0000_0001;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PairedSpeed {
    pub clips: usize,
    pub pairs: usize,
    /// Reference time over candidate time, the median over every pair: above 1, the candidate
    /// is faster.
    pub median_speedup: f64,
    /// A 95% bootstrap interval for `median_speedup`, resampling whole clips (pairs, when
    /// there's a single clip).
    pub ci95: (f64, f64),
    /// Two-sided Wilcoxon signed-rank p-value of the clips' median log ratios against zero
    /// (1 when no clip differs).
    pub wilcoxon_p: f64,
    /// The geometric mean over clips of each clip's median speedup: every clip weighs the same
    /// however many pairs it has.
    pub geomean_speedup_across_clips: f64,
}

/// Paired timings, per clip: (reference ms, candidate ms) for each pair. Pairs with a time
/// that isn't positive and finite are left out.
pub fn paired(clips: &[Vec<(f64, f64)>]) -> PairedSpeed {
    let logs: Vec<Vec<f64>> = clips
        .iter()
        .map(|pairs| {
            pairs
                .iter()
                .filter(|(r, c)| r.is_finite() && c.is_finite() && *r > 0. && *c > 0.)
                .map(|(r, c)| (r / c).ln())
                .collect::<Vec<f64>>()
        })
        .filter(|l| !l.is_empty())
        .collect();
    let all: Vec<f64> = logs.iter().flatten().copied().collect();
    if all.is_empty() {
        return PairedSpeed {
            median_speedup: 1.,
            ci95: (1., 1.),
            wilcoxon_p: 1.,
            geomean_speedup_across_clips: 1.,
            ..PairedSpeed::default()
        };
    }
    let per_clip: Vec<f64> = logs.iter().map(|l| median(l)).collect();
    let mut rng = Rng::new(SEED);
    let mut stats = Vec::with_capacity(BOOTSTRAP);
    let mut sample: Vec<f64> = Vec::with_capacity(all.len() * 2);
    for _ in 0..BOOTSTRAP {
        sample.clear();
        if logs.len() >= 2 {
            for _ in 0..logs.len() {
                sample.extend(&logs[rng.below(logs.len())]);
            }
        } else {
            for _ in 0..all.len() {
                sample.push(all[rng.below(all.len())]);
            }
        }
        stats.push(median_in_place(&mut sample));
    }
    stats.sort_by(f64::total_cmp);
    PairedSpeed {
        clips: logs.len(),
        pairs: all.len(),
        median_speedup: median(&all).exp(),
        ci95: (
            sorted_quantile(&stats, 0.025).exp(),
            sorted_quantile(&stats, 0.975).exp(),
        ),
        wilcoxon_p: wilcoxon(&per_clip).unwrap_or(1.),
        geomean_speedup_across_clips: (per_clip.iter().sum::<f64>() / per_clip.len() as f64).exp(),
    }
}

fn median_in_place(v: &mut [f64]) -> f64 {
    let n = v.len();
    let (_, upper, _) = v.select_nth_unstable_by(n / 2, f64::total_cmp);
    let upper = *upper;
    if n % 2 == 1 {
        upper
    } else {
        let lower = v[..n / 2].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (lower + upper) / 2.
    }
}

/// The speedup an A/A comparison (the same engine on both sides) appears to show, as a
/// fraction: the farthest from 1 of its median and its interval's ends, 0.03 for ±3%. A
/// measured speedup `s` is real when `s > 1 + floor` or `s < 1 / (1 + floor)`.
pub fn noise_floor(aa_pairs: &[Vec<(f64, f64)>]) -> f64 {
    let s = paired(aa_pairs);
    [s.median_speedup, s.ci95.0, s.ci95.1]
        .iter()
        .map(|x| x.ln().abs())
        .fold(0., f64::max)
        .exp()
        - 1.
}

/// Two-sided p-value of the Wilcoxon signed-rank test that `x` is centred on zero. Zeros are
/// dropped; exact below 26 values without ties, else the normal approximation with tie and
/// continuity corrections.
pub fn wilcoxon(x: &[f64]) -> Option<f64> {
    let mut v: Vec<f64> = x
        .iter()
        .copied()
        .filter(|d| d.is_finite() && d.abs() > 1e-12)
        .collect();
    let n = v.len();
    if n == 0 {
        return None;
    }
    v.sort_by(|a, b| a.abs().total_cmp(&b.abs()));
    // Average ranks over ties of |x|.
    let mut ranks = vec![0f64; n];
    let mut ties = Vec::new();
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && v[j + 1].abs() == v[i].abs() {
            j += 1;
        }
        let rank = (i + j + 2) as f64 / 2.;
        ranks[i..=j].fill(rank);
        if j > i {
            ties.push((j - i + 1) as f64);
        }
        i = j + 1;
    }
    let w: f64 = v
        .iter()
        .zip(&ranks)
        .filter(|(d, _)| **d > 0.)
        .map(|(_, r)| r)
        .sum();
    let nf = n as f64;
    if n <= 25 && ties.is_empty() {
        // Ways to reach each rank sum with a subset of 1..=n.
        let max = n * (n + 1) / 2;
        let mut ways = vec![0f64; max + 1];
        ways[0] = 1.;
        for k in 1..=n {
            for s in (k..=max).rev() {
                ways[s] += ways[s - k];
            }
        }
        let total = 2f64.powi(n as i32);
        let w = w.round() as usize;
        let below: f64 = ways[..=w].iter().sum::<f64>() / total;
        let above: f64 = ways[w..].iter().sum::<f64>() / total;
        return Some((2. * below.min(above)).min(1.));
    }
    let mean = nf * (nf + 1.) / 4.;
    let var = nf * (nf + 1.) * (2. * nf + 1.) / 24.
        - ties.iter().map(|t| t * t * t - t).sum::<f64>() / 48.;
    if var <= 0. {
        return Some(1.);
    }
    let z = ((w - mean).abs() - 0.5).max(0.) / var.sqrt();
    Some(erfc(z / std::f64::consts::SQRT_2).min(1.))
}

/// The complementary error function (Numerical Recipes' Chebyshev fit, relative error < 1.2e-7).
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1. / (1. + 0.5 * z);
    let r = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    if x >= 0. { r } else { 2. - r }
}
