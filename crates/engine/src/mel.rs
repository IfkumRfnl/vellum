//! Log-mel spectrograms, configured by a model's `stt.frontend.*` metadata, computed as
//! transcribe.cpp computes them (`src/transcribe-mel.cpp` there, as `arch/parakeet/model.cpp`
//! configures it), which is NeMo's `AudioToMelSpectrogramPreprocessor` at inference:
//!
//! 1. Pre-emphasis `y[i] = x[i] - 0.97·x[i-1]` (`y[0] = x[0]`), in f64. No dither: NeMo applies
//!    it only in training and the reference never does, whatever `stt.frontend.dither` says.
//! 2. Centering: `n_fft / 2` zeros on each side (Parakeet's `pad_mode = "constant"`), so there are
//!    `n / hop + 1` frames, frame `t` starting at padded sample `t·hop`.
//! 3. Each frame times a symmetric Hann window of `win_length` (`cos(2πk / (win - 1))`) centered
//!    in `n_fft` zeros, an f64 FFT, and the power `re² + im²` rounded to f32.
//! 4. A Slaney-scale, Slaney-normalized (librosa) triangular filterbank, accumulated in f64, then
//!    `ln(x + 2⁻²⁴)` rounded to f32.
//! 5. Per-feature normalization over the first `n / hop` frames (the last is the centering pad's
//!    artifact): subtract each bin's mean, divide by its unbiased standard deviation plus 1e-5,
//!    and zero the last frame.
//!
//! The reference runs its scalar path on Windows (no BLAS), so the sums here are taken in the
//! same order; the only difference left is the FFT's own f64 rounding, far below f32.

use crate::{Error, Result, cpu::Mat, gguf::Gguf};
use rayon::prelude::*;
use rustfft::{Fft, FftPlanner, num_complex::Complex64};
use std::{f64::consts::PI, sync::Arc};

/// NeMo's `log_zero_guard_value`, added to the mel power before the log.
const LOG_GUARD: f64 = 1. / 16_777_216.; // 2^-24
/// Added to each bin's standard deviation before dividing by it.
const NORM_EPS: f64 = 1e-5;

/// What happens to the log-mel after it's computed (`stt.frontend.normalize`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Normalize {
    /// Zero mean and unit (unbiased) variance per mel bin over the valid frames.
    PerFeature,
    /// Raw log-mel, with frames past the valid length zeroed (streaming checkpoints).
    None,
}

pub struct Mel {
    num_mels: usize,
    n_fft: usize,
    hop: usize,
    pre_emphasis: f64,
    normalize: Normalize,
    /// `n_fft` values: the Hann window centered in zeros.
    window: Vec<f64>,
    /// `num_mels` rows of `n_fft / 2 + 1` weights.
    filterbank: Vec<f32>,
    /// Per mel bin, the FFT bins `[lo, hi)` its filter covers, `lo` rounded down to a multiple
    /// of 4 as the reference does (its sums group four products at a time from there).
    bands: Vec<(usize, usize)>,
    fft: Arc<dyn Fft<f64>>,
}

impl Mel {
    pub fn from_gguf(file: &Gguf) -> Result<Mel> {
        let key = |name: &str| format!("stt.frontend.{name}");
        let missing = |name: &str| Error::Gguf(format!("no stt.frontend.{name}"));
        let int = |name: &str| {
            file.metadata_u64(&key(name))
                .map(|v| v as usize)
                .ok_or_else(|| missing(name))
        };
        // The reference reads these as f32; round the same way whatever the file stores.
        let real = |name: &str| {
            file.metadata_f64(&key(name))
                .map(|v| v as f32 as f64)
                .ok_or_else(|| missing(name))
        };
        // Where a file leaves a field out (Sortformer's has no type or band edges), the
        // reference's own defaults (`MelConfig`): a mel frontend over 0–8000 Hz.
        let real_or = |name: &str, default: f64| real(name).or(Ok::<f64, Error>(default));
        let text = |name: &str| file.metadata_str(&key(name)).ok_or_else(|| missing(name));

        let unsupported = |what: String| Err(Error::Unsupported(format!("{what} frontends")));
        let kind = file.metadata_str(&key("type")).unwrap_or("mel");
        if kind != "mel" {
            return unsupported(format!("\"{kind}\""));
        }
        if text("window")? != "hann" {
            return unsupported(format!("\"{}\"-window", text("window")?));
        }
        let normalize = match text("normalize")? {
            "per_feature" => Normalize::PerFeature,
            "none" => Normalize::None,
            other => return unsupported(format!("\"{other}\"-normalized")),
        };
        let sample_rate = int("sample_rate")?;
        let num_mels = int("num_mels")?;
        let n_fft = int("n_fft")?;
        let win_length = int("win_length")?;
        let hop = int("hop_length")?;
        if win_length > n_fft || hop == 0 || num_mels == 0 {
            return Err(Error::Gguf(format!(
                "frontend n_fft {n_fft}, win_length {win_length}, hop {hop}, {num_mels} mels"
            )));
        }

        let filterbank = slaney_filterbank(
            sample_rate,
            n_fft,
            num_mels,
            real_or("f_min", 0.)?,
            real_or("f_max", 8000.)?,
        );
        let n_freq = n_fft / 2 + 1;
        let bands = filterbank
            .chunks(n_freq)
            .map(|row| {
                let lo = row.iter().position(|&w| w != 0.);
                let hi = row.iter().rposition(|&w| w != 0.);
                match (lo, hi) {
                    (Some(lo), Some(hi)) => (lo - lo % 4, hi + 1),
                    _ => (0, 0),
                }
            })
            .collect();

        Ok(Mel {
            num_mels,
            n_fft,
            hop,
            pre_emphasis: real("pre_emphasis")?,
            normalize,
            window: hann_window(win_length, n_fft),
            filterbank,
            bands,
            fft: FftPlanner::new().plan_fft_forward(n_fft),
        })
    }

    /// Frames of features for `n_samples` of audio, counting the trailing centering-pad frame
    /// (`n / hop + 1`).
    pub fn frames(&self, n_samples: usize) -> usize {
        n_samples / self.hop + 1
    }

    /// Frames NeMo counts as real (`n / hop`): the reference drops the last one after dumping
    /// `enc.mel.in` and before the encoder.
    pub fn valid_frames(&self, n_samples: usize) -> usize {
        n_samples / self.hop
    }

    /// The features of 16 kHz audio: `num_mels` rows of one value per frame, `frames(n)` of
    /// them (the layout of the reference's `enc.mel.in` dump, padding frame included).
    pub fn compute(&self, pcm: &[f32]) -> Mat {
        let frames = self.frames(pcm.len());
        let mut out = self.log_mel(pcm, 0..frames);
        let valid = self.valid_frames(pcm.len());
        out.data
            .par_chunks_mut(frames)
            .for_each(|row| match self.normalize {
                Normalize::PerFeature => normalize_feature(row, valid),
                Normalize::None => row[valid.min(frames)..].fill(0.),
            });
        out
    }

    /// Frames `range` of the features, computed alone (exactly those columns of what
    /// [`compute`](Self::compute) gives): a long recording's features a window at a time,
    /// without holding all of them. Only for features normalized frame by frame, not over the
    /// whole recording (`normalize = "none"`); frames from the `n / hop`-th on are zero.
    pub fn compute_frames(&self, pcm: &[f32], range: std::ops::Range<usize>) -> Result<Mat> {
        if self.normalize != Normalize::None {
            return Err(Error::Unsupported(
                "windowed features normalized over the recording".into(),
            ));
        }
        let mut out = self.log_mel(pcm, range.clone());
        let valid = self.valid_frames(pcm.len());
        let first_zero = valid.clamp(range.start, range.end) - range.start;
        out.data
            .par_chunks_mut(range.len().max(1))
            .for_each(|row| row[first_zero..].fill(0.));
        Ok(out)
    }

    /// The unnormalized log-mel of frames `range`: `num_mels` rows of `range.len()` values.
    fn log_mel(&self, pcm: &[f32], range: std::ops::Range<usize>) -> Mat {
        let frames = range.len();
        let power = self.power_spectrum(pcm, range);
        let n_freq = self.n_fft / 2 + 1;
        let mut out = Mat::zeros(self.num_mels, frames);
        if frames == 0 {
            return out;
        }
        out.data
            .par_chunks_mut(frames)
            .zip(self.filterbank.par_chunks(n_freq))
            .zip(&self.bands)
            .for_each(|((row, weights), &(lo, hi))| {
                for (t, value) in row.iter_mut().enumerate() {
                    let spectrum = &power[t * n_freq..(t + 1) * n_freq];
                    *value = (mel_energy(weights, spectrum, lo, hi) + LOG_GUARD).ln() as f32;
                }
            });
        out
    }

    /// Pre-emphasis, centering, window and FFT: a row of `n_fft / 2 + 1` powers for each frame
    /// in `range`. The centered, pre-emphasized signal is read sample by sample (in f64), so
    /// no copy of the recording is made.
    fn power_spectrum(&self, pcm: &[f32], range: std::ops::Range<usize>) -> Vec<f32> {
        let pad = self.n_fft / 2;
        // Sample `j` of the zero-padded, pre-emphasized signal.
        let padded = |j: usize| -> f64 {
            match j.checked_sub(pad).filter(|&i| i < pcm.len()) {
                Some(0) => pcm[0] as f64,
                Some(i) => pcm[i] as f64 - self.pre_emphasis * pcm[i - 1] as f64,
                None => 0.,
            }
        };
        let n_freq = self.n_fft / 2 + 1;
        let mut power = vec![0f32; range.len() * n_freq];
        let scratch_len = self.fft.get_inplace_scratch_len();
        power.par_chunks_mut(n_freq).enumerate().for_each_init(
            || {
                let zero = Complex64::new(0., 0.);
                (vec![zero; self.n_fft], vec![zero; scratch_len])
            },
            |(buffer, scratch), (t, out)| {
                let start = (range.start + t) * self.hop;
                for (k, (b, &w)) in buffer.iter_mut().zip(&self.window).enumerate() {
                    *b = Complex64::new(padded(start + k) * w, 0.);
                }
                self.fft.process_with_scratch(buffer, scratch);
                for (p, c) in out.iter_mut().zip(buffer.iter()) {
                    *p = (c.re * c.re + c.im * c.im) as f32;
                }
            },
        );
        power
    }
}

/// One mel bin's energy in one frame, summed in f64 in the reference's order: groups of four
/// products from `lo`, then the remainder one by one.
fn mel_energy(weights: &[f32], power: &[f32], lo: usize, hi: usize) -> f64 {
    let product = |k: usize| weights[k] as f64 * power[k] as f64;
    let n_freq = power.len();
    let mut sum = 0.;
    let mut k = lo;
    while k < hi && k + 3 < n_freq {
        sum += product(k) + product(k + 1) + product(k + 2) + product(k + 3);
        k += 4;
    }
    while k < hi {
        sum += product(k);
        k += 1;
    }
    sum
}

/// NeMo's per-feature normalization of one mel bin over its first `valid` frames, in f64; the
/// frames after them (the centering pad's) become zero. Fewer than two valid frames have no
/// variance (the reference refuses or divides by zero there), so they give zeros.
fn normalize_feature(row: &mut [f32], valid: usize) {
    let (real, padding) = row.split_at_mut(valid.min(row.len()));
    padding.fill(0.);
    if real.len() < 2 {
        real.fill(0.);
        return;
    }
    let n = real.len() as f64;
    let mean = real.iter().map(|&v| v as f64).sum::<f64>() / n;
    let squares = real.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>();
    let scale = 1. / ((squares / (n - 1.)).sqrt() + NORM_EPS);
    for v in real {
        *v = ((*v as f64 - mean) * scale) as f32;
    }
}

/// `torch.hann_window(win, periodic=False)`, centered in `n_fft` zeros.
fn hann_window(win: usize, n_fft: usize) -> Vec<f64> {
    let mut window = vec![0.; n_fft];
    let offset = (n_fft - win) / 2;
    for k in 0..win {
        window[offset + k] = 0.5 - 0.5 * (2. * PI * k as f64 / (win - 1) as f64).cos();
    }
    window
}

/// librosa's `filters.mel(norm="slaney")` on the Slaney mel scale: `num_mels` triangles evenly
/// spaced in mel between `f_min` and `f_max`, each scaled to unit area, as `num_mels` rows of
/// `n_fft / 2 + 1` weights.
fn slaney_filterbank(
    sample_rate: usize,
    n_fft: usize,
    num_mels: usize,
    f_min: f64,
    f_max: f64,
) -> Vec<f32> {
    const LINEAR_HZ_PER_MEL: f64 = 200. / 3.;
    const LOG_START_HZ: f64 = 1000.;
    let log_start_mel = LOG_START_HZ / LINEAR_HZ_PER_MEL;
    let log_step = 6.4f64.ln() / 27.;
    let hz_to_mel = |hz: f64| {
        if hz < LOG_START_HZ {
            hz / LINEAR_HZ_PER_MEL
        } else {
            log_start_mel + (hz / LOG_START_HZ).ln() / log_step
        }
    };
    let mel_to_hz = |mel: f64| {
        if mel < log_start_mel {
            mel * LINEAR_HZ_PER_MEL
        } else {
            LOG_START_HZ * (log_step * (mel - log_start_mel)).exp()
        }
    };

    let n_freq = n_fft / 2 + 1;
    let (mel_min, mel_max) = (hz_to_mel(f_min), hz_to_mel(f_max));
    let edges: Vec<f64> = (0..num_mels + 2)
        .map(|m| mel_to_hz(mel_min + (mel_max - mel_min) * m as f64 / (num_mels + 1) as f64))
        .collect();

    let mut bank = vec![0f32; num_mels * n_freq];
    for (m, row) in bank.chunks_mut(n_freq).enumerate() {
        let (left, center, right) = (edges[m], edges[m + 1], edges[m + 2]);
        let area = 2. / (right - left);
        for (k, weight) in row.iter_mut().enumerate() {
            let hz = sample_rate as f64 * k as f64 / n_fft as f64;
            let rising = (hz - left) / (center - left);
            let falling = (right - hz) / (right - center);
            *weight = (rising.min(falling).max(0.) * area) as f32;
        }
    }
    bank
}
