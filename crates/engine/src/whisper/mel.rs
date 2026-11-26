//! Whisper's log-mel spectrogram, computed as transcribe.cpp computes it for this family (the
//! fp32 "mixed-radix" path of `src/transcribe-mel.cpp` there, configured by
//! `arch/whisper/weights.cpp`), which is `WhisperFeatureExtractor`:
//!
//! 1. Reflect padding by `n_fft / 2` on each side (no pre-emphasis, no dither), so there are
//!    `n / hop + 1` frames, frame `t` starting at padded sample `t·hop`.
//! 2. Each frame times the window the file carries (`frontend.window`, periodic Hann), then an
//!    f32 FFT: radix-2 halving down to odd lengths (400 → 25), which are naive DFTs, twiddles
//!    from an f32 table of `n_fft` cosines and sines. The power `re² + im²` in f32.
//! 3. The file's Slaney filterbank (`frontend.mel_filterbank`), accumulated in f64 in groups of
//!    four, then `log10(max(x, 1e-10))` rounded to f32.
//! 4. The last frame (the centering pad's) dropped; every value clamped to the maximum − 8 and
//!    mapped to `(x + 4) / 4`.
//!
//! Everything is in the reference's order of operations, so the result is the same to the bit
//! (the reference's compiler contracts no multiply-adds here).

use crate::{Error, Result, cpu::Mat, cpu::vector, gguf::Gguf};
use rayon::prelude::*;

pub struct LogMel {
    pub num_mels: usize,
    n_fft: usize,
    hop: usize,
    /// `n_fft` values.
    window: Vec<f32>,
    /// `num_mels` rows of `n_fft / 2 + 1` weights.
    filterbank: Vec<f32>,
    /// Per mel bin, the FFT bins `[lo, hi)` its filter covers, `lo` rounded down to a multiple
    /// of four (the reference's sums group four products at a time from there).
    bands: Vec<(usize, usize)>,
    /// `cos(2πi / n_fft)` and `sin(…)`, computed in f64 and rounded to f32.
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl LogMel {
    pub fn from_gguf(file: &Gguf) -> Result<LogMel> {
        let int = |name: &str| {
            file.metadata_u64(&format!("stt.frontend.{name}"))
                .map(|v| v as usize)
                .ok_or_else(|| Error::Gguf(format!("no stt.frontend.{name}")))
        };
        let num_mels = int("num_mels")?;
        let n_fft = int("n_fft")?;
        let hop = int("hop_length")?;
        if int("win_length")? != n_fft || n_fft.is_power_of_two() || hop == 0 {
            return Err(Error::Unsupported(format!(
                "Whisper frontends of n_fft {n_fft}, hop {hop}"
            )));
        }
        let n_freq = n_fft / 2 + 1;
        let window = vector(file, "frontend.window")?;
        let filterbank = vector(file, "frontend.mel_filterbank")?;
        if window.len() != n_fft || filterbank.len() != num_mels * n_freq {
            return Err(Error::Gguf(
                "frontend window or filterbank of the wrong size".into(),
            ));
        }
        let bands = filterbank
            .chunks(n_freq)
            .map(|row| {
                match (
                    row.iter().position(|&w| w != 0.),
                    row.iter().rposition(|&w| w != 0.),
                ) {
                    (Some(lo), Some(hi)) => (lo - lo % 4, hi + 1),
                    _ => (0, 0),
                }
            })
            .collect();
        let theta = |i: usize| 2. * std::f64::consts::PI * i as f64 / n_fft as f64;
        Ok(LogMel {
            num_mels,
            n_fft,
            hop,
            window,
            filterbank,
            bands,
            cos: (0..n_fft).map(|i| theta(i).cos() as f32).collect(),
            sin: (0..n_fft).map(|i| theta(i).sin() as f32).collect(),
        })
    }

    /// The features of 16 kHz audio (at least `n_fft / 2 + 1` samples): one row of `num_mels`
    /// values per frame, `n / hop` frames (the layout of the reference's `enc.mel.in`).
    pub fn compute(&self, pcm: &[f32]) -> Mat {
        let pad = self.n_fft / 2;
        let n = pcm.len();
        let mut padded = vec![0f32; n + 2 * pad];
        padded[pad..pad + n].copy_from_slice(pcm);
        for i in 0..pad {
            padded[i] = padded[2 * pad - i];
            padded[pad + n + i] = padded[pad + n - 2 - i];
        }

        let frames = n / self.hop + 1;
        let n_freq = self.n_fft / 2 + 1;
        let mut log_mel = Mat::zeros(frames, self.num_mels);
        log_mel
            .data
            .par_chunks_mut(self.num_mels)
            .enumerate()
            .for_each_init(
                || {
                    (
                        vec![0f32; 2 * self.n_fft],
                        vec![0f32; 8 * self.n_fft],
                        vec![0f32; n_freq],
                    )
                },
                |(input, output, power), (t, row)| {
                    let frame = &padded[t * self.hop..][..self.n_fft];
                    for ((x, &s), &w) in input.iter_mut().zip(frame).zip(&self.window) {
                        *x = s * w;
                    }
                    self.fft(input, self.n_fft, output);
                    for (k, p) in power.iter_mut().enumerate() {
                        let (re, im) = (output[2 * k], output[2 * k + 1]);
                        *p = re * re + im * im;
                    }
                    for (m, value) in row.iter_mut().enumerate() {
                        let weights = &self.filterbank[m * n_freq..][..n_freq];
                        let (lo, hi) = self.bands[m];
                        *value = mel_energy(weights, power, lo, hi).max(1e-10).log10() as f32;
                    }
                },
            );

        // Drop the centering pad's frame, then the dynamic-range clamp and scaling, in f64.
        let out_frames = frames - 1;
        log_mel.data.truncate(out_frames * self.num_mels);
        log_mel.rows = out_frames;
        let max = log_mel
            .data
            .iter()
            .fold(f64::NEG_INFINITY, |m, &v| m.max(v as f64));
        let floor = max - 8.;
        log_mel.map(|v| (((v as f64).max(floor) + 4.) / 4.) as f32);
        log_mel
    }

    /// The reference's `mixed_radix_fft_f32`: `input` holds `size` real values (and room for
    /// as many more, the even/odd split's scratch); the `size` complex results land
    /// interleaved at the start of `output` (`8·size` long, the rest the recursion's scratch).
    fn fft(&self, input: &mut [f32], size: usize, output: &mut [f32]) {
        if size == 1 {
            output[0] = input[0];
            output[1] = 0.;
            return;
        }
        if size % 2 == 1 {
            self.dft(&input[..size], output);
            return;
        }
        let half = size / 2;
        let (head, scratch) = input.split_at_mut(size);
        for i in 0..half {
            scratch[i] = head[2 * i];
        }
        // The halves' results land at 2·size and 3·size; each recursion's scratch runs on past
        // its result (the even one's over where the odd one's goes, before that is computed).
        let (out, rest) = output.split_at_mut(2 * size);
        self.fft(scratch, half, rest);
        for i in 0..half {
            scratch[i] = head[2 * i + 1];
        }
        let (even_fft, odd_fft) = rest.split_at_mut(size);
        self.fft(scratch, half, odd_fft);
        let step = self.n_fft / size;
        for k in 0..half {
            let w_re = self.cos[k * step];
            let w_im = -self.sin[k * step];
            let (re_odd, im_odd) = (odd_fft[2 * k], odd_fft[2 * k + 1]);
            let (re_even, im_even) = (even_fft[2 * k], even_fft[2 * k + 1]);
            out[2 * k] = re_even + w_re * re_odd - w_im * im_odd;
            out[2 * k + 1] = im_even + w_re * im_odd + w_im * re_odd;
            out[2 * (k + half)] = re_even - w_re * re_odd + w_im * im_odd;
            out[2 * (k + half) + 1] = im_even - w_re * im_odd - w_im * re_odd;
        }
    }

    /// The reference's `dft_naive_f32`, for odd lengths.
    fn dft(&self, input: &[f32], output: &mut [f32]) {
        let size = input.len();
        let stride = self.n_fft / size;
        for k in 0..size {
            let (mut re, mut im) = (0f32, 0f32);
            for (n, &x) in input.iter().enumerate() {
                let idx = (k * n * stride) % self.n_fft;
                re += x * self.cos[idx];
                im -= x * self.sin[idx];
            }
            output[2 * k] = re;
            output[2 * k + 1] = im;
        }
    }
}

/// One mel bin's energy in one frame, in f64 in the reference's order: groups of four products
/// from `lo`, then the rest one by one.
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
