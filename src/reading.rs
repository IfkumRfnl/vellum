//! How far through the transcript you are, and how fast you read.
//!
//! Progress comes from the scroll position. Reading speed comes from where the page rests: you
//! read while it's still and scroll on in short bursts, so the words between one resting place
//! and the next, over the time between them, are a reading pace. Jumps, skims, scrolling back
//! and long breaks don't count. The estimate starts from your speed on earlier transcripts and
//! moves toward what it measures as evidence builds.

/// Seconds of steady reading worth as much as the prior speed.
const PRIOR_WEIGHT: f64 = 90.0;
/// A step faster than this is skimming or jumping, not reading.
const SKIM_WPM: f64 = 700.0;
/// A pause longer than this is a break, not reading time.
const BREAK: f64 = 75.0;
/// Stillness after a scroll event that makes its position a resting place.
const SETTLE: f64 = 0.5;

pub struct Reading {
    pub words: usize,
    pub fraction: f32,
    prior_wpm: f64,
    measured_words: f64,
    measured_secs: f64,
    /// The latest scroll event and the latest resting place, as (seconds, words in).
    last: Option<(f64, f64)>,
    rest: Option<(f64, f64)>,
}

impl Reading {
    pub fn new(words: usize, prior_wpm: f32) -> Self {
        Self {
            words,
            fraction: 0.0,
            prior_wpm: f64::from(prior_wpm),
            measured_words: 0.0,
            measured_secs: 0.0,
            last: None,
            rest: None,
        }
    }

    /// A scroll position at `now` seconds: `offset` of `max` points scrolled.
    pub fn observe(&mut self, now: f64, offset: f32, max: f32) {
        self.fraction = if max <= 1.0 {
            1.0
        } else {
            (offset / max).clamp(0.0, 1.0)
        };
        let at = f64::from(self.fraction) * self.words as f64;
        if let Some((then, was)) = self.last
            && now - then > SETTLE
        {
            if let Some((start, from)) = self.rest {
                let (dt, dw) = (then - start, was - from);
                if dt > 0.0 && dw > 0.0 && dt < BREAK && dw / (dt / 60.0) < SKIM_WPM {
                    self.measured_words += dw;
                    self.measured_secs += dt;
                }
            }
            self.rest = Some((then, was));
        }
        self.last = Some((now, at));
    }

    /// Words per minute: the prior, pulled toward the measured pace by the reading time behind it.
    pub fn wpm(&self) -> f64 {
        let measured = if self.measured_secs > 0.0 {
            self.measured_words / (self.measured_secs / 60.0)
        } else {
            self.prior_wpm
        };
        let k = self.measured_secs / (self.measured_secs + PRIOR_WEIGHT);
        (self.prior_wpm * (1.0 - k) + measured * k).clamp(80.0, 900.0)
    }

    /// Whether there's enough reading behind the estimate to remember it.
    pub fn confident(&self) -> bool {
        self.measured_secs >= 45.0 && self.measured_words >= 120.0
    }

    pub fn words_read(&self) -> usize {
        (f64::from(self.fraction) * self.words as f64).round() as usize
    }

    pub fn words_left(&self) -> usize {
        self.words - self.words_read().min(self.words)
    }

    pub fn seconds_left(&self) -> f64 {
        self.words_left() as f64 / self.wpm() * 60.0
    }
}
