//! The steps from a link to a transcript, as the worker reports them and the progress screen
//! tells them.
//!
//! The worker sends [`Update`]s. [`Tracker`] turns them into one overall fraction, a stage
//! track, an estimate of the time left and live statistics. Stages that report no fraction of
//! their own (opening the video, separating voices) advance on an estimate that eases toward
//! completion, so the meter never sits still.

use crate::settings::Settings;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Fetch,
    Captions,
    Download,
    Convert,
    Model,
    Transcribe,
    Speakers,
}

impl Stage {
    /// Share of the whole job, roughly as long as each step takes.
    fn weight(self) -> f32 {
        match self {
            Stage::Fetch => 6.0,
            Stage::Captions => 8.0,
            Stage::Download => 14.0,
            Stage::Convert => 4.0,
            Stage::Model => 6.0,
            Stage::Transcribe => 56.0,
            Stage::Speakers => 12.0,
        }
    }

    /// Seconds the stage usually takes, for stages that report no fraction.
    fn expected(self, audio: Option<f64>) -> f64 {
        let audio = audio.unwrap_or(600.0);
        match self {
            Stage::Fetch => 4.0,
            Stage::Captions => 2.0,
            Stage::Download => 3.0 + audio / 90.0,
            Stage::Convert => 1.0 + audio / 400.0,
            Stage::Model => 3.0,
            Stage::Transcribe => 5.0 + audio / 10.0,
            Stage::Speakers => 4.0 + audio / 25.0,
        }
    }
}

/// The stages a job with these settings expects to run.
pub fn plan(settings: &Settings) -> Vec<Stage> {
    let mut plan = vec![Stage::Fetch];
    if settings.model == "captions" || settings.official_captions {
        plan.push(Stage::Captions);
    }
    if settings.model != "captions" || settings.speakers {
        plan.extend([Stage::Download, Stage::Convert]);
    }
    if settings.model != "captions" {
        plan.extend([Stage::Model, Stage::Transcribe]);
    }
    if settings.speakers {
        plan.push(Stage::Speakers);
    }
    plan
}

/// What the worker reports.
#[derive(Clone, Debug, PartialEq)]
pub enum Update {
    /// The video's metadata arrived.
    Video {
        title: String,
        duration: Option<f64>,
    },
    /// A stage began.
    Stage(Stage),
    /// How far the current stage is, 0 to 1.
    Fraction(f32),
    /// The current stage moved on to a part that reports no fraction: estimate from here.
    Estimated,
    /// The device doing the speech recognition.
    Device(String),
    /// A stretch of speech was recognized: `words` more words, ending with `text`, up to
    /// `until` seconds into the audio.
    Heard {
        text: String,
        words: usize,
        until: f64,
    },
    /// A passing remark ("Switching to CPU…").
    Note(String),
}

/// Recognized speech the progress screen keeps on show.
const HEARD_CHUNKS: usize = 3;

pub struct Tracker {
    pub plan: Vec<Stage>,
    pub done: Vec<(Stage, f64)>,
    pub current: Option<(Stage, f64)>,
    /// The last reported fraction and when it came, with the pace and step between reports,
    /// so the meter glides between a slow stage's reports instead of jumping.
    reported: Option<(f32, f64)>,
    pace: f32,
    step: f32,
    pub title: Option<String>,
    pub duration: Option<f64>,
    pub device: Option<String>,
    pub note: String,
    pub words: usize,
    /// The latest recognized chunks, oldest first, numbered from the start of the job.
    pub heard: Vec<(usize, String)>,
    chunks: usize,
    pub heard_until: f64,
    /// The overall fraction on show, eased toward the target and never moving backwards.
    pub shown: f32,
    pub finished: Option<f64>,
    /// The latest time-left estimate and when it was made, so the readout counts down through
    /// steps that can't estimate instead of disappearing.
    estimate: Option<(f64, f64)>,
}

impl Tracker {
    pub fn new(plan: Vec<Stage>) -> Self {
        Self {
            plan,
            done: Vec::new(),
            current: None,
            reported: None,
            pace: 0.0,
            step: 0.0,
            title: None,
            duration: None,
            device: None,
            note: String::new(),
            words: 0,
            heard: Vec::new(),
            chunks: 0,
            heard_until: 0.0,
            shown: 0.0,
            finished: None,
            estimate: None,
        }
    }

    pub fn apply(&mut self, update: Update, now: f64) {
        match update {
            Update::Video { title, duration } => {
                self.title = Some(title);
                self.duration = duration.filter(|d| *d > 0.0);
            }
            Update::Stage(stage) => self.enter(stage, now),
            Update::Fraction(f) => {
                let f = f.clamp(0.0, 1.0);
                if let Some((last, at)) = self.reported
                    && f > last
                    && now > at
                {
                    let pace = (f - last) / (now - at) as f32;
                    self.pace = if self.pace > 0.0 {
                        self.pace * 0.6 + pace * 0.4
                    } else {
                        pace
                    };
                    self.step = f - last;
                }
                self.reported = Some((f, now));
            }
            Update::Estimated => {
                self.reported = None;
                if let Some((_, since)) = &mut self.current {
                    *since = now;
                }
            }
            Update::Device(device) => self.device = Some(device),
            Update::Note(note) => self.note = note,
            Update::Heard { text, words, until } => {
                self.words += words;
                self.heard_until = until;
                self.heard.push((self.chunks, text.trim().to_owned()));
                self.chunks += 1;
                if self.heard.len() > HEARD_CHUNKS {
                    self.heard.remove(0);
                }
            }
        }
    }

    fn enter(&mut self, stage: Stage, now: f64) {
        if self.current.is_some_and(|(s, _)| s == stage) {
            return;
        }
        self.complete_current(now);
        if let Some(at) = self.plan.iter().position(|s| *s == stage) {
            // Stages the job skipped on its way here (captions found, no transcription needed).
            let done = &self.done;
            let mut index = 0;
            self.plan.retain(|s| {
                let keep = index >= at || done.iter().any(|(d, _)| d == s);
                index += 1;
                keep
            });
        } else {
            let after = self
                .plan
                .iter()
                .rposition(|s| self.done.iter().any(|(d, _)| d == s))
                .map_or(0, |i| i + 1);
            self.plan.insert(after, stage);
        }
        self.current = Some((stage, now));
        self.reported = None;
        self.pace = 0.0;
        self.step = 0.0;
        self.note.clear();
    }

    fn complete_current(&mut self, now: f64) {
        if let Some((stage, _)) = self.current.take()
            && !self.done.iter().any(|(s, _)| *s == stage)
        {
            self.done.push((stage, now));
        }
    }

    pub fn finish(&mut self, now: f64) {
        self.complete_current(now);
        self.plan.retain(|s| self.done.iter().any(|(d, _)| d == s));
        self.finished = Some(now);
    }

    pub fn is_done(&self, stage: Stage) -> bool {
        self.done.iter().any(|(s, _)| *s == stage)
    }

    /// The current stage's fraction: reported, or estimated from how long it has run.
    pub fn stage_fraction(&self, now: f64) -> f32 {
        let Some((stage, since)) = self.current else {
            return 0.0;
        };
        match self.reported {
            Some((f, at)) => {
                let glide = (self.pace * (now - at).max(0.0) as f32).min(self.step * 0.95);
                (f + glide).min(1.0)
            }
            None => {
                let expected = stage.expected(self.duration);
                (0.92 * (1.0 - (-(now - since).max(0.0) / expected).exp())) as f32
            }
        }
    }

    /// The overall fraction the meter is heading for.
    pub fn target(&self, now: f64) -> f32 {
        if self.finished.is_some() {
            return 1.0;
        }
        let total: f32 = self.plan.iter().map(|s| s.weight()).sum();
        if total <= 0.0 {
            return 0.0;
        }
        let done: f32 = self
            .plan
            .iter()
            .filter(|s| self.is_done(**s))
            .map(|s| s.weight())
            .sum();
        let current = self
            .current
            .map_or(0.0, |(s, _)| s.weight() * self.stage_fraction(now));
        ((done + current) / total).clamp(0.0, 0.995)
    }

    /// Moves the meter toward its target. True when what's on show changed.
    pub fn tick(&mut self, now: f64, dt: f64) -> bool {
        if let Some(left) = self.remaining(now) {
            self.estimate = Some((now, left));
        }
        let target = self.target(now);
        let before = self.shown;
        let k = 1.0 - (-dt.max(0.0) * if self.finished.is_some() { 7.0 } else { 4.5 }).exp();
        let eased = self.shown + (target - self.shown) * k as f32;
        self.shown = if (target - eased).abs() < 0.0005 {
            target
        } else {
            eased
        }
        .max(self.shown);
        (self.shown - before).abs() > 1e-5
    }

    /// Seconds until the transcript is ready, from the latest estimate, counting down.
    pub fn time_left(&self, now: f64) -> Option<f64> {
        let (at, left) = self.estimate?;
        Some((left - (now - at)).max(1.0))
    }

    /// Seconds until the transcript is ready, once the current stage can say.
    fn remaining(&self, now: f64) -> Option<f64> {
        let (stage, since) = self.current?;
        let fraction = f64::from(self.reported?.0);
        let elapsed = now - since;
        if fraction < 0.03 || elapsed < 2.0 {
            return None;
        }
        let mut left = elapsed * (1.0 - fraction) / fraction;
        let after = self.plan.iter().skip_while(|s| **s != stage).skip(1);
        for later in after {
            left += match later {
                // Recognition runs at the measured speed.
                Stage::Transcribe => self.duration.unwrap_or(600.0) / 12.0,
                s => s.expected(self.duration),
            };
        }
        Some(left)
    }
}
