//! A recording at a time on the GPU: windows' mel features made on a thread of their own, the
//! encoder taking them a batch at a time, and the decoder decoding every window that has a
//! slot in lockstep, a token per step for all of them at once.
//!
//! Each window keeps its decoder slot from its first pass's encoding until it's done: its
//! passes (a pass that ends early on a timestamp pair is followed by one from there, encoded
//! into the same slot) and its attempts (a pass whose greedy decoding fails the thresholds is
//! tried again at the next temperature in the same slot, the GPU then reading that slot's
//! logits back each step so its tokens are sampled on the CPU with the window's generator).
//! While every decoding window is greedy, steps go to the GPU sixteen at a time, two runs in
//! flight, and only the state and choices come back; while one samples, every step is a round
//! trip (the greedy windows still advance a step each).
//!
//! Without a language, the app's rule holds: the first window with any text decides it for the
//! rest. Windows are encoded ahead regardless, but only the first unfinished one decodes (and
//! detects its language) until it has, so each window's result is exactly what
//! [`Whisper::run`] gives it alone: nothing a window computes depends on the others in its
//! batch, or on the slot it's in.

use super::{
    Attempt, Backend, Choices, MAX_NEW_TOKENS, Whisper, Window, decoding::Rng, gpu::GpuWhisper,
    gpu::decoder::Mode, gpu::decoder::Pending, ms, temperatures,
};
use crate::{Error, NoTrace, Result, RunOptions, Timings, Transcript, cpu::Mat};
use std::{collections::VecDeque, sync::mpsc, time::Instant};

/// Greedy steps per submit while every decoding window is greedy, two submits in flight.
const STEPS: usize = 16;

/// Where a window is.
enum Stage {
    /// Its next pass (from `seek`) waits for the encoder.
    Waiting,
    /// Encoded into its slot; its attempts wait for the language.
    Encoded,
    /// An attempt in progress: greedy on the GPU, or sampled here.
    Decoding {
        tier: usize,
        epoch: u64,
        sampled: Option<Sampled>,
    },
    Done,
}

/// A sampling attempt's choices so far.
struct Sampled {
    temperature: f32,
    choices: Choices,
    generated: Vec<i32>,
}

struct Job<'a> {
    /// Taken when the window's transcript is handed on.
    window: Option<Window<'a>>,
    mel: Mat,
    seek: usize,
    language: Option<i32>,
    slot: Option<usize>,
    /// Prompt positions its slot's cache holds for this pass (language detection's first).
    primed: usize,
    stage: Stage,
    no_speech: f32,
    timings: Timings,
    started: Option<Instant>,
}

impl<'a> Job<'a> {
    fn window(&mut self) -> &mut Window<'a> {
        self.window
            .as_mut()
            .expect("the window until it's handed on")
    }
}

impl Whisper {
    /// [`run_each`](Self::run_each) on the GPU (see the module's doc).
    pub(super) fn pipeline(
        &self,
        windows: &[&[f32]],
        options: &RunOptions,
        stop: &(dyn Fn() -> bool + Sync),
        each: &mut dyn FnMut(usize, Transcript) -> Result<()>,
    ) -> Result<()> {
        let Backend::Gpu(g) = &self.backend else {
            unreachable!("the pipeline is the GPU's")
        };
        let _exclusive = self.gpu_runs.lock().unwrap_or_else(|e| e.into_inner());
        let mut language = self.language_token(options)?;
        let slots = g.decoder.slots();
        std::thread::scope(|s| -> Result<()> {
            let (mel_tx, mel_rx) = mpsc::sync_channel::<(Mat, f64)>(slots + 2);
            s.spawn(move || {
                for pcm in windows {
                    if stop() {
                        break;
                    }
                    let start = Instant::now();
                    let mel = self.features(pcm);
                    if mel_tx.send((mel, ms(start))).is_err() {
                        break;
                    }
                }
            });
            let mut jobs: Vec<Job> = Vec::with_capacity(windows.len());
            let mut free: Vec<usize> = (0..slots).rev().collect();
            let mut epochs = vec![0u64; slots];
            let mut in_flight: VecDeque<(Pending, Vec<u64>)> = VecDeque::new();
            let mut delivered = 0;
            let mut run = || -> Result<()> {
                while delivered < windows.len() {
                    if stop() {
                        return Err(Error::Cancelled);
                    }
                    // Mel features as they come, waiting for one only when nothing else can
                    // move.
                    let take = |mel: Mat, mel_ms: f64, jobs: &mut Vec<Job>| {
                        jobs.push(Job {
                            window: Some(Window {
                                // Nothing traces a pipelined window (a NoTrace is zero-sized:
                                // leaking one allocates nothing).
                                trace: Box::leak(Box::new(NoTrace)),
                                rng: Rng::new(0x5EED),
                                detected: None,
                                text_ids: Vec::new(),
                                words: Vec::new(),
                                tokens: Vec::new(),
                            }),
                            mel,
                            seek: 0,
                            language: None,
                            slot: None,
                            primed: 0,
                            stage: Stage::Waiting,
                            no_speech: 0.,
                            timings: Timings {
                                mel_ms,
                                ..Timings::default()
                            },
                            started: None,
                        })
                    };
                    while let Ok((mel, mel_ms)) = mel_rx.try_recv() {
                        take(mel, mel_ms, &mut jobs);
                    }
                    let busy = !in_flight.is_empty()
                        || jobs.iter().any(|j| match j.stage {
                            Stage::Decoding { .. } | Stage::Encoded => true,
                            Stage::Waiting => j.slot.is_some() || !free.is_empty(),
                            Stage::Done => false,
                        });
                    if !busy && jobs.len() < windows.len() {
                        match mel_rx.recv() {
                            Ok((mel, mel_ms)) => take(mel, mel_ms, &mut jobs),
                            Err(_) => return Err(Error::Cancelled),
                        }
                    }

                    // The encoder: the waiting windows' next passes, a batch at a time, the
                    // earliest first, each into its own slot or a free one.
                    let mut batch = Vec::new();
                    for (i, job) in jobs.iter_mut().enumerate() {
                        if batch.len() == g.encoder.max_batch() {
                            break;
                        }
                        if !matches!(job.stage, Stage::Waiting) {
                            continue;
                        }
                        if job.slot.is_none() {
                            match free.pop() {
                                Some(slot) => job.slot = Some(slot),
                                None => break,
                            }
                        }
                        batch.push(i);
                    }
                    if !batch.is_empty() {
                        let start = Instant::now();
                        let chunks: Vec<Mat> =
                            batch.iter().map(|&i| self.pass_chunk(&jobs[i])).collect();
                        let pairs: Vec<(&Mat, usize)> = batch
                            .iter()
                            .zip(&chunks)
                            .map(|(&i, c)| (c, jobs[i].slot.expect("a slot")))
                            .collect();
                        g.encode(&pairs)?;
                        let each_ms = ms(start) / batch.len() as f64;
                        for &i in &batch {
                            jobs[i].timings.encode_ms += each_ms;
                            jobs[i].stage = Stage::Encoded;
                            jobs[i].started.get_or_insert_with(Instant::now);
                        }
                    }

                    // Attempts for the encoded windows the language allows: all of them once
                    // it's known; before, only the first unfinished window, which detects it.
                    let head = jobs.iter().position(|j| !matches!(j.stage, Stage::Done));
                    for (i, job) in jobs.iter_mut().enumerate() {
                        if !matches!(job.stage, Stage::Encoded) {
                            continue;
                        }
                        if job.seek == 0 && self.hp.multilingual {
                            if language.is_some() {
                                job.language = language;
                            } else if head == Some(i) {
                                let slot = job.slot.expect("a slot");
                                let (id, code) =
                                    self.detect_language_gpu(g, slot, job.window().trace)?;
                                job.language = Some(id);
                                job.window().detected = Some(code);
                                job.primed = 1;
                            } else {
                                continue;
                            }
                        }
                        self.start_attempt(g, job, 0, &mut epochs);
                    }

                    // The decoder.
                    let decoding: Vec<usize> = (0..jobs.len())
                        .filter(|&i| matches!(jobs[i].stage, Stage::Decoding { .. }))
                        .collect();
                    if decoding.is_empty() && in_flight.is_empty() {
                        self.deliver(&mut jobs, &mut delivered, stop, each)?;
                        continue;
                    }
                    let rows = decoding
                        .iter()
                        .filter_map(|&i| jobs[i].slot)
                        .max()
                        .map_or(1, |s| s + 1);
                    let host: Vec<usize> = decoding
                        .iter()
                        .filter(|&&i| {
                            matches!(
                                jobs[i].stage,
                                Stage::Decoding {
                                    sampled: Some(_),
                                    ..
                                }
                            )
                        })
                        .map(|&i| jobs[i].slot.expect("a slot"))
                        .collect();
                    if host.is_empty() {
                        while in_flight.len() < 2 && !decoding.is_empty() {
                            let pending = g.decoder.submit(rows, STEPS, &[])?;
                            in_flight.push_back((pending, epochs.clone()));
                        }
                    } else if in_flight.is_empty() {
                        in_flight.push_back((g.decoder.submit(rows, 1, &host)?, epochs.clone()));
                    }
                    let (pending, at) = in_flight.pop_front().expect("a run in flight");
                    let snapshot = g.decoder.finish(pending)?;
                    for i in decoding {
                        let slot = jobs[i].slot.expect("a slot");
                        let Stage::Decoding {
                            tier,
                            epoch,
                            ref sampled,
                        } = jobs[i].stage
                        else {
                            continue;
                        };
                        let sampling = sampled.is_some();
                        if at[slot] != epoch || slot >= snapshot.slots.len() {
                            continue;
                        }
                        let view = &snapshot.slots[slot];
                        let choices = if sampling {
                            if view.len < view.prompt_len {
                                continue;
                            }
                            let Some((_, row)) = snapshot.logits.iter().find(|(s, _)| *s == slot)
                            else {
                                continue;
                            };
                            match self.sample_step(g, &mut jobs[i], slot, row.clone()) {
                                Some(choices) => choices,
                                None => continue,
                            }
                        } else {
                            if view.mode != Mode::Idle {
                                continue;
                            }
                            if tier == 0 {
                                jobs[i].no_speech = view.no_speech;
                            }
                            view.choices
                                .iter()
                                .map(|c| (c.token, c.logprob, c.margin))
                                .collect()
                        };
                        let job = &mut jobs[i];
                        let seek = job.seek;
                        let attempt = self.replay(job.window(), &choices, seek, tier);
                        self.attempt_done(g, job, tier, attempt, &mut epochs, &mut free);
                        if matches!(job.stage, Stage::Done) && language.is_none() {
                            let window = job.window();
                            if !self.transcript_text(window).trim().is_empty() {
                                language = window
                                    .detected
                                    .as_ref()
                                    .and_then(|code| self.languages.iter().find(|(c, _)| c == code))
                                    .map(|(_, id)| *id);
                            }
                        }
                    }
                    self.deliver(&mut jobs, &mut delivered, stop, each)?;
                }
                Ok(())
            };
            let result = run();
            // Whatever happened, nothing is left decoding or in flight.
            for slot in 0..slots {
                g.decoder.stop(slot);
            }
            for (pending, _) in in_flight {
                let _ = g.decoder.finish(pending);
            }
            drop(mel_rx);
            result
        })
    }

    /// Every finished window not yet handed on, in order, as far as they go.
    fn deliver(
        &self,
        jobs: &mut [Job],
        delivered: &mut usize,
        stop: &(dyn Fn() -> bool + Sync),
        each: &mut dyn FnMut(usize, Transcript) -> Result<()>,
    ) -> Result<()> {
        while *delivered < jobs.len() && matches!(jobs[*delivered].stage, Stage::Done) {
            if stop() {
                return Err(Error::Cancelled);
            }
            let job = &mut jobs[*delivered];
            let mut timings = job.timings;
            timings.decode_ms = job.started.map_or(0., ms) - timings.encode_ms;
            let window = job.window.take().expect("handed on once");
            job.mel = Mat::zeros(0, 0);
            each(*delivered, self.transcript(window, timings))?;
            *delivered += 1;
        }
        Ok(())
    }

    /// A job's next pass: 30 s of mel features from its seek, zero-padded.
    fn pass_chunk(&self, job: &Job) -> Mat {
        let mel = &job.mel;
        let frames = (mel.rows - job.seek).min(self.hp.frames);
        let mut chunk = Mat::zeros(self.hp.frames, mel.cols);
        chunk.data[..frames * mel.cols]
            .copy_from_slice(&mel.data[job.seek * mel.cols..(job.seek + frames) * mel.cols]);
        chunk
    }

    /// Starts a job's attempt at temperature tier `tier` in its slot.
    fn start_attempt(&self, g: &GpuWhisper, job: &mut Job, tier: usize, epochs: &mut [u64]) {
        let slot = job.slot.expect("a slot");
        let temperature = temperatures()[tier];
        let sampled = (temperature > 0.).then(|| Sampled {
            temperature,
            choices: Vec::new(),
            generated: Vec::new(),
        });
        let mode = if sampled.is_some() {
            Mode::Host
        } else {
            Mode::Greedy
        };
        // As `pass` does: the first attempt after what is primed, the rest from the prompt's
        // last position (the earlier attempt left the rest in the cache).
        let prompt = self.prompt(job.language);
        let at = if tier == 0 {
            job.primed
        } else {
            prompt.len() - 1
        };
        g.decoder
            .resume(slot, &prompt, at, mode, MAX_NEW_TOKENS + 1);
        epochs[slot] += 1;
        job.stage = Stage::Decoding {
            tier,
            epoch: epochs[slot],
            sampled,
        };
    }

    /// A sampling job's next token from its slot's raw logits, as the oracle samples it; its
    /// choices once it's done.
    fn sample_step(
        &self,
        g: &GpuWhisper,
        job: &mut Job,
        slot: usize,
        mut row: Vec<f32>,
    ) -> Option<Choices> {
        let Job {
            window: Some(window),
            stage:
                Stage::Decoding {
                    sampled: Some(sampled),
                    ..
                },
            ..
        } = job
        else {
            return None;
        };
        if sampled.choices.is_empty() {
            for &id in &self.hp.begin_suppress {
                if let Some(v) = row.get_mut(id as usize) {
                    *v = f32::NEG_INFINITY;
                }
            }
        }
        let next = self.choose(
            &mut row,
            &sampled.generated,
            sampled.temperature,
            &mut window.rng,
        );
        sampled.choices.push(next);
        let done = sampled.choices.len() > MAX_NEW_TOKENS || next.0 == self.hp.eos;
        g.decoder.set_input(slot, (!done).then_some(next.0));
        if done {
            Some(std::mem::take(&mut sampled.choices))
        } else {
            sampled.generated.push(next.0);
            None
        }
    }

    /// An attempt judged as [`pass`](Self::pass) judges it: the next temperature, or the pass
    /// accepted (or skipped as silence) and the window on to its next pass or done.
    fn attempt_done(
        &self,
        g: &GpuWhisper,
        job: &mut Job,
        tier: usize,
        attempt: Attempt,
        epochs: &mut [u64],
        free: &mut Vec<usize>,
    ) {
        let temperatures = temperatures();
        let (seek, no_speech) = (job.seek, job.no_speech);
        let frames = (job.mel.rows - seek).min(self.hp.frames);
        let (passes, skip) =
            self.judge(job.window(), &attempt, no_speech, seek, temperatures[tier]);
        if !(skip || passes) && tier + 1 < temperatures.len() {
            self.start_attempt(g, job, tier + 1, epochs);
            return;
        }
        let advance = if skip {
            frames
        } else {
            self.accept(job.window(), &attempt, seek, frames)
        };
        job.seek += advance;
        job.primed = 0;
        if job.seek < job.mel.rows {
            job.stage = Stage::Waiting;
        } else {
            job.stage = Stage::Done;
            free.push(job.slot.take().expect("a slot"));
        }
    }

    /// The text a window has so far (as its transcript will have it).
    fn transcript_text(&self, window: &Window) -> String {
        let text = self.tokenizer.decode_bytes(window.text_ids.iter().copied());
        String::from_utf8_lossy(super::tokenizer::trim(&text)).into_owned()
    }
}
