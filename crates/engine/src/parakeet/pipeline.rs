//! A recording at a time, with its three stages in flight at once (`GPU.md`, "The execution
//! model"): a mel thread makes the features of the windows ahead (on a small pool of its own),
//! the encoder stage turns them into frames for the joint network on the calling thread, and
//! decoder threads decode the windows already encoded, one window each. The queues between
//! them are bounded, so no stage runs far ahead of the next, and the results come back in
//! window order.
//!
//! The CPU threads are budgeted, not left to contend: the mel pool's few threads, the encoder
//! stage's own (all of them for the CPU encoder, about one for a GPU's, which mostly waits),
//! and decoder threads for the rest. Nothing in the decoders or the GPU stage's feeder uses the
//! global rayon pool. When there are fewer windows than decoder threads, each window gets a
//! [`Team`] that shares its decoding.
//!
//! Results go to the caller in window order as they're done, on the calling thread (which runs
//! the encoder stage), between batches. A stop check is polled by every stage between windows.
//!
//! A window's result is exactly what [`Parakeet::run`] gives for it alone: every stage works
//! on one window at a time (or, for an encoder that batches, computes each window as if alone),
//! and nothing a stage computes depends on the threads it ran on.

use super::{Parakeet, gpu_encoder::Pending, team::Team};
use crate::{Error, NoTrace, Result, Transcript, cpu::Mat};
use std::{
    collections::VecDeque,
    sync::{
        Mutex,
        atomic::{AtomicIsize, AtomicUsize, Ordering::SeqCst},
        mpsc,
    },
    time::Instant,
};

/// The pipeline's middle stage: windows' mel features to their encoder frames, projected into
/// the joint network's space (the part of the joint that doesn't depend on the tokens, which
/// is all the decoder needs of the encoder). The CPU encoder is one; a GPU encoder that takes
/// a batch of windows, and keeps a batch queued behind the one it's encoding, is another.
pub trait EncoderStage {
    /// Windows per [`submit`](Self::submit) at most: one for the CPU, a batch for a GPU.
    fn batch(&self) -> usize;

    /// CPU threads the stage keeps busy while it works: all of them for the CPU encoder, about
    /// one for a GPU's. The decoder threads get the rest (two at least).
    fn cpu_threads(&self) -> usize;

    /// Starts on a batch of windows' mel features (`num_mels` × the frames the encoder sees, as
    /// [`Parakeet::features`] makes them) and returns the windows of earlier batches that are
    /// done since the last call, oldest first: each window's encoder frames in the joint's
    /// space (⌈frames / 8⌉ × `joint_hidden`). A stage may keep batches in flight; their
    /// results come from later calls or [`flush`](Self::flush). A window's result must not
    /// depend on the other windows in its batch.
    fn submit(&mut self, mels: Vec<Mat>) -> Result<Vec<Mat>>;

    /// Waits for every batch still in flight; their windows, oldest first.
    fn flush(&mut self) -> Result<Vec<Mat>>;
}

/// The CPU encoder and the joint's encoder projection, one window at a time, on the global
/// rayon pool (every core: it's where the time goes).
pub struct CpuEncoder<'a> {
    pub model: &'a Parakeet,
    pub encoder: &'a super::encoder::Encoder,
}

impl EncoderStage for CpuEncoder<'_> {
    fn batch(&self) -> usize {
        1
    }

    fn cpu_threads(&self) -> usize {
        rayon::current_num_threads()
    }

    fn submit(&mut self, mels: Vec<Mat>) -> Result<Vec<Mat>> {
        Ok(mels
            .iter()
            .map(|mel| {
                let encoded = self.encoder.forward(mel, &mut NoTrace);
                self.model.decoder.project(&encoded)
            })
            .collect())
    }

    fn flush(&mut self) -> Result<Vec<Mat>> {
        Ok(Vec::new())
    }
}

/// The GPU encoder, the joint's encoder projection included, with two batches in flight: each
/// call submits its batch and then waits for the one before it, so the GPU always has the next
/// batch queued while the CPU reads one back and hands it on. It keeps about one CPU thread
/// busy; the decoders get the rest.
pub struct GpuStage<'a> {
    pub encoder: &'a super::gpu_encoder::GpuEncoder,
    in_flight: VecDeque<Pending>,
}

impl<'a> GpuStage<'a> {
    pub fn new(encoder: &'a super::gpu_encoder::GpuEncoder) -> GpuStage<'a> {
        GpuStage {
            encoder,
            in_flight: VecDeque::new(),
        }
    }
}

impl EncoderStage for GpuStage<'_> {
    fn batch(&self) -> usize {
        self.encoder.max_batch()
    }

    fn cpu_threads(&self) -> usize {
        1
    }

    fn submit(&mut self, mels: Vec<Mat>) -> Result<Vec<Mat>> {
        self.in_flight
            .push_back(self.encoder.submit_projected(&mels)?);
        match self.in_flight.len() {
            1 => Ok(Vec::new()),
            _ => self.encoder.finish(self.in_flight.pop_front().unwrap()),
        }
    }

    fn flush(&mut self) -> Result<Vec<Mat>> {
        let mut out = Vec::new();
        while let Some(pending) = self.in_flight.pop_front() {
            out.extend(self.encoder.finish(pending)?);
        }
        Ok(out)
    }
}

impl Drop for GpuStage<'_> {
    /// Batches left in flight by an error give their readback buffers back.
    fn drop(&mut self) {
        while let Some(pending) = self.in_flight.pop_front() {
            let _ = self.encoder.finish(pending);
        }
    }
}

/// A window between stages: its index, what the last stage made of it, the times so far, and
/// whether its decoding should get a team.
struct InFlight {
    index: usize,
    data: Mat,
    timings: crate::Timings,
    team: bool,
}

/// Threads in the pool that makes mel features. A window's take 5 ms on two, 3 ms on four:
/// either keeps well ahead of the GPU encoder (26 ms a window), but four fill its second batch
/// sooner at the start, while the decoders have nothing to do yet. They sleep the rest of the
/// time.
const MEL_THREADS: usize = 4;

impl Parakeet {
    /// Every window through the three stages, with `stage` as the encoder; each result to
    /// `each` in window order, as soon as it and those before it are decoded. When `stop`
    /// says so, every stage gives up after the window (or batch) it's on and the run returns
    /// [`Error::Cancelled`].
    pub fn pipeline(
        &self,
        windows: &[&[f32]],
        stage: &mut dyn EncoderStage,
        stop: &(dyn Fn() -> bool + Sync),
        each: &mut dyn FnMut(usize, Transcript) -> Result<()>,
    ) -> Result<()> {
        if windows.is_empty() {
            return Ok(());
        }
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let budget = threads
            .saturating_sub(stage.cpu_threads() + MEL_THREADS)
            .max(2);
        let decoders = budget.min(windows.len());
        // Threads free to help decode: each decoding window holds one, its team's helpers one
        // each, and a window takes helpers only from what the windows yet to start decoding
        // leave over, so the decoders never run more threads than the budget (spinning helpers
        // on an oversubscribed CPU wait on members that aren't running, and starve the
        // encoder's feeder). A window gets helpers only when the decoders would otherwise sit
        // idle: in a short recording, and once every window is in the encoder; during a long
        // one the CPU stays as quiet as the work allows, leaving the laptop's power to the GPU.
        let free = AtomicIsize::new(budget as isize);
        let unstarted = AtomicUsize::new(windows.len());
        let short = windows.len() <= budget;
        let batch = stage.batch().max(1);
        let mel_pool = self.mel_pool()?;
        // Decoded windows that came out of order, until those before them are done.
        let mut results: Vec<Option<Transcript>> = vec![None; windows.len()];
        let mut next = 0;
        // Encoded windows, for whichever decoder is free.
        let (encoded_tx, encoded) = mpsc::sync_channel::<InFlight>(2 * decoders);
        let encoded = Mutex::new(encoded);
        let outcome = std::thread::scope(|s| -> Result<()> {
            // Mel features, in order, a couple of batches ahead of the encoder.
            let (features_tx, features) = mpsc::sync_channel::<InFlight>(2 * batch);
            s.spawn(move || {
                mel_pool.install(|| {
                    for (index, pcm) in windows.iter().enumerate() {
                        if stop() {
                            break; // the encoder stage stops too
                        }
                        let start = Instant::now();
                        let data = self.features(pcm, &mut NoTrace);
                        let timings = crate::Timings {
                            mel_ms: ms(start),
                            ..Default::default()
                        };
                        let window = InFlight {
                            index,
                            data,
                            timings,
                            team: false,
                        };
                        if features_tx.send(window).is_err() {
                            break; // the encoder stage stopped
                        }
                    }
                })
            });
            // Decoders: each takes the next encoded window and decodes it alone (with its team).
            let (done_tx, done) = mpsc::channel::<(usize, Transcript)>();
            for _ in 0..decoders {
                let (encoded, done_tx) = (&encoded, done_tx.clone());
                let (free, unstarted) = (&free, &unstarted);
                s.spawn(move || {
                    // Until every window is encoded, or the encoder stopped. The lock is
                    // held only while waiting: the guard is dropped inside `and_then`.
                    while let Ok(mut window) = encoded
                        .lock()
                        .map_err(drop)
                        .and_then(|queue| queue.recv().map_err(drop))
                    {
                        // Stopped: drain the queue without decoding, so the encoder stage
                        // never waits on a full one.
                        if stop() {
                            continue;
                        }
                        let start = Instant::now();
                        free.fetch_sub(1, SeqCst);
                        let after = unstarted.fetch_sub(1, SeqCst) - 1;
                        let helpers = if window.team {
                            take(free, super::TEAM_SIZE - 1, after)
                        } else {
                            0
                        };
                        let team = self.take_team(1 + helpers);
                        let transcript = self.decode(&window.data, &mut NoTrace, &team);
                        self.put_team(team);
                        free.fetch_add(1 + helpers as isize, SeqCst);
                        window.timings.decode_ms = ms(start);
                        if done_tx
                            .send((
                                window.index,
                                Transcript {
                                    timings: window.timings,
                                    ..transcript
                                },
                            ))
                            .is_err()
                        {
                            break;
                        }
                    }
                });
            }
            drop(done_tx);
            // The encoder, here: a batch at a time, as many windows as are ready (at least
            // one), the stage handing back earlier batches' windows as they're done.
            let mut submitted: VecDeque<InFlight> = VecDeque::new();
            let mut started = Instant::now();
            // Hands encoded windows on to the decoders, with teams once every window is in the
            // encoder (or the recording is short).
            let hand_on = |projected: Vec<Mat>,
                           submitted: &mut VecDeque<InFlight>,
                           started: Instant,
                           all_submitted: bool|
             -> Result<()> {
                if projected.is_empty() {
                    return Ok(());
                }
                if projected.len() > submitted.len() {
                    return Err(Error::Other(format!(
                        "the encoder gave {} results for {} windows",
                        projected.len(),
                        submitted.len()
                    )));
                }
                // The time since the last hand-off, split evenly: each window's share of the
                // stage while it was the stage's bottleneck.
                let each = ms(started) / projected.len() as f64;
                for data in projected {
                    let mut window = submitted.pop_front().unwrap();
                    window.data = data;
                    window.timings.encode_ms = each;
                    window.team = short || all_submitted;
                    // The decoders only stop when this sender is dropped.
                    let _ = encoded_tx.send(window);
                }
                Ok(())
            };
            // Hands the caller every decoded window it can have yet: `wait` blocks for the
            // rest, until the decoders are done.
            let mut deliver = |wait: bool| -> Result<()> {
                loop {
                    let got = if wait {
                        done.recv().ok()
                    } else {
                        done.try_recv().ok()
                    };
                    let Some((index, transcript)) = got else {
                        return Ok(());
                    };
                    results[index] = Some(transcript);
                    while let Some(transcript) = results.get_mut(next).and_then(Option::take) {
                        if stop() {
                            return Err(Error::Cancelled);
                        }
                        each(next, transcript)?;
                        next += 1;
                    }
                }
            };
            let mut count = 0;
            loop {
                if stop() {
                    return Err(Error::Cancelled);
                }
                // Toward the end the batches halve, so the last windows come out of the
                // encoder a few at a time and decode on big teams as they come, instead of
                // all at once after the GPU is done.
                let size = batch.min((windows.len() - count).div_ceil(2)).max(1);
                // With a batch in flight there's time to fill this one; with none, the GPU
                // waits, so it goes with whatever is ready.
                let wait_for = if submitted.is_empty() { 1 } else { size };
                let mut pending = Vec::with_capacity(size);
                while pending.len() < size {
                    let next = if pending.len() < wait_for {
                        features.recv().ok()
                    } else {
                        features.try_recv().ok()
                    };
                    match next {
                        Some(window) => pending.push(window),
                        None => break,
                    }
                }
                if pending.is_empty() {
                    break; // every window is submitted
                }
                let mels: Vec<Mat> = pending
                    .iter_mut()
                    .map(|w| std::mem::replace(&mut w.data, Mat::zeros(0, 0)))
                    .collect();
                count += pending.len();
                submitted.extend(pending);
                let projected = stage.submit(mels)?;
                if !projected.is_empty() {
                    hand_on(projected, &mut submitted, started, count == windows.len())?;
                    started = Instant::now();
                }
                deliver(false)?;
            }
            if stop() {
                return Err(Error::Cancelled);
            }
            let projected = stage.flush()?;
            hand_on(projected, &mut submitted, started, true)?;
            if !submitted.is_empty() {
                return Err(Error::Other(format!(
                    "the encoder kept {} windows",
                    submitted.len()
                )));
            }
            drop(encoded_tx);
            deliver(true)
        });
        outcome?;
        if stop() {
            return Err(Error::Cancelled);
        }
        match next {
            n if n == windows.len() => Ok(()),
            n => Err(Error::Other(format!("window {n} wasn't decoded"))),
        }
    }

    /// The pool mel features are made on during a pipeline: made at first use, kept.
    fn mel_pool(&self) -> Result<&rayon::ThreadPool> {
        if let Some(pool) = self.mel_pool.get() {
            return Ok(pool);
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(MEL_THREADS)
            .thread_name(|i| format!("mel {i}"))
            .build()
            .map_err(|e| Error::Other(format!("mel threads: {e}")))?;
        Ok(self.mel_pool.get_or_init(|| pool))
    }

    /// A team of `size` for one decoder: a kept one if there is one, else new.
    pub(super) fn take_team(&self, size: usize) -> Team {
        let kept = {
            let mut teams = self.teams.lock().unwrap();
            teams
                .iter()
                .position(|t| t.size() == size)
                .map(|i| teams.swap_remove(i))
        };
        kept.unwrap_or_else(|| Team::new(size))
    }

    /// Keeps a team for the next call, its helpers parked.
    pub(super) fn put_team(&self, team: Team) {
        team.rest();
        if team.size() > 1 {
            self.teams.lock().unwrap().push(team);
        }
    }
}

/// Up to `want` of the `free` threads, leaving `reserve` (none when there are no more).
fn take(free: &AtomicIsize, want: usize, reserve: usize) -> usize {
    let mut have = free.load(SeqCst);
    loop {
        let n = (have - reserve as isize).clamp(0, want as isize);
        if n == 0 {
            return 0;
        }
        match free.compare_exchange(have, have - n, SeqCst, SeqCst) {
            Ok(_) => return n as usize,
            Err(now) => have = now,
        }
    }
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1e3
}
