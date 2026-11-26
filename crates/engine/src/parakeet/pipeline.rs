//! A recording at a time, with its three stages in flight at once (`GPU.md`, "The execution
//! model"): one thread makes the mel features of the windows ahead, the encoder stage turns
//! them into frames for the joint network on the calling thread, and decoder threads decode
//! the windows already encoded, one window each. The queues between them are bounded, so no
//! stage runs far ahead of the next, and the results come back in window order.
//!
//! A window's result is exactly what [`Parakeet::run`] gives for it alone: every stage works
//! on one window at a time (or, for an encoder that batches, computes each window as if alone),
//! and nothing a stage computes depends on the threads it ran on.

use super::Parakeet;
use crate::{Error, NoTrace, Result, Transcript, cpu::Mat};
use std::{
    sync::{Mutex, mpsc},
    time::Instant,
};

/// The pipeline's middle stage: windows' mel features to their encoder frames, projected into
/// the joint network's space (the part of the joint that doesn't depend on the tokens, which
/// is all the decoder needs of the encoder). The CPU encoder is one; a GPU encoder that takes
/// a batch of windows is another, and plugs in without changing the other stages.
pub trait EncoderStage {
    /// Windows per [`encode`](Self::encode) call: one for the CPU, a batch for a GPU. Fewer
    /// arrive at the end of a recording.
    fn batch(&self) -> usize;

    /// CPU threads the stage keeps busy while it works: all of them for the CPU encoder, about
    /// one for a GPU's. The decoder threads get the rest (two at least).
    fn cpu_threads(&self) -> usize;

    /// Each window's mel features (`num_mels` × the frames the encoder sees, as
    /// [`Parakeet::features`] makes them) to its encoder frames in the joint's space
    /// (⌈frames / 8⌉ × `joint_hidden`), in the same order. A window's result must not depend on
    /// the other windows in its batch.
    fn encode(&mut self, mels: &[Mat]) -> Result<Vec<Mat>>;
}

/// The CPU encoder and the joint's encoder projection, one window at a time, on the global
/// rayon pool (every core: it's where the time goes).
pub struct CpuEncoder<'a> {
    pub model: &'a Parakeet,
}

impl EncoderStage for CpuEncoder<'_> {
    fn batch(&self) -> usize {
        1
    }

    fn cpu_threads(&self) -> usize {
        rayon::current_num_threads()
    }

    fn encode(&mut self, mels: &[Mat]) -> Result<Vec<Mat>> {
        Ok(mels
            .iter()
            .map(|mel| {
                let encoded = self.model.encoder.forward(mel, &mut NoTrace);
                self.model.decoder.project(&encoded)
            })
            .collect())
    }
}

/// The GPU encoder: batches of windows encoded together on the device, the joint's encoder
/// projection on the CPU. It keeps about one CPU thread busy feeding the GPU; the decoders get
/// the rest.
pub struct GpuStage<'a> {
    pub model: &'a Parakeet,
    pub encoder: &'a super::gpu_encoder::GpuEncoder,
}

impl EncoderStage for GpuStage<'_> {
    fn batch(&self) -> usize {
        self.encoder.max_batch()
    }

    fn cpu_threads(&self) -> usize {
        1
    }

    fn encode(&mut self, mels: &[Mat]) -> Result<Vec<Mat>> {
        Ok(self
            .encoder
            .encode(mels)?
            .iter()
            .map(|encoded| self.model.decoder.project(encoded))
            .collect())
    }
}

/// A window between stages: its index, what the last stage made of it, and the times so far.
struct InFlight {
    index: usize,
    data: Mat,
    timings: crate::Timings,
}

impl Parakeet {
    /// Every window through the three stages, with `stage` as the encoder; results in window
    /// order.
    pub fn pipeline(
        &self,
        windows: &[&[f32]],
        stage: &mut dyn EncoderStage,
    ) -> Result<Vec<Transcript>> {
        if windows.is_empty() {
            return Ok(Vec::new());
        }
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let decoders = threads
            .saturating_sub(stage.cpu_threads())
            .max(2)
            .min(windows.len());
        let batch = stage.batch().max(1);
        let mut results: Vec<Option<Transcript>> = vec![None; windows.len()];
        // Encoded windows, for whichever decoder is free.
        let (encoded_tx, encoded) = mpsc::sync_channel::<InFlight>(2 * decoders);
        let encoded = Mutex::new(encoded);
        let outcome = std::thread::scope(|s| -> Result<()> {
            // Mel features, in order, a couple of batches ahead of the encoder.
            let (features_tx, features) = mpsc::sync_channel::<InFlight>(2 * batch);
            s.spawn(move || {
                for (index, pcm) in windows.iter().enumerate() {
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
                    };
                    if features_tx.send(window).is_err() {
                        break; // the encoder stage stopped
                    }
                }
            });
            // Decoders: each takes the next encoded window and decodes it alone.
            let (done_tx, done) = mpsc::channel::<(usize, Transcript)>();
            for _ in 0..decoders {
                let (encoded, done_tx) = (&encoded, done_tx.clone());
                s.spawn(move || {
                    // Until every window is encoded, or the encoder stopped. The lock is
                    // held only while waiting: the guard is dropped inside `and_then`.
                    while let Ok(mut window) = encoded
                        .lock()
                        .map_err(drop)
                        .and_then(|queue| queue.recv().map_err(drop))
                    {
                        let start = Instant::now();
                        let emitted = self.decoder.greedy(&window.data, &mut NoTrace);
                        let mut transcript =
                            self.decoder
                                .transcript(&emitted, &self.tokenizer, &self.hparams);
                        window.timings.decode_ms = ms(start);
                        transcript.timings = window.timings;
                        if done_tx.send((window.index, transcript)).is_err() {
                            break;
                        }
                    }
                });
            }
            drop(done_tx);
            // The encoder, here: a batch at a time, as the features arrive.
            let mut pending: Vec<InFlight> = Vec::with_capacity(batch);
            let mut features = features.into_iter().peekable();
            while features.peek().is_some() {
                pending.extend(features.by_ref().take(batch));
                let start = Instant::now();
                let mels: Vec<Mat> = pending
                    .iter_mut()
                    .map(|w| std::mem::replace(&mut w.data, Mat::zeros(0, 0)))
                    .collect();
                let projected = stage.encode(&mels)?;
                if projected.len() != mels.len() {
                    return Err(Error::Other(format!(
                        "the encoder gave {} results for {} windows",
                        projected.len(),
                        mels.len()
                    )));
                }
                let each = ms(start) / mels.len() as f64;
                for (mut window, data) in pending.drain(..).zip(projected) {
                    window.data = data;
                    window.timings.encode_ms = each;
                    // The decoders only stop when this sender is dropped.
                    let _ = encoded_tx.send(window);
                }
            }
            drop(encoded_tx);
            for (index, transcript) in done {
                results[index] = Some(transcript);
            }
            Ok(())
        });
        outcome?;
        results
            .into_iter()
            .enumerate()
            .map(|(i, r)| r.ok_or_else(|| Error::Other(format!("window {i} wasn't decoded"))))
            .collect()
    }
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1e3
}
