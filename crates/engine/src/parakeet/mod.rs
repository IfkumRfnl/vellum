//! Parakeet (FastConformer encoder, RNN-T decoder) as transcribe.cpp runs it
//! (`src/arch/parakeet/` there): log-mel features, 8× convolutional subsampling, 24 conformer
//! blocks, then greedy RNN-T decoding of SentencePiece tokens.
//!
//! Checkpoints carry the reference's dump names (`enc.mel.in`, `enc.pre_encode.out`,
//! `enc.pos_emb`, `enc.block.<L>.out`, `enc.final`, `dec.*`) so `cargo lab probe` lines them up.

mod decoder;
pub mod encoder;
pub mod gpu_encoder;
mod pipeline;
mod team;

pub use pipeline::{CpuEncoder, EncoderStage, GpuStage};

use crate::{
    Error, ModelInfo, Result, RunOptions, Timings, Trace, Transcript, cpu::Mat, gguf::Gguf,
    mel::Mel, tokenizer::Tokenizer, trace::checkpoint,
};
use std::{
    sync::{Arc, Mutex, OnceLock},
    time::Instant,
};
use team::Team;

/// The most threads one window's decoding is shared among. On the i5-13450HX (6 performance
/// cores, 4 efficiency cores) a short window decodes 4× as fast on 6 as on one; on 8 it's
/// slower again, the team then spanning a core's two hyperthreads or an efficiency core, whose
/// pace every step waits for.
const TEAM_SIZE: usize = 6;

/// The model's shape, from its `stt.parakeet.*` metadata (all of it, whether the CPU path
/// reads a field or not).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct HParams {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub conv_kernel: usize,
    pub subsampling_factor: usize,
    pub subsampling_channels: usize,
    pub xscaling: bool,
    pub pos_emb_max_len: usize,
    pub predictor_hidden: usize,
    pub predictor_layers: usize,
    pub joint_hidden: usize,
    pub vocab: usize,
    pub blank: usize,
}

impl HParams {
    pub fn from_gguf(file: &Gguf) -> Result<HParams> {
        let n = |key: &str| {
            file.metadata_u64(&format!("stt.parakeet.{key}"))
                .map(|v| v as usize)
                .ok_or_else(|| Error::Gguf(format!("no stt.parakeet.{key}")))
        };
        Ok(HParams {
            d_model: n("encoder.d_model")?,
            n_heads: n("encoder.n_heads")?,
            n_layers: n("encoder.n_layers")?,
            d_ff: n("encoder.d_ff")?,
            conv_kernel: n("encoder.conv_kernel")?,
            subsampling_factor: n("encoder.subsampling_factor")?,
            subsampling_channels: n("encoder.subsampling_channels")?,
            xscaling: file
                .metadata
                .get("stt.parakeet.encoder.xscaling")
                .is_some_and(|v| v.as_u64() == Some(1)),
            pos_emb_max_len: n("encoder.pos_emb_max_len")?,
            predictor_hidden: n("predictor.hidden")?,
            predictor_layers: n("predictor.n_layers")?,
            joint_hidden: n("joint.hidden")?,
            vocab: n("predictor.vocab")?,
            blank: file
                .metadata_u64("tokenizer.ggml.blank_token_id")
                .ok_or_else(|| Error::Gguf("no tokenizer.ggml.blank_token_id".into()))?
                as usize,
        })
    }
}

pub struct Parakeet {
    pub hparams: HParams,
    mel: Mel,
    encoder: Encoder,
    decoder: decoder::Decoder,
    tokenizer: Tokenizer,
    pub info: ModelInfo,
    /// The threads a pipeline makes mel features on (made at first use).
    mel_pool: OnceLock<rayon::ThreadPool>,
    /// Decoder teams kept between calls, their helpers parked.
    teams: Mutex<Vec<Team>>,
}

/// Where the encoder runs. Only one is loaded: the GPU's needs nothing of the CPU's.
// One per model: its size doesn't matter.
#[allow(clippy::large_enum_variant)]
enum Encoder {
    Cpu(encoder::Encoder),
    Gpu(Arc<crate::gpu::Gpu>, gpu_encoder::GpuEncoder),
}

impl Parakeet {
    /// The model, its encoder on `gpu` when there's one, else on the CPU.
    pub fn load(file: &Gguf, gpu: Option<Arc<crate::gpu::Gpu>>) -> Result<Parakeet> {
        let hparams = HParams::from_gguf(file)?;
        // The decoder loads beside the encoder, off its critical path.
        let (encoder, decoder) = rayon::join(
            || -> Result<Encoder> {
                Ok(match gpu {
                    Some(gpu) => {
                        let encoder = gpu_encoder::GpuEncoder::load(&gpu, file, &hparams)?;
                        Encoder::Gpu(gpu, encoder)
                    }
                    None => Encoder::Cpu(encoder::Encoder::load(file, &hparams)?),
                })
            },
            || decoder::Decoder::load(file, &hparams),
        );
        let encoder = encoder?;
        let device = match &encoder {
            Encoder::Cpu(_) => "CPU".into(),
            Encoder::Gpu(gpu, _) => gpu.name(),
        };
        Ok(Parakeet {
            encoder,
            mel: Mel::from_gguf(file)?,
            decoder: decoder?,
            tokenizer: Tokenizer::from_gguf(file)?,
            info: ModelInfo {
                architecture: "parakeet".into(),
                variant: file.metadata_str("stt.variant").unwrap_or_default().into(),
                device,
            },
            hparams,
            mel_pool: OnceLock::new(),
            teams: Mutex::new(Vec::new()),
        })
    }

    /// One window of 16 kHz audio to words. Untraced, the encoder's frames come back already
    /// projected into the joint's space (on a GPU, the projection is its last multiply), as
    /// in [`run_each`](Self::run_each); a trace that records or substitutes checkpoints gets
    /// the encoder's output itself, projected on the CPU.
    pub fn run(
        &self,
        pcm: &[f32],
        options: &RunOptions,
        trace: &mut dyn Trace,
    ) -> Result<Transcript> {
        self.run_until(pcm, options, trace, &|| false)
    }

    /// [`run`](Self::run), with `stop` polled between the CPU encoder's blocks (a window there
    /// takes seconds); once it says so, the run returns [`Error::Cancelled`].
    fn run_until(
        &self,
        pcm: &[f32],
        _options: &RunOptions,
        trace: &mut dyn Trace,
        stop: &(dyn Fn() -> bool + Sync),
    ) -> Result<Transcript> {
        let mut timings = Timings::default();
        let start = Instant::now();
        let mel = self.features(pcm, trace);
        timings.mel_ms = start.elapsed().as_secs_f64() * 1e3;
        let start = Instant::now();
        let enc_proj = if self.traced(trace) {
            // A trace sees (or substitutes) the encoder's own output, projected here. Testing
            // the decoder alone on the reference's encoder output needs no encoding.
            let mut encoded = if trace.injects("enc.final") {
                Mat::zeros(0, self.hparams.d_model)
            } else {
                match &self.encoder {
                    Encoder::Gpu(_, gpu) => gpu
                        .encode_traced(std::slice::from_ref(&mel), trace)?
                        .remove(0),
                    Encoder::Cpu(cpu) => cpu.forward_until(&mel, trace, stop)?,
                }
            };
            checkpoint(trace, "enc.final", &mut encoded);
            checkpoint(trace, "dec.enc_out", &mut encoded);
            self.decoder.project(&encoded)
        } else {
            match &self.encoder {
                // The projection is the GPU encoder's last multiply, as in the pipeline.
                Encoder::Gpu(_, gpu) => gpu.encode_projected(std::slice::from_ref(&mel))?.remove(0),
                Encoder::Cpu(cpu) => self.decoder.project(&cpu.forward_until(&mel, trace, stop)?),
            }
        };
        timings.encode_ms = start.elapsed().as_secs_f64() * 1e3;
        let start = Instant::now();
        let team = self.take_team(TEAM_SIZE);
        let mut transcript = self.decode(&enc_proj, trace, &team);
        self.put_team(team);
        timings.decode_ms = start.elapsed().as_secs_f64() * 1e3;
        transcript.timings = timings;
        Ok(transcript)
    }

    /// A window's projected encoder frames to its transcript, `team` sharing the work.
    fn decode(&self, enc_proj: &Mat, trace: &mut dyn Trace, team: &Team) -> Transcript {
        let emitted = self.decoder.greedy(enc_proj, trace, team);
        self.decoder
            .transcript(&emitted, &self.tokenizer, &self.hparams)
    }

    /// Every window of a recording, in order, each with exactly the result [`run`](Self::run)
    /// gives it, but with mel, encoder and decoder working on different windows at once (see
    /// [`pipeline`](Self::pipeline)); each handed to `each` as soon as it and the windows
    /// before it are done. `stop` is polled between windows and batches, and on the CPU between
    /// the encoder's blocks. A trace that records
    /// or substitutes tensors gets the windows one by one instead, so its checkpoints keep
    /// their meaning.
    pub fn run_each(
        &self,
        windows: &[&[f32]],
        options: &RunOptions,
        trace: &mut dyn Trace,
        stop: &(dyn Fn() -> bool + Sync),
        each: &mut dyn FnMut(usize, Transcript) -> Result<()>,
    ) -> Result<()> {
        // One window has nothing to overlap: it runs as `run` runs it.
        if self.traced(trace) || windows.len() == 1 {
            for (index, pcm) in windows.iter().enumerate() {
                if stop() {
                    return Err(Error::Cancelled);
                }
                each(index, self.run_until(pcm, options, trace, stop)?)?;
            }
            return Ok(());
        }
        match &self.encoder {
            Encoder::Gpu(_, encoder) => {
                self.pipeline(windows, &mut GpuStage::new(encoder), stop, each)
            }
            Encoder::Cpu(encoder) => self.pipeline(
                windows,
                &mut CpuEncoder {
                    model: self,
                    encoder,
                    stop,
                },
                stop,
                each,
            ),
        }
    }

    /// The GPU the encoder runs on, if it runs on one.
    pub fn gpu(&self) -> Option<&crate::gpu::Gpu> {
        match &self.encoder {
            Encoder::Gpu(gpu, _) => Some(gpu),
            Encoder::Cpu(_) => None,
        }
    }

    /// Whether `trace` records or substitutes any checkpoint.
    fn traced(&self, trace: &dyn Trace) -> bool {
        let fixed = [
            "enc.mel.in",
            "enc.pre_encode.out",
            "enc.pos_emb",
            "enc.final",
            "dec.enc_out",
            "dec.embed.0",
            "dec.lstm.0.h.0",
            "dec.lstm.0.c.0",
            "dec.joint.0",
        ];
        let blocks = (0..self.hparams.n_layers).map(|l| format!("enc.block.{l}.out"));
        fixed
            .into_iter()
            .map(str::to_owned)
            .chain(blocks)
            .any(|name| trace.wants(&name) || trace.injects(&name))
    }

    /// A window's mel features as the encoder takes them (`num_mels` × frames). The last frame
    /// is the centering pad's: the reference dumps it (`enc.mel.in`) but encodes without it.
    pub fn features(&self, pcm: &[f32], trace: &mut dyn Trace) -> Mat {
        let mut mel = self.mel.compute(pcm);
        checkpoint(trace, "enc.mel.in", &mut mel);
        let valid = self.mel.valid_frames(pcm.len()).min(mel.cols);
        if valid < mel.cols {
            let mut trimmed = Mat::zeros(mel.rows, valid);
            for r in 0..mel.rows {
                trimmed.row_mut(r).copy_from_slice(&mel.row(r)[..valid]);
            }
            mel = trimmed;
        }
        mel
    }
}
