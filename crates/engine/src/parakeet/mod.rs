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

pub use pipeline::{CpuEncoder, EncoderStage, GpuStage};

use crate::{
    Error, ModelInfo, Result, RunOptions, Timings, Trace, Transcript, gguf::Gguf, mel::Mel,
    tokenizer::Tokenizer, trace::checkpoint,
};
use std::{sync::Arc, time::Instant};

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
    encoder: encoder::Encoder,
    /// The encoder on the GPU, when the model runs on one. The CPU encoder stays loaded: it's
    /// the oracle, and the fallback.
    gpu: Option<(Arc<crate::gpu::Gpu>, gpu_encoder::GpuEncoder)>,
    decoder: decoder::Decoder,
    tokenizer: Tokenizer,
    pub info: ModelInfo,
}

impl Parakeet {
    /// The model, with its encoder also on `gpu` when there's one.
    pub fn load(file: &Gguf, gpu: Option<Arc<crate::gpu::Gpu>>) -> Result<Parakeet> {
        let hparams = HParams::from_gguf(file)?;
        let gpu = match gpu {
            Some(gpu) => {
                let encoder = gpu_encoder::GpuEncoder::load(&gpu, file, &hparams)?;
                Some((gpu, encoder))
            }
            None => None,
        };
        let device = gpu.as_ref().map_or("CPU".into(), |(g, _)| g.name());
        Ok(Parakeet {
            gpu,
            mel: Mel::from_gguf(file)?,
            encoder: encoder::Encoder::load(file, &hparams)?,
            decoder: decoder::Decoder::load(file, &hparams)?,
            tokenizer: Tokenizer::from_gguf(file)?,
            info: ModelInfo {
                architecture: "parakeet".into(),
                variant: file.metadata_str("stt.variant").unwrap_or_default().into(),
                device,
            },
            hparams,
        })
    }

    /// One window of 16 kHz audio to words.
    pub fn run(
        &self,
        pcm: &[f32],
        _options: &RunOptions,
        trace: &mut dyn Trace,
    ) -> Result<Transcript> {
        let mut timings = Timings::default();
        let start = Instant::now();
        let mel = self.features(pcm, trace);
        timings.mel_ms = start.elapsed().as_secs_f64() * 1e3;
        let start = Instant::now();
        // Testing the decoder alone on the reference's encoder output needs no encoding.
        let mut encoded = if trace.injects("enc.final") {
            crate::cpu::Mat::zeros(0, self.hparams.d_model)
        } else if let Some((_, gpu)) = &self.gpu {
            gpu.encode_traced(std::slice::from_ref(&mel), trace)?
                .remove(0)
        } else {
            self.encoder.forward(&mel, trace)
        };
        checkpoint(trace, "enc.final", &mut encoded);
        timings.encode_ms = start.elapsed().as_secs_f64() * 1e3;
        let start = Instant::now();
        checkpoint(trace, "dec.enc_out", &mut encoded);
        let enc_proj = self.decoder.project(&encoded);
        let emitted = self.decoder.greedy(&enc_proj, trace);
        let mut transcript = self
            .decoder
            .transcript(&emitted, &self.tokenizer, &self.hparams);
        timings.decode_ms = start.elapsed().as_secs_f64() * 1e3;
        transcript.timings = timings;
        Ok(transcript)
    }

    /// Every window of a recording, in order, each with exactly the result [`run`](Self::run)
    /// gives it, but with mel, encoder and decoder working on different windows at once (see
    /// [`pipeline`](Self::pipeline)). A trace that records or substitutes tensors gets the
    /// windows one by one instead, so its checkpoints keep their meaning.
    pub fn run_all(
        &self,
        windows: &[&[f32]],
        options: &RunOptions,
        trace: &mut dyn Trace,
    ) -> Result<Vec<Transcript>> {
        if self.traced(trace) {
            return windows
                .iter()
                .map(|pcm| self.run(pcm, options, trace))
                .collect();
        }
        match &self.gpu {
            Some((_, encoder)) => self.pipeline(
                windows,
                &mut GpuStage {
                    model: self,
                    encoder,
                },
            ),
            None => self.pipeline(windows, &mut CpuEncoder { model: self }),
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
    pub fn features(&self, pcm: &[f32], trace: &mut dyn Trace) -> crate::cpu::Mat {
        let mut mel = self.mel.compute(pcm);
        checkpoint(trace, "enc.mel.in", &mut mel);
        let valid = self.mel.valid_frames(pcm.len()).min(mel.cols);
        if valid < mel.cols {
            let mut trimmed = crate::cpu::Mat::zeros(mel.rows, valid);
            for r in 0..mel.rows {
                trimmed.row_mut(r).copy_from_slice(&mel.row(r)[..valid]);
            }
            mel = trimmed;
        }
        mel
    }
}
