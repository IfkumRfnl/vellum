//! NVIDIA's Streaming Sortformer (4 speakers, v2.1) as transcribe.cpp runs it for the app
//! (`src/arch/sortformer/` there, `SortformerPreset::VeryHighLatency`, the whole recording in
//! one call): unnormalized log-mel features, the FastConformer encoder Parakeet has (17 blocks
//! of 512 here), a projection to 192, an 18-layer post-LN Transformer, and a sigmoid per
//! speaker per 80 ms frame.
//!
//! The recording is diarized in chunks of 27.2 s, each subsampled alone (with one frame of
//! left and up to 40 of right context) and then encoded joined to a speaker cache and a FIFO
//! of earlier frames ([`stream`]), so a speaker keeps its index across the recording. Speaker
//! turns are the runs of frames whose probability is over ½.
//!
//! Checkpoints carry the reference's dump names: `diar.probs` (the streamed probabilities),
//! `compress.<NNN>.*` (each speaker-cache compression, with `TRANSCRIBE_SORTFORMER_COMPRESS_DUMP`
//! set) and, with `TRANSCRIBE_SORTFORMER_OFFLINE_DUMP` set, an extra full-context pass over the
//! whole recording (on the CPU only): `enc.mel.in`, `enc.fastconformer.out`,
//! `enc.encoder_proj.out`, `enc.transformer.out`, `diar.preds_offline`.

mod gpu;
mod stream;
mod transformer;

use crate::{
    Error, ModelInfo, NoTrace, Result, SpeakerTurn, Trace,
    cpu::Mat,
    gguf::Gguf,
    mel::Mel,
    parakeet::{self, encoder::Encoder},
    trace::checkpoint,
};
use gpu::GpuSortformer;
use std::{ops::Range, sync::Arc};
use stream::{State, StreamParams};
use transformer::Transformer;

/// The model's shape, from its `stt.sortformer.*` metadata.
#[derive(Debug, Clone)]
pub struct HParams {
    pub max_speakers: usize,
    /// Audio samples per diarization frame (the mel hop times the subsampling).
    pub frame_hop: usize,
    pub sample_rate: usize,
    pub hop_length: usize,
    /// The FastConformer, in the terms Parakeet's encoder takes.
    pub encoder: parakeet::HParams,
    pub tf_layers: usize,
    pub tf_heads: usize,
}

impl HParams {
    pub fn from_gguf(file: &Gguf) -> Result<HParams> {
        let n = |key: &str| {
            file.metadata_u64(key)
                .map(|v| v as usize)
                .ok_or_else(|| Error::Gguf(format!("no {key}")))
        };
        let enc = |key: &str| n(&format!("stt.sortformer.encoder.{key}"));
        let text = |key: &str| file.metadata_str(key).unwrap_or_default();
        if text("stt.sortformer.encoder.conv_norm_type") != "batch_norm" {
            return Err(Error::Unsupported(
                "Sortformer without batch norm in its convolutions".into(),
            ));
        }
        if text("stt.sortformer.transformer.activation") != "relu"
            || file
                .metadata
                .get("stt.sortformer.transformer.pre_ln")
                .is_some_and(|v| v.as_u64() == Some(1))
        {
            return Err(Error::Unsupported(
                "Sortformer with a pre-LN or non-ReLU transformer".into(),
            ));
        }
        Ok(HParams {
            max_speakers: n("stt.sortformer.max_speakers")?,
            frame_hop: n("stt.sortformer.frame_hop")?,
            sample_rate: n("stt.frontend.sample_rate")?,
            hop_length: n("stt.frontend.hop_length")?,
            // NeMo's NEST FastConformer: x-scaling, biases, full attention, batch norm.
            encoder: parakeet::HParams {
                d_model: enc("d_model")?,
                n_heads: enc("n_heads")?,
                n_layers: enc("n_layers")?,
                d_ff: enc("d_ff")?,
                conv_kernel: enc("conv_kernel")?,
                subsampling_factor: enc("subsampling_factor")?,
                subsampling_channels: enc("subsampling_channels")?,
                xscaling: true,
                pos_emb_max_len: enc("pos_emb_max_len")?,
                predictor_hidden: 0,
                predictor_layers: 0,
                joint_hidden: 0,
                vocab: 0,
                blank: 0,
            },
            tf_layers: n("stt.sortformer.transformer.n_layers")?,
            tf_heads: n("stt.sortformer.transformer.n_heads")?,
        })
    }

    /// Milliseconds per diarization frame (80).
    pub fn frame_ms(&self) -> f64 {
        1000. * self.frame_hop as f64 / self.sample_rate as f64
    }
}

pub struct Sortformer {
    pub hparams: HParams,
    mel: Mel,
    backend: Backend,
    pub info: ModelInfo,
}

/// Where the encoder and transformer run. Only one is loaded.
// One per model: its size doesn't matter.
#[allow(clippy::large_enum_variant)]
enum Backend {
    Cpu {
        encoder: Encoder,
        transformer: Transformer,
    },
    Gpu(GpuSortformer),
}

/// One chunk of the stream: its mel window, and how many diarization frames of it are left
/// and right context.
struct Chunk {
    window: Range<usize>,
    lc: usize,
    rc: usize,
}

/// Whether the reference would make its full-context dumps (`TRANSCRIBE_SORTFORMER_OFFLINE_DUMP`).
fn offline_dump() -> bool {
    std::env::var_os("TRANSCRIBE_SORTFORMER_OFFLINE_DUMP").is_some()
}

impl Sortformer {
    /// The model, on `gpu` when there's one, else on the CPU.
    pub fn load(file: &Gguf, gpu: Option<Arc<crate::gpu::Gpu>>) -> Result<Sortformer> {
        let hparams = HParams::from_gguf(file)?;
        let backend = match gpu {
            Some(gpu) => {
                // Sized for the longest chunk window and the longest concatenation.
                let p = StreamParams::very_high_latency();
                let sub = hparams.encoder.subsampling_factor;
                let max_mel = (p.chunk_len + p.chunk_left_context + p.chunk_right_context) * sub;
                let max_frames =
                    (p.spkcache_len + p.fifo_len + max_mel.div_ceil(sub)).next_multiple_of(64);
                Backend::Gpu(GpuSortformer::load(
                    &gpu, file, &hparams, max_mel, max_frames,
                )?)
            }
            None => {
                let (encoder, transformer) = rayon::join(
                    // The stream's conformer keeps keys and values in f32 (`kv_type`).
                    || Encoder::load_keys(file, &hparams.encoder, false),
                    || Transformer::load(file, hparams.tf_layers, hparams.tf_heads),
                );
                Backend::Cpu {
                    encoder: encoder?,
                    transformer: transformer?,
                }
            }
        };
        let device = match &backend {
            Backend::Cpu { .. } => "CPU".into(),
            Backend::Gpu(g) => g.gpu.name(),
        };
        Ok(Sortformer {
            mel: Mel::from_gguf(file)?,
            backend,
            info: ModelInfo {
                architecture: "sortformer".into(),
                variant: file.metadata_str("stt.variant").unwrap_or_default().into(),
                device,
            },
            hparams,
        })
    }

    /// A whole recording of 16 kHz audio to speaker turns. `stop` is polled before every
    /// chunk (and on the CPU between the encoder's blocks); once it says so, the run returns
    /// [`Error::Cancelled`].
    pub fn run(
        &self,
        pcm: &[f32],
        trace: &mut dyn Trace,
        stop: &(dyn Fn() -> bool + Sync),
    ) -> Result<Vec<SpeakerTurn>> {
        let mut probs = self.probabilities(pcm, trace, stop)?;
        checkpoint(trace, "diar.probs", &mut probs);
        Ok(turns(&probs, self.hparams.frame_ms()))
    }

    /// The chunks a recording of `feat_len` mel frames is streamed in (NeMo's
    /// `streaming_feat_loader`): `chunk_len` diarization frames each, with up to
    /// `chunk_left_context` and `chunk_right_context` more on either side.
    fn chunks(&self, params: &StreamParams, feat_len: usize) -> Vec<Chunk> {
        let sub = self.hparams.encoder.subsampling_factor;
        let mut chunks = Vec::new();
        let (mut start, mut end) = (0, 0);
        while end < feat_len {
            let left = (params.chunk_left_context * sub).min(start);
            end = (start + params.chunk_len * sub).min(feat_len);
            let right = (params.chunk_right_context * sub).min(feat_len - end);
            chunks.push(Chunk {
                window: start - left..end + right,
                // `left` is a whole number of diarization frames; `right` may be short at the
                // end of the recording.
                lc: (left + sub / 2) / sub,
                rc: right.div_ceil(sub),
            });
            start = end;
        }
        chunks
    }

    /// The GPU the model runs on, if it runs on one.
    pub fn gpu(&self) -> Option<&crate::gpu::Gpu> {
        match &self.backend {
            Backend::Cpu { .. } => None,
            Backend::Gpu(g) => Some(&g.gpu),
        }
    }

    /// The speaker probabilities of every 80 ms frame (frames × speakers), streamed.
    fn probabilities(
        &self,
        pcm: &[f32],
        trace: &mut dyn Trace,
        stop: &(dyn Fn() -> bool + Sync),
    ) -> Result<Mat> {
        let n_spk = self.hparams.max_speakers;
        let sub = self.hparams.encoder.subsampling_factor;
        // NeMo's Sortformer frames the audio as ⌈n / hop⌉ (no trailing centering frame).
        let feat_len = pcm.len().div_ceil(self.hparams.hop_length);
        // The whole mel only when a checkpoint needs it; the stream makes a window at a time.
        let offline = offline_dump() && trace.wants("diar.preds_offline");
        let full = if offline || trace.injects("enc.mel.in") {
            let mut mel = self.mel.compute_frames(pcm, 0..feat_len)?;
            checkpoint(trace, "enc.mel.in", &mut mel);
            Some(mel)
        } else {
            None
        };
        if let (Some(mel), true) = (&full, offline) {
            self.offline(mel, trace);
        }

        let params = StreamParams::very_high_latency();
        let chunks = self.chunks(&params, feat_len);
        // A chunk doesn't depend on the stream until it's encoded: a thread makes the next
        // batch's features and subsamples them while this one encodes the last batch's chunks,
        // one by one, with the stream's state.
        let batch = match &self.backend {
            Backend::Cpu { .. } => 1,
            Backend::Gpu(g) => g.max_batch(),
        };
        let mut state = State::new(n_spk, self.hparams.encoder.d_model);
        std::thread::scope(|s| -> Result<()> {
            let (tx, rx) = std::sync::mpsc::sync_channel(1);
            let (full, chunks) = (&full, &chunks);
            s.spawn(move || {
                for group in chunks.chunks(batch) {
                    let embs = self.subsample(pcm, full.as_ref(), group);
                    let failed = embs.is_err();
                    // Stop once the stream has (it returned early) or this failed.
                    if tx.send(embs).is_err() || failed {
                        break;
                    }
                }
            });
            for (b, group) in chunks.chunks(batch).enumerate() {
                let embs = rx
                    .recv()
                    .map_err(|_| Error::Other("subsampling stopped".into()))??;
                for (i, (chunk, embs)) in group.iter().zip(&embs).enumerate() {
                    if stop() {
                        return Err(Error::Cancelled);
                    }
                    let concat = state.concat(embs);
                    trace.event(
                        "sortformer.chunk",
                        serde_json::json!({
                            "index": b * batch + i,
                            "mel": [chunk.window.start, chunk.window.end],
                            "lc": chunk.lc, "rc": chunk.rc,
                            "frames": embs.rows, "concat": concat.rows,
                        }),
                    );
                    let preds = self.predict(concat, stop)?;
                    state.update(&params, embs, &preds, (chunk.lc, chunk.rc), trace);
                }
            }
            Ok(())
        })?;
        let frames = (state.total_preds.len() / n_spk).min(feat_len.div_ceil(sub));
        state.total_preds.truncate(frames * n_spk);
        Ok(Mat::from_vec(frames, n_spk, state.total_preds))
    }

    /// Chunks' pre-encode embeddings (frames × 512 each): their mel windows, from the whole
    /// recording's features when there are any, then the subsampling.
    fn subsample(&self, pcm: &[f32], full: Option<&Mat>, chunks: &[Chunk]) -> Result<Vec<Mat>> {
        let mels = chunks
            .iter()
            .map(|c| match full {
                Some(mel) => Ok(columns(mel, c.window.clone())),
                None => self.mel.compute_frames(pcm, c.window.clone()),
            })
            .collect::<Result<Vec<_>>>()?;
        match &self.backend {
            Backend::Cpu { encoder, .. } => Ok(mels.iter().map(|m| encoder.subsample(m)).collect()),
            Backend::Gpu(g) => g.subsample(&mels),
        }
    }

    /// Speaker probabilities (frames × speakers) for pre-encode embeddings (frames × 512):
    /// the x-scaling, the conformer blocks, the projection, the transformer and the head.
    fn predict(&self, embs: Mat, stop: &(dyn Fn() -> bool + Sync)) -> Result<Mat> {
        match &self.backend {
            Backend::Cpu {
                encoder,
                transformer,
            } => {
                let x = encoder.conformer_until(embs, &mut NoTrace, stop)?;
                let t = transformer.encode(transformer.project(&x));
                Ok(transformer.speakers(&t))
            }
            Backend::Gpu(g) => {
                // Scaled as the CPU's conformer scales its input.
                let mut x = embs;
                let s = (self.hparams.encoder.d_model as f32).sqrt();
                x.map(|v| v * s);
                g.predict(&x)
            }
        }
    }

    /// The reference's full-context pass over the whole recording, for its checkpoints only
    /// (the product is the stream). On the CPU only: the GPU is sized for chunks.
    fn offline(&self, mel: &Mat, trace: &mut dyn Trace) {
        let Backend::Cpu {
            encoder,
            transformer,
        } = &self.backend
        else {
            return;
        };
        let mut x = encoder.forward(mel, &mut NoTrace);
        checkpoint(trace, "enc.fastconformer.out", &mut x);
        let mut t = transformer.project(&x);
        checkpoint(trace, "enc.encoder_proj.out", &mut t);
        let mut t = transformer.encode(t);
        checkpoint(trace, "enc.transformer.out", &mut t);
        let mut p = transformer.speakers(&t);
        checkpoint(trace, "diar.preds_offline", &mut p);
    }
}

/// Columns `range` of `m`.
fn columns(m: &Mat, range: Range<usize>) -> Mat {
    let mut out = Mat::zeros(m.rows, range.len());
    for r in 0..m.rows {
        out.row_mut(r).copy_from_slice(&m.row(r)[range.clone()]);
    }
    out
}

/// One turn per run of frames a speaker's probability is over ½ in, speaker by speaker and
/// in time order (speakers numbered from 1), as the reference's `probs_to_segments` makes
/// them. The model gives no confidence per turn.
fn turns(probs: &Mat, frame_ms: f64) -> Vec<SpeakerTurn> {
    let mut out = Vec::new();
    for s in 0..probs.cols {
        let mut run: Option<usize> = None;
        for t in 0..=probs.rows {
            let active = t < probs.rows && probs.row(t)[s] > 0.5;
            match (active, run) {
                (true, None) => run = Some(t),
                (false, Some(start)) => {
                    out.push(SpeakerTurn {
                        t0_ms: (start as f64 * frame_ms).round() as i64,
                        t1_ms: (t as f64 * frame_ms).round() as i64,
                        speaker: s as i32 + 1,
                        p: f32::NAN,
                    });
                    run = None;
                }
                _ => {}
            }
        }
    }
    out
}
