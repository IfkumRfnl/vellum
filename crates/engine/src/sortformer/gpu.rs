//! Sortformer on the GPU: the FastConformer is Parakeet's GPU encoder (its subsampling run on
//! a batch of chunk windows at once; its blocks on each `[speaker cache | FIFO | chunk]`
//! concatenation), and the projection, the 18 transformer blocks and the speaker head run
//! after the last block in the same submission ([`Tail`]), so only the 4 logits per frame come
//! back. The streaming state stays on the CPU, exactly as in [`super::stream`].
//!
//! Weights in f16, everything else in f32, as for Parakeet.

use super::HParams;
use crate::{
    Error, Result,
    cpu::{Mat, vector},
    gguf::Gguf,
    gpu::{
        Gpu, Kernel, Tensor,
        gemm::{Activation, Epilogue, Gemm, Operands, Weights},
    },
    parakeet::gpu_encoder::{
        GpuEncoder, GpuEncoderOptions, LAYER_NORM, Tail, record::Recorder, stacked, upload_linear,
    },
};
use std::sync::{Arc, Mutex};

const TF_ATTEND: &str = include_str!("tf_attend.wgsl");
const RELU: &str = include_str!("relu.wgsl");

/// Every layer norm's epsilon.
const EPS: f32 = 1e-5;

pub struct GpuSortformer {
    pub gpu: Arc<Gpu>,
    encoder: GpuEncoder,
    tail: Transformer,
}

impl GpuSortformer {
    /// The model on `gpu`, sized for chunk windows of up to `max_mel_frames` and
    /// concatenations of up to `max_frames` encoder frames.
    pub fn load(
        gpu: &Arc<Gpu>,
        file: &Gguf,
        hparams: &HParams,
        max_mel_frames: usize,
        max_frames: usize,
    ) -> Result<GpuSortformer> {
        let options = GpuEncoderOptions {
            max_mel_frames,
            max_frames: Some(max_frames),
            // A chunk is one window of up to ~600 frames: submit every few blocks, so the GPU
            // starts while the rest is recorded.
            rows_per_submit: 2000,
            ..GpuEncoderOptions::default()
        };
        let encoder = GpuEncoder::load_with(gpu, file, &hparams.encoder, &options)?;
        let tail = Transformer::load(gpu, file, hparams, encoder.gemm_variant(), max_frames)?;
        Ok(GpuSortformer {
            gpu: gpu.clone(),
            encoder,
            tail,
        })
    }

    /// Windows per subsampling batch.
    pub fn max_batch(&self) -> usize {
        self.encoder.max_batch()
    }

    /// Chunk windows' mel features to their pre-encode embeddings (frames × 512 each).
    pub fn subsample(&self, mels: &[Mat]) -> Result<Vec<Mat>> {
        self.encoder.subsample(mels)
    }

    /// Speaker probabilities (frames × speakers) for x-scaled pre-encode embeddings.
    pub fn predict(&self, x: &Mat) -> Result<Mat> {
        let mut p = self.encoder.encode_rows(x, &self.tail)?;
        p.map(crate::cpu::sigmoid);
        Ok(p)
    }
}

struct Norm {
    weight: Tensor,
    bias: Tensor,
}

struct Block {
    /// q, k and v stacked.
    qkv: Weights,
    out: Weights,
    norm1: Norm,
    ff_in: Weights,
    ff_out: Weights,
    norm2: Norm,
}

/// The activations, for the longest concatenation.
struct Buffers {
    x: [Tensor; 2],
    qkv: Tensor,
    context: Tensor,
    hidden: Tensor,
    logits: Tensor,
}

/// The projection, the post-LN transformer and the speaker head, recorded after the encoder.
struct Transformer {
    gpu: Arc<Gpu>,
    d: usize,
    heads: usize,
    gemm: Gemm,
    layer_norm: Arc<Kernel>,
    attend: Arc<Kernel>,
    relu: Arc<Kernel>,
    proj: Weights,
    blocks: Vec<Block>,
    fc1: Weights,
    head: Weights,
    buffers: Mutex<Buffers>,
}

impl Transformer {
    fn load(
        gpu: &Arc<Gpu>,
        file: &Gguf,
        hparams: &HParams,
        variant: crate::gpu::gemm::Variant,
        max_frames: usize,
    ) -> Result<Transformer> {
        let proj = upload_linear(gpu, file, "diar.encoder_proj", "encoder_proj")?;
        let d = proj.n;
        let heads = hparams.tf_heads;
        let d_head = d / heads.max(1);
        if heads * d_head != d || d_head > 64 || !d.is_multiple_of(4) {
            return Err(Error::Unsupported(format!(
                "a GPU transformer of {d} with {heads} heads"
            )));
        }
        let norm = |name: &str| -> Result<Norm> {
            Ok(Norm {
                weight: gpu.upload_f32(name, &vector(file, &format!("{name}.weight"))?),
                bias: gpu.upload_f32(name, &vector(file, &format!("{name}.bias"))?),
            })
        };
        let blocks = (0..hparams.tf_layers)
            .map(|l| {
                let name = |part: &str| format!("tf.blocks.{l}.{part}");
                Ok(Block {
                    qkv: stacked(
                        gpu,
                        file,
                        "tf.qkv",
                        &["q", "k", "v"].map(|p| name(&format!("attn.{p}"))),
                    )?,
                    out: upload_linear(gpu, file, &name("attn.out"), "tf.out")?,
                    norm1: norm(&name("norm_1"))?,
                    ff_in: upload_linear(gpu, file, &name("ff.in"), "tf.ff_in")?,
                    ff_out: upload_linear(gpu, file, &name("ff.out"), "tf.ff_out")?,
                    norm2: norm(&name("norm_2"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let fc1 = upload_linear(gpu, file, "diar.fc1", "fc1")?;
        let head = upload_linear(gpu, file, "diar.single_spk_head", "single_spk_head")?;
        let d_ff = blocks.first().map_or(0, |b| b.ff_in.n);
        let attention = [
            ("HEADS", heads as u32),
            ("D_HEAD", d_head as u32),
            ("D_MODEL", d as u32),
        ];
        let rows = max_frames;
        Ok(Transformer {
            gpu: gpu.clone(),
            d,
            heads,
            gemm: Gemm::new(gpu, variant)?,
            layer_norm: gpu.kernel(
                "enc_layer_norm",
                LAYER_NORM,
                &[("D4", (d / 4) as u32)],
                &[false, false, false, true],
                8,
            )?,
            attend: gpu.kernel("tf_attend", TF_ATTEND, &attention, &[false, true], 8)?,
            relu: gpu.kernel("relu", RELU, &[], &[true], 4)?,
            buffers: Mutex::new(Buffers {
                x: [
                    gpu.zeros_f32("tf x 0", rows * d),
                    gpu.zeros_f32("tf x 1", rows * d),
                ],
                qkv: gpu.zeros_f32("tf qkv", rows * 3 * d),
                context: gpu.zeros_f32("tf context", rows * d),
                hidden: gpu.zeros_f32("tf hidden", rows * d_ff.max(d)),
                logits: gpu.zeros_f32("logits", rows * head.n),
            }),
            proj,
            blocks,
            fc1,
            head,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        rec: &mut Recorder,
        label: &str,
        x: &Tensor,
        weights: &Weights,
        y: &Tensor,
        m: usize,
        epilogue: Epilogue,
    ) {
        let gpu = &*self.gpu;
        rec.with_pass(label, |pass| {
            self.gemm
                .encode(gpu, pass, &Operands { x, weights, y, m }, epilogue);
        });
    }

    fn layer_norm(&self, rec: &mut Recorder, x: &Tensor, norm: &Norm, out: &Tensor, rows: usize) {
        rec.dispatch(
            "tf/norm",
            &self.layer_norm,
            &[x, &norm.weight, &norm.bias, out],
            &[rows as u32, EPS.to_bits()],
            [rows as u32, 1, 1],
        );
    }
}

impl Tail for Transformer {
    fn record(&self, rec: &mut Recorder, x: &Tensor, rows: usize, staging: &wgpu::Buffer) -> usize {
        let b = self.buffers.lock().unwrap();
        let add = Epilogue {
            accumulate: true,
            ..Epilogue::default()
        };
        let relu = Epilogue {
            activation: Activation::Relu,
            ..Epilogue::default()
        };
        let d_head = self.d / self.heads;
        let scale = 1. / (d_head as f32).sqrt().sqrt();
        self.gemm(
            rec,
            "tf/proj",
            x,
            &self.proj,
            &b.x[0],
            rows,
            Epilogue::default(),
        );
        for block in &self.blocks {
            // Attention, added back, then layer-normed (post-LN).
            self.gemm(
                rec,
                "tf/qkv",
                &b.x[0],
                &block.qkv,
                &b.qkv,
                rows,
                Epilogue::default(),
            );
            rec.dispatch(
                "tf/attend",
                &self.attend,
                &[&b.qkv, &b.context],
                &[rows as u32, scale.to_bits()],
                [rows.div_ceil(32) as u32, self.heads as u32, 1],
            );
            self.gemm(rec, "tf/out", &b.context, &block.out, &b.x[0], rows, add);
            self.layer_norm(rec, &b.x[0], &block.norm1, &b.x[1], rows);
            // Feed-forward, the same.
            self.gemm(
                rec,
                "tf/ff_in",
                &b.x[1],
                &block.ff_in,
                &b.hidden,
                rows,
                relu,
            );
            self.gemm(
                rec,
                "tf/ff_out",
                &b.hidden,
                &block.ff_out,
                &b.x[1],
                rows,
                add,
            );
            self.layer_norm(rec, &b.x[1], &block.norm2, &b.x[0], rows);
        }
        // The head: ReLU, fc1 + ReLU, the speaker logits (the sigmoid is the CPU's).
        let len = (rows * self.d) as u32;
        rec.dispatch(
            "tf/relu",
            &self.relu,
            &[&b.x[0]],
            &[len],
            [len.div_ceil(256), 1, 1],
        );
        self.gemm(rec, "tf/fc1", &b.x[0], &self.fc1, &b.x[1], rows, relu);
        self.gemm(
            rec,
            "tf/head",
            &b.x[1],
            &self.head,
            &b.logits,
            rows,
            Epilogue::default(),
        );
        rec.copy(&b.logits, 0, staging, 0, (rows * self.head.n * 4) as u64);
        self.head.n
    }
}
