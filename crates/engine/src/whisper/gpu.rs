//! Whisper on the GPU: the same computation as the CPU oracle (`encoder.rs`, `decoder.rs`,
//! `decoding.rs`), checked against it by `cargo lab whisper`.
//!
//! The encoder ([`encoder`]) takes a batch of 30 s windows at once: the convolutions as
//! matrix multiplies over their receptive fields, the blocks' linear layers on tensor cores
//! (f16 weights, f32 accumulation, bias, GELU and residual adds in the epilogues), attention
//! flash-style on tensor cores, the residual stream, layer-norm statistics and softmax in f32.
//! Each window's encoder keys and values for every decoder layer are made right after, into a
//! decoder slot.
//!
//! The decoder ([`decoder`]) decodes the windows in its slots in lockstep, a token per step:
//! weights kept as the file's Q8_0 (exact, and half the bytes a step streams), keys and values
//! cached on the GPU in f16 as the oracle rounds them, and the logits rules and greedy choice
//! made on the GPU too, so a run of steps is recorded and submitted at once and only the
//! choices come back. A window sampling at a temperature has its logits read back each step
//! and its token chosen by the oracle's own code, with its seeded generator.

pub mod decoder;
pub mod encoder;

use super::HParams;
use crate::{
    Result,
    cpu::vector,
    gguf::Gguf,
    gpu::{
        Gpu, Kernel, Tensor,
        gemm::{Kind, Variant},
    },
    parakeet::gpu_encoder::record::Recorder,
};
use std::sync::Arc;

/// The epsilon of every layer norm (the reference's `kLayerNormEps`).
const EPS: f32 = 1e-5;

/// How the GPU model is built.
#[derive(Debug, Clone, Copy)]
pub struct GpuOptions {
    /// Windows the encoder takes per batch.
    pub max_batch: usize,
    /// Windows the decoder decodes together (each a slot with its caches).
    pub slots: usize,
    /// The encoder's matrix multiply (default: tensor cores where the adapter has them).
    pub gemm: Option<Variant>,
}

impl Default for GpuOptions {
    fn default() -> GpuOptions {
        GpuOptions {
            max_batch: 4,
            slots: 8,
            gemm: None,
        }
    }
}

/// The model on the GPU: its encoder and its decoder, which share the device and meet in the
/// decoder's cross cache.
pub struct GpuWhisper {
    pub gpu: Arc<Gpu>,
    pub encoder: encoder::Encoder,
    pub decoder: decoder::Decoder,
}

impl GpuWhisper {
    pub fn load(
        gpu: &Arc<Gpu>,
        file: &Gguf,
        hp: &HParams,
        options: &GpuOptions,
    ) -> Result<GpuWhisper> {
        if !gpu.has(crate::gpu::wgpu::Features::SUBGROUP)
            || !gpu.has(crate::gpu::wgpu::Features::SHADER_F16)
            || !gpu.has_f16_tensor_cores()
        {
            return Err(crate::Error::Unsupported(format!(
                "Whisper on {} (it needs f16 tensor cores and subgroups)",
                gpu.info.name
            )));
        }
        let encoder = encoder::Encoder::load(gpu, file, hp, options)?;
        let decoder = decoder::Decoder::load(gpu, file, hp, options)?;
        // The loads' bind groups (over temporary buffers) aren't needed again.
        gpu.clear_bind_groups();
        Ok(GpuWhisper {
            gpu: gpu.clone(),
            encoder,
            decoder,
        })
    }

    /// Windows' mel features (3000 frames × mels each) through the encoder, each window's
    /// keys and values into its decoder slot, in batches of the encoder's size. Nothing is
    /// read back; the decoder's next steps see the slots once the GPU gets to them.
    pub fn encode(&self, windows: &[(&crate::cpu::Mat, usize)]) -> Result<()> {
        for batch in windows.chunks(self.encoder.max_batch()) {
            let mels: Vec<&crate::cpu::Mat> = batch.iter().map(|(m, _)| *m).collect();
            let slots: Vec<usize> = batch.iter().map(|(_, s)| *s).collect();
            let mut rec = Recorder::new(
                &self.gpu,
                crate::parakeet::gpu_encoder::record::Timing::None,
            );
            self.encoder.record(&mut rec, &mels, None)?;
            for (i, &slot) in slots.iter().enumerate() {
                self.decoder
                    .record_cross(&mut rec, self.encoder.output(), i, slot);
            }
            rec.submit();
        }
        Ok(())
    }
}

/// The default multiply: tensor cores (64×128 tiles, K steps of 64, as Parakeet measures best).
fn default_variant() -> Variant {
    Variant {
        kind: Kind::Coop,
        bm: 64,
        bn: 128,
        bk: 64,
        tm: 32,
        tn: 64,
    }
}

/// A layer norm's scale and shift.
pub(crate) struct Norm {
    weight: Tensor,
    bias: Tensor,
}

impl Norm {
    fn load(gpu: &Gpu, file: &Gguf, name: &str) -> Result<Norm> {
        Ok(Norm {
            weight: gpu.upload_f32(name, &vector(file, &format!("{name}.weight"))?),
            bias: gpu.upload_f32(name, &vector(file, &format!("{name}.bias"))?),
        })
    }

    /// Records the norm of the first `rows` rows of `x` into `out`.
    fn record(&self, rec: &mut Recorder, kernel: &Kernel, x: &Tensor, out: &Tensor, rows: usize) {
        rec.dispatch(
            "norm",
            kernel,
            &[x, &self.weight, &self.bias, out],
            &[rows as u32, EPS.to_bits()],
            [rows as u32, 1, 1],
        );
    }
}

/// Workgroups for `threads` threads of 256, spread over two dimensions when there are more
/// than one dimension takes.
fn groups_2d(threads: usize) -> [u32; 3] {
    let groups = threads.div_ceil(256) as u32;
    if groups <= 65535 {
        [groups, 1, 1]
    } else {
        let x = 32768;
        [x, groups.div_ceil(x), 1]
    }
}
