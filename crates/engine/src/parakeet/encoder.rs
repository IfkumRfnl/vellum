//! The FastConformer encoder: 8× convolutional subsampling (`enc.pre_encode.*`), relative
//! positional encodings, and the conformer blocks (`enc.blocks.<L>.*`), as transcribe.cpp builds
//! them (`src/arch/parakeet/encoder.cpp`, `src/conformer/conformer.cpp` there) for an offline
//! run of one window.
//!
//! This model (unified-en) is trained with chunked attention (`chunked_limited_with_rc`), but
//! the reference only applies a chunk mask when it streams; an offline window gets full
//! attention, no mask, and the ordinary non-causal subsampling. So does this.

mod attention;
mod block;
pub(crate) mod rounding;
mod subsampling;

use super::HParams;
use crate::{Result, Trace, cpu::Mat, gguf::Gguf, trace::checkpoint};
use block::Block;
use subsampling::Subsampling;

pub struct Encoder {
    d_model: usize,
    /// NeMo's `xscaling`: the subsampled features are multiplied by √d_model before block 0.
    xscale: f32,
    subsampling: Subsampling,
    blocks: Vec<Block>,
}

impl Encoder {
    pub fn load(file: &Gguf, hparams: &HParams) -> Result<Encoder> {
        if hparams.subsampling_factor != 8 {
            return Err(crate::Error::Unsupported(format!(
                "Parakeet with {}× subsampling",
                hparams.subsampling_factor
            )));
        }
        // The convolution module's padding: centred unless the file says otherwise (the
        // streaming variants pad causally, which this offline port doesn't do).
        let context =
            |side: &str| file.metadata_u64(&format!("stt.parakeet.encoder.conv_context_{side}"));
        let centred = Some((hparams.conv_kernel / 2) as u64);
        if [context("left"), context("right")]
            .iter()
            .any(|c| c.is_some() && *c != centred)
        {
            return Err(crate::Error::Unsupported(
                "Parakeet with a causal convolution module".into(),
            ));
        }
        let blocks = (0..hparams.n_layers)
            .map(|l| Block::load(file, &format!("enc.blocks.{l}"), hparams))
            .collect::<Result<_>>()?;
        Ok(Encoder {
            d_model: hparams.d_model,
            xscale: if hparams.xscaling {
                (hparams.d_model as f32).sqrt()
            } else {
                1.
            },
            subsampling: Subsampling::load(file, hparams)?,
            blocks,
        })
    }

    /// Mel features (`num_mels` × frames) to encoder states (⌈frames / 8⌉ × `d_model`). The
    /// frames are the ones the encoder sees: the reference drops the mel's trailing
    /// centering-pad frame (keeping `Mel::valid_frames`) after its `enc.mel.in` dump.
    pub fn forward(&self, mel: &Mat, trace: &mut dyn Trace) -> Mat {
        let mut x = self.subsample(mel);
        checkpoint(trace, "enc.pre_encode.out", &mut x);
        self.conformer(x, trace)
    }

    /// The 8× subsampling alone: mel features (`num_mels` × frames) to ⌈frames / 8⌉ ×
    /// `d_model` (NeMo's `pre_encode`, before the x-scaling).
    pub fn subsample(&self, mel: &Mat) -> Mat {
        self.subsampling.forward(mel)
    }

    /// The conformer on subsampled features (frames × `d_model`): the x-scaling, the
    /// positional encodings of the frames given, every block. Streaming Sortformer runs it on
    /// its speaker cache and newest chunk joined, which [`subsample`](Self::subsample) made
    /// apart.
    pub fn conformer(&self, mut x: Mat, trace: &mut dyn Trace) -> Mat {
        if self.xscale != 1. {
            let s = self.xscale;
            x.map(|v| v * s);
        }
        let mut pos = positional_encodings(x.rows, self.d_model);
        checkpoint(trace, "enc.pos_emb", &mut pos);
        for (l, block) in self.blocks.iter().enumerate() {
            x = block.forward(x, &pos, l, trace);
        }
        x
    }
}

/// The sinusoidal encodings of every relative position a window of `frames` frames has, from
/// `frames - 1` (row 0) down to `-(frames - 1)` (the last row): NeMo's `RelPositionalEncoding`,
/// computed in f32 exactly as the reference fills `pos_emb.in`.
fn positional_encodings(frames: usize, d_model: usize) -> Mat {
    let len = 2 * frames - 1;
    let ln_10000 = 10000f32.ln();
    let div: Vec<f32> = (0..d_model / 2)
        .map(|k| ((2 * k) as f32 * (-ln_10000 / d_model as f32)).exp())
        .collect();
    let mut pe = Mat::zeros(len, d_model);
    for i in 0..len {
        let position = (frames as i64 - 1 - i as i64) as f32;
        let row = pe.row_mut(i);
        for (k, d) in div.iter().enumerate() {
            row[2 * k] = (position * d).sin();
            row[2 * k + 1] = (position * d).cos();
        }
    }
    pe
}
