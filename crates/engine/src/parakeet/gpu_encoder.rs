//! The FastConformer encoder on the GPU: the same computation as the CPU oracle
//! ([`super::encoder`]), for a batch of windows at once, checked against it checkpoint by
//! checkpoint by `cargo lab encoder`.
//!
//! The design (`GPU.md`): weights in f16, everything else — the residual stream, every
//! accumulator, layer-norm statistics, softmax — in f32. A batch of windows is packed as one
//! sequence of rows with a table of where each window starts; the linear layers, layer norms
//! and feed-forward run over all rows at once, and only the subsampling, attention and the
//! depthwise convolution stop at window edges, so a window's result doesn't depend on its
//! batch. Per block: five layer norms, seven matrix multiplies (q, k and v fused into one;
//! bias, SiLU, the ½ scale and every residual add in their epilogues), the attention scores
//! with relative positions computed by index from a table of `W_pos · pos_emb` made at load,
//! softmax, the weighted values, and GLU + depthwise convolution + batch norm + SiLU in one
//! kernel. Buffers and bind groups are made once; shapes are immediates; no pipeline depends
//! on a window's length or the batch size.

pub(crate) mod record;

use super::HParams;
use crate::{
    Error, Result, Trace,
    cpu::{Linear, Mat, vector},
    gguf::Gguf,
    gpu::{
        Gpu, Kernel, Profiler, Tensor, f16,
        gemm::{self, Activation, Epilogue, Gemm, Kind, Operands, Variant, Weights},
    },
};
use rayon::prelude::*;
use record::{Recorder, Timing};
use std::sync::{Arc, Mutex};

const PRE_CONV: &str = include_str!("gpu_encoder/pre_conv.wgsl");
const DEPTHWISE2D: &str = include_str!("gpu_encoder/depthwise2d.wgsl");
pub(crate) const LAYER_NORM: &str = include_str!("gpu_encoder/layer_norm.wgsl");
const SCORES: &str = include_str!("gpu_encoder/scores.wgsl");
const ATTEND: &str = include_str!("gpu_encoder/attend.wgsl");
const CONV_MODULE: &str = include_str!("gpu_encoder/conv_module.wgsl");

/// The epsilon of every layer norm and of the batch norm.
const EPS: f32 = 1e-5;

/// How the encoder is built.
#[derive(Debug, Clone)]
pub struct GpuEncoderOptions {
    /// The matrix multiply (default: the tensor-core kernel where the adapter has one, else
    /// the portable one; [`Gemm::autotune`] picks the best tile shape per adapter).
    pub gemm: Option<Variant>,
    /// Windows per batch the buffers are sized for (more are encoded in several batches).
    pub max_batch: usize,
    /// The longest window, in mel frames (10 ms each); the positional table and the buffers
    /// are sized for it.
    pub max_mel_frames: usize,
    /// Work per submit, in rows × blocks: a submit goes out once this much is recorded, so
    /// no single one runs long (about 15 ms on an RTX 4060, well under a second on an iGPU).
    pub rows_per_submit: usize,
    /// The most encoder frames a window can have, when it's more than the longest mel window
    /// subsampled: Sortformer encodes its speaker cache and newest chunk together.
    pub max_frames: Option<usize>,
}

impl Default for GpuEncoderOptions {
    fn default() -> GpuEncoderOptions {
        GpuEncoderOptions {
            gemm: None,
            max_batch: 8,
            // 30.72 s: the engine takes windows up to 30 s (and one centering frame).
            max_mel_frames: 3072,
            rows_per_submit: 6000,
            max_frames: None,
        }
    }
}

/// The tile shapes that measure best on an RTX 4060 (`cargo lab kernels`). K steps of 64
/// rather than 32 make no difference to a long recording and take a short window's encoding
/// from 13.5 to 11.5 ms (it's bound by streaming the weights, at any length).
fn default_variant(gpu: &Gpu) -> Variant {
    let (kind, bm, bn, bk, tm, tn) = if gpu.has_f16_tensor_cores() {
        (Kind::Coop, 64, 128, 64, 32, 64)
    } else {
        (Kind::Simt, 128, 64, 16, 8, 4)
    };
    Variant {
        kind,
        bm,
        bn,
        bk,
        tm,
        tn,
    }
}

/// Frames after one stride-2 stage: ⌈n / 2⌉.
fn halve(n: usize) -> usize {
    n.div_ceil(2)
}

struct Kernels {
    pre_conv: Arc<Kernel>,
    depthwise2d: Arc<Kernel>,
    layer_norm: Arc<Kernel>,
    scores: Arc<Kernel>,
    attend: Arc<Kernel>,
    conv_module: Arc<Kernel>,
}

struct Norm {
    weight: Tensor,
    bias: Tensor,
}

struct PreEncode {
    /// conv.0, conv.2, conv.5: `[tap][channel]` weights and biases.
    conv0: (Tensor, Tensor),
    depthwise: [(Tensor, Tensor); 2],
    /// conv.3, conv.6.
    pointwise: [Weights; 2],
    /// The projection to d_model, its inputs reordered frequency-major (see `load`).
    out: Weights,
}

struct Block {
    norm_ff1: Norm,
    ff1: [Weights; 2],
    norm_attn: Norm,
    /// q, k and v stacked: 3·d_model outputs.
    qkv: Weights,
    out: Weights,
    /// `W_pos · pos_emb` for every offset the longest window has, `[2R − 1][d_model]`.
    pos: Tensor,
    bias_u: Tensor,
    bias_v: Tensor,
    norm_conv: Norm,
    pointwise1: Weights,
    /// The depthwise kernel with batch norm folded in, `[tap][channel]`, and its bias.
    depthwise: (Tensor, Tensor),
    pointwise2: Weights,
    norm_ff2: Norm,
    ff2: [Weights; 2],
    norm_out: Norm,
}

/// The persistent activations, sized for the largest batch.
struct Buffers {
    mel: Tensor,
    /// `(first row, frames)` per window, u32.
    windows: Tensor,
    /// Subsampling scratch for one window at a time.
    pre: [Tensor; 2],
    /// The residual stream, ping-ponged by the blocks' final layer norm.
    residual: [Tensor; 2],
    normed: Tensor,
    qkv: Tensor,
    /// The feed-forward hidden layer, the first pointwise product, the flattened subsampling.
    hidden: Tensor,
    /// The attention context, the convolution module's output.
    context: Tensor,
    scores: Tensor,
}

/// The encoder's shape.
#[derive(Debug, Clone, Copy)]
struct Dims {
    d_model: usize,
    heads: usize,
    d_ff: usize,
    kernel: usize,
    channels: usize,
    mels: usize,
    /// Frequencies after subsampling.
    freq: usize,
    /// R: the most encoder frames a window can have.
    max_frames: usize,
    max_mel_frames: usize,
    max_batch: usize,
}

pub struct GpuEncoder {
    gpu: Arc<Gpu>,
    dims: Dims,
    xscale: f32,
    rows_per_submit: usize,
    gemm: Gemm,
    kernels: Kernels,
    pre: PreEncode,
    blocks: Vec<Block>,
    /// The joint network's encoder projection (`joint.enc`, d_model → joint hidden), applied
    /// to the last block's output on the fast path, so only its narrower result comes back
    /// (Parakeet's; a model without one has none).
    joint: Option<Weights>,
    buffers: Mutex<Buffers>,
    /// Mapped-readback buffers for the batches in flight (two: one being encoded while the
    /// last one's result is read).
    staging: Mutex<Vec<wgpu::Buffer>>,
}

/// A timed pass: its label and GPU milliseconds.
pub type GpuTime = (String, f64);

/// A batch submitted to the GPU, its result not yet read: [`GpuEncoder::finish`] waits for it.
/// Up to two can be in flight, so the GPU works on one while the last one is read back.
pub struct Pending {
    staging: wgpu::Buffer,
    windows: Vec<Window>,
    submission: wgpu::SubmissionIndex,
    /// Values per row of the result: d_model, or the joint's width when projected.
    width: usize,
}

/// What a batch reads back.
#[derive(Clone, Copy)]
enum Output<'a> {
    /// The encoder states (`enc.final`).
    Encoded,
    /// The encoder states through the joint's encoder projection.
    Projected,
    /// The subsampling's output (`enc.pre_encode.out`, before the x-scaling); no blocks run.
    PreEncoded,
    /// Whatever a [`Tail`] makes of the encoder states.
    Tail(&'a dyn Tail),
}

/// What a batch is made of.
#[derive(Clone, Copy)]
enum Input<'a> {
    /// Each window's mel features.
    Mels(&'a [Mat]),
    /// One window already subsampled and x-scaled: frames × d_model.
    Rows(&'a Mat),
}

/// Layers that run on the GPU after the encoder's last block, in the same submission, their
/// result being what's read back: Sortformer's transformer and speaker head.
pub(crate) trait Tail {
    /// Records the layers on the first `rows` rows of `x` (d_model wide) and copies their
    /// result into the start of `staging`; returns its width per row.
    fn record(&self, rec: &mut Recorder, x: &Tensor, rows: usize, staging: &wgpu::Buffer) -> usize;
}

/// One window of a batch: where its mel and its rows are.
#[derive(Clone, Copy)]
struct Window {
    mel_frames: usize,
    mel_offset: usize,
    start: usize,
    frames: usize,
}

impl GpuEncoder {
    pub fn load(gpu: &Arc<Gpu>, file: &Gguf, hparams: &HParams) -> Result<GpuEncoder> {
        GpuEncoder::load_with(gpu, file, hparams, &GpuEncoderOptions::default())
    }

    pub fn load_with(
        gpu: &Arc<Gpu>,
        file: &Gguf,
        hparams: &HParams,
        options: &GpuEncoderOptions,
    ) -> Result<GpuEncoder> {
        let mels = file
            .metadata_u64("stt.frontend.num_mels")
            .ok_or_else(|| Error::Gguf("no stt.frontend.num_mels".into()))?
            as usize;
        let max_frames =
            halve(halve(halve(options.max_mel_frames))).max(options.max_frames.unwrap_or(0));
        let dims = Dims {
            d_model: hparams.d_model,
            heads: hparams.n_heads,
            d_ff: hparams.d_ff,
            kernel: hparams.conv_kernel,
            channels: hparams.subsampling_channels,
            mels,
            freq: halve(halve(halve(mels))),
            max_frames,
            max_mel_frames: options.max_mel_frames,
            max_batch: options.max_batch.max(1),
        };
        let d_head = dims.d_model / dims.heads.max(1);
        // Layer norms take any width; the convolution module needs a multiple of 256 channels,
        // the attention heads 64 features or a multiple of 128.
        if hparams.subsampling_factor != 8
            || !dims.d_model.is_multiple_of(256)
            || dims.heads * d_head != dims.d_model
            || !(d_head == 64 || d_head.is_multiple_of(128))
            || !dims.channels.is_multiple_of(16)
            || dims.kernel.is_multiple_of(2)
        {
            return Err(Error::Unsupported(format!(
                "the GPU encoder for d_model {}, {} heads, {}× subsampling, {} channels",
                dims.d_model, dims.heads, hparams.subsampling_factor, dims.channels
            )));
        }
        let gemm = Gemm::new(gpu, options.gemm.unwrap_or_else(|| default_variant(gpu)))?;
        let c4 = (dims.channels / 4) as u32;
        let m4 = (dims.d_model / 4) as u32;
        let attention = [
            ("HEADS", dims.heads as u32),
            ("D_HEAD", d_head as u32),
            ("D_MODEL", dims.d_model as u32),
        ];
        let kernels = Kernels {
            pre_conv: gpu.kernel(
                "enc_pre_conv",
                PRE_CONV,
                &[("C4", c4)],
                &[false, false, false, false, false, true],
                20,
            )?,
            depthwise2d: gpu.kernel(
                "enc_depthwise2d",
                DEPTHWISE2D,
                &[("C4", c4)],
                &[false, false, false, true],
                16,
            )?,
            layer_norm: gpu.kernel(
                "enc_layer_norm",
                LAYER_NORM,
                &[("D4", m4)],
                &[false, false, false, true],
                8,
            )?,
            scores: gpu.kernel(
                "enc_scores",
                SCORES,
                &attention,
                &[false, false, false, false, false, true],
                12,
            )?,
            attend: gpu.kernel(
                "enc_attend",
                ATTEND,
                &attention,
                &[false, false, false, true],
                4,
            )?,
            conv_module: gpu.kernel(
                "enc_conv_module",
                CONV_MODULE,
                &[("C4", m4), ("K", dims.kernel as u32)],
                &[false, false, false, false, true],
                8,
            )?,
        };

        let pre = load_pre_encode(gpu, file, &dims)?;
        let pe = gpu.upload_f32(
            "pos_emb",
            &positional_encodings(dims.max_frames, dims.d_model),
        );
        // The blocks' weights converted to f16 on every core (most of the load's time).
        let loaded: Vec<(Block, Weights)> = (0..hparams.n_layers)
            .into_par_iter()
            .map(|l| -> Result<(Block, Weights)> {
                let name = format!("enc.blocks.{l}");
                Ok((
                    load_block(gpu, file, &name, &dims)?,
                    upload_linear(gpu, file, &format!("{name}.attn.linear_pos"), "linear_pos")?,
                ))
            })
            .collect::<Result<_>>()?;
        let mut blocks = Vec::with_capacity(hparams.n_layers);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        for (block, linear_pos) in loaded {
            // The layer's projected encodings, once: every window reads its offsets from them.
            let mut pass = encoder.begin_compute_pass(&Default::default());
            gemm.encode(
                gpu,
                &mut pass,
                &Operands {
                    x: &pe,
                    weights: &linear_pos,
                    y: &block.pos,
                    m: 2 * dims.max_frames - 1,
                },
                Epilogue::default(),
            );
            drop(pass);
            blocks.push(block);
        }
        gpu.queue.submit([encoder.finish()]);
        gpu.wait()?;
        // The load's bind groups (over the encodings and the positional weights) aren't
        // needed again.
        gpu.clear_bind_groups();

        let joint = match file.tensor("joint.enc.weight") {
            Some(_) => Some(upload_linear(gpu, file, "joint.enc", "joint.enc")?),
            None => None,
        };
        if let Some(joint) = &joint
            && (joint.k != dims.d_model || joint.n > dims.d_model)
        {
            return Err(Error::Gguf(format!(
                "joint.enc is {}×{}, for an encoder of {}",
                joint.n, joint.k, dims.d_model
            )));
        }
        let rows = dims.max_batch * dims.max_frames;
        let pre_points = halve(halve(dims.max_mel_frames)) * halve(halve(dims.mels));
        let scores = dims.max_batch * dims.heads * dims.max_frames * dims.max_frames;
        let buffers = Buffers {
            mel: gpu.zeros_f32("mel", dims.max_batch * dims.mels * dims.max_mel_frames),
            windows: gpu.zeros_f32("windows", 2 * dims.max_batch),
            pre: [
                gpu.zeros_f32("pre_encode 1", pre_points * dims.channels),
                gpu.zeros_f32("pre_encode 2", pre_points * dims.channels),
            ],
            residual: [
                gpu.zeros_f32("residual 0", rows * dims.d_model),
                gpu.zeros_f32("residual 1", rows * dims.d_model),
            ],
            normed: gpu.zeros_f32("normed", rows * dims.d_model),
            qkv: gpu.zeros_f32("qkv", rows * 3 * dims.d_model),
            hidden: gpu.zeros_f32(
                "hidden",
                rows * dims
                    .d_ff
                    .max(2 * dims.d_model)
                    .max(dims.channels * dims.freq),
            ),
            context: gpu.zeros_f32("context", rows * dims.d_model),
            scores: gpu.zeros_f32("scores", scores),
        };
        let staging = (0..2)
            .map(|_| {
                gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("encoder readback"),
                    size: (rows * dims.d_model * 4) as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        Ok(GpuEncoder {
            gpu: gpu.clone(),
            dims,
            xscale: if hparams.xscaling {
                (hparams.d_model as f32).sqrt()
            } else {
                1.
            },
            rows_per_submit: options.rows_per_submit.max(1),
            gemm,
            kernels,
            pre,
            blocks,
            joint,
            buffers: Mutex::new(buffers),
            staging: Mutex::new(staging),
        })
    }

    /// The matrix multiply in use.
    pub fn gemm_variant(&self) -> Variant {
        self.gemm.variant
    }

    /// Windows per GPU batch.
    pub fn max_batch(&self) -> usize {
        self.dims.max_batch
    }

    /// The longest window [`encode`](Self::encode) takes, in mel frames.
    pub fn max_mel_frames(&self) -> usize {
        self.dims.max_mel_frames
    }

    /// Each window's mel features (`num_mels` × the frames the encoder sees) to its encoder
    /// states (⌈frames / 8⌉ × d_model), in order, [`max_batch`](Self::max_batch) windows per
    /// GPU batch, the next batch submitted before the last one is read back. Nothing is read
    /// back until a batch is done.
    pub fn encode(&self, mels: &[Mat]) -> Result<Vec<Mat>> {
        let mut out = Vec::with_capacity(mels.len());
        let mut last: Option<Pending> = None;
        for batch in mels.chunks(self.dims.max_batch) {
            let pending = self.submit(batch)?;
            if let Some(last) = last.replace(pending) {
                out.extend(self.finish(last)?);
            }
        }
        if let Some(last) = last {
            out.extend(self.finish(last)?);
        }
        Ok(out)
    }

    /// Records and submits one batch (at most [`max_batch`](Self::max_batch) windows) without
    /// waiting for it. At most two can be pending at once.
    pub fn submit(&self, mels: &[Mat]) -> Result<Pending> {
        self.submit_as(mels, Output::Encoded)
    }

    /// [`submit`](Self::submit), the result being each window's frames through the joint's
    /// encoder projection (⌈frames / 8⌉ × joint hidden): all the decoder needs of the encoder.
    pub fn submit_projected(&self, mels: &[Mat]) -> Result<Pending> {
        self.submit_as(mels, Output::Projected)
    }

    /// [`encode`](Self::encode), projected as [`submit_projected`](Self::submit_projected)
    /// projects.
    pub fn encode_projected(&self, mels: &[Mat]) -> Result<Vec<Mat>> {
        let mut out = Vec::with_capacity(mels.len());
        for batch in mels.chunks(self.dims.max_batch) {
            out.extend(self.finish(self.submit_projected(batch)?)?);
        }
        Ok(out)
    }

    fn submit_as(&self, mels: &[Mat], output: Output) -> Result<Pending> {
        self.check_batch(mels)?;
        if matches!(output, Output::Projected) && self.joint.is_none() {
            return Err(Error::Other("this encoder has no joint projection".into()));
        }
        let staging = self.take_staging()?;
        let r = self.record(Input::Mels(mels), None, Timing::None, output, &staging);
        self.pending(staging, r)
    }

    /// Each window's subsampled features (⌈frames / 8⌉ × d_model, NeMo's `pre_encode` before
    /// the x-scaling), [`max_batch`](Self::max_batch) windows per GPU batch; no block runs.
    pub fn subsample(&self, mels: &[Mat]) -> Result<Vec<Mat>> {
        let mut out = Vec::with_capacity(mels.len());
        for batch in mels.chunks(self.dims.max_batch) {
            out.extend(self.finish(self.submit_as(batch, Output::PreEncoded)?)?);
        }
        Ok(out)
    }

    /// The blocks on one window of subsampled, x-scaled features (frames × d_model), then
    /// `tail`'s layers on their output, all in one submission; `tail`'s result comes back.
    pub(crate) fn encode_rows(&self, x: &Mat, tail: &dyn Tail) -> Result<Mat> {
        if x.cols != self.dims.d_model || x.rows == 0 || x.rows > self.dims.max_frames {
            return Err(Error::Other(format!(
                "{}×{} features for an encoder of {} and at most {} frames",
                x.rows, x.cols, self.dims.d_model, self.dims.max_frames
            )));
        }
        let staging = self.take_staging()?;
        let r = self.record(
            Input::Rows(x),
            None,
            Timing::None,
            Output::Tail(tail),
            &staging,
        );
        let mut out = self.finish(self.pending(staging, r)?)?;
        Ok(out.remove(0))
    }

    fn pending(
        &self,
        staging: wgpu::Buffer,
        recorded: Result<(Vec<Window>, wgpu::SubmissionIndex, usize)>,
    ) -> Result<Pending> {
        match recorded {
            Ok((windows, submission, width)) => Ok(Pending {
                staging,
                windows,
                submission,
                width,
            }),
            Err(e) => {
                self.staging.lock().unwrap().push(staging);
                Err(e)
            }
        }
    }

    /// Waits for a submitted batch and reads its windows' encoder states back.
    pub fn finish(&self, pending: Pending) -> Result<Vec<Mat>> {
        let Pending {
            staging,
            windows,
            submission,
            width,
        } = pending;
        let rows: usize = windows.iter().map(|w| w.frames).sum();
        let spans: Vec<(usize, usize)> = windows
            .iter()
            .map(|w| (w.start * width, w.frames * width))
            .collect();
        let data = map_read(
            &self.gpu,
            &staging,
            (rows * width * 4) as u64,
            Some(submission),
            &spans,
        );
        self.staging.lock().unwrap().push(staging);
        Ok(windows
            .iter()
            .zip(data?)
            .map(|(w, data)| Mat::from_vec(w.frames, width, data))
            .collect())
    }

    fn take_staging(&self) -> Result<wgpu::Buffer> {
        self.staging
            .lock()
            .unwrap()
            .pop()
            .ok_or_else(|| Error::Other("more than two GPU encoder batches in flight".into()))
    }

    /// [`encode`](Self::encode) of one batch, reading back window 0's checkpoints that `trace`
    /// wants (`enc.pre_encode.out`, `enc.block.<L>.ff1`, `.attn`, `.conv`, `.ff2`, `.out`),
    /// as the CPU oracle names them. Debug only: each read waits for the GPU.
    pub fn encode_traced(&self, mels: &[Mat], trace: &mut dyn Trace) -> Result<Vec<Mat>> {
        self.check_batch(mels)?;
        let staging = self.take_staging()?;
        let recorded = self.record(
            Input::Mels(mels),
            Some(trace),
            Timing::None,
            Output::Encoded,
            &staging,
        );
        self.finish(self.pending(staging, recorded)?)
    }

    /// [`encode`](Self::encode) of one batch with GPU timestamps: per submit (`per_dispatch`
    /// false: the fast path's own time, labels `batch`) or per dispatch (labels
    /// `family/name`, each its own pass: slower, for the breakdown). The labels and
    /// milliseconds come back with the result.
    pub fn encode_timed(
        &self,
        mels: &[Mat],
        per_dispatch: bool,
    ) -> Result<(Vec<Mat>, Vec<GpuTime>)> {
        self.check_batch(mels)?;
        let mut profiler = Profiler::new(&self.gpu, 2048);
        let timing = if per_dispatch {
            Timing::Dispatches(&mut profiler)
        } else {
            Timing::Passes(&mut profiler)
        };
        let staging = self.take_staging()?;
        let recorded = self.record(Input::Mels(mels), None, timing, Output::Encoded, &staging);
        let out = self.finish(self.pending(staging, recorded)?)?;
        let times = profiler.read(&self.gpu)?;
        Ok((out, times))
    }

    fn check_batch(&self, mels: &[Mat]) -> Result<()> {
        if mels.len() > self.dims.max_batch {
            return Err(Error::Other(format!(
                "{} windows in one GPU batch of at most {}",
                mels.len(),
                self.dims.max_batch
            )));
        }
        Ok(())
    }

    /// Records and submits one batch, its `output` copied into `staging`; returns the windows,
    /// the last submission and the output's width per row.
    fn record(
        &self,
        input: Input,
        mut trace: Option<&mut dyn Trace>,
        timing: Timing,
        output: Output,
        staging: &wgpu::Buffer,
    ) -> Result<(Vec<Window>, wgpu::SubmissionIndex, usize)> {
        let d = self.dims;
        let gpu = &*self.gpu;
        let mels = match input {
            Input::Mels(mels) => mels,
            Input::Rows(_) => &[],
        };
        let mut windows = Vec::with_capacity(mels.len().max(1));
        let (mut start, mut mel_offset) = (0, 0);
        if let Input::Rows(x) = input {
            windows.push(Window {
                mel_frames: 0,
                mel_offset: 0,
                start: 0,
                frames: x.rows,
            });
            start = x.rows;
        }
        for mel in mels {
            if mel.rows != d.mels || mel.cols == 0 || mel.cols > d.max_mel_frames {
                return Err(Error::Other(format!(
                    "mel features of {}×{} for an encoder of {} mels and at most {} frames",
                    mel.rows, mel.cols, d.mels, d.max_mel_frames
                )));
            }
            let frames = halve(halve(halve(mel.cols)));
            windows.push(Window {
                mel_frames: mel.cols,
                mel_offset,
                start,
                frames,
            });
            start += frames;
            mel_offset += mel.data.len();
        }
        let rows = start;
        let tm = windows.iter().map(|w| w.frames).max().unwrap_or(0);
        let buffers = self.buffers.lock().unwrap();
        let b = &*buffers;

        // The batch's inputs: every window's mel, packed, and the window table.
        for (mel, w) in mels.iter().zip(&windows) {
            gpu.queue.write_buffer(
                &b.mel.buffer,
                (w.mel_offset * 4) as u64,
                f32_bytes(&mel.data),
            );
        }
        let table: Vec<u8> = windows
            .iter()
            .flat_map(|w| [w.start as u32, w.frames as u32])
            .flat_map(u32::to_le_bytes)
            .collect();
        gpu.queue.write_buffer(&b.windows.buffer, 0, &table);

        let mut rec = Recorder::new(gpu, timing);
        let wanted = |trace: &Option<&mut dyn Trace>, name: &str| {
            trace.as_ref().is_some_and(|t| t.wants(name))
        };

        // Subsampling, window by window, each window's flattened frames copied into its rows
        // of `hidden`; then the projection to d_model over the whole batch, with ×√d_model
        // (unless it's the output). Rows given as input go straight to the residual stream.
        let flat = d.channels * d.freq;
        if let Input::Rows(x) = input {
            gpu.queue
                .write_buffer(&b.residual[0].buffer, 0, f32_bytes(&x.data));
        }
        if !mels.is_empty() {
            for w in &windows {
                self.pre_encode(&mut rec, b, w);
                let bytes = (w.frames * flat * 4) as u64;
                rec.copy(
                    &b.pre[1],
                    0,
                    &b.hidden.buffer,
                    (w.start * flat * 4) as u64,
                    bytes,
                );
            }
            let scale = match output {
                Output::PreEncoded => 1.,
                _ => self.xscale,
            };
            self.gemm(
                &mut rec,
                "gemm/pre_encode.out",
                &b.hidden,
                &self.pre.out,
                &b.residual[0],
                rows,
                Epilogue {
                    scale,
                    ..Epilogue::default()
                },
            );
        }
        if let Output::PreEncoded = output {
            rec.copy(&b.residual[0], 0, staging, 0, (rows * d.d_model * 4) as u64);
            let submission = rec
                .submit()
                .ok_or_else(|| Error::Other("nothing recorded".into()))?;
            return Ok((windows, submission, d.d_model));
        }
        if let Some(trace) = trace.as_deref_mut()
            && trace.wants("enc.pre_encode.out")
        {
            let mut x = self.read_window0(&mut rec, &b.residual[0], &windows[0], staging)?;
            // The checkpoint is before the scaling (a power of two: exact either way).
            let s = 1. / self.xscale;
            x.map(|v| v * s);
            trace.tensor("enc.pre_encode.out", &x.shape(), &x.data);
        }

        let mut recorded = 0;
        for (l, block) in self.blocks.iter().enumerate() {
            let x = &b.residual[l % 2];
            let next = &b.residual[(l + 1) % 2];
            let mut check = |rec: &mut Recorder, name: &str, t: &Tensor| -> Result<()> {
                let name = format!("enc.block.{l}.{name}");
                if wanted(&trace, &name) {
                    let m = self.read_window0(rec, t, &windows[0], staging)?;
                    if let Some(trace) = trace.as_deref_mut() {
                        trace.tensor(&name, &m.shape(), &m.data);
                    }
                }
                Ok(())
            };
            let half = Epilogue {
                scale: 0.5,
                accumulate: true,
                ..Epilogue::default()
            };
            let add = Epilogue {
                accumulate: true,
                ..Epilogue::default()
            };
            let silu = Epilogue {
                activation: Activation::Silu,
                ..Epilogue::default()
            };

            // Feed-forward, half step.
            self.layer_norm(&mut rec, "norm/ff1", x, &block.norm_ff1, &b.normed, rows);
            self.gemm(
                &mut rec,
                "gemm/ff.linear1",
                &b.normed,
                &block.ff1[0],
                &b.hidden,
                rows,
                silu,
            );
            self.gemm(
                &mut rec,
                "gemm/ff.linear2",
                &b.hidden,
                &block.ff1[1],
                x,
                rows,
                half,
            );
            check(&mut rec, "ff1", x)?;

            // Self-attention with relative positions.
            self.layer_norm(&mut rec, "norm/attn", x, &block.norm_attn, &b.normed, rows);
            self.gemm(
                &mut rec,
                "gemm/attn.qkv",
                &b.normed,
                &block.qkv,
                &b.qkv,
                rows,
                Epilogue::default(),
            );
            self.attention(&mut rec, b, block, windows.len(), tm);
            self.gemm(
                &mut rec,
                "gemm/attn.out",
                &b.context,
                &block.out,
                x,
                rows,
                add,
            );
            check(&mut rec, "attn", x)?;

            // Convolution module.
            self.layer_norm(&mut rec, "norm/conv", x, &block.norm_conv, &b.normed, rows);
            self.gemm(
                &mut rec,
                "gemm/conv.pointwise1",
                &b.normed,
                &block.pointwise1,
                &b.hidden,
                rows,
                Epilogue::default(),
            );
            rec.dispatch(
                "conv/depthwise",
                &self.kernels.conv_module,
                &[
                    &b.hidden,
                    &block.depthwise.0,
                    &block.depthwise.1,
                    &b.windows,
                    &b.context,
                ],
                &[rows as u32, windows.len() as u32],
                [rows as u32, 1, 1],
            );
            self.gemm(
                &mut rec,
                "gemm/conv.pointwise2",
                &b.context,
                &block.pointwise2,
                x,
                rows,
                add,
            );
            check(&mut rec, "conv", x)?;

            // Feed-forward, half step.
            self.layer_norm(&mut rec, "norm/ff2", x, &block.norm_ff2, &b.normed, rows);
            self.gemm(
                &mut rec,
                "gemm/ff.linear1",
                &b.normed,
                &block.ff2[0],
                &b.hidden,
                rows,
                silu,
            );
            self.gemm(
                &mut rec,
                "gemm/ff.linear2",
                &b.hidden,
                &block.ff2[1],
                x,
                rows,
                half,
            );
            check(&mut rec, "ff2", x)?;

            self.layer_norm(&mut rec, "norm/out", x, &block.norm_out, next, rows);
            check(&mut rec, "out", next)?;

            recorded += rows;
            if recorded >= self.rows_per_submit {
                rec.submit();
                recorded = 0;
            }
        }

        let out = &b.residual[self.blocks.len() % 2];
        let width = match output {
            Output::Encoded | Output::PreEncoded => {
                rec.copy(out, 0, staging, 0, (rows * d.d_model * 4) as u64);
                d.d_model
            }
            Output::Projected => {
                let joint = self.joint.as_ref().expect("checked when submitted");
                // Into `normed`, free once the last block's layer norm has run.
                self.gemm(
                    &mut rec,
                    "gemm/joint.enc",
                    out,
                    joint,
                    &b.normed,
                    rows,
                    Epilogue::default(),
                );
                rec.copy(&b.normed, 0, staging, 0, (rows * joint.n * 4) as u64);
                joint.n
            }
            Output::Tail(tail) => tail.record(&mut rec, out, rows, staging),
        };
        let submission = rec
            .submit()
            .ok_or_else(|| Error::Other("nothing recorded".into()))?;
        Ok((windows, submission, width))
    }

    /// One window's subsampling up to its flattened frames, in `pre[1]` (frames × freq ×
    /// channels).
    fn pre_encode(&self, rec: &mut Recorder, b: &Buffers, w: &Window) {
        let d = self.dims;
        let c = d.channels;
        let relu = Epilogue {
            activation: Activation::Relu,
            ..Epilogue::default()
        };
        // 256 threads per workgroup, a thread per 4 channels: 1024 / c points each.
        let groups = |points: usize| [points.div_ceil(1024 / c) as u32, 1, 1];
        // conv.0 + ReLU and conv.2 (depthwise), fused: mel → pre[0].
        let (t1, f1) = (halve(w.mel_frames), halve(d.mels));
        let (t2, f2) = (halve(t1), halve(f1));
        rec.dispatch(
            "pre_encode/conv",
            &self.kernels.pre_conv,
            &[
                &b.mel,
                &self.pre.conv0.0,
                &self.pre.conv0.1,
                &self.pre.depthwise[0].0,
                &self.pre.depthwise[0].1,
                &b.pre[0],
            ],
            &[
                w.mel_frames as u32,
                d.mels as u32,
                w.mel_offset as u32,
                t2 as u32,
                f2 as u32,
            ],
            groups(t2 * f2),
        );
        // conv.3 (pointwise) + ReLU → pre[1].
        self.gemm(
            rec,
            "gemm/pre_encode.pointwise",
            &b.pre[0],
            &self.pre.pointwise[0],
            &b.pre[1],
            t2 * f2,
            relu,
        );
        // conv.5 (depthwise) → pre[0]; conv.6 (pointwise) + ReLU → pre[1].
        let (t3, f3) = (halve(t2), halve(f2));
        rec.dispatch(
            "pre_encode/conv",
            &self.kernels.depthwise2d,
            &[
                &b.pre[1],
                &self.pre.depthwise[1].0,
                &self.pre.depthwise[1].1,
                &b.pre[0],
            ],
            &[t2 as u32, f2 as u32, t3 as u32, f3 as u32],
            groups(t3 * f3),
        );
        self.gemm(
            rec,
            "gemm/pre_encode.pointwise",
            &b.pre[0],
            &self.pre.pointwise[1],
            &b.pre[1],
            t3 * f3,
            relu,
        );
    }

    /// Scores, then softmax and the weighted values, for every window and head, into
    /// `context`.
    fn attention(&self, rec: &mut Recorder, b: &Buffers, block: &Block, windows: usize, tm: usize) {
        let d = self.dims;
        let d_head = d.d_model / d.heads;
        let tiles = tm.div_ceil(64) as u32;
        let z = (windows * d.heads) as u32;
        let scale = 1. / (d_head as f32).sqrt();
        rec.dispatch(
            "attn/scores",
            &self.kernels.scores,
            &[
                &b.qkv,
                &block.pos,
                &block.bias_u,
                &block.bias_v,
                &b.windows,
                &b.scores,
            ],
            &[tm as u32, d.max_frames as u32, scale.to_bits()],
            [tiles, tiles, z],
        );
        rec.dispatch(
            "attn/softmax_values",
            &self.kernels.attend,
            &[&b.qkv, &b.scores, &b.windows, &b.context],
            &[tm as u32],
            // Tiles of 128 of a head's features, or one of 64.
            [(d_head / 4 / (d_head / 4).min(32)) as u32, tiles, z],
        );
    }

    fn layer_norm(
        &self,
        rec: &mut Recorder,
        label: &str,
        x: &Tensor,
        norm: &Norm,
        out: &Tensor,
        rows: usize,
    ) {
        rec.dispatch(
            label,
            &self.kernels.layer_norm,
            &[x, &norm.weight, &norm.bias, out],
            &[rows as u32, EPS.to_bits()],
            [rows as u32, 1, 1],
        );
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

    /// Window 0's rows of `t` (d_model wide), read back now.
    fn read_window0(
        &self,
        rec: &mut Recorder,
        t: &Tensor,
        w: &Window,
        staging: &wgpu::Buffer,
    ) -> Result<Mat> {
        let d = self.dims.d_model;
        let bytes = (w.frames * d * 4) as u64;
        rec.copy(t, (w.start * d * 4) as u64, staging, 0, bytes);
        let submission = rec.submit();
        let data = map_read(&self.gpu, staging, bytes, submission, &[(0, w.frames * d)])?;
        Ok(Mat::from_vec(w.frames, d, data.into_iter().next().unwrap()))
    }
}

fn load_pre_encode(gpu: &Gpu, file: &Gguf, d: &Dims) -> Result<PreEncode> {
    let c = d.channels;
    let p = "enc.pre_encode";
    // A 3×3 kernel per channel, stored channel, time tap, frequency tap: to [tap][channel].
    let taps = |name: &str| -> Result<(Tensor, Tensor)> {
        let w = vector(file, &format!("{p}.{name}.weight"))?;
        let bias = vector(file, &format!("{p}.{name}.bias"))?;
        if w.len() != 9 * c || bias.len() != c {
            return Err(Error::Gguf(format!(
                "{p}.{name}.weight: {} values",
                w.len()
            )));
        }
        let t: Vec<f32> = (0..9)
            .flat_map(|tap| (0..c).map(move |ch| (tap, ch)))
            .map(|(tap, ch)| w[ch * 9 + tap])
            .collect();
        Ok((gpu.upload_f32(name, &t), gpu.upload_f32(name, &bias)))
    };
    let pointwise = |name: &str| -> Result<Weights> {
        Ok(Weights::upload(
            gpu,
            name,
            &Linear::load(file, &format!("{p}.{name}"))?,
        ))
    };
    // The projection's inputs are each frame's features channel-major (channel · freq + f);
    // the pointwise products leave them frequency-major (f · channels + channel), so the
    // weight's columns are reordered to match and nothing is transposed per window.
    let out = Linear::load(file, &format!("{p}.out"))?;
    if out.cols != c * d.freq {
        return Err(Error::Gguf(format!(
            "{p}.out takes {} inputs, not {} channels × {} frequencies",
            out.cols, c, d.freq
        )));
    }
    let (w, inexact) = gemm::to_f16(&out);
    let mut reordered = vec![f16::ZERO; w.len()];
    for (dst, src) in reordered
        .chunks_exact_mut(out.cols)
        .zip(w.chunks_exact(out.cols))
    {
        for f in 0..d.freq {
            for ch in 0..c {
                dst[f * c + ch] = src[ch * d.freq + f];
            }
        }
    }
    let out = Weights {
        n: out.rows,
        k: out.cols,
        weight: gpu.upload_f16("pre_encode.out", &reordered),
        bias: out
            .bias
            .as_ref()
            .map(|b| gpu.upload_f32("pre_encode.out", b)),
        inexact,
    };
    Ok(PreEncode {
        conv0: taps("conv.0")?,
        depthwise: [taps("conv.2")?, taps("conv.5")?],
        pointwise: [pointwise("conv.3")?, pointwise("conv.6")?],
        out,
    })
}

fn load_block(gpu: &Gpu, file: &Gguf, name: &str, d: &Dims) -> Result<Block> {
    let norm = |n: &str| -> Result<Norm> {
        Ok(Norm {
            weight: gpu.upload_f32(n, &vector(file, &format!("{name}.{n}.weight"))?),
            bias: gpu.upload_f32(n, &vector(file, &format!("{name}.{n}.bias"))?),
        })
    };
    let linear = |n: &str| upload_linear(gpu, file, &format!("{name}.{n}"), n);
    let qkv = stacked(
        gpu,
        file,
        "attn.qkv",
        &["q", "k", "v"].map(|p| format!("{name}.attn.linear_{p}")),
    )?;

    // Batch norm folded into the depthwise convolution: `scale = γ / √(var + ε)`, weights
    // times scale, bias times scale plus `β − mean · scale`.
    let c = format!("{name}.conv");
    let w = vector(file, &format!("{c}.depthwise.weight"))?;
    let bias = vector(file, &format!("{c}.depthwise.bias"))?;
    let gamma = vector(file, &format!("{c}.bn.weight"))?;
    let beta = vector(file, &format!("{c}.bn.bias"))?;
    let mean = vector(file, &format!("{c}.bn.running_mean"))?;
    let var = vector(file, &format!("{c}.bn.running_var"))?;
    let channels = bias.len();
    if channels != d.d_model || w.len() != d.kernel * channels {
        return Err(Error::Gguf(format!(
            "{c}.depthwise.weight: {} values",
            w.len()
        )));
    }
    let scale: Vec<f32> = gamma
        .iter()
        .zip(&var)
        .map(|(g, v)| g / (v + EPS).sqrt())
        .collect();
    let folded_w: Vec<f32> = (0..d.kernel)
        .flat_map(|k| (0..channels).map(move |ch| (k, ch)))
        .map(|(k, ch)| w[ch * d.kernel + k] * scale[ch])
        .collect();
    let folded_b: Vec<f32> = (0..channels)
        .map(|ch| bias[ch] * scale[ch] + (beta[ch] - mean[ch] * scale[ch]))
        .collect();

    let rows = 2 * d.max_frames - 1;
    Ok(Block {
        norm_ff1: norm("norm_ff1")?,
        ff1: [linear("ff1.linear1")?, linear("ff1.linear2")?],
        norm_attn: norm("norm_attn")?,
        qkv,
        out: linear("attn.linear_out")?,
        pos: gpu.zeros_f32("pos", rows * d.d_model),
        bias_u: gpu.upload_f32(
            "pos_bias_u",
            &vector(file, &format!("{name}.attn.pos_bias_u"))?,
        ),
        bias_v: gpu.upload_f32(
            "pos_bias_v",
            &vector(file, &format!("{name}.attn.pos_bias_v"))?,
        ),
        norm_conv: norm("norm_conv")?,
        pointwise1: linear("conv.pointwise1")?,
        depthwise: (
            gpu.upload_f32("depthwise", &folded_w),
            gpu.upload_f32("depthwise", &folded_b),
        ),
        pointwise2: linear("conv.pointwise2")?,
        norm_ff2: norm("norm_ff2")?,
        ff2: [linear("ff2.linear1")?, linear("ff2.linear2")?],
        norm_out: norm("norm_out")?,
    })
}

/// Layers with the same inputs as one: their output rows (and biases) one after another.
pub(crate) fn stacked(gpu: &Gpu, file: &Gguf, label: &str, names: &[String]) -> Result<Weights> {
    let mut weight = Vec::new();
    let mut bias = Vec::new();
    let (mut k, mut inexact) = (None, 0);
    for name in names {
        let layer = F16Linear::load(file, name)?;
        if *k.get_or_insert(layer.cols) != layer.cols {
            return Err(Error::Gguf(format!("{label}: layers of different widths")));
        }
        weight.extend(layer.weight);
        inexact += layer.inexact;
        match layer.bias {
            Some(b) => bias.extend(b),
            None => bias.extend(std::iter::repeat_n(0., layer.rows)),
        }
    }
    Ok(Weights {
        n: bias.len(),
        k: k.unwrap_or(0),
        weight: gpu.upload_f16(label, &weight),
        bias: Some(gpu.upload_f32(label, &bias)),
        inexact,
    })
}

/// A linear layer `name` (its `.weight` and any `.bias`) uploaded for [`Gemm`].
pub(crate) fn upload_linear(gpu: &Gpu, file: &Gguf, name: &str, label: &str) -> Result<Weights> {
    let layer = F16Linear::load(file, name)?;
    Ok(Weights {
        n: layer.rows,
        k: layer.cols,
        weight: gpu.upload_f16(label, &layer.weight),
        bias: layer.bias.map(|b| gpu.upload_f32(label, &b)),
        inexact: layer.inexact,
    })
}

/// A linear layer's weight as the GPU takes it: `rows × cols` row-major f16, exactly what
/// [`gemm::to_f16`] makes of the CPU's [`Linear`] (each value rounded to the nearest f16), but
/// converted straight from the file's Q8_0 blocks, eight values at a time where the CPU
/// converts in hardware: it's most of the model's load time otherwise.
struct F16Linear {
    rows: usize,
    cols: usize,
    weight: Vec<f16>,
    bias: Option<Vec<f32>>,
    /// Values f16 couldn't hold exactly.
    inexact: usize,
}

impl F16Linear {
    fn load(file: &Gguf, name: &str) -> Result<F16Linear> {
        let info = crate::cpu::tensor(file, &format!("{name}.weight"))?;
        // As `Linear::load` reads shapes: a leading kernel dimension of one isn't an input.
        let mut dims: Vec<usize> = info.dims.iter().map(|d| *d as usize).collect();
        while dims.len() > 2 && dims[0] == 1 {
            dims.remove(0);
        }
        let cols = dims[0];
        let rows: usize = dims[1..].iter().product();
        if info.dtype != crate::gguf::Dtype::Q8_0 || !cols.is_multiple_of(32) {
            let linear = Linear::load(file, name)?;
            let (weight, inexact) = gemm::to_f16(&linear);
            return Ok(F16Linear {
                rows,
                cols,
                weight,
                bias: linear.bias,
                inexact,
            });
        }
        let blocks = file.data(info).as_chunks::<34>().0;
        if blocks.len() * 32 != rows * cols {
            return Err(Error::Gguf(format!(
                "{name}.weight: {} blocks",
                blocks.len()
            )));
        }
        let mut weight = vec![f16::ZERO; rows * cols];
        // Chunks of whole blocks, one task each.
        const PER_TASK: usize = 2048;
        let inexact = weight
            .par_chunks_mut(32 * PER_TASK)
            .zip(blocks.par_chunks(PER_TASK))
            .map(|(out, blocks)| q8_to_f16(blocks, out))
            .sum();
        let bias = match file.tensor(&format!("{name}.bias")) {
            Some(b) => Some(file.dequantize(b)?),
            None => None,
        };
        if bias.as_ref().is_some_and(|b| b.len() != rows) {
            return Err(Error::Gguf(format!("{name}.bias: not {rows} values")));
        }
        Ok(F16Linear {
            rows,
            cols,
            weight,
            bias,
            inexact,
        })
    }
}

/// Q8_0 blocks (an f16 scale, 32 int8s) to f16 `scale · q`, rounded to nearest (the product is
/// exact in f32); returns how many rounded.
fn q8_to_f16(blocks: &[[u8; 34]], out: &mut [f16]) -> usize {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("f16c") && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the CPU has the features; `out` holds 32 values per block.
        return unsafe { q8_to_f16_f16c(blocks, out) };
    }
    let mut inexact = 0;
    for (b, out) in blocks.iter().zip(out.as_chunks_mut::<32>().0) {
        let scale = f16::from_le_bytes([b[0], b[1]]).to_f32();
        for (o, &q) in out.iter_mut().zip(&b[2..]) {
            let v = scale * q as i8 as f32;
            *o = f16::from_f32(v);
            inexact += usize::from(o.to_f32() != v);
        }
    }
    inexact
}

/// [`q8_to_f16`] with F16C: the same rounding (to nearest, ties to even), eight at a time.
///
/// # Safety
/// The CPU has AVX2 and F16C, and `out` holds 32 values per block.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn q8_to_f16_f16c(blocks: &[[u8; 34]], out: &mut [f16]) -> usize {
    use std::arch::x86_64::*;
    assert!(out.len() >= 32 * blocks.len());
    let mut inexact = 0;
    for (b, out) in blocks.iter().zip(out.as_chunks_mut::<32>().0) {
        let scale = _mm256_set1_ps(f16::from_le_bytes([b[0], b[1]]).to_f32());
        for part in 0..4 {
            // SAFETY: bytes 2 + 8·part .. + 8 are in the block; `out` has 8 values there.
            unsafe {
                let q = _mm_loadl_epi64(b.as_ptr().add(2 + 8 * part).cast());
                let v = _mm256_mul_ps(scale, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q)));
                let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v);
                _mm_storeu_si128(out.as_mut_ptr().add(8 * part).cast(), h);
                let back = _mm256_cvtph_ps(h);
                let differ = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_NEQ_UQ>(back, v));
                inexact += differ.count_ones() as usize;
            }
        }
    }
    inexact
}

/// The sinusoidal encodings of offsets `frames − 1` (row 0) down to `−(frames − 1)`, exactly
/// as the CPU oracle computes them for a window (so the rows a window reads are the same).
fn positional_encodings(frames: usize, d_model: usize) -> Vec<f32> {
    let len = 2 * frames - 1;
    let ln_10000 = 10000f32.ln();
    let div: Vec<f32> = (0..d_model / 2)
        .map(|k| ((2 * k) as f32 * (-ln_10000 / d_model as f32)).exp())
        .collect();
    let mut pe = vec![0f32; len * d_model];
    for (i, row) in pe.chunks_exact_mut(d_model).enumerate() {
        let position = (frames as i64 - 1 - i as i64) as f32;
        for (k, d) in div.iter().enumerate() {
            row[2 * k] = (position * d).sin();
            row[2 * k + 1] = (position * d).cos();
        }
    }
    pe
}

/// The bytes of some f32s, as the GPU takes them (every target the engine builds for is
/// little-endian).
pub(crate) fn f32_bytes(data: &[f32]) -> &[u8] {
    const { assert!(cfg!(target_endian = "little")) };
    // SAFETY: f32 has no padding and every byte is a valid u8; the same memory, as bytes.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) }
}

/// Waits for `submission` (or everything submitted), then reads spans (`(first value,
/// values)`) of the first `bytes` of a mappable buffer as f32s, each its own vector.
pub(crate) fn map_read(
    gpu: &Gpu,
    buffer: &wgpu::Buffer,
    bytes: u64,
    submission: Option<wgpu::SubmissionIndex>,
    spans: &[(usize, usize)],
) -> Result<Vec<Vec<f32>>> {
    let slice = buffer.slice(..bytes);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| drop(tx.send(r)));
    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: submission,
            timeout: None,
        })
        .map_err(|e| Error::Other(format!("GPU: {e}")))?;
    rx.recv()
        .map_err(|_| Error::Other("readback never mapped".into()))?
        .map_err(|e| Error::Other(format!("readback: {e}")))?;
    let data = {
        let view = slice
            .get_mapped_range()
            .map_err(|e| Error::Other(format!("readback: {e}")))?;
        spans
            .iter()
            .map(|&(first, len)| {
                let mut out = vec![0f32; len];
                // SAFETY: as in `f32_bytes`: the vector's memory as bytes, exactly as long.
                unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), len * 4) }
                    .copy_from_slice(&view[first * 4..(first + len) * 4]);
                out
            })
            .collect()
    };
    buffer.unmap();
    Ok(data)
}
