//! Whisper's encoder on the GPU, a batch of 30 s windows at once: the two convolutions as
//! matrix multiplies over their receptive fields (`im2col`, weights reordered tap-major to
//! match) with GELU in the epilogue, the second one adding into the learned positions; then
//! per block a layer norm, the fused q/k/v product, flash attention on tensor cores, the
//! output projection added into the residual stream, a layer norm, the GELU feed-forward
//! added in; a final layer norm. Windows are rows one after another; nothing depends on the
//! batch (the lab checks a window in a batch against the same window alone).

use super::{GpuOptions, Norm, groups_2d};
use crate::{
    Error, Result, Trace,
    cpu::{Mat, tensor, vector},
    gguf::Gguf,
    gpu::{
        Gpu, Kernel, Tensor, f16,
        gemm::{Activation, Epilogue, Gemm, Operands, Weights},
    },
    parakeet::gpu_encoder::{
        LAYER_NORM, f32_bytes, map_read,
        record::{Recorder, Timing},
        stacked, upload_linear,
    },
    whisper::HParams,
};
use rayon::prelude::*;
use std::sync::Arc;

const IM2COL: &str = include_str!("im2col.wgsl");
const ATTEND: &str = include_str!("attend.wgsl");

/// Mel frames and encoder positions of one window.
const MEL_FRAMES: usize = 3000;
const FRAMES: usize = 1500;

struct Block {
    norm_attn: Norm,
    /// q, k and v stacked (k has no bias: zeros).
    qkv: Weights,
    out: Weights,
    norm_ffn: Norm,
    fc1: Weights,
    fc2: Weights,
}

/// The activations, sized for the largest batch. `hidden` also holds both convolutions'
/// receptive fields, `qkv` the first convolution's output.
struct Buffers {
    mel: Tensor,
    residual: Tensor,
    normed: Tensor,
    qkv: Tensor,
    context: Tensor,
    hidden: Tensor,
}

pub struct Encoder {
    gpu: Arc<Gpu>,
    d: usize,
    heads: usize,
    mels: usize,
    max_batch: usize,
    gemm: Gemm,
    im2col: Arc<Kernel>,
    attend: Arc<Kernel>,
    layer_norm: Arc<Kernel>,
    /// The convolutions over tap-major receptive fields; the first's padded to a multiple of
    /// the multiply's K step with zero columns.
    conv1: Weights,
    conv2: Weights,
    positions: Tensor,
    blocks: Vec<Block>,
    final_norm: Norm,
    buffers: Buffers,
}

/// Bytes the encoder's activations (a batch's residual stream, multiply outputs and the
/// like) may take on the GPU.
const ACTIVATION_BUDGET: usize = 192 << 20;

impl Encoder {
    pub fn load(
        gpu: &Arc<Gpu>,
        file: &Gguf,
        hp: &HParams,
        options: &GpuOptions,
    ) -> Result<Encoder> {
        let d = hp.d_model;
        let heads = hp.enc_heads;
        if heads * 64 != d || !d.is_multiple_of(64) || hp.frames != MEL_FRAMES {
            return Err(Error::Unsupported(format!(
                "the GPU encoder for d_model {d}, {heads} heads, {} frames",
                hp.frames
            )));
        }
        let variant = options.gemm.unwrap_or_else(super::default_variant);
        let gemm = Gemm::new(gpu, variant)?;
        let bk = variant.bk as usize;
        let attention = [
            ("HEADS", heads as u32),
            ("D_MODEL", d as u32),
            ("T", FRAMES as u32),
        ];
        let im2col = gpu.kernel("whisper_im2col", IM2COL, &[], &[false, true], 24)?;
        let attend = gpu.kernel(
            "whisper_attend",
            &attend_shader(),
            &attention,
            &[false, true],
            0,
        )?;
        let layer_norm = gpu.kernel(
            "enc_layer_norm",
            LAYER_NORM,
            &[("D4", (d / 4) as u32)],
            &[false, false, false, true],
            8,
        )?;
        let conv1 = conv(gpu, file, "enc.conv.0", bk)?;
        let mels = conv1.k_in;
        let conv1 = conv1.weights;
        let conv2 = conv(gpu, file, "enc.conv.1", bk)?.weights;
        let pos = tensor(file, "enc.pos_emb.weight")?;
        let positions = file.dequantize(pos)?;
        if positions.len() < FRAMES * d {
            return Err(Error::Gguf("enc.pos_emb is too short".into()));
        }
        let positions = gpu.upload_f32("enc.pos_emb", &positions[..FRAMES * d]);
        let blocks = (0..hp.enc_layers)
            .into_par_iter()
            .map(|l| -> Result<Block> {
                let name = |part: &str| format!("enc.blocks.{l}.{part}");
                Ok(Block {
                    norm_attn: Norm::load(gpu, file, &name("norm_attn"))?,
                    qkv: stacked(
                        gpu,
                        file,
                        "attn.qkv",
                        &["q", "k", "v"].map(|p| name(&format!("attn.{p}"))),
                    )?,
                    out: upload_linear(gpu, file, &name("attn.out"), "attn.out")?,
                    norm_ffn: Norm::load(gpu, file, &name("norm_ffn"))?,
                    fc1: upload_linear(gpu, file, &name("ffn.fc1"), "ffn.fc1")?,
                    fc2: upload_linear(gpu, file, &name("ffn.fc2"), "ffn.fc2")?,
                })
            })
            .collect::<Result<Vec<Block>>>()?;
        let d_ff = blocks.first().map_or(4 * d, |b| b.fc1.n);
        // A window costs the same GPU time in any batch, so the batch is only as large as its
        // activations' budget allows: 4 windows for Small, 2 for Turbo (whose activations are
        // 79 MB a window, memory the driver also charges to the system's commit).
        let per_window = 4
            * (MEL_FRAMES * mels
                + 3 * FRAMES * d
                + (FRAMES * 3 * d).max(MEL_FRAMES * d)
                + (FRAMES * d_ff)
                    .max(FRAMES * 3 * d)
                    .max(MEL_FRAMES * conv1.k));
        let b = options.max_batch.min(ACTIVATION_BUDGET / per_window).max(1);
        let rows = b * FRAMES;
        let hidden = (rows * d_ff)
            .max(rows * 3 * d)
            .max(b * MEL_FRAMES * conv1.k);
        let buffers = Buffers {
            mel: gpu.zeros_f32("mel", b * MEL_FRAMES * mels),
            residual: gpu.zeros_f32("residual", rows * d),
            normed: gpu.zeros_f32("normed", rows * d),
            qkv: gpu.zeros_f32("qkv", (rows * 3 * d).max(b * MEL_FRAMES * d)),
            context: gpu.zeros_f32("context", rows * d),
            hidden: gpu.zeros_f32("hidden", hidden),
        };
        Ok(Encoder {
            gpu: gpu.clone(),
            d,
            heads,
            mels,
            max_batch: b,
            gemm,
            im2col,
            attend,
            layer_norm,
            conv1,
            conv2,
            positions,
            blocks,
            final_norm: Norm::load(gpu, file, "enc.final_norm")?,
            buffers,
        })
    }

    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    /// Where the last batch's encoder states are (window i's 1500 rows from row 1500·i).
    pub(crate) fn output(&self) -> &Tensor {
        &self.buffers.normed
    }

    /// Records a batch of windows (3000 frames × mels each) through the encoder; their states
    /// end in [`output`](Self::output). With a trace, window 0's checkpoints it wants are read
    /// back as they're made (`enc.conv1.out`, `enc.embed.out`, `enc.block.<L>.out`,
    /// `enc.final`): debug only, each read waits for the GPU.
    pub fn record(
        &self,
        rec: &mut Recorder,
        mels: &[&Mat],
        mut trace: Option<&mut dyn Trace>,
    ) -> Result<()> {
        let n = mels.len();
        if n == 0 || n > self.max_batch {
            return Err(Error::Other(format!(
                "{n} windows for a GPU encoder batch of at most {}",
                self.max_batch
            )));
        }
        let gpu = &*self.gpu;
        let d = self.d;
        let b = &self.buffers;
        for (i, mel) in mels.iter().enumerate() {
            if mel.rows != MEL_FRAMES || mel.cols != self.mels {
                return Err(Error::Other(format!(
                    "mel features of {}×{}, not {MEL_FRAMES}×{}",
                    mel.rows, mel.cols, self.mels
                )));
            }
            gpu.queue.write_buffer(
                &b.mel.buffer,
                (i * MEL_FRAMES * self.mels * 4) as u64,
                f32_bytes(&mel.data),
            );
        }
        let mut read = |rec: &mut Recorder, name: &str, t: &Tensor, rows: usize| -> Result<()> {
            if let Some(trace) = trace.as_deref_mut()
                && trace.wants(name)
            {
                let data = read_rows(gpu, rec, t, rows * d)?;
                trace.tensor(name, &[rows, d], &data);
            }
            Ok(())
        };
        let gelu = Epilogue {
            activation: Activation::Gelu,
            ..Epilogue::default()
        };
        let add = Epilogue {
            accumulate: true,
            ..Epilogue::default()
        };
        let rows = n * FRAMES;

        // conv1 + GELU: mel → receptive fields (hidden) → qkv.
        self.im2col(
            rec,
            &b.mel,
            &b.hidden,
            self.mels,
            MEL_FRAMES,
            MEL_FRAMES,
            1,
            self.conv1.k,
            n,
        );
        self.gemm(
            rec,
            "gemm/conv1",
            &b.hidden,
            &self.conv1,
            &b.qkv,
            n * MEL_FRAMES,
            gelu,
        );
        read(rec, "enc.conv1.out", &b.qkv, MEL_FRAMES)?;
        // conv2 + GELU, added into the positions: qkv → receptive fields → residual.
        self.im2col(
            rec,
            &b.qkv,
            &b.hidden,
            d,
            MEL_FRAMES,
            FRAMES,
            2,
            self.conv2.k,
            n,
        );
        for i in 0..n {
            rec.copy(
                &self.positions,
                0,
                &b.residual.buffer,
                (i * FRAMES * d * 4) as u64,
                (FRAMES * d * 4) as u64,
            );
        }
        let embed = Epilogue {
            activation: Activation::Gelu,
            accumulate: true,
            ..Epilogue::default()
        };
        self.gemm(
            rec,
            "gemm/conv2",
            &b.hidden,
            &self.conv2,
            &b.residual,
            rows,
            embed,
        );
        read(rec, "enc.embed.out", &b.residual, FRAMES)?;

        // Work per submit: about a window-block of the large model at a time, so no submit
        // runs long.
        let per_submit = (24 / n).max(1);
        for (l, block) in self.blocks.iter().enumerate() {
            block
                .norm_attn
                .record(rec, &self.layer_norm, &b.residual, &b.normed, rows);
            self.gemm(
                rec,
                "gemm/attn.qkv",
                &b.normed,
                &block.qkv,
                &b.qkv,
                rows,
                Epilogue::default(),
            );
            rec.dispatch(
                "attn/flash",
                &self.attend,
                &[&b.qkv, &b.context],
                &[],
                [FRAMES.div_ceil(64) as u32, self.heads as u32, n as u32],
            );
            self.gemm(
                rec,
                "gemm/attn.out",
                &b.context,
                &block.out,
                &b.residual,
                rows,
                add,
            );
            block
                .norm_ffn
                .record(rec, &self.layer_norm, &b.residual, &b.normed, rows);
            self.gemm(
                rec,
                "gemm/ffn.fc1",
                &b.normed,
                &block.fc1,
                &b.hidden,
                rows,
                gelu,
            );
            self.gemm(
                rec,
                "gemm/ffn.fc2",
                &b.hidden,
                &block.fc2,
                &b.residual,
                rows,
                add,
            );
            read(rec, &format!("enc.block.{l}.out"), &b.residual, FRAMES)?;
            if (l + 1) % per_submit == 0 {
                rec.submit();
            }
        }
        self.final_norm
            .record(rec, &self.layer_norm, &b.residual, &b.normed, rows);
        read(rec, "enc.final", &b.normed, FRAMES)?;
        Ok(())
    }

    /// One batch encoded and its windows' states read back (lab and checkpoints).
    pub fn encode(&self, mels: &[&Mat], trace: Option<&mut dyn Trace>) -> Result<Vec<Mat>> {
        let mut rec = Recorder::new(&self.gpu, Timing::None);
        self.record(&mut rec, mels, trace)?;
        let out = self.output();
        let data = read_rows(&self.gpu, &mut rec, out, mels.len() * FRAMES * self.d)?;
        Ok(data
            .chunks_exact(FRAMES * self.d)
            .map(|w| Mat::from_vec(FRAMES, self.d, w.to_vec()))
            .collect())
    }

    /// One batch with GPU timestamps: per submit (`per_dispatch` false: the batch's own GPU
    /// time) or per dispatch (labels `family/name`). Nothing is read back but the times.
    pub fn encode_timed(&self, mels: &[&Mat], per_dispatch: bool) -> Result<Vec<(String, f64)>> {
        let mut profiler = crate::gpu::Profiler::new(&self.gpu, 2048);
        let timing = if per_dispatch {
            Timing::Dispatches(&mut profiler)
        } else {
            Timing::Passes(&mut profiler)
        };
        let mut rec = Recorder::new(&self.gpu, timing);
        self.record(&mut rec, mels, None)?;
        rec.submit();
        drop(rec);
        profiler.read(&self.gpu)
    }

    #[allow(clippy::too_many_arguments)]
    fn im2col(
        &self,
        rec: &mut Recorder,
        x: &Tensor,
        out: &Tensor,
        channels: usize,
        frames_in: usize,
        frames_out: usize,
        stride: usize,
        width: usize,
        windows: usize,
    ) {
        let rows = windows * frames_out;
        rec.dispatch(
            "conv/im2col",
            &self.im2col,
            &[x, out],
            &[
                frames_in as u32,
                (channels / 4) as u32,
                frames_out as u32,
                stride as u32,
                (width / 4) as u32,
                rows as u32,
            ],
            groups_2d(rows * width / 4),
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
}

/// A convolution's weights as the multiply takes them, and its input channels.
struct Conv {
    weights: Weights,
    k_in: usize,
}

/// A width-3 convolution's kernel (`[3, in, out]` in ggml's order) as `out` rows of
/// tap-major inputs (`tap · in + channel`), zero-padded to a multiple of `bk`, in f16.
fn conv(gpu: &Gpu, file: &Gguf, name: &str, bk: usize) -> Result<Conv> {
    let info = tensor(file, &format!("{name}.weight"))?;
    let (taps, inputs, outputs) = (
        info.dims[0] as usize,
        info.dims[1] as usize,
        info.dims[2] as usize,
    );
    if taps != 3 || !inputs.is_multiple_of(4) {
        return Err(Error::Gguf(format!("{name}.weight: {:?}", info.dims)));
    }
    let w = file.dequantize(info)?;
    let width = (3 * inputs).next_multiple_of(bk);
    let mut out = vec![f16::ZERO; outputs * width];
    for (o, row) in out.chunks_exact_mut(width).enumerate() {
        for c in 0..inputs {
            for tap in 0..3 {
                row[tap * inputs + c] = f16::from_f32(w[(o * inputs + c) * 3 + tap]);
            }
        }
    }
    let bias = vector(file, &format!("{name}.bias"))?;
    Ok(Conv {
        weights: Weights {
            n: outputs,
            k: width,
            weight: gpu.upload_f16(name, &out),
            bias: Some(gpu.upload_f32(name, &bias)),
            inexact: 0,
        },
        k_in: inputs,
    })
}

/// The flash-attention kernel with its cooperative matrices declared and its products
/// unrolled (see `attend.wgsl`).
fn attend_shader() -> String {
    let mut load_q = String::new();
    let mut zero_o = String::new();
    let mut store_o = String::new();
    let mut load_o = String::new();
    let mut scores = String::new();
    let mut values = String::new();
    for i in 0..4 {
        let at = 16 * i;
        load_q.push_str(&format!(
            "    let qa{i} = coopLoadT<coop_mat16x16<f16, A>>(&qp[r0 * LD + {at}u], LD);\n"
        ));
        zero_o.push_str(&format!("    var o{i} = coop_mat16x16<f32, C>();\n"));
        store_o.push_str(&format!(
            "        coopStoreT(o{i}, &so[sg * PATCH + {at}u], LS);\n"
        ));
        load_o.push_str(&format!(
            "        o{i} = coopLoadT<coop_mat16x16<f32, C>>(&so[sg * PATCH + {at}u], LS);\n"
        ));
        scores.push_str(&format!("        var s{i} = coop_mat16x16<f32, C>();\n"));
    }
    for kk in 0..4 {
        for j in 0..4 {
            scores.push_str(&format!(
                "        s{j} = coopMultiplyAdd(qa{kk}, coopLoad<coop_mat16x16<f16, B>>(&ks[{}u * LD + {}u], LD), s{j});\n",
                16 * j,
                16 * kk
            ));
        }
    }
    for j in 0..4 {
        scores.push_str(&format!(
            "        coopStoreT(s{j}, &so[sg * PATCH + {}u], LS);\n",
            16 * j
        ));
    }
    for kk in 0..4 {
        values.push_str(&format!(
            "        let pa{kk} = coopLoadT<coop_mat16x16<f16, A>>(&qp[r0 * LD + {}u], LD);\n",
            16 * kk
        ));
        for n in 0..4 {
            values.push_str(&format!(
                "        o{n} = coopMultiplyAdd(pa{kk}, coopLoadT<coop_mat16x16<f16, B>>(&vs[{}u * LD + {}u], LD), o{n});\n",
                16 * kk,
                16 * n
            ));
        }
    }
    ATTEND
        .replace("    //@LOAD_Q\n", &load_q)
        .replace("    //@ZERO_O\n", &zero_o)
        .replace(
            "            //@STORE_O\n",
            &store_o.replace("        ", "            "),
        )
        .replace("    //@STORE_O\n", &store_o.replace("        ", "    "))
        .replace(
            "            //@LOAD_O\n",
            &load_o.replace("        ", "            "),
        )
        .replace("        //@SCORES\n", &scores)
        .replace("        //@VALUES\n", &values)
}

/// The first `len` values of `t`, read back now (submitting what's recorded).
pub(crate) fn read_rows(gpu: &Gpu, rec: &mut Recorder, t: &Tensor, len: usize) -> Result<Vec<f32>> {
    let bytes = (len * 4) as u64;
    use crate::gpu::wgpu;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    rec.copy(t, 0, &staging, 0, bytes);
    let submission = rec.submit();
    Ok(map_read(gpu, &staging, bytes, submission, &[(0, len)])?.remove(0))
}
