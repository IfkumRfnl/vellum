//! Whisper's decoder on the GPU: windows decoded in lockstep, one token per step, each in a
//! slot with its own caches (self-attention keys and values up to the 448 positions, the
//! encoder's keys and values for every layer, both f16 as the CPU oracle rounds them).
//!
//! A step: each slot's input token embedded with its position; per layer a layer norm, the
//! fused q/k/v product, causal self-attention over the slot's cache, the output projection
//! added in, a layer norm, the cross-attention query, cross-attention over the slot's encoder
//! keys and values, its projection added in, a layer norm, the GELU feed-forward added in; the
//! final layer norm, the tied output head, and the choice (`choose.wgsl`): the next prompt
//! token, or for a greedy slot the next token by the oracle's rules, appended to the slot's
//! choices. The linear layers are matrix-vector products over the file's Q8_0 weights
//! (`gemv.wgsl`).
//!
//! Each slot's decoding state lives on the GPU (`state`: word 0, whether any slot is decoding,
//! then [`SLOT_WORDS`] words per slot), so a run of steps is recorded and submitted at once
//! and only the state and choices come back. Once every slot is done, the steps left in a run
//! cost almost nothing: every kernel checks word 0 first.

use super::{GpuOptions, Norm, groups_2d};
use crate::{
    Error, Result,
    cpu::{Mat, tensor},
    gguf::{Dtype, Gguf},
    gpu::{
        Gpu, Kernel, Profiler, Tensor,
        gemm::{Epilogue, Gemm, Operands, Weights},
        wgpu,
    },
    parakeet::gpu_encoder::{
        f32_bytes, map_read,
        record::{Recorder, Timing},
        stacked,
    },
    whisper::HParams,
};
use std::sync::{Arc, Mutex};

const EMBED: &str = include_str!("embed.wgsl");
const GEMV: &str = include_str!("gemv.wgsl");
const SELF_ATTEND: &str = include_str!("self_attend.wgsl");
const CROSS_ATTEND: &str = include_str!("cross_attend.wgsl");
const CROSS_COMBINE: &str = include_str!("cross_combine.wgsl");
const CHOOSE: &str = include_str!("choose.wgsl");
const CROSS_STORE: &str = include_str!("cross_store.wgsl");

/// Encoder positions of a window.
const FRAMES: usize = 1500;
/// The most slots a step takes (the matrix-vector kernel's rows of x).
pub const MAX_SLOTS: usize = 8;
/// Parts each window's cross-attention is split into over the encoder positions.
const CROSS_SPLITS: usize = 6;
/// Choices kept per slot: a pass takes at most 257.
pub const MAX_CHOICES: usize = 260;

/// Words of the state per slot, and what each holds.
pub const SLOT_WORDS: usize = 16;
const MODE: usize = 0;
const LEN: usize = 1;
const CHOICES: usize = 2;
const PROMPT_LEN: usize = 3;
/// Four words of prompt tokens.
const PROMPT: usize = 4;
const INPUT: usize = 8;
const LAST: usize = 9;
const PENULT: usize = 10;
/// The last timestamp token generated (0: none yet).
const LAST_TS: usize = 11;
/// The probability of `<|nospeech|>` at the first position (f32 bits).
const NO_SPEECH: usize = 12;
/// The most choices before the slot is done.
const LIMIT: usize = 13;

/// What a slot is doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Idle = 0,
    /// The GPU chooses its tokens (temperature 0).
    Greedy = 1,
    /// The host reads its logits after each step and writes its next token.
    Host = 2,
}

/// One token chosen on the GPU: its log-probability and the gap between the best and the
/// second-best masked logits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Choice {
    pub token: i32,
    pub logprob: f32,
    pub margin: f32,
}

/// A slot as a finished run of steps left it.
#[derive(Debug, Clone)]
pub struct SlotView {
    pub mode: Mode,
    /// Positions in its cache.
    pub len: usize,
    pub prompt_len: usize,
    pub choices: Vec<Choice>,
    pub no_speech: f32,
}

/// Every slot after a run of steps, and the raw logits of the slots asked for.
pub struct Snapshot {
    pub slots: Vec<SlotView>,
    pub logits: Vec<(usize, Vec<f32>)>,
}

/// A run of steps submitted, not yet read back.
pub struct Pending {
    staging: wgpu::Buffer,
    submission: Option<wgpu::SubmissionIndex>,
    batch: usize,
    host: Vec<usize>,
}

/// A linear layer as the file keeps it, Q8_0: `n` rows of `k` int8 values (four per word)
/// and an f16 scale per 32 (two per word).
struct Q8 {
    n: usize,
    k: usize,
    qs: Tensor,
    scales: Tensor,
    bias: Option<Tensor>,
}

struct Block {
    norm_self: Norm,
    qkv: Q8,
    self_out: Q8,
    norm_cross: Norm,
    cross_q: Q8,
    cross_out: Q8,
    norm_ffn: Norm,
    fc1: Q8,
    fc2: Q8,
}

struct Kernels {
    embed: Arc<Kernel>,
    /// By the most windows a dispatch takes (1, 2, 4, 8: fewer accumulators and reductions
    /// per lane for fewer windows, each output's arithmetic the same), rows per subgroup (1,
    /// 2, 4), without and with the layer norm.
    gemv: [[[Arc<Kernel>; 2]; 3]; 4],
    self_attend: Arc<Kernel>,
    cross_attend: Arc<Kernel>,
    cross_combine: Arc<Kernel>,
    choose: Arc<Kernel>,
    cross_store: Arc<Kernel>,
}

struct Buffers {
    state: Tensor,
    x: Tensor,
    qkv: Tensor,
    context: Tensor,
    hidden: Tensor,
    /// The cross-attention's parts: [slot][head][split][max, sum, 64 weighted values].
    parts: Tensor,
    logits: Tensor,
    choices: Tensor,
    /// [layer][key, value][slot][ctx][d_model] f16 pairs.
    self_cache: Tensor,
    /// [layer][key, value][slot][1500][d_model] f16 pairs.
    cross_cache: Tensor,
    /// One window's encoder states, and their keys and values for every layer (f32).
    encoded: Tensor,
    stacked: Tensor,
}

pub struct Decoder {
    gpu: Arc<Gpu>,
    d: usize,
    heads: usize,
    vocab: usize,
    ctx: usize,
    slots: usize,
    kernels: Kernels,
    gemm: Gemm,
    /// The token embedding, also the output head.
    embedding: Q8,
    positions: Tensor,
    blocks: Vec<Block>,
    final_norm: Norm,
    /// Every layer's cross-attention key and value weights, stacked (f16, for the multiply).
    cross_kv: Weights,
    /// Bits of the suppressed tokens, then of the tokens suppressed at the first choice.
    masks: Tensor,
    buffers: Buffers,
    staging: Mutex<Vec<wgpu::Buffer>>,
    staging_bytes: u64,
}

impl Decoder {
    pub fn load(
        gpu: &Arc<Gpu>,
        file: &Gguf,
        hp: &HParams,
        options: &GpuOptions,
    ) -> Result<Decoder> {
        let d = hp.d_model;
        let heads = hp.dec_heads;
        let slots = options.slots.clamp(1, MAX_SLOTS);
        if heads * 64 != d || !d.is_multiple_of(64) {
            return Err(Error::Unsupported(format!(
                "the GPU decoder for d_model {d}, {heads} heads"
            )));
        }
        let vocab = hp.vocab;
        let (eos, no_timestamps) = (hp.eos as u32, hp.no_timestamps as u32);
        let positions = file.dequantize(tensor(file, "dec.pos_emb.weight")?)?;
        let ctx = positions.len() / d;
        let mask_words = vocab.div_ceil(32);
        let state_consts = [
            ("SLOT_WORDS", SLOT_WORDS as u32),
            ("MODE", MODE as u32),
            ("LEN", LEN as u32),
        ];
        let attention = |frames: (&'static str, usize)| {
            let mut c = vec![
                ("HEADS", heads as u32),
                ("D_MODEL", d as u32),
                (frames.0, frames.1 as u32),
            ];
            c.extend(state_consts);
            c
        };
        let kernels = Kernels {
            embed: gpu.kernel(
                "whisper_embed",
                EMBED,
                &[
                    ("D_MODEL", d as u32),
                    ("SLOT_WORDS", SLOT_WORDS as u32),
                    ("MODE", MODE as u32),
                    ("LEN", LEN as u32),
                    ("INPUT", INPUT as u32),
                ],
                &[false, false, false, true, true],
                4,
            )?,
            gemv: {
                let kernel = |windows: u32, rows: u32, norm: u32| {
                    gpu.kernel(
                        "whisper_gemv",
                        GEMV,
                        &[("MAX_B", windows), ("ROWS", rows), ("NORM", norm)],
                        &[false, false, false, false, false, false, false, true],
                        20,
                    )
                };
                let by_rows = |windows: u32| -> Result<[[Arc<Kernel>; 2]; 3]> {
                    Ok([
                        [kernel(windows, 1, 0)?, kernel(windows, 1, 1)?],
                        [kernel(windows, 2, 0)?, kernel(windows, 2, 1)?],
                        [kernel(windows, 4, 0)?, kernel(windows, 4, 1)?],
                    ])
                };
                const { assert!(MAX_SLOTS == 8) };
                [by_rows(1)?, by_rows(2)?, by_rows(4)?, by_rows(8)?]
            },
            self_attend: gpu.kernel(
                "whisper_self_attend",
                SELF_ATTEND,
                &attention(("CTX", ctx)),
                &[false, true, false, true],
                8,
            )?,
            cross_attend: gpu.kernel(
                "whisper_cross_attend",
                CROSS_ATTEND,
                &[
                    attention(("FRAMES", FRAMES)),
                    vec![("SPLITS", CROSS_SPLITS as u32)],
                ]
                .concat(),
                &[false, false, false, false, true],
                8,
            )?,
            cross_combine: gpu.kernel(
                "whisper_cross_combine",
                CROSS_COMBINE,
                &attention(("SPLITS", CROSS_SPLITS)),
                &[false, false, true],
                0,
            )?,
            choose: gpu.kernel(
                "whisper_choose",
                CHOOSE,
                &[
                    ("VOCAB", vocab as u32),
                    ("EOS", eos),
                    ("NO_TIMESTAMPS", no_timestamps),
                    ("TS_BEGIN", no_timestamps + 1),
                    ("MAX_INITIAL", super::super::MAX_INITIAL_TIMESTAMP as u32),
                    ("NO_SPEECH_TOKEN", no_timestamps - 1),
                    ("MASK_WORDS", mask_words as u32),
                    ("MAX_CHOICES", MAX_CHOICES as u32),
                    ("CTX", ctx as u32),
                    ("SLOT_WORDS", SLOT_WORDS as u32),
                    ("MODE", MODE as u32),
                    ("LEN", LEN as u32),
                    ("CHOICES", CHOICES as u32),
                    ("PROMPT_LEN", PROMPT_LEN as u32),
                    ("PROMPT", PROMPT as u32),
                    ("INPUT", INPUT as u32),
                    ("LAST", LAST as u32),
                    ("PENULT", PENULT as u32),
                    ("LAST_TS", LAST_TS as u32),
                    ("NO_SPEECH", NO_SPEECH as u32),
                    ("LIMIT", LIMIT as u32),
                ],
                &[false, false, true, true],
                0,
            )?,
            cross_store: gpu.kernel(
                "whisper_cross_store",
                CROSS_STORE,
                &[
                    ("D_MODEL", d as u32),
                    ("LAYERS", hp.dec_layers as u32),
                    ("FRAMES", FRAMES as u32),
                ],
                &[false, true],
                8,
            )?,
        };
        let gemm = Gemm::new(gpu, options.gemm.unwrap_or_else(super::default_variant))?;

        let blocks = (0..hp.dec_layers)
            .map(|l| -> Result<Block> {
                let name = |part: &str| format!("dec.blocks.{l}.{part}");
                let lin = |part: &str| q8(gpu, file, &[name(part)], part);
                Ok(Block {
                    norm_self: Norm::load(gpu, file, &name("norm_self"))?,
                    qkv: q8(
                        gpu,
                        file,
                        &["q", "k", "v"].map(|p| name(&format!("self_attn.{p}"))),
                        "self_attn.qkv",
                    )?,
                    self_out: lin("self_attn.out")?,
                    norm_cross: Norm::load(gpu, file, &name("norm_cross"))?,
                    cross_q: lin("cross_attn.q")?,
                    cross_out: lin("cross_attn.out")?,
                    norm_ffn: Norm::load(gpu, file, &name("norm_ffn"))?,
                    fc1: lin("ffn.fc1")?,
                    fc2: lin("ffn.fc2")?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let d_ff = blocks.first().map_or(4 * d, |b| b.fc1.n);
        let cross_names: Vec<String> = (0..hp.dec_layers)
            .flat_map(|l| ["k", "v"].map(move |p| format!("dec.blocks.{l}.cross_attn.{p}")))
            .collect();
        let cross_kv = stacked(gpu, file, "cross_attn.kv", &cross_names)?;
        let embedding = q8(gpu, file, &["dec.token_embd".to_owned()], "dec.token_embd")?;
        if embedding.n != vocab || embedding.k != d {
            return Err(Error::Gguf(format!(
                "dec.token_embd is {}×{}",
                embedding.n, embedding.k
            )));
        }
        let mut masks = vec![0u32; 2 * mask_words];
        for (list, offset) in [(&hp.suppress, 0), (&hp.begin_suppress, mask_words)] {
            for &id in list.iter().filter(|&&i| (0..vocab as i32).contains(&i)) {
                masks[offset + id as usize / 32] |= 1 << (id % 32);
            }
        }
        let layers = hp.dec_layers;
        let s = slots;
        let buffers = Buffers {
            state: gpu.zeros_u32("decoder state", 1 + s * SLOT_WORDS),
            x: gpu.zeros_f32("decoder x", s * d),
            qkv: gpu.zeros_f32("decoder qkv", s * 3 * d),
            context: gpu.zeros_f32("decoder context", s * d),
            hidden: gpu.zeros_f32("decoder hidden", s * d_ff),
            parts: gpu.zeros_f32("cross-attention parts", s * heads * CROSS_SPLITS * 66),
            logits: gpu.zeros_f32("logits", s * vocab),
            choices: gpu.zeros_u32("choices", s * MAX_CHOICES * 3),
            self_cache: gpu.zeros_u32("self cache", layers * 2 * s * ctx * d / 2),
            cross_cache: gpu.zeros_u32("cross cache", layers * 2 * s * FRAMES * d / 2),
            encoded: gpu.zeros_f32("encoded", FRAMES * d),
            stacked: gpu.zeros_f32("cross keys and values", FRAMES * cross_kv.n),
        };
        let staging_bytes = (((1 + s * SLOT_WORDS) + s * MAX_CHOICES * 3 + s * vocab) * 4) as u64;
        // Readback buffers for the runs in flight (two, and one being read), made now so a
        // run never allocates.
        let staging = (0..3).map(|_| new_staging(gpu, staging_bytes)).collect();
        Ok(Decoder {
            gpu: gpu.clone(),
            d,
            heads,
            vocab,
            ctx,
            slots,
            kernels,
            gemm,
            embedding,
            positions: gpu.upload_f32("dec.pos_emb", &positions),
            blocks,
            final_norm: Norm::load(gpu, file, "dec.final_norm")?,
            cross_kv,
            masks: gpu.upload_u32("token masks", &masks),
            buffers,
            staging: Mutex::new(staging),
            staging_bytes,
        })
    }

    /// Slots the decoder has.
    pub fn slots(&self) -> usize {
        self.slots
    }

    /// Positions a slot's self-attention cache holds.
    pub fn context(&self) -> usize {
        self.ctx
    }

    /// Records the keys and values of window `window` of the encoder's output (its rows from
    /// 1500·window) for every layer into `slot`'s cross cache.
    pub fn record_cross(&self, rec: &mut Recorder, encoded: &Tensor, window: usize, slot: usize) {
        let bytes = (FRAMES * self.d * 4) as u64;
        rec.copy(
            encoded,
            window as u64 * bytes,
            &self.buffers.encoded.buffer,
            0,
            bytes,
        );
        self.record_cross_store(rec, slot);
    }

    /// [`record_cross`](Self::record_cross) from encoder states on the CPU (1500 × d_model).
    pub fn load_cross(&self, encoded: &Mat, slot: usize) -> Result<()> {
        if encoded.rows != FRAMES || encoded.cols != self.d {
            return Err(Error::Other(format!(
                "encoder states of {}×{}",
                encoded.rows, encoded.cols
            )));
        }
        self.gpu
            .queue
            .write_buffer(&self.buffers.encoded.buffer, 0, f32_bytes(&encoded.data));
        let mut rec = Recorder::new(&self.gpu, Timing::None);
        self.record_cross_store(&mut rec, slot);
        rec.submit();
        Ok(())
    }

    fn record_cross_store(&self, rec: &mut Recorder, slot: usize) {
        let b = &self.buffers;
        let gpu = &*self.gpu;
        rec.with_pass("gemm/cross.kv", |pass| {
            self.gemm.encode(
                gpu,
                pass,
                &Operands {
                    x: &b.encoded,
                    weights: &self.cross_kv,
                    y: &b.stacked,
                    m: FRAMES,
                },
                Epilogue::default(),
            );
        });
        rec.dispatch(
            "cross/store",
            &self.kernels.cross_store,
            &[&b.stacked, &b.cross_cache],
            &[slot as u32, self.slots as u32],
            groups_2d(self.blocks.len() * 2 * FRAMES * self.d / 2),
        );
    }

    /// Starts `slot` on `prompt` (one to four tokens) from position 0, choosing for itself
    /// (greedy) or by the host, for at most `limit` choices. Takes effect at the next submit.
    pub fn start(&self, slot: usize, prompt: &[i32], mode: Mode, limit: usize) {
        self.resume(slot, prompt, 0, mode, limit);
        self.write_words(slot, NO_SPEECH, &[0]);
    }

    /// [`start`](Self::start) from position `at` of the prompt: the slot's cache already
    /// holds the prompt's keys and values before it (an earlier run of the same prompt on the
    /// same encoder states left them, and they'd come out the same), and its no-speech
    /// probability stays.
    pub fn resume(&self, slot: usize, prompt: &[i32], at: usize, mode: Mode, limit: usize) {
        assert!((1..=4).contains(&prompt.len()) && at < prompt.len() && slot < self.slots);
        let mut words = [0u32; NO_SPEECH];
        words[MODE] = mode as u32;
        words[LEN] = at as u32;
        words[PROMPT_LEN] = prompt.len() as u32;
        for (i, &t) in prompt.iter().enumerate() {
            words[PROMPT + i] = t as u32;
        }
        words[INPUT] = prompt[at] as u32;
        self.write_words(slot, 0, &words);
        self.write_words(slot, LIMIT, &[limit.min(MAX_CHOICES) as u32]);
    }

    /// A host slot's next input token, or that it's done (`None`).
    pub fn set_input(&self, slot: usize, token: Option<i32>) {
        match token {
            Some(t) => self.write_words(slot, INPUT, &[t as u32]),
            None => self.write_words(slot, MODE, &[Mode::Idle as u32]),
        }
    }

    /// Stops a slot.
    pub fn stop(&self, slot: usize) {
        self.write_words(slot, MODE, &[Mode::Idle as u32]);
    }

    fn write_words(&self, slot: usize, word: usize, words: &[u32]) {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.gpu.queue.write_buffer(
            &self.buffers.state.buffer,
            ((1 + slot * SLOT_WORDS + word) * 4) as u64,
            &bytes,
        );
    }

    /// Records and submits `steps` steps of the first `batch` slots, then the copies of the
    /// state, the choices and the raw logits of the `host` slots that [`finish`](Self::finish)
    /// reads.
    pub fn submit(&self, batch: usize, steps: usize, host: &[usize]) -> Result<Pending> {
        let staging = self.take_staging();
        let mut rec = Recorder::new(&self.gpu, Timing::None);
        for _ in 0..steps {
            self.record_step(&mut rec, batch);
        }
        self.record_readback(&mut rec, &staging, host);
        Ok(Pending {
            staging,
            submission: rec.submit(),
            batch,
            host: host.to_vec(),
        })
    }

    /// Waits for a run of steps and reads its state, choices and logits back.
    pub fn finish(&self, pending: Pending) -> Result<Snapshot> {
        let Pending {
            staging,
            submission,
            batch,
            host,
        } = pending;
        let state_words = 1 + self.slots * SLOT_WORDS;
        let choice_words = self.slots * MAX_CHOICES * 3;
        let mut spans = vec![(0, state_words), (state_words, choice_words)];
        for i in 0..host.len() {
            spans.push((state_words + choice_words + i * self.vocab, self.vocab));
        }
        let bytes = ((state_words + choice_words + host.len() * self.vocab) * 4) as u64;
        let data = map_read(&self.gpu, &staging, bytes, submission, &spans);
        self.staging.lock().unwrap().push(staging);
        let mut data = data?.into_iter();
        let state: Vec<u32> = data.next().unwrap().iter().map(|v| v.to_bits()).collect();
        let choices: Vec<u32> = data.next().unwrap().iter().map(|v| v.to_bits()).collect();
        let slots = (0..batch)
            .map(|s| {
                let w = &state[1 + s * SLOT_WORDS..1 + (s + 1) * SLOT_WORDS];
                let n = (w[CHOICES] as usize).min(MAX_CHOICES);
                SlotView {
                    mode: match w[MODE] {
                        1 => Mode::Greedy,
                        2 => Mode::Host,
                        _ => Mode::Idle,
                    },
                    len: w[LEN] as usize,
                    prompt_len: w[PROMPT_LEN] as usize,
                    choices: (0..n)
                        .map(|i| {
                            let c = &choices[(s * MAX_CHOICES + i) * 3..][..3];
                            Choice {
                                token: c[0] as i32,
                                logprob: f32::from_bits(c[1]),
                                margin: f32::from_bits(c[2]),
                            }
                        })
                        .collect(),
                    no_speech: f32::from_bits(w[NO_SPEECH]),
                }
            })
            .collect();
        Ok(Snapshot {
            slots,
            logits: host.into_iter().zip(data).collect(),
        })
    }

    fn take_staging(&self) -> wgpu::Buffer {
        self.staging
            .lock()
            .unwrap()
            .pop()
            .unwrap_or_else(|| new_staging(&self.gpu, self.staging_bytes))
    }

    fn record_readback(&self, rec: &mut Recorder, staging: &wgpu::Buffer, host: &[usize]) {
        let b = &self.buffers;
        let state_bytes = ((1 + self.slots * SLOT_WORDS) * 4) as u64;
        let choice_bytes = (self.slots * MAX_CHOICES * 3 * 4) as u64;
        rec.copy(&b.state, 0, staging, 0, state_bytes);
        rec.copy(&b.choices, 0, staging, state_bytes, choice_bytes);
        let row = (self.vocab * 4) as u64;
        for (i, &s) in host.iter().enumerate() {
            rec.copy(
                &b.logits,
                s as u64 * row,
                staging,
                state_bytes + choice_bytes + i as u64 * row,
                row,
            );
        }
    }

    /// One step of the first `batch` slots, recorded.
    fn record_step(&self, rec: &mut Recorder, batch: usize) {
        let b = &self.buffers;
        let k = &self.kernels;
        let s = self.slots as u32;
        let heads = self.heads as u32;
        rec.dispatch(
            "embed",
            &k.embed,
            &[
                &self.embedding.qs,
                &self.embedding.scales,
                &self.positions,
                &b.state,
                &b.x,
            ],
            &[batch as u32],
            [1, 1, 1],
        );
        for (l, block) in self.blocks.iter().enumerate() {
            let l = l as u32;
            let norm = |n| Some(n);
            self.gemv(
                rec,
                "gemv/self.qkv",
                &b.x,
                norm(&block.norm_self),
                &block.qkv,
                &b.qkv,
                batch,
                false,
                false,
            );
            rec.dispatch(
                "attn/self",
                &k.self_attend,
                &[&b.qkv, &b.self_cache, &b.state, &b.context],
                &[l, s],
                [heads, batch as u32, 1],
            );
            self.gemv(
                rec,
                "gemv/self.out",
                &b.context,
                None,
                &block.self_out,
                &b.x,
                batch,
                false,
                true,
            );
            self.gemv(
                rec,
                "gemv/cross.q",
                &b.x,
                norm(&block.norm_cross),
                &block.cross_q,
                &b.qkv,
                batch,
                false,
                false,
            );
            rec.dispatch(
                "attn/cross",
                &k.cross_attend,
                &[&b.qkv, &b.cross_cache, &b.cross_cache, &b.state, &b.parts],
                &[l, s],
                [heads, CROSS_SPLITS as u32, batch as u32],
            );
            rec.dispatch(
                "attn/combine",
                &k.cross_combine,
                &[&b.parts, &b.state, &b.context],
                &[],
                [heads, batch as u32, 1],
            );
            self.gemv(
                rec,
                "gemv/cross.out",
                &b.context,
                None,
                &block.cross_out,
                &b.x,
                batch,
                false,
                true,
            );
            self.gemv(
                rec,
                "gemv/ffn.fc1",
                &b.x,
                norm(&block.norm_ffn),
                &block.fc1,
                &b.hidden,
                batch,
                true,
                false,
            );
            self.gemv(
                rec,
                "gemv/ffn.fc2",
                &b.hidden,
                None,
                &block.fc2,
                &b.x,
                batch,
                false,
                true,
            );
        }
        self.gemv(
            rec,
            "gemv/head",
            &b.x,
            Some(&self.final_norm),
            &self.embedding,
            &b.logits,
            batch,
            false,
            false,
        );
        rec.dispatch(
            "choose",
            &k.choose,
            &[&b.logits, &self.masks, &b.state, &b.choices],
            &[],
            [batch as u32, 1, 1],
        );
    }

    /// Records `y = epilogue(W · x + b)` for the first `batch` rows of `x`, layer-normed on
    /// the way in when there's a `norm`.
    #[allow(clippy::too_many_arguments)]
    fn gemv(
        &self,
        rec: &mut Recorder,
        label: &str,
        x: &Tensor,
        norm: Option<&Norm>,
        w: &Q8,
        y: &Tensor,
        batch: usize,
        gelu: bool,
        accumulate: bool,
    ) {
        let flags = u32::from(w.bias.is_some())
            | if gelu { 3 << 1 } else { 0 }
            | (u32::from(accumulate) << 3);
        let bias = w.bias.as_ref().unwrap_or(&w.scales);
        // A fused layer norm costs each workgroup its rows' statistics: fewer, larger ones.
        let rows = if norm.is_some() { 4 } else { gemv_rows(batch) };
        let windows = batch.next_power_of_two().min(MAX_SLOTS);
        let kernel = &self.kernels.gemv[windows.trailing_zeros() as usize]
            [rows.trailing_zeros() as usize][usize::from(norm.is_some())];
        let (gamma, beta) = norm.map_or((&w.scales, &w.scales), |n| (&n.weight, &n.bias));
        rec.dispatch(
            label,
            kernel,
            &[
                x,
                &w.qs,
                &w.scales,
                bias,
                &self.buffers.state,
                gamma,
                beta,
                y,
            ],
            &[
                w.n as u32,
                w.k as u32,
                batch as u32,
                flags,
                super::EPS.to_bits(),
            ],
            [w.n.div_ceil(8 * rows) as u32, 1, 1],
        );
    }

    /// GPU milliseconds of one step of `batch` slots, per submit or per dispatch (labels
    /// `family/name`), for the lab. The slots should be decoding (else the kernels skip).
    pub fn step_timed(
        &self,
        batch: usize,
        steps: usize,
        per_dispatch: bool,
    ) -> Result<Vec<(String, f64)>> {
        let mut profiler = Profiler::new(&self.gpu, 2048);
        let timing = if per_dispatch {
            Timing::Dispatches(&mut profiler)
        } else {
            Timing::Passes(&mut profiler)
        };
        let mut rec = Recorder::new(&self.gpu, timing);
        for _ in 0..steps {
            self.record_step(&mut rec, batch);
        }
        rec.submit();
        drop(rec);
        profiler.read(&self.gpu)
    }
}

/// A buffer the decoder's state, choices and logits are copied into and read back from.
fn new_staging(gpu: &Gpu, bytes: u64) -> wgpu::Buffer {
    gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("decoder readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Output rows per subgroup of the matrix-vector kernel for a batch: one for a window or
/// two (more workgroups, to stream the weights at full bandwidth), more as each weight is
/// used for more rows. `VELLUM_GEMV_ROWS` overrides it (an experiment).
fn gemv_rows(batch: usize) -> usize {
    static ROWS: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let forced = *ROWS.get_or_init(|| {
        std::env::var("VELLUM_GEMV_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|r| [1, 2, 4].contains(r))
    });
    forced.unwrap_or(match batch {
        0..=2 => 1,
        3..=4 => 2,
        _ => 4,
    })
}

/// Layers with the same inputs, their rows one after another, as [`Q8`] (zero biases for a
/// layer without one, when another has one).
fn q8(gpu: &Gpu, file: &Gguf, names: &[String], label: &str) -> Result<Q8> {
    let mut qs: Vec<u32> = Vec::new();
    let mut scales: Vec<u32> = Vec::new();
    let mut biases: Vec<Option<Vec<f32>>> = Vec::new();
    let (mut n, mut k) = (0, None);
    for name in names {
        let info = tensor(file, &format!("{name}.weight"))?;
        let cols = info.dims[0] as usize;
        let rows: usize = info.dims[1..].iter().map(|d| *d as usize).product();
        if info.dtype != Dtype::Q8_0 || !cols.is_multiple_of(64) {
            return Err(Error::Unsupported(format!(
                "{name}.weight as {:?} for the GPU decoder",
                info.dtype
            )));
        }
        if *k.get_or_insert(cols) != cols {
            return Err(Error::Gguf(format!("{label}: layers of different widths")));
        }
        let blocks = file.data(info).as_chunks::<34>().0;
        if blocks.len() * 32 != rows * cols {
            return Err(Error::Gguf(format!(
                "{name}.weight: {} blocks",
                blocks.len()
            )));
        }
        qs.reserve(rows * cols / 4);
        scales.reserve(rows * cols / 64);
        for pair in blocks.as_chunks::<2>().0 {
            let scale = |b: &[u8; 34]| u16::from_le_bytes([b[0], b[1]]) as u32;
            scales.push(scale(&pair[0]) | (scale(&pair[1]) << 16));
            for b in pair {
                for w in b[2..].as_chunks::<4>().0 {
                    qs.push(u32::from_le_bytes(*w));
                }
            }
        }
        biases.push(match file.tensor(&format!("{name}.bias")) {
            Some(b) => Some(file.dequantize(b)?),
            None => None,
        });
        n += rows;
    }
    let bias = if biases.iter().any(Option::is_some) {
        let mut all = Vec::with_capacity(n);
        let mut rows = names.iter().map(|name| {
            tensor(file, &format!("{name}.weight"))
                .map(|i| i.dims[1..].iter().map(|d| *d as usize).product::<usize>())
        });
        for b in &biases {
            let r = rows.next().unwrap()?;
            match b {
                Some(b) if b.len() == r => all.extend_from_slice(b),
                Some(_) => return Err(Error::Gguf(format!("{label}: a bias of the wrong size"))),
                None => all.extend(std::iter::repeat_n(0., r)),
            }
        }
        Some(gpu.upload_f32(label, &all))
    } else {
        None
    };
    Ok(Q8 {
        n,
        k: k.unwrap_or(0),
        qs: gpu.upload_u32(label, &qs),
        scales: gpu.upload_u32(label, &scales),
        bias,
    })
}
