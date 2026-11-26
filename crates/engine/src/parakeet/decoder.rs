//! Greedy RNN-T decoding: the predictor (`pred.*`, an embedding and LSTM layers), the joint
//! network (`joint.*`), and the transcript the tokens make, as transcribe.cpp decodes
//! (`src/arch/parakeet/decoder.cpp`, `model.cpp` there).
//!
//! The reference runs the whole decoder in f32 on weights dequantized once at load (the
//! per-step products are matrix–vector, where quantized activations would buy nothing), so this
//! does the same and agrees with it to rounding. It decides exactly as a step-by-step loop
//! would, only faster:
//!
//! - the weights stay as Q8_0 blocks, read as `scale · q` (exact) — 7.4 MB for the LSTM instead
//!   of 26 MB of f32, so a step's weights come from the last-level cache, not memory;
//! - layer 0's input product `W_x · embed[token]` is a table made at load, one row per token;
//! - the joint's predictor projection is kept until a token is emitted;
//! - through a run of blanks the joint is evaluated for several frames in one matrix product,
//!   and the frames past the first token are thrown away.
//!
//! Every product computes each output exactly as [`crate::cpu::dot`] does, whatever the
//! batching, so none of this changes a single bit of a logit.

use super::HParams;
use crate::{
    Error, Result, Token, Trace, Transcript, Word,
    cpu::{Linear, Mat, Weight, expand, products, relu, sigmoid},
    gguf::Gguf,
    tokenizer::{Tokenizer, WORD_MARKER},
};
use rayon::prelude::*;
use serde_json::json;

/// The most tokens the reference emits on one encoder frame before it moves on, whatever the
/// joint says (its RNN-T `max_symbols`, fixed at 10 for this head).
const MAX_SYMBOLS_PER_FRAME: usize = 10;

/// Frames the joint is evaluated for at once after a token: the frame it was emitted on (whose
/// next decision is usually blank) and the next. Each batch that's all blank doubles the next,
/// up to [`MAX_FRAMES_AHEAD`]. Speech emits a token every two or three frames, so longer
/// batches mostly compute logits that are thrown away (measured on the probe windows: this
/// evaluates 1.3 frames per decision and takes a fifth less time than one frame at a time).
const FRAMES_AHEAD_AFTER_TOKEN: usize = 2;
const MAX_FRAMES_AHEAD: usize = 4;

/// A token the decoder emitted, at the encoder frame it emitted it on.
#[derive(Debug, Clone, Copy)]
pub struct Emitted {
    pub token: usize,
    pub frame: usize,
    /// Its confidence as the reference reports it: one minus the entropy of the softmax over
    /// every joint output (blank included), relative to the entropy of a uniform one.
    pub p: f32,
}

/// A weight matrix of `rows` outputs × `cols` inputs as the file stores it, with an optional
/// bias.
struct Matrix {
    rows: usize,
    cols: usize,
    weight: Weight,
    bias: Option<Vec<f32>>,
}

impl Matrix {
    /// The tensor `name` (ggml dims `[cols, rows]`), and `bias` if given.
    fn load(file: &Gguf, name: &str, bias: Option<&str>) -> Result<Matrix> {
        let (rows, cols, weight) = crate::cpu::weight(file, name)?;
        let bias = match bias {
            Some(b) => {
                let b = crate::cpu::vector(file, b)?;
                if b.len() != rows {
                    return Err(Error::Gguf(format!(
                        "{name}: bias of {} for {rows} rows",
                        b.len()
                    )));
                }
                Some(b)
            }
            None => None,
        };
        Ok(Matrix {
            rows,
            cols,
            weight,
            bias,
        })
    }

    /// `W·x` (without the bias) for each input row, into `out` (`rows` per input). Serial: at
    /// the decoder's sizes, one window per thread is the parallelism that pays.
    fn product(&self, xs: &[&[f32]], out: &mut [f32]) {
        products(
            xs,
            self.weight.rows(0..self.rows, self.cols),
            self.cols,
            out,
        );
    }

    /// `W·x + b` for each input row.
    fn apply(&self, xs: &[&[f32]], out: &mut [f32]) {
        self.product(xs, out);
        if let Some(b) = &self.bias {
            for row in out.chunks_exact_mut(self.rows) {
                for (y, b) in row.iter_mut().zip(b) {
                    *y += b;
                }
            }
        }
    }

    /// Row `r` as f32 (exact).
    fn row(&self, r: usize) -> Vec<f32> {
        let mut out = vec![0.; self.cols];
        match &self.weight {
            Weight::F32(w) => out.copy_from_slice(&w[r * self.cols..(r + 1) * self.cols]),
            Weight::Q8(blocks) => {
                let per = self.cols / 32;
                expand(&blocks[r * per..(r + 1) * per], &mut out);
            }
        }
        out
    }
}

/// One LSTM layer: gates `[i, f, g, o]` (PyTorch's order) stacked in one `4H × H` matrix for
/// the input and one for the previous hidden state, with their biases pre-summed.
struct LstmLayer {
    input: Matrix,
    recurrent: Matrix,
    bias: Vec<f32>,
}

/// The predictor's state: each layer's hidden and cell vectors.
#[derive(Clone)]
struct State {
    h: Vec<Vec<f32>>,
    c: Vec<Vec<f32>>,
}

/// What the predictor steps on.
enum Input<'a> {
    /// The start: zeros.
    Start,
    /// A token: its embedding, whose layer-0 input product is in the table.
    Token(usize),
    /// Any other vector (an injected embedding).
    Vector(&'a [f32]),
}

pub struct Decoder {
    hidden: usize,
    /// One row per token (the blank's row is never read: the start state embeds as zeros).
    embed: Matrix,
    lstm: Vec<LstmLayer>,
    /// `W_x · embed[token]` for layer 0, one row of `4H` per token: exactly the product the
    /// step would compute.
    layer0_inputs: Vec<f32>,
    /// Encoder frame → joint space; applied to all frames at once.
    joint_enc: Linear,
    /// Predictor output → joint space.
    joint_pred: Matrix,
    /// Joint space → one logit per token, blank last.
    joint_out: Matrix,
    blank: usize,
    /// Milliseconds between mel frames (hop / sample rate: 10 ms here).
    hop_ms: f64,
}

impl Decoder {
    pub fn load(file: &Gguf, hparams: &HParams) -> Result<Decoder> {
        let hidden = hparams.predictor_hidden;
        let embed = Matrix::load(file, "pred.embed.weight", None)?;
        if embed.cols != hidden || embed.rows != hparams.vocab {
            return Err(Error::Gguf(format!(
                "pred.embed.weight is {}×{}, expected {}×{hidden}",
                embed.rows, embed.cols, hparams.vocab
            )));
        }
        let lstm = (0..hparams.predictor_layers)
            .map(|l| {
                let layer = LstmLayer {
                    input: Matrix::load(file, &format!("pred.lstm.{l}.Wx"), None)?,
                    recurrent: Matrix::load(file, &format!("pred.lstm.{l}.Wh"), None)?,
                    bias: crate::cpu::vector(file, &format!("pred.lstm.{l}.bias"))?,
                };
                for m in [&layer.input, &layer.recurrent] {
                    if m.rows != 4 * hidden || m.cols != hidden {
                        return Err(Error::Gguf(format!("pred.lstm.{l} isn't 4×{hidden} gates")));
                    }
                }
                if layer.bias.len() != 4 * hidden {
                    return Err(Error::Gguf(format!("pred.lstm.{l}.bias isn't 4×{hidden}")));
                }
                Ok(layer)
            })
            .collect::<Result<Vec<_>>>()?;
        if lstm.is_empty() {
            return Err(Error::Gguf("a predictor without LSTM layers".into()));
        }
        let joint_enc = Linear::load(file, "joint.enc")?;
        let joint_pred = Matrix::load(file, "joint.pred.weight", Some("joint.pred.bias"))?;
        let joint_out = Matrix::load(file, "joint.out.weight", Some("joint.out.bias"))?;
        if joint_out.rows != hparams.vocab || hparams.blank != hparams.vocab - 1 {
            return Err(Error::Gguf(format!(
                "joint.out has {} outputs for {} tokens with blank {}",
                joint_out.rows, hparams.vocab, hparams.blank
            )));
        }
        let joint = hparams.joint_hidden;
        if joint_enc.rows != joint
            || joint_enc.cols != hparams.d_model
            || joint_pred.rows != joint
            || joint_out.cols != joint
        {
            return Err(Error::Gguf("joint network shapes don't line up".into()));
        }
        let frontend = |key: &str| {
            file.metadata_u64(&format!("stt.frontend.{key}"))
                .ok_or_else(|| Error::Gguf(format!("no stt.frontend.{key}")))
        };
        let hop_ms = 1000. * frontend("hop_length")? as f64 / frontend("sample_rate")? as f64;
        let layer0_inputs = input_table(&embed, &lstm[0].input);
        Ok(Decoder {
            hidden,
            embed,
            lstm,
            layer0_inputs,
            joint_enc,
            joint_pred,
            joint_out,
            blank: hparams.blank,
            hop_ms,
        })
    }

    /// Encoder frames (frames × d_model) to the joint's space (frames × joint hidden): the
    /// part of the joint that doesn't depend on the tokens, done once for every frame.
    pub fn project(&self, encoded: &Mat) -> Mat {
        self.joint_enc.forward(encoded)
    }

    /// One predictor step from `state` on `input`. Checkpoints the first step as the reference
    /// dumps it.
    fn predict(&self, input: Input, state: &State, first: bool, trace: &mut dyn Trace) -> State {
        let h = self.hidden;
        let mut injected = None;
        if first {
            let mut x = match input {
                Input::Start => vec![0.; h],
                Input::Token(t) => self.embed.row(t),
                Input::Vector(v) => v.to_vec(),
            };
            let before = x.clone();
            checkpoint_vector(trace, "dec.embed.0", &mut x);
            if x != before {
                injected = Some(x);
            }
        }
        let input = match &injected {
            Some(x) => Input::Vector(x),
            None => input,
        };
        let mut next = State {
            h: Vec::with_capacity(self.lstm.len()),
            c: Vec::with_capacity(self.lstm.len()),
        };
        let mut from_input = vec![0.; 4 * h];
        let mut gates = vec![0.; 4 * h];
        for (l, layer) in self.lstm.iter().enumerate() {
            // `W_x · x`: the table's row for a token, zeros (as the product of zeros is) at the
            // start, else computed.
            let x_product: &[f32] = match (l, &input) {
                (0, Input::Start) => {
                    from_input.fill(0.);
                    &from_input
                }
                (0, Input::Token(t)) => &self.layer0_inputs[t * 4 * h..(t + 1) * 4 * h],
                (0, Input::Vector(x)) => {
                    layer.input.product(&[x], &mut from_input);
                    &from_input
                }
                _ => {
                    layer.input.product(&[&next.h[l - 1]], &mut from_input);
                    &from_input
                }
            };
            layer.recurrent.product(&[&state.h[l]], &mut gates);
            // `(W_x·x + W_h·h) + b`, summed in the reference's order.
            for ((g, x), b) in gates.iter_mut().zip(x_product).zip(&layer.bias) {
                *g = (x + *g) + b;
            }
            let (mut hl, mut cl) = (vec![0.; h], vec![0.; h]);
            for k in 0..h {
                let i = sigmoid(gates[k]);
                let f = sigmoid(gates[h + k]);
                let g = gates[2 * h + k].tanh();
                let o = sigmoid(gates[3 * h + k]);
                cl[k] = f * state.c[l][k] + i * g;
                hl[k] = o * cl[k].tanh();
            }
            if first {
                checkpoint_vector(trace, &format!("dec.lstm.{l}.h.0"), &mut hl);
                checkpoint_vector(trace, &format!("dec.lstm.{l}.c.0"), &mut cl);
            }
            next.h.push(hl);
            next.c.push(cl);
        }
        next
    }

    /// The joint's logits for frames `start..start + n` of `enc_proj` with one predictor
    /// output (already projected): `out · relu(enc + pred) + b`, one row of logits per frame.
    fn joint(
        &self,
        enc_proj: &Mat,
        start: usize,
        n: usize,
        pred: &[f32],
        hidden: &mut Vec<f32>,
        logits: &mut Vec<f32>,
    ) {
        let width = self.joint_out.cols;
        hidden.resize(n * width, 0.);
        for (k, row) in hidden.chunks_exact_mut(width).enumerate() {
            for ((s, e), p) in row.iter_mut().zip(enc_proj.row(start + k)).zip(pred) {
                *s = relu(e + p);
            }
        }
        let rows: Vec<&[f32]> = hidden.chunks_exact(width).collect();
        logits.resize(n * self.joint_out.rows, 0.);
        self.joint_out.apply(&rows, logits);
    }

    /// Greedy RNN-T over `enc_proj` (frames × joint hidden, from [`project`](Self::project)):
    /// on each frame, emit the joint's best token and stay, until it says blank (or
    /// [`MAX_SYMBOLS_PER_FRAME`] tokens are out), then move to the next frame. The predictor
    /// only steps when a token is emitted, so its output is reused across blanks, and so are
    /// the logits of frames evaluated ahead.
    pub fn greedy(&self, enc_proj: &Mat, trace: &mut dyn Trace) -> Vec<Emitted> {
        let frames = enc_proj.rows;
        let mut emitted = Vec::new();
        if frames == 0 {
            return emitted;
        }
        let zeros = vec![0.; self.hidden];
        let mut state = State {
            h: vec![zeros.clone(); self.lstm.len()],
            c: vec![zeros; self.lstm.len()],
        };
        let mut next = self.predict(Input::Start, &state, true, trace);
        let mut pred_proj = vec![0.; self.joint_pred.rows];
        self.joint_pred
            .apply(&[next.h.last().unwrap()], &mut pred_proj);
        let vocab = self.joint_out.rows;
        // Logits of frames `ahead_start..ahead_start + ahead_len` for the current predictor
        // output, and how many frames the next evaluation takes.
        let (mut ahead, mut hidden) = (Vec::new(), Vec::new());
        let (mut ahead_start, mut ahead_len, mut run) = (0, 0, FRAMES_AHEAD_AFTER_TOKEN);
        let mut logits = vec![0.; vocab];
        let (mut frame, mut symbols) = (0, 0);
        // The reference's guard against a decoder that never says blank.
        let max_steps = 16 * frames + 1024;
        for step in 0..max_steps {
            if frame >= frames {
                break;
            }
            if !(ahead_start..ahead_start + ahead_len).contains(&frame) {
                let n = run.min(frames - frame);
                self.joint(enc_proj, frame, n, &pred_proj, &mut hidden, &mut ahead);
                (ahead_start, ahead_len) = (frame, n);
                run = (2 * run).min(MAX_FRAMES_AHEAD);
            }
            let k = frame - ahead_start;
            logits.copy_from_slice(&ahead[k * vocab..(k + 1) * vocab]);
            if step == 0 {
                // The reference dumps the first step's joint as log-probabilities (NeMo's form),
                // and this decides on them: the same decision, as the shift is uniform.
                log_softmax(&mut logits);
                checkpoint_vector(trace, "dec.joint.0", &mut logits);
            }
            let (token, second) = top2(&logits);
            if token == self.blank {
                frame += 1;
                symbols = 0;
                continue;
            }
            let p = confidence(&logits);
            trace.event(
                "rnnt.step",
                json!({
                    "step": step,
                    "frame": frame,
                    "token": token,
                    "p": p,
                    "logit": logits[token],
                    "second": second,
                    "top2_margin": logits[token] - logits[second],
                    "blank_margin": logits[token] - logits[self.blank],
                }),
            );
            emitted.push(Emitted { token, frame, p });
            state = next;
            next = self.predict(Input::Token(token), &state, false, trace);
            self.joint_pred
                .apply(&[next.h.last().unwrap()], &mut pred_proj);
            // A new predictor output: the logits ahead are stale.
            (ahead_len, run) = (0, FRAMES_AHEAD_AFTER_TOKEN);
            symbols += 1;
            if symbols >= MAX_SYMBOLS_PER_FRAME {
                frame += 1;
                symbols = 0;
            }
        }
        emitted
    }

    /// The transcript the emitted tokens make, as the reference builds it: control and unknown
    /// pieces dropped; each token spans its encoder frame (a frame is `subsampling` mel hops, so
    /// `t0 = frame · 80 ms`, `t1 = t0 + 80 ms`, rounded half away from zero as `llround` does);
    /// a word starts at the first token and at each piece beginning with `▁`, and ends where
    /// its last token ends; the text is every token decoded together with runs of spaces
    /// collapsed and the ends trimmed. Nothing emitted (silence) gives an empty transcript.
    pub fn transcript(
        &self,
        emitted: &[Emitted],
        tokenizer: &Tokenizer,
        hparams: &HParams,
    ) -> Transcript {
        let frame_ms = hparams.subsampling_factor as f64 * self.hop_ms;
        let ms = |frame: usize| (frame_ms * frame as f64).round() as i64;
        let kept: Vec<&Emitted> = emitted
            .iter()
            .filter(|e| {
                !tokenizer.is_special(e.token) && !is_language_tag(tokenizer.piece(e.token))
            })
            .collect();
        let tokens: Vec<Token> = kept
            .iter()
            .map(|e| Token {
                id: e.token as i32,
                p: e.p,
                t0_ms: ms(e.frame),
                t1_ms: ms(e.frame + 1),
                text: tokenizer.decode([e.token]),
            })
            .collect();
        // Each word as the range of tokens it spans.
        let mut spans: Vec<std::ops::Range<usize>> = Vec::new();
        for (i, e) in kept.iter().enumerate() {
            if i == 0 || tokenizer.piece(e.token).starts_with(WORD_MARKER) {
                spans.push(i..i + 1);
            } else if let Some(last) = spans.last_mut() {
                last.end = i + 1;
            }
        }
        let words = spans
            .into_iter()
            .map(|span| {
                let text = tokenizer.decode(kept[span.clone()].iter().map(|e| e.token));
                Word {
                    t0_ms: tokens[span.start].t0_ms,
                    t1_ms: tokens[span.end - 1].t1_ms,
                    text: match text.strip_prefix(' ') {
                        Some(rest) => rest.to_owned(),
                        None => text,
                    },
                }
            })
            .collect();
        let text = normalize_spaces(&tokenizer.decode(kept.iter().map(|e| e.token)));
        Transcript {
            text,
            language: None,
            words,
            tokens,
            ..Transcript::default()
        }
    }
}

/// A one-dimensional checkpoint (the reference dumps the decoder's vectors with shape `[n]`).
fn checkpoint_vector(trace: &mut dyn Trace, name: &str, x: &mut Vec<f32>) {
    if let Some((_, data)) = trace.inject(name, &[x.len()]) {
        if data.len() == x.len() {
            *x = data;
        } else {
            eprintln!(
                "injecting {name}: {} values for {}; not used",
                data.len(),
                x.len()
            );
        }
    }
    if trace.wants(name) {
        trace.tensor(name, &[x.len()], x);
    }
}

/// The best and second-best entries; ties go to the first, as the reference's argmax does.
fn top2(x: &[f32]) -> (usize, usize) {
    let (mut best, mut second) = (0, usize::MAX);
    for (i, &v) in x.iter().enumerate().skip(1) {
        if v > x[best] {
            second = best;
            best = i;
        } else if second == usize::MAX || v > x[second] {
            second = i;
        }
    }
    (best, if second == usize::MAX { best } else { second })
}

/// Log-softmax in place, accumulating in f64 as the reference does.
fn log_softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = x.iter().map(|&v| ((v - max) as f64).exp()).sum();
    let log_sum = sum.ln() as f32 + max;
    for v in x {
        *v -= log_sum;
    }
}

/// The reference's per-token confidence (NeMo's entropy measure): with `p = softmax(logits)`,
/// `1 − H(p) / ln(n)`, mixing f32 and f64 exactly where it does.
fn confidence(logits: &[f32]) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits.iter().map(|&v| (v - max).exp() as f64).sum();
    let inv_sum = (1. / sum) as f32;
    let entropy: f64 = logits
        .iter()
        .map(|&v| {
            let p = (v - max).exp() * inv_sum;
            -(p as f64) * (p as f64 + 1e-10).ln()
        })
        .sum();
    let max_entropy = (logits.len() as f64).ln();
    if max_entropy <= 0. {
        return 1.;
    }
    (1. - entropy / max_entropy) as f32
}

/// A multilingual locale piece such as `<en-US>` (2–3 lowercase letters, a dash, 2–4
/// letters), which the reference drops from results. None occur in English vocabularies.
fn is_language_tag(piece: &str) -> bool {
    let Some(inner) = piece.strip_prefix('<').and_then(|p| p.strip_suffix('>')) else {
        return false;
    };
    let Some((language, region)) = inner.split_once('-') else {
        return false;
    };
    (2..=3).contains(&language.len())
        && language.bytes().all(|b| b.is_ascii_lowercase())
        && (2..=4).contains(&region.len())
        && region.bytes().all(|b| b.is_ascii_alphabetic())
}

/// Runs of ASCII spaces collapsed to one, and both ends trimmed of them.
fn normalize_spaces(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        if ch == ' ' && out.ends_with(' ') {
            continue;
        }
        out.push(ch);
    }
    out.trim_matches(' ').to_owned()
}

/// Layer 0's input product for every token's embedding, `vocab` rows of `4H`: what each step
/// would compute, made once at load (10.5 MB, a few milliseconds).
fn input_table(embed: &Matrix, input: &Matrix) -> Vec<f32> {
    let rows: Vec<Vec<f32>> = (0..embed.rows).map(|t| embed.row(t)).collect();
    let mut table = vec![0.; embed.rows * input.rows];
    table
        .par_chunks_mut(16 * input.rows)
        .zip(rows.par_chunks(16))
        .for_each(|(out, xs)| {
            let xs: Vec<&[f32]> = xs.iter().map(Vec::as_slice).collect();
            input.product(&xs, out);
        });
    table
}
