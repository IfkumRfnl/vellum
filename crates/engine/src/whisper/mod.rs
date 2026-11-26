//! Whisper (Small, Large v3 Turbo) on the CPU as transcribe.cpp runs it for the app's call
//! (`src/arch/whisper/` there; `RunOptions::default()` with or without a language): the log-mel
//! of the window padded to 30 s, the encoder, language detection if no language is given, then
//! greedy decoding with timestamps and Whisper's temperature fallback, cut into segments.
//!
//! Checkpoints carry the reference's dump names, for the first decoding pass of a window's
//! first 30 s only, as the reference dumps them: `enc.mel.in`, `enc.conv1.out`,
//! `enc.conv2.out`, `enc.pos_emb`, `enc.embed.out`, `enc.block.<L>.out`, `enc.final`, then the
//! prompt pass's `dec.token_emb`, `dec.pos_emb`, `dec.embed_sum`, `dec.block.<L>.out`,
//! `dec.out_before_head`, `dec.logits_raw`, `dec.logits`, and the 20th step's
//! `dec.logits_raw.gen20`. Every decoder decision is a `whisper.step` event with its margin.

mod decoder;
mod decoding;
mod encoder;
pub mod gpu;
mod mel;
mod ops;
mod pipeline;
mod tokenizer;

use crate::{
    Error, ModelInfo, Result, RunOptions, Timings, Token, Trace, Transcript, Word, cpu::Mat,
    gguf::Gguf, gguf::Value, trace::checkpoint,
};
use decoding::{Rng, Specials, TEXT_END};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::Instant,
};
use tokenizer::Tokenizer;

/// The most tokens decoded per 30 s pass (the reference's `k_max_new_tokens`).
const MAX_NEW_TOKENS: usize = 256;
/// Whisper's recipe, transcribe.cpp's defaults (`transcribe_whisper_run_ext_init`).
const TEMPERATURE_STEP: f32 = 0.2;
const COMPRESSION_RATIO_THRESHOLD: f32 = 2.4;
const LOGPROB_THRESHOLD: f32 = -1.;
const NO_SPEECH_THRESHOLD: f32 = 0.6;
/// The first timestamp may be at most 1 s in (in 20 ms steps).
const MAX_INITIAL_TIMESTAMP: usize = 50;

/// The model's shape and vocabulary, from its `stt.whisper.*` and `stt.frontend.*` metadata.
#[derive(Debug, Clone)]
pub struct HParams {
    pub d_model: usize,
    pub enc_layers: usize,
    pub enc_heads: usize,
    pub dec_layers: usize,
    pub dec_heads: usize,
    pub vocab: usize,
    pub sot: i32,
    pub transcribe: i32,
    pub no_timestamps: i32,
    pub eos: i32,
    pub suppress: Vec<i32>,
    pub begin_suppress: Vec<i32>,
    /// Samples and mel frames of one 30 s pass.
    pub samples: usize,
    pub frames: usize,
    /// Whether the vocabulary has language tokens (not the `.en` models).
    pub multilingual: bool,
}

impl HParams {
    pub fn from_gguf(file: &Gguf) -> Result<HParams> {
        let n = |key: &str| {
            file.metadata_u64(key)
                .map(|v| v as usize)
                .ok_or_else(|| Error::Gguf(format!("no {key}")))
        };
        let w = |key: &str| n(&format!("stt.whisper.{key}"));
        let ids = |key: &str| match file.metadata.get(&format!("stt.whisper.{key}")) {
            Some(Value::Array(items)) => items
                .iter()
                .filter_map(|v| match *v {
                    Value::I32(i) => Some(i),
                    ref other => other.as_u64().map(|i| i as i32),
                })
                .collect(),
            _ => Vec::new(),
        };
        let d_model = w("encoder.d_model")?;
        if w("decoder.d_model")? != d_model {
            return Err(Error::Gguf("encoder and decoder widths differ".into()));
        }
        Ok(HParams {
            d_model,
            enc_layers: w("encoder.n_layers")?,
            enc_heads: w("encoder.n_heads")?,
            dec_layers: w("decoder.n_layers")?,
            dec_heads: w("decoder.n_heads")?,
            vocab: w("decoder.vocab_size")?,
            sot: w("decoder_start_token_id")? as i32,
            transcribe: w("transcribe_token_id")? as i32,
            no_timestamps: w("no_timestamps_token_id")? as i32,
            eos: file
                .metadata_u64("tokenizer.ggml.eos_token_id")
                .map_or(TEXT_END, |v| v as i32),
            suppress: ids("suppress_tokens"),
            begin_suppress: ids("begin_suppress_tokens"),
            samples: n("stt.frontend.n_samples").unwrap_or(480_000),
            frames: n("stt.frontend.nb_max_frames").unwrap_or(3000),
            multilingual: file
                .metadata
                .get("stt.capability.lang_detect")
                .is_none_or(|v| v.as_u64() == Some(1)),
        })
    }
}

pub struct Whisper {
    hp: HParams,
    mel: mel::LogMel,
    backend: Backend,
    tokenizer: Tokenizer,
    /// Each language the model knows (ISO code) and its token.
    languages: Vec<(String, i32)>,
    pub info: ModelInfo,
    /// One run at a time on the GPU: its buffers and slots are the model's.
    gpu_runs: Mutex<()>,
}

/// Where the model runs. Only one is loaded: the GPU's needs nothing of the CPU's.
// One per model: its size doesn't matter.
#[allow(clippy::large_enum_variant)]
enum Backend {
    Cpu {
        encoder: encoder::Encoder,
        decoder: decoder::Decoder,
    },
    Gpu(gpu::GpuWhisper),
}

/// Where a pass's encoder states are: the CPU's cross cache, or a GPU decoder slot (and how
/// many of the prompt's positions its cache already holds: language detection leaves
/// `<|startoftranscript|>`'s).
enum Encoded<'a> {
    Cpu(&'a decoder::CrossCache),
    Gpu(&'a gpu::GpuWhisper, usize, usize),
}

/// Greedy steps per GPU submit, two submits in flight: a window's last run is mostly steps
/// past its end (about 0.1 ms each once it's done).
const STEPS_PER_SUBMIT: usize = 8;

/// One window's decoding state across its 30 s passes.
struct Window<'a> {
    trace: &'a mut dyn Trace,
    rng: Rng,
    detected: Option<String>,
    /// Accepted tokens of every pass, text only (for the transcript's text).
    text_ids: Vec<i32>,
    words: Vec<Word>,
    tokens: Vec<Token>,
}

/// An attempt's choices in order: each token, its log-probability and the gap between the
/// best and second-best logits it was chosen from.
type Choices = Vec<(i32, f32, f32)>;

/// One temperature's attempt at a pass.
struct Attempt {
    generated: Vec<i32>,
    /// Each generated token's log-probability.
    logprobs: Vec<f32>,
    avg_logprob: f32,
    compression: f32,
}

impl Whisper {
    /// The model on the CPU.
    pub fn load(file: &Gguf) -> Result<Whisper> {
        Whisper::load_on(file, None)
    }

    /// The model on `gpu` when there's one (and it has what the GPU path needs), else on the
    /// CPU.
    pub fn load_on(file: &Gguf, gpu: Option<Arc<crate::gpu::Gpu>>) -> Result<Whisper> {
        Whisper::load_with(file, gpu, &gpu::GpuOptions::default())
    }

    /// [`load_on`](Self::load_on) with the GPU model's options.
    pub fn load_with(
        file: &Gguf,
        gpu: Option<Arc<crate::gpu::Gpu>>,
        options: &gpu::GpuOptions,
    ) -> Result<Whisper> {
        let hp = HParams::from_gguf(file)?;
        let tokenizer = Tokenizer::from_gguf(file)?;
        let mut languages = Vec::new();
        if hp.multilingual
            && let Some(Value::Array(codes)) = file.metadata.get("general.languages")
        {
            for code in codes.iter().filter_map(Value::as_str) {
                let id = tokenizer
                    .special(&format!("<|{code}|>"))
                    .ok_or_else(|| Error::Gguf(format!("no token for language {code}")))?;
                languages.push((code.to_owned(), id));
            }
        }
        // A GPU without what the GPU path needs (f16 tensor cores, subgroups) runs it on the
        // CPU instead.
        let model = match gpu {
            Some(device) => match gpu::GpuWhisper::load(&device, file, &hp, options) {
                Ok(model) => Some((model, device.name())),
                Err(Error::Unsupported(_)) => None,
                Err(e) => return Err(e),
            },
            None => None,
        };
        let (backend, device) = match model {
            Some((model, name)) => (Backend::Gpu(model), name),
            None => {
                let (encoder, decoder) = rayon::join(
                    || encoder::Encoder::load(file, &hp),
                    || decoder::Decoder::load(file, &hp),
                );
                (
                    Backend::Cpu {
                        encoder: encoder?,
                        decoder: decoder?,
                    },
                    "CPU".into(),
                )
            }
        };
        Ok(Whisper {
            mel: mel::LogMel::from_gguf(file)?,
            backend,
            tokenizer,
            languages,
            info: ModelInfo {
                architecture: "whisper".into(),
                variant: file.metadata_str("stt.variant").unwrap_or_default().into(),
                device,
            },
            hp,
            gpu_runs: Mutex::new(()),
        })
    }

    /// The GPU the model runs on, if it runs on one.
    pub fn gpu(&self) -> Option<&crate::gpu::Gpu> {
        match &self.backend {
            Backend::Gpu(g) => Some(&g.gpu),
            Backend::Cpu { .. } => None,
        }
    }

    /// The GPU model, if the model runs on one (for the lab).
    pub fn gpu_model(&self) -> Option<&gpu::GpuWhisper> {
        match &self.backend {
            Backend::Gpu(g) => Some(g),
            Backend::Cpu { .. } => None,
        }
    }

    pub fn hparams(&self) -> &HParams {
        &self.hp
    }

    fn specials(&self) -> Specials {
        Specials {
            eos: self.hp.eos,
            no_timestamps: self.hp.no_timestamps,
            timestamp_begin: self.hp.no_timestamps + 1,
            vocab: self.hp.vocab as i32,
        }
    }

    /// Every window in order, each as [`run`](Self::run) gives it. Without a language, the
    /// first window with any text decides it for the rest, as the app does.
    pub fn run_each(
        &self,
        windows: &[&[f32]],
        options: &RunOptions,
        trace: &mut dyn Trace,
        stop: &(dyn Fn() -> bool + Sync),
        each: &mut dyn FnMut(usize, Transcript) -> Result<()>,
    ) -> Result<()> {
        // On the GPU, several windows at once (`pipeline.rs`), unless a trace wants the
        // encoder's checkpoints, which a run gives window by window (as
        // `VELLUM_WHISPER_SEQUENTIAL=1` asks, for comparisons).
        if matches!(self.backend, Backend::Gpu(_))
            && windows.len() > 1
            && !encoder_traced(trace, self.hp.enc_layers)
            && std::env::var_os("VELLUM_WHISPER_SEQUENTIAL").is_none()
        {
            return self.pipeline(windows, options, stop, each);
        }
        let mut options = options.clone();
        for (i, pcm) in windows.iter().enumerate() {
            if stop() {
                return Err(Error::Cancelled);
            }
            let transcript = self.run(pcm, &options, trace)?;
            if options.language.is_none() && !transcript.text.trim().is_empty() {
                options.language = transcript.language.clone();
            }
            each(i, transcript)?;
        }
        Ok(())
    }

    /// One window of 16 kHz audio to its transcript: segments as words (the reference's
    /// result has segment times only, which the app then uses), and the detected language
    /// when none was given.
    pub fn run(
        &self,
        pcm: &[f32],
        options: &RunOptions,
        trace: &mut dyn Trace,
    ) -> Result<Transcript> {
        if pcm.is_empty() {
            return Err(Error::Other("no audio".into()));
        }
        let _exclusive = match &self.backend {
            Backend::Gpu(_) => Some(self.gpu_runs.lock().unwrap_or_else(|e| e.into_inner())),
            Backend::Cpu { .. } => None,
        };
        let mut timings = Timings::default();
        let start = Instant::now();
        let mel = self.features(pcm);
        timings.mel_ms = ms(start);
        let mut language = self.language_token(options)?;

        let mut window = Window {
            trace,
            rng: Rng::new(0x5EED),
            detected: None,
            text_ids: Vec::new(),
            words: Vec::new(),
            tokens: Vec::new(),
        };
        let total = mel.rows;
        let mut seek = 0;
        let decode_start = Instant::now();
        let mut encode_ms = 0.;
        while seek < total {
            let first = seek == 0;
            let frames = (total - seek).min(self.hp.frames);
            let mut chunk = Mat::zeros(self.hp.frames, mel.cols);
            chunk.data[..frames * mel.cols]
                .copy_from_slice(&mel.data[seek * mel.cols..(seek + frames) * mel.cols]);
            let start = Instant::now();
            let detect = first && self.hp.multilingual && language.is_none();
            let advance = match &self.backend {
                Backend::Cpu { encoder, decoder } => {
                    let encoded = if first {
                        checkpoint(window.trace, "enc.mel.in", &mut chunk);
                        if window.trace.injects("enc.final") {
                            let mut e = Mat::zeros(0, self.hp.d_model);
                            checkpoint(window.trace, "enc.final", &mut e);
                            e
                        } else {
                            encoder.forward(&chunk, window.trace)
                        }
                    } else {
                        encoder.forward(&chunk, &mut crate::NoTrace)
                    };
                    encode_ms += ms(start);
                    if detect {
                        let (id, code) = self.detect_language(decoder, &encoded, window.trace);
                        language = Some(id);
                        window.detected = Some(code);
                    }
                    let cross = decoder.cross_cache(&encoded, true);
                    let encoded = Encoded::Cpu(&cross);
                    self.pass(&mut window, &encoded, language, seek, frames, first)?
                }
                Backend::Gpu(g) => {
                    const SLOT: usize = 0;
                    self.encode_gpu(g, &mut chunk, SLOT, first, window.trace)?;
                    encode_ms += ms(start);
                    if detect {
                        let (id, code) = self.detect_language_gpu(g, SLOT, window.trace)?;
                        language = Some(id);
                        window.detected = Some(code);
                    }
                    let encoded = Encoded::Gpu(g, SLOT, usize::from(detect));
                    self.pass(&mut window, &encoded, language, seek, frames, first)?
                }
            };
            seek += advance;
        }
        timings.encode_ms = encode_ms;
        timings.decode_ms = ms(decode_start) - encode_ms;
        Ok(self.transcript(window, timings))
    }

    /// A window's log-mel features (frames × mels): padded to 30 s first when it fits one
    /// pass, as transformers does.
    fn features(&self, pcm: &[f32]) -> Mat {
        if pcm.len() <= self.hp.samples {
            let mut padded = pcm.to_vec();
            padded.resize(self.hp.samples, 0.);
            self.mel.compute(&padded)
        } else {
            self.mel.compute(pcm)
        }
    }

    /// The token of the language asked for, if any.
    fn language_token(&self, options: &RunOptions) -> Result<Option<i32>> {
        match &options.language {
            Some(code) if self.hp.multilingual => Ok(Some(
                self.languages
                    .iter()
                    .find(|(c, _)| c == code)
                    .map(|(_, id)| *id)
                    .ok_or_else(|| Error::Other(format!("Whisper doesn't know \"{code}\"")))?,
            )),
            Some(code) if code != "en" => Err(Error::Other(format!(
                "an English-only model, not \"{code}\""
            ))),
            _ => Ok(None),
        }
    }

    /// A finished window's transcript.
    fn transcript(&self, window: Window, timings: Timings) -> Transcript {
        let text = self.tokenizer.decode_bytes(window.text_ids.iter().copied());
        Transcript {
            text: String::from_utf8_lossy(tokenizer::trim(&text)).into_owned(),
            language: window.detected,
            words: window.words,
            tokens: window.tokens,
            timings,
        }
    }

    /// A pass's 30 s of mel features through the GPU encoder into decoder slot `slot`. The
    /// first pass of a traced window takes `enc.mel.in` and `enc.final` as checkpoints (read
    /// back, or substituted), and the encoder's own as it makes them.
    fn encode_gpu(
        &self,
        g: &gpu::GpuWhisper,
        chunk: &mut Mat,
        slot: usize,
        first: bool,
        trace: &mut dyn Trace,
    ) -> Result<()> {
        let traced = first && encoder_traced(trace, self.hp.enc_layers);
        if !traced {
            g.encode(&[(chunk, slot)])?;
            return g.gpu.wait();
        }
        checkpoint(trace, "enc.mel.in", chunk);
        let mut encoded = if trace.injects("enc.final") {
            let mut e = Mat::zeros(0, self.hp.d_model);
            checkpoint(trace, "enc.final", &mut e);
            e
        } else {
            g.encoder.encode(&[chunk], Some(trace))?.remove(0)
        };
        if encoded.rows == 0 {
            encoded = Mat::zeros(1500, self.hp.d_model);
        }
        g.decoder.load_cross(&encoded, slot)?;
        g.gpu.wait()
    }

    /// The language whose token the decoder rates highest after `<|startoftranscript|>`
    /// alone (keys and values in f32, as the reference's one-off prefill graph has them).
    fn detect_language(
        &self,
        decoder: &decoder::Decoder,
        encoded: &Mat,
        trace: &mut dyn Trace,
    ) -> (i32, String) {
        let cross = decoder.cross_cache(encoded, false);
        let mut cache = decoder.self_cache(1, false);
        let logits = decoder.forward(&[self.hp.sot], &mut cache, &cross, None);
        self.pick_language(logits.row(0), trace)
    }

    /// [`detect_language`](Self::detect_language) on the GPU, from slot `slot`'s encoder keys
    /// and values (f16, the pass's own: the one difference from the oracle's).
    fn detect_language_gpu(
        &self,
        g: &gpu::GpuWhisper,
        slot: usize,
        trace: &mut dyn Trace,
    ) -> Result<(i32, String)> {
        let dec = &g.decoder;
        dec.start(slot, &[self.hp.sot], gpu::decoder::Mode::Host, 1);
        let snapshot = dec.finish(dec.submit(slot + 1, 1, &[slot])?)?;
        dec.stop(slot);
        let (_, row) = snapshot
            .logits
            .first()
            .ok_or_else(|| Error::Other("no logits came back".into()))?;
        Ok(self.pick_language(row, trace))
    }

    /// The language whose token is highest in a row of logits (the first of equals).
    fn pick_language(&self, row: &[f32], trace: &mut dyn Trace) -> (i32, String) {
        let mut best: Option<(f32, usize)> = None;
        for (i, (_, id)) in self.languages.iter().enumerate() {
            let v = row[*id as usize];
            if best.is_none_or(|(b, _)| v > b) {
                best = Some((v, i));
            }
        }
        let (_, i) = best.expect("a multilingual model has languages");
        let mut sorted: Vec<f32> = self
            .languages
            .iter()
            .map(|(_, id)| row[*id as usize])
            .collect();
        sorted.sort_by(|a, b| b.total_cmp(a));
        trace.event(
            "whisper.language",
            serde_json::json!({
                "language": self.languages[i].0,
                "token": self.languages[i].1,
                "margin": sorted[0] - sorted.get(1).copied().unwrap_or(f32::NEG_INFINITY),
            }),
        );
        (self.languages[i].1, self.languages[i].0.clone())
    }

    /// The prompt of every pass: `<|startoftranscript|>`, and the language and
    /// `<|transcribe|>` when there's a language.
    fn prompt(&self, language: Option<i32>) -> Vec<i32> {
        let mut prompt = vec![self.hp.sot];
        if let Some(language) = language {
            prompt.push(language);
            prompt.push(self.hp.transcribe);
        }
        prompt
    }

    /// One 30 s pass from mel frame `seek` (`frames` of them real): decoding at rising
    /// temperatures until one passes, its segments into the window. Returns how many frames
    /// the window advances.
    fn pass(
        &self,
        window: &mut Window,
        encoded: &Encoded,
        language: Option<i32>,
        seek: usize,
        frames: usize,
        first: bool,
    ) -> Result<usize> {
        let prompt = self.prompt(language);
        let mut no_speech = 0f32;
        let mut accepted = None;
        let mut skip = false;
        for (tier, &temperature) in temperatures().iter().enumerate() {
            let traced = first && tier == 0;
            let wanted = (tier == 0).then_some(&mut no_speech);
            let attempt = match encoded {
                Encoded::Cpu(cross) => self.attempt(
                    window,
                    cross,
                    &prompt,
                    temperature,
                    traced,
                    wanted,
                    seek,
                    tier,
                ),
                // Every attempt after the first finds the prompt's keys and values but the
                // last position's in the slot's cache already.
                Encoded::Gpu(g, slot, primed) => self.attempt_gpu(
                    g,
                    (*slot, if tier == 0 { *primed } else { prompt.len() - 1 }),
                    window,
                    &prompt,
                    temperature,
                    wanted,
                    seek,
                    tier,
                )?,
            };
            let (passes, skips) = self.judge(window, &attempt, no_speech, seek, temperature);
            skip = skips;
            accepted = Some(attempt);
            if skip || passes {
                break;
            }
        }
        if skip {
            return Ok(frames);
        }
        let attempt = accepted.expect("at least one temperature");
        Ok(self.accept(window, &attempt, seek, frames))
    }

    /// Whether an attempt passes (not too repetitive, likely enough) and whether the pass is
    /// skipped as silence, recorded as a `whisper.pass` event.
    fn judge(
        &self,
        window: &mut Window,
        attempt: &Attempt,
        no_speech: f32,
        seek: usize,
        temperature: f32,
    ) -> (bool, bool) {
        let passes = attempt.compression <= COMPRESSION_RATIO_THRESHOLD
            && attempt.avg_logprob >= LOGPROB_THRESHOLD;
        let skip = no_speech > NO_SPEECH_THRESHOLD && attempt.avg_logprob < LOGPROB_THRESHOLD;
        window.trace.event(
            "whisper.pass",
            serde_json::json!({
                "seek": seek,
                "temperature": temperature,
                "compression_ratio": attempt.compression,
                "avg_logprob": attempt.avg_logprob,
                "no_speech_prob": no_speech,
                "no_speech_skip": skip,
                "tokens": attempt.generated.len(),
            }),
        );
        (passes, skip)
    }

    /// An accepted attempt's segments into the window; how many frames the window advances.
    fn accept(&self, window: &mut Window, attempt: &Attempt, seek: usize, frames: usize) -> usize {
        let specials = self.specials();
        let offset_ms = seek as i64 * 10;
        let (segments, advance) =
            decoding::segments(&attempt.generated, &specials, offset_ms, frames);
        for segment in &segments {
            let ids: Vec<i32> = attempt.generated[segment.tokens.clone()]
                .iter()
                .copied()
                .filter(|&id| (0..TEXT_END).contains(&id) && !specials.is_timestamp(id))
                .collect();
            let bytes = self.tokenizer.decode_bytes(ids);
            let text = tokenizer::trim(&bytes);
            if !text.is_empty() {
                window.words.push(Word {
                    t0_ms: segment.t0_ms,
                    t1_ms: segment.t1_ms,
                    text: String::from_utf8_lossy(text).into_owned(),
                });
            }
            for i in segment.tokens.clone() {
                let id = attempt.generated[i];
                if (0..TEXT_END).contains(&id) {
                    window.tokens.push(Token {
                        id,
                        p: attempt.logprobs[i].exp(),
                        t0_ms: segment.t0_ms,
                        t1_ms: segment.t1_ms,
                        text: String::from_utf8_lossy(&self.tokenizer.decode_bytes([id]))
                            .into_owned(),
                    });
                }
            }
        }
        window.text_ids.extend(
            attempt
                .generated
                .iter()
                .copied()
                .filter(|&id| (0..TEXT_END).contains(&id) && !specials.is_timestamp(id)),
        );
        if advance == 0 { frames } else { advance }
    }

    /// Decoding at one temperature: the prompt, then a token at a time until the end token,
    /// 256 tokens or a full context. `no_speech`, on the first attempt, receives the
    /// probability of `<|nospeech|>` after `<|startoftranscript|>`.
    #[allow(clippy::too_many_arguments)]
    fn attempt(
        &self,
        window: &mut Window,
        cross: &decoder::CrossCache,
        prompt: &[i32],
        temperature: f32,
        traced: bool,
        no_speech: Option<&mut f32>,
        seek: usize,
        tier: usize,
    ) -> Attempt {
        let Backend::Cpu { decoder, .. } = &self.backend else {
            unreachable!("a CPU cross cache comes from the CPU decoder")
        };
        let specials = self.specials();
        let context = decoder.max_positions();
        let mut cache = decoder.self_cache(context, true);
        let mut taps = decoder::Taps::default();
        let mut logits = decoder.forward(prompt, &mut cache, cross, traced.then_some(&mut taps));
        if traced {
            let trace = &mut *window.trace;
            let mut dump = |name: &str, m: Option<Mat>| {
                if let Some(mut m) = m {
                    checkpoint(trace, name, &mut m);
                }
            };
            dump("dec.token_emb", taps.token_emb.take());
            dump("dec.pos_emb", taps.pos_emb.take());
            dump("dec.embed_sum", taps.embed_sum.take());
            for (l, b) in std::mem::take(&mut taps.blocks).into_iter().enumerate() {
                dump(&format!("dec.block.{l}.out"), Some(b));
            }
            dump("dec.out_before_head", taps.before_head.take());
            checkpoint(trace, "dec.logits_raw", &mut logits);
            if trace.wants("dec.logits") {
                let mut log_probs = logits.clone();
                for r in 0..log_probs.rows {
                    log_softmax(log_probs.row_mut(r));
                }
                checkpoint(trace, "dec.logits", &mut log_probs);
            }
        }
        if let Some(no_speech) = no_speech {
            // The raw row of `<|startoftranscript|>` (the prompt's first token here).
            let row = logits.row(0);
            *no_speech = probability(row, (self.hp.no_timestamps - 1) as usize);
        }

        let mut row = logits.row(logits.rows - 1).to_vec();
        for &id in &self.hp.begin_suppress {
            if let Some(v) = row.get_mut(id as usize) {
                *v = f32::NEG_INFINITY;
            }
        }
        let mut choices = Vec::new();
        let mut generated = Vec::new();
        let mut next = self.choose(&mut row, &generated, temperature, &mut window.rng);
        loop {
            choices.push(next);
            if choices.len() > MAX_NEW_TOKENS || next.0 == specials.eos {
                break;
            }
            generated.push(next.0);
            if cache.len + 1 > context {
                break;
            }
            let mut logits = decoder.forward(&[next.0], &mut cache, cross, None);
            if traced && choices.len() == 20 {
                // The reference dumps this one as a vector.
                let name = "dec.logits_raw.gen20";
                if let Some((_, data)) = window.trace.inject(name, &[logits.cols]) {
                    logits = Mat::from_vec(1, data.len(), data);
                }
                if window.trace.wants(name) {
                    window.trace.tensor(name, &[logits.cols], &logits.data);
                }
            }
            row.copy_from_slice(logits.row(0));
            next = self.choose(&mut row, &generated, temperature, &mut window.rng);
        }
        self.replay(window, &choices, seek, tier)
    }

    /// [`attempt`](Self::attempt) on the GPU, the window's encoder keys and values in slot
    /// `slot`, its cache holding the prompt's first `at` positions: at temperature 0 the GPU
    /// chooses every token, runs of steps at a time; above, each step's logits come back and
    /// the token is sampled here, as the oracle samples it.
    #[allow(clippy::too_many_arguments)]
    fn attempt_gpu(
        &self,
        g: &gpu::GpuWhisper,
        (slot, at): (usize, usize),
        window: &mut Window,
        prompt: &[i32],
        temperature: f32,
        no_speech: Option<&mut f32>,
        seek: usize,
        tier: usize,
    ) -> Result<Attempt> {
        let choices = if temperature <= 0. {
            let (choices, ns) = self.greedy_gpu(g, slot, prompt, at)?;
            if let Some(no_speech) = no_speech {
                *no_speech = ns;
            }
            choices
        } else {
            // Only the first attempt asks for the no-speech probability, and it's greedy.
            self.sampled_gpu(g, (slot, at), window, prompt, temperature)?
        };
        Ok(self.replay(window, &choices, seek, tier))
    }

    /// Greedy decoding of `prompt` in slot `slot` (the only one decoding), the GPU choosing
    /// every token: its choices and the no-speech probability.
    fn greedy_gpu(
        &self,
        g: &gpu::GpuWhisper,
        slot: usize,
        prompt: &[i32],
        at: usize,
    ) -> Result<(Choices, f32)> {
        use gpu::decoder::Mode;
        let dec = &g.decoder;
        let batch = slot + 1;
        dec.resume(slot, prompt, at, Mode::Greedy, MAX_NEW_TOKENS + 1);
        let mut in_flight = VecDeque::new();
        for _ in 0..2 {
            in_flight.push_back(dec.submit(batch, STEPS_PER_SUBMIT, &[])?);
        }
        loop {
            let snapshot = dec.finish(in_flight.pop_front().expect("a run in flight"))?;
            let view = &snapshot.slots[slot];
            if view.mode == Mode::Idle {
                for pending in in_flight {
                    dec.finish(pending)?;
                }
                let choices = view
                    .choices
                    .iter()
                    .map(|c| (c.token, c.logprob, c.margin))
                    .collect();
                return Ok((choices, view.no_speech));
            }
            in_flight.push_back(dec.submit(batch, STEPS_PER_SUBMIT, &[])?);
        }
    }

    /// Decoding of `prompt` in slot `slot` at a temperature: each step's logits read back and
    /// the token sampled with the window's generator, as the oracle does.
    fn sampled_gpu(
        &self,
        g: &gpu::GpuWhisper,
        (slot, at): (usize, usize),
        window: &mut Window,
        prompt: &[i32],
        temperature: f32,
    ) -> Result<Choices> {
        use gpu::decoder::Mode;
        let dec = &g.decoder;
        let limit = MAX_NEW_TOKENS + 1;
        dec.resume(slot, prompt, at, Mode::Host, limit);
        let mut choices = Vec::new();
        let mut generated = Vec::new();
        loop {
            let mut snapshot = dec.finish(dec.submit(slot + 1, 1, &[slot])?)?;
            let view = &snapshot.slots[slot];
            if view.len < view.prompt_len {
                continue;
            }
            let (_, mut row) = snapshot.logits.remove(0);
            if choices.is_empty() {
                for &id in &self.hp.begin_suppress {
                    if let Some(v) = row.get_mut(id as usize) {
                        *v = f32::NEG_INFINITY;
                    }
                }
            }
            let next = self.choose(&mut row, &generated, temperature, &mut window.rng);
            choices.push(next);
            let done = choices.len() >= limit || next.0 == self.hp.eos;
            dec.set_input(slot, (!done).then_some(next.0));
            if done {
                return Ok(choices);
            }
            generated.push(next.0);
        }
    }

    /// An attempt from its choices (token, log-probability, margin) in order, as the
    /// reference's loop scores them: every choice's log-probability counts, the end token's
    /// and the one past the most tokens included.
    fn replay(
        &self,
        window: &mut Window,
        choices: &[(i32, f32, f32)],
        seek: usize,
        tier: usize,
    ) -> Attempt {
        let specials = self.specials();
        let mut generated = Vec::new();
        let mut logprobs = Vec::new();
        let mut sum_logprob = 0f64;
        let mut samples = 0;
        let mut hit_eos = false;
        for (step, &(token, logprob, margin)) in choices.iter().enumerate() {
            sum_logprob += logprob as f64;
            samples += 1;
            window.trace.event(
                "whisper.step",
                serde_json::json!({
                    "seek": seek, "tier": tier, "step": step, "token": token,
                    "logprob": logprob, "top2_margin": margin,
                }),
            );
            if step == MAX_NEW_TOKENS {
                break;
            }
            if token == specials.eos {
                hit_eos = true;
                break;
            }
            generated.push(token);
            logprobs.push(logprob);
        }
        let mut tail = generated.clone();
        if hit_eos {
            tail.push(specials.eos);
        }
        Attempt {
            compression: decoding::compression_ratio(&tail, specials.vocab),
            avg_logprob: (sum_logprob / samples.max(1) as f64) as f32,
            generated,
            logprobs,
        }
    }

    /// The next token from a row of raw logits: suppressed tokens and the timestamp rules
    /// applied, then the best one at temperature 0, else a sample. Returns it, its
    /// log-probability and the top-2 margin of the masked logits.
    fn choose(
        &self,
        row: &mut [f32],
        generated: &[i32],
        temperature: f32,
        rng: &mut Rng,
    ) -> (i32, f32, f32) {
        for &id in &self.hp.suppress {
            if let Some(v) = row.get_mut(id as usize) {
                *v = f32::NEG_INFINITY;
            }
        }
        decoding::timestamp_rules(row, generated, &self.specials(), MAX_INITIAL_TIMESTAMP);
        let margin = decoding::top2_margin(row).1;
        let (id, logprob) = if temperature <= 0. {
            decoding::argmax_logprob(row)
        } else {
            decoding::sample(row, temperature, rng)
        };
        (id as i32, logprob, margin)
    }

    /// The first pass's 30 s of mel features of a window (3000 frames × mels), for the lab.
    pub fn chunk_features(&self, pcm: &[f32]) -> Mat {
        let mel = self.features(pcm);
        let frames = mel.rows.min(self.hp.frames);
        let mut chunk = Mat::zeros(self.hp.frames, mel.cols);
        chunk.data[..frames * mel.cols].copy_from_slice(&mel.data[..frames * mel.cols]);
        chunk
    }

    /// A pass's mel features through the encoder (on the CPU or the GPU), its checkpoints to
    /// `trace`, for the lab.
    pub fn encode_checked(&self, chunk: &Mat, trace: &mut dyn Trace) -> Result<Mat> {
        match &self.backend {
            Backend::Cpu { encoder, .. } => Ok(encoder.forward(chunk, trace)),
            Backend::Gpu(g) => {
                let _exclusive = self.gpu_runs.lock().unwrap_or_else(|e| e.into_inner());
                Ok(g.encoder.encode(&[chunk], Some(trace))?.remove(0))
            }
        }
    }

    /// The prompt for a language (`None`: `<|startoftranscript|>` alone).
    pub fn prompt_for(&self, language: Option<&str>) -> Result<Vec<i32>> {
        let options = RunOptions {
            language: language.map(str::to_owned),
        };
        Ok(self.prompt(self.language_token(&options)?))
    }

    /// The raw logits after each of `tokens` (from position 0), given the encoder's states:
    /// the decoder alone, teacher-forced, for the lab.
    pub fn teacher_forced(&self, encoded: &Mat, tokens: &[i32]) -> Result<Mat> {
        match &self.backend {
            Backend::Cpu { decoder, .. } => {
                let cross = decoder.cross_cache(encoded, true);
                let mut cache = decoder.self_cache(decoder.max_positions(), true);
                Ok(decoder.forward(tokens, &mut cache, &cross, None))
            }
            Backend::Gpu(g) => {
                let _exclusive = self.gpu_runs.lock().unwrap_or_else(|e| e.into_inner());
                let dec = &g.decoder;
                dec.load_cross(encoded, 0)?;
                dec.start(0, &tokens[..1], gpu::decoder::Mode::Host, tokens.len() + 1);
                let mut out = Mat::zeros(tokens.len(), self.hp.vocab);
                for i in 0..tokens.len() {
                    let snapshot = dec.finish(dec.submit(1, 1, &[0])?)?;
                    out.row_mut(i).copy_from_slice(&snapshot.logits[0].1);
                    dec.set_input(0, tokens.get(i + 1).copied());
                }
                Ok(out)
            }
        }
    }

    /// The first attempt's choices (token, log-probability, top-2 margin) at temperature 0
    /// after `prompt`, given the encoder's states: the decoder alone, for the lab.
    pub fn greedy_choices(&self, encoded: &Mat, prompt: &[i32]) -> Result<Choices> {
        let mut trace = Steps::default();
        let mut window = Window {
            trace: &mut trace,
            rng: Rng::new(0x5EED),
            detected: None,
            text_ids: Vec::new(),
            words: Vec::new(),
            tokens: Vec::new(),
        };
        match &self.backend {
            Backend::Cpu { decoder, .. } => {
                let cross = decoder.cross_cache(encoded, true);
                self.attempt(&mut window, &cross, prompt, 0., false, None, 0, 0);
            }
            Backend::Gpu(g) => {
                let _exclusive = self.gpu_runs.lock().unwrap_or_else(|e| e.into_inner());
                g.decoder.load_cross(encoded, 0)?;
                self.attempt_gpu(g, (0, 0), &mut window, prompt, 0., None, 0, 0)?;
            }
        }
        Ok(trace.0)
    }

    pub fn max_audio_ms(&self) -> u32 {
        (self.hp.samples as u64 * 1000 / crate::SAMPLE_RATE as u64) as u32
    }
}

/// `exp(row[id]) / Σ exp(row)`, in f64 over finite logits (the reference's no-speech
/// probability).
fn probability(row: &[f32], id: usize) -> f32 {
    let max = row
        .iter()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    if !max.is_finite() {
        return 0.;
    }
    let mut sum = 0f64;
    let mut ns = 0f64;
    for (i, &l) in row.iter().enumerate() {
        if l.is_finite() {
            let e = ((l - max) as f64).exp();
            sum += e;
            if i == id {
                ns = e;
            }
        }
    }
    if sum > 0. { (ns / sum) as f32 } else { 0. }
}

/// A row of logits to log-probabilities, in f32 (ggml's `log(soft_max(x))`).
fn log_softmax(row: &mut [f32]) {
    let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let sum: f64 = row.iter().map(|&v| ((v - max) as f64).exp()).sum();
    let inv = (1. / sum) as f32;
    for v in row.iter_mut() {
        *v = ((*v - max).exp() * inv).ln();
    }
}

/// Whisper's temperatures: 0, then up by 0.2 to 1.
fn temperatures() -> Vec<f32> {
    let mut temperatures = vec![0f32];
    let mut t = TEMPERATURE_STEP;
    while t <= 1. + 1e-4 {
        temperatures.push(t);
        t += TEMPERATURE_STEP;
    }
    temperatures
}

/// Whether `trace` records or substitutes any of the encoder's checkpoints.
fn encoder_traced(trace: &dyn Trace, layers: usize) -> bool {
    let fixed = ["enc.mel.in", "enc.conv1.out", "enc.embed.out", "enc.final"];
    fixed
        .into_iter()
        .map(str::to_owned)
        .chain((0..layers).map(|l| format!("enc.block.{l}.out")))
        .any(|name| trace.wants(&name) || trace.injects(&name))
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1e3
}

/// The `whisper.step` events of a run: each choice's token, log-probability and margin.
#[derive(Default)]
struct Steps(Choices);

impl Trace for Steps {
    fn event(&mut self, kind: &str, fields: serde_json::Value) {
        if kind == "whisper.step" {
            let f = |k: &str| fields[k].as_f64().unwrap_or(f64::NAN) as f32;
            self.0.push((
                fields["token"].as_i64().unwrap_or(-1) as i32,
                f("logprob"),
                f("top2_margin"),
            ));
        }
    }
}
