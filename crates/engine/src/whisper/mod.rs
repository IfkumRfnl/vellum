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
mod mel;
mod ops;
mod tokenizer;

use crate::{
    Error, ModelInfo, Result, RunOptions, Timings, Token, Trace, Transcript, Word, cpu::Mat,
    gguf::Gguf, gguf::Value, trace::checkpoint,
};
use decoding::{Rng, Specials, TEXT_END};
use std::time::Instant;
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
    encoder: encoder::Encoder,
    decoder: decoder::Decoder,
    tokenizer: Tokenizer,
    /// Each language the model knows (ISO code) and its token.
    languages: Vec<(String, i32)>,
    pub info: ModelInfo,
}

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

/// One temperature's attempt at a pass.
struct Attempt {
    generated: Vec<i32>,
    /// Each generated token's log-probability.
    logprobs: Vec<f32>,
    avg_logprob: f32,
    compression: f32,
}

impl Whisper {
    pub fn load(file: &Gguf) -> Result<Whisper> {
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
        let (encoder, decoder) = rayon::join(
            || encoder::Encoder::load(file, &hp),
            || decoder::Decoder::load(file, &hp),
        );
        Ok(Whisper {
            mel: mel::LogMel::from_gguf(file)?,
            encoder: encoder?,
            decoder: decoder?,
            tokenizer,
            languages,
            info: ModelInfo {
                architecture: "whisper".into(),
                variant: file.metadata_str("stt.variant").unwrap_or_default().into(),
                device: "CPU".into(),
            },
            hp,
        })
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
        let mut timings = Timings::default();
        let start = Instant::now();
        // A window that fits one pass is padded to 30 s before the mel, as transformers does.
        let mel = if pcm.len() <= self.hp.samples {
            let mut padded = pcm.to_vec();
            padded.resize(self.hp.samples, 0.);
            self.mel.compute(&padded)
        } else {
            self.mel.compute(pcm)
        };
        timings.mel_ms = start.elapsed().as_secs_f64() * 1e3;

        let mut language = match &options.language {
            Some(code) if self.hp.multilingual => Some(
                self.languages
                    .iter()
                    .find(|(c, _)| c == code)
                    .map(|(_, id)| *id)
                    .ok_or_else(|| Error::Other(format!("Whisper doesn't know \"{code}\"")))?,
            ),
            Some(code) if code != "en" => {
                return Err(Error::Other(format!(
                    "an English-only model, not \"{code}\""
                )));
            }
            _ => None,
        };

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
            let encoded = if first {
                checkpoint(window.trace, "enc.mel.in", &mut chunk);
                if window.trace.injects("enc.final") {
                    let mut e = Mat::zeros(0, self.hp.d_model);
                    checkpoint(window.trace, "enc.final", &mut e);
                    e
                } else {
                    self.encoder.forward(&chunk, window.trace)
                }
            } else {
                self.encoder.forward(&chunk, &mut crate::NoTrace)
            };
            encode_ms += start.elapsed().as_secs_f64() * 1e3;

            if first && self.hp.multilingual && language.is_none() {
                let (id, code) = self.detect_language(&encoded, window.trace);
                language = Some(id);
                window.detected = Some(code);
            }
            let advance = self.pass(&mut window, &encoded, language, seek, frames, first);
            seek += advance;
        }
        timings.encode_ms = encode_ms;
        timings.decode_ms = decode_start.elapsed().as_secs_f64() * 1e3 - encode_ms;

        let text = self.tokenizer.decode_bytes(window.text_ids.iter().copied());
        Ok(Transcript {
            text: String::from_utf8_lossy(tokenizer::trim(&text)).into_owned(),
            language: window.detected,
            words: window.words,
            tokens: window.tokens,
            timings,
        })
    }

    /// The language whose token the decoder rates highest after `<|startoftranscript|>`
    /// alone (keys and values in f32, as the reference's one-off prefill graph has them).
    fn detect_language(&self, encoded: &Mat, trace: &mut dyn Trace) -> (i32, String) {
        let cross = self.decoder.cross_cache(encoded, false);
        let mut cache = self.decoder.self_cache(1, false);
        let logits = self
            .decoder
            .forward(&[self.hp.sot], &mut cache, &cross, None);
        let row = logits.row(0);
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

    /// One 30 s pass from mel frame `seek` (`frames` of them real): decoding at rising
    /// temperatures until one passes, its segments into the window. Returns how many frames
    /// the window advances.
    fn pass(
        &self,
        window: &mut Window,
        encoded: &Mat,
        language: Option<i32>,
        seek: usize,
        frames: usize,
        first: bool,
    ) -> usize {
        let specials = self.specials();
        let mut prompt = vec![self.hp.sot];
        if let Some(language) = language {
            prompt.push(language);
            prompt.push(self.hp.transcribe);
        }
        let cross = self.decoder.cross_cache(encoded, true);

        let mut temperatures = vec![0f32];
        let mut t = TEMPERATURE_STEP;
        while t <= 1. + 1e-4 {
            temperatures.push(t);
            t += TEMPERATURE_STEP;
        }
        let mut no_speech = 0f32;
        let mut accepted = None;
        let mut skip = false;
        for (tier, &temperature) in temperatures.iter().enumerate() {
            let traced = first && tier == 0;
            let attempt = self.attempt(
                window,
                &cross,
                &prompt,
                temperature,
                traced,
                (tier == 0).then_some(&mut no_speech),
                seek,
                tier,
            );
            let passes = attempt.compression <= COMPRESSION_RATIO_THRESHOLD
                && attempt.avg_logprob >= LOGPROB_THRESHOLD;
            skip = no_speech > NO_SPEECH_THRESHOLD && attempt.avg_logprob < LOGPROB_THRESHOLD;
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
            accepted = Some(attempt);
            if skip || passes {
                break;
            }
        }
        if skip {
            return frames;
        }
        let attempt = accepted.expect("at least one temperature");
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
        let specials = self.specials();
        let context = self.decoder.max_positions();
        let mut cache = self.decoder.self_cache(context, true);
        let mut taps = decoder::Taps::default();
        let mut logits =
            self.decoder
                .forward(prompt, &mut cache, cross, traced.then_some(&mut taps));
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
        let mut generated = Vec::new();
        let mut logprobs = Vec::new();
        let mut sum_logprob = 0f64;
        let mut samples = 0;
        let mut next = self.choose(&mut row, &generated, temperature, &mut window.rng);
        let mut hit_eos = false;
        let mut step = 0;
        loop {
            sum_logprob += next.1 as f64;
            samples += 1;
            window.trace.event(
                "whisper.step",
                serde_json::json!({
                    "seek": seek, "tier": tier, "step": step, "token": next.0,
                    "logprob": next.1, "top2_margin": next.2,
                }),
            );
            if step == MAX_NEW_TOKENS {
                break;
            }
            if next.0 == specials.eos {
                hit_eos = true;
                break;
            }
            generated.push(next.0);
            logprobs.push(next.1);
            if cache.len + 1 > context {
                break;
            }
            let mut logits = self.decoder.forward(&[next.0], &mut cache, cross, None);
            if traced && step == 19 {
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
            step += 1;
        }
        let mut tail = generated.clone();
        if hit_eos {
            tail.push(specials.eos);
        }
        Attempt {
            compression: decoding::compression_ratio(&tail, specials.vocab),
            avg_logprob: (sum_logprob / samples as f64) as f32,
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
