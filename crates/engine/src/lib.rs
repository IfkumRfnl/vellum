//! Vellum's speech engine in Rust. It reads the same GGUF files as transcribe.cpp and is
//! measured against it by the engine lab (`cargo lab`, see `crates/lab`).
//!
//! The public surface is deliberately small: a [`Transcriber`] turns one window of 16 kHz
//! mono audio into words, a [`Diarizer`] turns a whole recording into speaker turns. Both take
//! a [`Trace`] so the lab can compare every intermediate tensor and decoder decision with the
//! reference engine's.

pub mod cpu;
pub mod gguf;
pub mod gpu;
pub mod mel;
pub mod parakeet;
pub mod tokenizer;
pub mod trace;

use serde::{Deserialize, Serialize};
use std::path::Path;
pub use trace::{NoTrace, Trace};

/// The sample rate every model takes.
pub const SAMPLE_RATE: u32 = 16_000;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("invalid GGUF file: {0}")]
    Gguf(String),
    #[error("{0} models aren't supported yet")]
    Unsupported(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Where a model runs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Device {
    /// The best GPU, else the CPU.
    #[default]
    Auto,
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, Default)]
pub struct Options {
    pub device: Device,
}

#[derive(Debug, Clone, Default)]
pub struct RunOptions {
    /// The language to transcribe (ISO code), or `None` to detect it.
    pub language: Option<String>,
}

/// One recognized word. Times are milliseconds from the start of the window.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Word {
    pub t0_ms: i64,
    pub t1_ms: i64,
    pub text: String,
}

/// One decoded token, with the model's confidence in it (NaN when it gives none).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Token {
    pub id: i32,
    pub p: f32,
    pub t0_ms: i64,
    pub t1_ms: i64,
    pub text: String,
}

/// Milliseconds spent in each stage of a run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Timings {
    pub mel_ms: f64,
    pub encode_ms: f64,
    pub decode_ms: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Transcript {
    pub text: String,
    /// The language the model heard, if it detects one.
    pub language: Option<String>,
    pub words: Vec<Word>,
    pub tokens: Vec<Token>,
    pub timings: Timings,
}

/// Who spoke when. Times are milliseconds from the start of the recording.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct SpeakerTurn {
    pub t0_ms: i64,
    pub t1_ms: i64,
    pub speaker: i32,
    pub p: f32,
}

/// The facts of a loaded model the lab reports beside its results.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelInfo {
    pub architecture: String,
    pub variant: String,
    /// "CPU", or the GPU's name and API.
    pub device: String,
}

fn architecture(path: &Path) -> Result<(gguf::Gguf, String)> {
    let file = gguf::Gguf::open(path)?;
    let arch = file
        .metadata_str("general.architecture")
        .ok_or_else(|| Error::Gguf("no general.architecture".into()))?
        .to_owned();
    Ok((file, arch))
}

/// The GPU a model runs on, by the request: none for `Cpu`; the best adapter for `Gpu`
/// (an error if there's none); for `Auto`, the best adapter only if it's a discrete card or
/// Apple's (an integrated GPU elsewhere is often slower than the CPU — this laptop's is), unless
/// `VELLUM_GPU` names one.
fn gpu(device: Device) -> Result<Option<std::sync::Arc<gpu::Gpu>>> {
    let options = gpu::GpuOptions::from_env();
    match device {
        Device::Cpu => Ok(None),
        Device::Gpu => Ok(Some(std::sync::Arc::new(gpu::Gpu::new(&options)?))),
        Device::Auto => {
            let Ok(gpu) = gpu::Gpu::new(&options) else {
                return Ok(None);
            };
            let fast = gpu.info.device_type == gpu::wgpu::DeviceType::DiscreteGpu
                || gpu.info.backend == gpu::wgpu::Backend::Metal
                || options.adapter.is_some();
            Ok(fast.then(|| std::sync::Arc::new(gpu)))
        }
    }
}

/// A speech model, loaded.
pub struct Transcriber {
    model: SpeechModel,
}

enum SpeechModel {
    Parakeet(parakeet::Parakeet),
}

impl Transcriber {
    pub fn load(path: &Path, options: &Options) -> Result<Self> {
        let (file, arch) = architecture(path)?;
        let model = match arch.as_str() {
            "parakeet" => {
                SpeechModel::Parakeet(parakeet::Parakeet::load(&file, gpu(options.device)?)?)
            }
            _ => return Err(Error::Unsupported(arch)),
        };
        Ok(Transcriber { model })
    }

    pub fn info(&self) -> &ModelInfo {
        match &self.model {
            SpeechModel::Parakeet(m) => &m.info,
        }
    }

    /// The longest window of audio [`run`](Self::run) takes, in milliseconds.
    pub fn max_audio_ms(&self) -> u32 {
        30_000
    }

    pub fn run(
        &mut self,
        pcm: &[f32],
        options: &RunOptions,
        trace: &mut dyn Trace,
    ) -> Result<Transcript> {
        match &self.model {
            SpeechModel::Parakeet(m) => m.run(pcm, options, trace),
        }
    }

    /// Every window of a recording at once, in a language known in advance, results in window
    /// order: each exactly what [`run`](Self::run) gives that window, sooner, as the model can
    /// work on several windows at once (making features for the next while it encodes this one
    /// and decodes the last). Stage times are per window; a stage that takes a batch splits its
    /// time evenly.
    pub fn run_all(
        &mut self,
        windows: &[&[f32]],
        options: &RunOptions,
        trace: &mut dyn Trace,
    ) -> Result<Vec<Transcript>> {
        match &self.model {
            SpeechModel::Parakeet(m) => m.run_all(windows, options, trace),
        }
    }
}

/// A speaker model, loaded.
pub struct Diarizer {
    info: ModelInfo,
}

impl Diarizer {
    pub fn load(path: &Path, _options: &Options) -> Result<Self> {
        let (_file, arch) = architecture(path)?;
        Err(Error::Unsupported(arch))
    }

    pub fn info(&self) -> &ModelInfo {
        &self.info
    }

    pub fn run(&mut self, _pcm: &[f32], _trace: &mut dyn Trace) -> Result<Vec<SpeakerTurn>> {
        Err(Error::Unsupported(self.info.architecture.clone()))
    }
}
