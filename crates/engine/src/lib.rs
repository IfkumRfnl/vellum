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
pub mod sortformer;
pub mod tokenizer;
pub mod trace;
pub mod whisper;

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
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
    /// The caller's stop check asked a run to stop.
    #[error("cancelled")]
    Cancelled,
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
    /// Where a GPU keeps its compiled kernels between runs (Vulkan's pipeline cache, one file
    /// per adapter and driver), so loading a model again can skip the driver's compiles
    /// (NVIDIA's driver keeps a cache of its own, so there it saves little). `None`: compile
    /// them every time.
    pub cache_dir: Option<PathBuf>,
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
/// `VELLUM_GPU` names one. Its kernels are cached in the options' `cache_dir`, if any.
fn gpu(options: &Options) -> Result<Option<std::sync::Arc<gpu::Gpu>>> {
    let device = options.device;
    let options = gpu::GpuOptions {
        cache_dir: options.cache_dir.clone(),
        ..gpu::GpuOptions::from_env()
    };
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

/// Asks Windows not to slow this process down while it isn't in the foreground. Windows 11
/// otherwise moves a background process's threads to efficiency cores at low clocks ("EcoQoS"
/// power throttling) whenever it sees fit: measured in the lab, an 18-minute recording then
/// took 2.5 s instead of 1.2 s, its decoding seven times as long. A transcription the user
/// asked for is foreground work even when its window is not.
#[cfg(windows)]
fn keep_full_speed() {
    use std::ffi::c_void;
    /// `PROCESS_POWER_THROTTLING_STATE`.
    #[repr(C)]
    struct PowerThrottling {
        version: u32,
        control_mask: u32,
        state_mask: u32,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn SetProcessInformation(
            process: *mut c_void,
            class: i32,
            information: *const c_void,
            size: u32,
        ) -> i32;
    }
    const PROCESS_POWER_THROTTLING: i32 = 4;
    const EXECUTION_SPEED: u32 = 1;
    // Control the execution-speed throttling, and turn it off.
    let state = PowerThrottling {
        version: 1,
        control_mask: EXECUTION_SPEED,
        state_mask: 0,
    };
    // SAFETY: the pseudo-handle of this process and a valid state of the size given. A
    // failure (an older Windows) changes nothing, so its result isn't needed.
    unsafe {
        SetProcessInformation(
            GetCurrentProcess(),
            PROCESS_POWER_THROTTLING,
            (&raw const state).cast(),
            size_of::<PowerThrottling>() as u32,
        );
    }
}

#[cfg(not(windows))]
fn keep_full_speed() {}

/// Writes the kernels a model just compiled at load to the pipeline cache, when there is one
/// (`Options::cache_dir`). A cache that can't be written only costs the next load its
/// compiles, so a failure is ignored.
fn keep_kernels(gpu: Option<&gpu::Gpu>) {
    if let Some(gpu) = gpu {
        let _ = gpu.save_pipeline_cache();
    }
}

/// A speech model, loaded.
pub struct Transcriber {
    model: SpeechModel,
}

// One per transcriber: its size doesn't matter.
#[allow(clippy::large_enum_variant)]
enum SpeechModel {
    Parakeet(parakeet::Parakeet),
    Whisper(whisper::Whisper),
}

impl Transcriber {
    pub fn load(path: &Path, options: &Options) -> Result<Self> {
        keep_full_speed();
        let (file, arch) = architecture(path)?;
        let model = match arch.as_str() {
            "parakeet" => SpeechModel::Parakeet(parakeet::Parakeet::load(&file, gpu(options)?)?),
            "whisper" => SpeechModel::Whisper(whisper::Whisper::load_on(&file, gpu(options)?)?),
            _ => return Err(Error::Unsupported(arch)),
        };
        let transcriber = Transcriber { model };
        keep_kernels(transcriber.gpu());
        Ok(transcriber)
    }

    pub fn info(&self) -> &ModelInfo {
        match &self.model {
            SpeechModel::Parakeet(m) => &m.info,
            SpeechModel::Whisper(m) => &m.info,
        }
    }

    /// The longest window of audio [`run`](Self::run) takes, in milliseconds.
    pub fn max_audio_ms(&self) -> u32 {
        match &self.model {
            SpeechModel::Parakeet(_) => 30_000,
            SpeechModel::Whisper(m) => m.max_audio_ms(),
        }
    }

    pub fn run(
        &mut self,
        pcm: &[f32],
        options: &RunOptions,
        trace: &mut dyn Trace,
    ) -> Result<Transcript> {
        match &self.model {
            SpeechModel::Parakeet(m) => m.run(pcm, options, trace),
            SpeechModel::Whisper(m) => m.run(pcm, options, trace),
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
        let mut out = Vec::with_capacity(windows.len());
        self.run_each(windows, options, trace, &|| false, &mut |_, t| {
            out.push(t);
            Ok(())
        })?;
        Ok(out)
    }

    /// [`run_all`](Self::run_all) for an app: each window's transcript goes to `each` (with
    /// its index) as soon as it and every window before it are done, on the calling thread,
    /// so a long recording shows progress as it goes. `stop` is polled between windows and
    /// batches; once it says so, the run gives up within about a window's work and returns
    /// [`Error::Cancelled`]. An error from `each` stops the run the same way and is returned.
    pub fn run_each(
        &mut self,
        windows: &[&[f32]],
        options: &RunOptions,
        trace: &mut dyn Trace,
        stop: &(dyn Fn() -> bool + Sync),
        each: &mut dyn FnMut(usize, Transcript) -> Result<()>,
    ) -> Result<()> {
        match &self.model {
            SpeechModel::Parakeet(m) => m.run_each(windows, options, trace, stop, each),
            SpeechModel::Whisper(m) => m.run_each(windows, options, trace, stop, each),
        }
    }

    /// The GPU the model runs on, if it runs on one.
    pub fn gpu(&self) -> Option<&gpu::Gpu> {
        match &self.model {
            SpeechModel::Parakeet(m) => m.gpu(),
            SpeechModel::Whisper(m) => m.gpu(),
        }
    }
}

/// A speaker model, loaded.
pub struct Diarizer {
    model: SpeakerModel,
}

enum SpeakerModel {
    Sortformer(sortformer::Sortformer),
}

impl Diarizer {
    pub fn load(path: &Path, options: &Options) -> Result<Self> {
        keep_full_speed();
        let (file, arch) = architecture(path)?;
        let model = match arch.as_str() {
            "sortformer" => {
                SpeakerModel::Sortformer(sortformer::Sortformer::load(&file, gpu(options)?)?)
            }
            _ => return Err(Error::Unsupported(arch)),
        };
        let diarizer = Diarizer { model };
        keep_kernels(diarizer.gpu());
        Ok(diarizer)
    }

    pub fn info(&self) -> &ModelInfo {
        match &self.model {
            SpeakerModel::Sortformer(m) => &m.info,
        }
    }

    /// A whole recording of 16 kHz mono audio to speaker turns.
    pub fn run(&mut self, pcm: &[f32], trace: &mut dyn Trace) -> Result<Vec<SpeakerTurn>> {
        self.run_until(pcm, trace, &|| false)
    }

    /// [`run`](Self::run) for an app: `stop` is polled between the recording's chunks (half
    /// a minute of audio each; on the CPU also inside them, between the encoder's blocks).
    /// Once it says so, the run gives up within a fraction of a second and returns
    /// [`Error::Cancelled`].
    pub fn run_until(
        &mut self,
        pcm: &[f32],
        trace: &mut dyn Trace,
        stop: &(dyn Fn() -> bool + Sync),
    ) -> Result<Vec<SpeakerTurn>> {
        match &self.model {
            SpeakerModel::Sortformer(m) => m.run(pcm, trace, stop),
        }
    }

    /// The GPU the model runs on, if it runs on one.
    pub fn gpu(&self) -> Option<&gpu::Gpu> {
        match &self.model {
            SpeakerModel::Sortformer(m) => m.gpu(),
        }
    }
}
