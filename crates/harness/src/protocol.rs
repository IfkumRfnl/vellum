use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Bumped whenever a field's meaning changes, so cached results from an older lab are redone.
pub const SCHEMA: u32 = 1;

/// One model loaded once, run over some clips.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub schema: u32,
    pub task: Task,
    pub device: DeviceRequest,
    pub clips: Vec<ClipJob>,
    /// Unmeasured runs per clip before the measured ones, to compile kernels and warm caches.
    pub warmup: u32,
    /// Measured runs per clip; at least one. The first run's output is the one reported.
    pub repeats: u32,
    /// Tensor and decoder-step dumps. Changes how the reference executes, so a dump job
    /// is never timed, and it has exactly one clip, run once.
    pub dump: Option<DumpRequest>,
    /// Tensors to substitute at checkpoints (see `vellum_engine::Trace::inject`): a dump
    /// directory, usually the reference's for the same clip. Our engine only.
    #[serde(default)]
    pub inject: Option<InjectRequest>,
    /// Engine settings as environment variables, set before the engine starts: the
    /// reference's `TRANSCRIBE_NO_FLASH`, `TRANSCRIBE_CONV_DIRECT_DW`, thread counts…
    #[serde(default)]
    pub env: Vec<(String, String)>,
    /// Where the runner writes `report.json`.
    pub out_dir: PathBuf,
}

impl Job {
    /// The environment the runner process starts with: the job's settings and, for a dump job,
    /// transcribe.cpp's dump switches. They're given to the process when it starts because
    /// the C++ runtime may not see variables set later (on Windows it keeps its own copy).
    pub fn environment(&self) -> Vec<(String, String)> {
        let mut env = self.env.clone();
        if let Some(dump) = &self.dump {
            env.push((
                "TRANSCRIBE_DUMP_DIR".into(),
                dump.dir.to_string_lossy().into_owned(),
            ));
            if dump.all_blocks {
                env.push(("TRANSCRIBE_DUMP_ALL_BLOCKS".into(), "1".into()));
            }
            if !dump.sub_blocks.is_empty() {
                let blocks: Vec<String> = dump.sub_blocks.iter().map(u32::to_string).collect();
                env.push(("TRANSCRIBE_DUMP_SUB_BLOCKS".into(), blocks.join(",")));
            }
        }
        env
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Task {
    /// Speech to words, window by window as the app reads a recording.
    Transcribe {
        model: PathBuf,
        /// The app's id for the model (`parakeet_unified_en`, `small`, `turbo`).
        model_id: String,
        /// The language to transcribe, or `None` to detect it and keep the first one heard.
        language: Option<String>,
        windowing: Windowing,
    },
    /// Who spoke when, over a whole recording in one call.
    Diarize { model: PathBuf },
}

/// How a recording is cut into windows the model takes one at a time.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Windowing {
    /// The longest window, before the model's own limit.
    pub max_ms: u32,
}

impl Default for Windowing {
    fn default() -> Self {
        // Vellum's: a little under Whisper's 30 s.
        Windowing { max_ms: 28_000 }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum DeviceRequest {
    /// The best GPU, else the CPU, as the app asks.
    #[default]
    Auto,
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipJob {
    pub id: String,
    /// 16 kHz mono WAV or FLAC.
    pub audio: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DumpRequest {
    pub dir: PathBuf,
    /// Every encoder block's output, not just the stage boundaries.
    pub all_blocks: bool,
    /// Block indices to dump sub-layer activations of (attention, conv, feed-forward…).
    pub sub_blocks: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectRequest {
    pub dir: PathBuf,
    /// The checkpoints to substitute, by tensor name; the others run as computed.
    pub names: Vec<String>,
}

/// What a runner writes to `<out_dir>/report.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JobReport {
    pub schema: u32,
    pub engine: EngineInfo,
    /// Milliseconds from process start until the model was ready.
    pub ready_ms: f64,
    /// Milliseconds the model load itself took.
    pub load_ms: f64,
    /// The process's peak resident memory, in bytes (0 when unknown).
    pub peak_rss: u64,
    /// The whole job's wall time, in milliseconds.
    pub wall_ms: f64,
    pub clips: Vec<ClipReport>,
    /// Why the job failed as a whole (an unsupported model, a device that wouldn't open…).
    pub error: Option<String>,
    /// Anything else the engine measured: kernel timings, dispatch counts, shapes…
    pub profile: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EngineInfo {
    /// `transcribe.cpp 0.4.2` or `vellum-engine 0.0.0`.
    pub name: String,
    /// What the binary was built with: features, CPU flags, validation hooks.
    pub build: String,
    /// The device the model actually runs on.
    pub device: String,
    pub architecture: String,
    pub variant: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClipReport {
    pub id: String,
    pub seconds: f64,
    /// SHA-256 of the decoded samples, so results follow the audio, not the file.
    pub pcm_sha256: String,
    pub warmup_ms: Vec<f64>,
    pub runs: Vec<RunTiming>,
    /// The first measured run's output.
    pub output: Option<Output>,
    /// Whether every run produced the same output.
    pub deterministic: bool,
    pub error: Option<String>,
}

/// One measured run of a clip.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunTiming {
    /// From samples in memory to results in memory, all windows.
    pub wall_ms: f64,
    /// Per window, in order.
    pub windows_ms: Vec<f64>,
    /// The engine's own stage times, summed over windows (0 when it doesn't say).
    pub mel_ms: f64,
    pub encode_ms: f64,
    pub decode_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Output {
    Transcript(Transcript),
    Diarization(Diarization),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Transcript {
    pub text: String,
    /// The language used for the recording: given, or the first one detected.
    pub language: Option<String>,
    /// Words, timed in milliseconds from the start of the recording.
    pub words: Vec<Word>,
    pub tokens: Vec<Token>,
    pub windows: Vec<WindowInfo>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Word {
    pub t0_ms: i64,
    pub t1_ms: i64,
    pub text: String,
    /// The window it came from.
    pub window: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Token {
    pub id: i32,
    /// Confidence; `None` when the engine gives none.
    pub p: Option<f32>,
    pub t0_ms: i64,
    pub t1_ms: i64,
    pub text: String,
    pub window: u32,
}

/// One window as it was fed to the model, so any window can be replayed alone.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct WindowInfo {
    pub start_sample: u64,
    pub end_sample: u64,
    /// Skipped because every sample was zero, as the app does.
    pub silent: bool,
    /// The language the window was run with.
    pub language: Option<String>,
    /// The language the model detected in it.
    pub detected: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Diarization {
    pub turns: Vec<Turn>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct Turn {
    pub t0_ms: i64,
    pub t1_ms: i64,
    pub speaker: i32,
    pub p: Option<f32>,
}
