//! `vellum-ref run-job <job.json>`: runs a lab job on transcribe.cpp, as the app calls it.
//! `vellum-ref info`: the build and the devices it sees, as JSON.

use anyhow::{Context, Result};
use std::{env, ffi::OsString, fs, path::PathBuf, process::ExitCode, time::Instant};
use transcribe_cpp::{
    Backend, DiarizeExtension, DiarizeOptions, DiarizeSession, Model, ModelOptions, RunOptions,
    Session, SortformerDiarizeOptions, SortformerPreset, TimestampKind,
};
use vellum_harness::{DeviceRequest, Engine, EngineInfo, Job, Task, Token, Turn, WindowResult};

fn main() -> ExitCode {
    let started = Instant::now();
    let args: Vec<OsString> = env::args_os().skip(1).collect();
    match (args.first().and_then(|a| a.to_str()), args.get(1)) {
        (Some("run-job"), Some(path)) => {
            vellum_harness::run_job(&PathBuf::from(path), started, load)
        }
        (Some("info"), _) => match info() {
            Ok(text) => {
                println!("{text}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("{error:#}");
                ExitCode::FAILURE
            }
        },
        _ => {
            eprintln!("usage: vellum-ref run-job <job.json> | vellum-ref info");
            ExitCode::FAILURE
        }
    }
}

/// How this binary was built, for the report and the golden cache key.
fn build() -> String {
    format!(
        "transcribe.cpp {} ({}) cmake[{}]",
        transcribe_cpp::version(),
        transcribe_cpp::version_commit(),
        option_env!("TRANSCRIBE_CMAKE_ARGS").unwrap_or("")
    )
}

fn info() -> Result<String> {
    transcribe_cpp::init_backends_default()?;
    let devices: Vec<_> = transcribe_cpp::devices()
        .into_iter()
        .map(|d| {
            serde_json::json!({
                "name": d.name,
                "description": d.description,
                "kind": d.kind,
                "id": d.device_id,
                "memory_total": d.memory_total,
            })
        })
        .collect();
    Ok(serde_json::to_string_pretty(&serde_json::json!({
        "build": build(),
        "devices": devices,
    }))?)
}

fn backend(device: DeviceRequest) -> Backend {
    match device {
        DeviceRequest::Auto => Backend::Auto,
        DeviceRequest::Cpu => Backend::Cpu,
        DeviceRequest::Gpu if cfg!(target_os = "macos") => Backend::Metal,
        DeviceRequest::Gpu => Backend::Vulkan,
    }
}

fn load(job: &Job) -> Result<Box<dyn Engine>> {
    if let Some(dump) = &job.dump {
        fs::create_dir_all(&dump.dir)?;
    }
    // transcribe.cpp reads its settings from the environment it started with (see
    // `Job::environment`); the lab sets them on this process. Check they arrived.
    for (key, value) in job.environment() {
        anyhow::ensure!(
            env::var(&key).is_ok_and(|v| v == value),
            "{key} isn't set to {value:?}: start vellum-ref with the job's environment"
        );
    }
    transcribe_cpp::init_backends_default().context("initializing devices")?;
    let options = ModelOptions {
        backend: backend(job.device),
        ..Default::default()
    };
    Ok(match &job.task {
        Task::Transcribe { model, .. } => {
            let model = Model::load_with(model, &options)?;
            let session = model.session()?;
            let max_audio_ms = session.limits()?.effective_max_audio_ms.max(0) as u32;
            let info = engine_info(&model)?;
            Box::new(Speech {
                _model: model,
                session,
                max_audio_ms,
                info,
            })
        }
        Task::Diarize { model } => {
            let model = Model::load_with(model, &options)?;
            let session = model.diarize_session()?;
            let info = engine_info(&model)?;
            Box::new(Speakers {
                _model: model,
                session,
                info,
            })
        }
    })
}

fn engine_info(model: &Model) -> Result<EngineInfo> {
    let device = model.device()?;
    Ok(EngineInfo {
        name: format!("transcribe.cpp {}", transcribe_cpp::version()),
        build: build(),
        device: if device.kind == "cpu" {
            "CPU".into()
        } else {
            format!("{} · {}", device.kind, device.description)
        },
        architecture: model.arch(),
        variant: model.variant(),
    })
}

struct Speech {
    _model: Model,
    session: Session,
    max_audio_ms: u32,
    info: EngineInfo,
}

impl Engine for Speech {
    fn info(&self) -> EngineInfo {
        self.info.clone()
    }

    fn max_audio_ms(&self) -> u32 {
        self.max_audio_ms
    }

    fn transcribe(&mut self, pcm: &[f32], language: Option<&str>) -> Result<WindowResult> {
        let options = RunOptions {
            language: language.map(str::to_owned),
            ..Default::default()
        };
        let result = self.session.run(pcm, &options)?;
        // As the app reads a result: words when the model times words, else segments.
        let words = if matches!(
            result.timestamp_kind,
            TimestampKind::Word | TimestampKind::Token
        ) && !result.words.is_empty()
        {
            result
                .words
                .into_iter()
                .map(|w| (w.t0_ms, w.t1_ms, w.text))
                .collect()
        } else {
            result
                .segments
                .into_iter()
                .map(|s| (s.t0_ms, s.t1_ms, s.text))
                .collect()
        };
        Ok(WindowResult {
            text: result.text,
            language: result.language,
            words,
            tokens: result
                .tokens
                .into_iter()
                .map(|t| Token {
                    id: t.id,
                    p: (!t.p.is_nan()).then_some(t.p),
                    t0_ms: t.t0_ms,
                    t1_ms: t.t1_ms,
                    text: t.text,
                    window: 0,
                })
                .collect(),
            mel_ms: result.timings.mel_ms as f64,
            encode_ms: result.timings.encode_ms as f64,
            decode_ms: result.timings.decode_ms as f64,
            wall_ms: 0.,
        })
    }
}

struct Speakers {
    _model: Model,
    session: DiarizeSession,
    info: EngineInfo,
}

impl Engine for Speakers {
    fn info(&self) -> EngineInfo {
        self.info.clone()
    }

    fn diarize(&mut self, pcm: &[f32]) -> Result<Vec<Turn>> {
        // The app's operating point: half-minute windows, the most accurate.
        let options = DiarizeOptions {
            family: Some(DiarizeExtension::Sortformer(SortformerDiarizeOptions {
                preset: Some(SortformerPreset::VeryHighLatency),
            })),
        };
        Ok(self
            .session
            .run(pcm, &options)?
            .into_iter()
            .map(|s| Turn {
                t0_ms: s.t0_ms,
                t1_ms: s.t1_ms,
                speaker: s.speaker_id,
                p: (!s.p.is_nan()).then_some(s.p),
            })
            .collect())
    }

    fn profile(&mut self) -> Option<serde_json::Value> {
        let t = self.session.timings();
        Some(serde_json::json!({
            "load_ms": t.load_ms,
            "mel_ms": t.mel_ms,
            "encode_ms": t.encode_ms,
            "decode_ms": t.decode_ms,
        }))
    }
}
