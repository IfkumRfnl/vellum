//! `lab run-job <job.json>`: a lab job on Vellum's own engine. The lab starts itself as a child
//! process for this, as it starts `vellum-ref` for the reference, so both engines are measured
//! the same way and a crash in either only fails its job.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use vellum_engine::{Device, Diarizer, Options, RunOptions, Trace, Transcriber};
use vellum_harness::{
    DeviceRequest, Engine, EngineInfo, Job, Task, Token, Turn, WindowResult, dump,
};

pub fn load(job: &Job) -> Result<Box<dyn Engine>> {
    let options = Options {
        device: match job.device {
            DeviceRequest::Auto => Device::Auto,
            DeviceRequest::Cpu => Device::Cpu,
            DeviceRequest::Gpu => Device::Gpu,
        },
        ..Options::default()
    };
    let trace = LabTrace::new(job)?;
    Ok(match &job.task {
        Task::Transcribe { model, .. } => {
            let model = Transcriber::load(model, &options)?;
            Box::new(Speech { model, trace })
        }
        Task::Diarize { model } => {
            let model = Diarizer::load(model, &options)?;
            Box::new(Speakers { model, trace })
        }
    })
}

fn info(model: &vellum_engine::ModelInfo) -> EngineInfo {
    EngineInfo {
        name: format!("vellum-engine {}", env!("CARGO_PKG_VERSION")),
        build: format!(
            "vellum-engine {} ({})",
            env!("CARGO_PKG_VERSION"),
            if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            }
        ),
        device: model.device.clone(),
        architecture: model.architecture.clone(),
        variant: model.variant.clone(),
    }
}

struct Speech {
    model: Transcriber,
    trace: LabTrace,
}

impl Engine for Speech {
    fn info(&self) -> EngineInfo {
        info(self.model.info())
    }

    fn max_audio_ms(&self) -> u32 {
        self.model.max_audio_ms()
    }

    fn transcribe(&mut self, pcm: &[f32], language: Option<&str>) -> Result<WindowResult> {
        let options = RunOptions {
            language: language.map(str::to_owned),
        };
        Ok(window_result(self.model.run(
            pcm,
            &options,
            &mut self.trace,
        )?))
    }

    /// The whole recording to the engine at once, which pipelines it.
    fn transcribe_all(
        &mut self,
        windows: &[&[f32]],
        language: Option<&str>,
    ) -> Result<Vec<WindowResult>> {
        let options = RunOptions {
            language: language.map(str::to_owned),
        };
        Ok(self
            .model
            .run_all(windows, &options, &mut self.trace)?
            .into_iter()
            .map(window_result)
            .collect())
    }
}

fn window_result(result: vellum_engine::Transcript) -> WindowResult {
    WindowResult {
        text: result.text,
        language: result.language,
        words: result
            .words
            .into_iter()
            .map(|w| (w.t0_ms, w.t1_ms, w.text))
            .collect(),
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
        mel_ms: result.timings.mel_ms,
        encode_ms: result.timings.encode_ms,
        decode_ms: result.timings.decode_ms,
        wall_ms: 0.,
    }
}

struct Speakers {
    model: Diarizer,
    trace: LabTrace,
}

impl Engine for Speakers {
    fn info(&self) -> EngineInfo {
        info(self.model.info())
    }

    fn diarize(&mut self, pcm: &[f32]) -> Result<Vec<Turn>> {
        Ok(self
            .model
            .run(pcm, &mut self.trace)?
            .into_iter()
            .map(|t| Turn {
                t0_ms: t.t0_ms,
                t1_ms: t.t1_ms,
                speaker: t.speaker,
                p: (!t.p.is_nan()).then_some(t.p),
            })
            .collect())
    }
}

/// Dumps what the job asks for, and substitutes the tensors it names from another dump.
struct LabTrace {
    dump: Option<PathBuf>,
    /// Blocks whose sub-layers are dumped.
    sub_blocks: Vec<u32>,
    inject: Option<(PathBuf, Vec<String>)>,
}

impl LabTrace {
    fn new(job: &Job) -> Result<LabTrace> {
        if let Some(dump) = &job.dump {
            std::fs::create_dir_all(&dump.dir)?;
        }
        Ok(LabTrace {
            dump: job.dump.as_ref().map(|d| d.dir.clone()),
            sub_blocks: job
                .dump
                .as_ref()
                .map(|d| d.sub_blocks.clone())
                .unwrap_or_default(),
            inject: job
                .inject
                .as_ref()
                .map(|i| (i.dir.clone(), i.names.clone())),
        })
    }
}

impl Trace for LabTrace {
    fn wants(&self, name: &str) -> bool {
        if self.dump.is_none() {
            return false;
        }
        // A block's sub-layers only when the job asks for that block, as the reference does.
        match name
            .strip_prefix("enc.block.")
            .and_then(|r| r.split_once('.'))
        {
            Some((block, part)) if part != "out" => block
                .parse::<u32>()
                .is_ok_and(|b| self.sub_blocks.contains(&b)),
            _ => true,
        }
    }

    fn tensor(&mut self, name: &str, shape: &[usize], data: &[f32]) {
        if let Some(dir) = &self.dump
            && let Err(error) = dump::write_tensor(dir, name, shape, data, "vellum")
        {
            eprintln!("dumping {name}: {error:#}");
        }
    }

    fn event(&mut self, kind: &str, fields: serde_json::Value) {
        if let Some(dir) = &self.dump
            && let Err(error) = dump::write_event(dir, kind, fields)
        {
            eprintln!("recording {kind}: {error:#}");
        }
    }

    fn injects(&self, name: &str) -> bool {
        self.inject
            .as_ref()
            .is_some_and(|(_, names)| names.iter().any(|n| n == name))
    }

    fn inject(&mut self, name: &str, shape: &[usize]) -> Option<(Vec<usize>, Vec<f32>)> {
        let (dir, names) = self.inject.as_ref()?;
        if !names.iter().any(|n| n == name) {
            return None;
        }
        match injected(dir, name, shape) {
            Ok(data) => Some(data),
            Err(error) => {
                eprintln!("injecting {name}: {error:#}");
                None
            }
        }
    }
}

fn injected(dir: &Path, name: &str, shape: &[usize]) -> Result<(Vec<usize>, Vec<f32>)> {
    let found = dump::list(dir)?
        .into_iter()
        .find(|d| d.sidecar.name == name)
        .with_context(|| format!("no {name} in {}", dir.display()))?;
    if found.sidecar.shape != shape {
        eprintln!(
            "injecting {name}: the reference's is {:?}, ours was {shape:?}; using the reference's",
            found.sidecar.shape
        );
    }
    Ok((found.sidecar.shape.clone(), found.load()?))
}
