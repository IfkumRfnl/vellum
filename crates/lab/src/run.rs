//! Engines as the lab names them (`ref`, `ref-cpu`, `vellum-gpu`, …) and their jobs, each run in
//! a child process of its own.

use crate::{reference, system};
use anyhow::{Context, Result, bail};
use std::{
    fmt, fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    str::FromStr,
    time::Duration,
};
use vellum_harness::{DeviceRequest, Job, JobReport};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Engine {
    /// transcribe.cpp.
    Reference,
    /// Vellum's own engine.
    Vellum,
}

/// An engine on a device: what one side of a comparison runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Candidate {
    pub engine: Engine,
    pub device: DeviceRequest,
}

impl fmt::Display for Candidate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let engine = match self.engine {
            Engine::Reference => "ref",
            Engine::Vellum => "vellum",
        };
        match self.device {
            DeviceRequest::Auto => write!(f, "{engine}"),
            DeviceRequest::Cpu => write!(f, "{engine}-cpu"),
            DeviceRequest::Gpu => write!(f, "{engine}-gpu"),
        }
    }
}

impl FromStr for Candidate {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Candidate> {
        let (engine, device) = s.split_once('-').unwrap_or((s, ""));
        let engine = match engine {
            "ref" | "reference" | "cpp" => Engine::Reference,
            "vellum" | "rust" => Engine::Vellum,
            _ => bail!("unknown engine {engine:?}: use ref or vellum"),
        };
        let device = match device {
            "" | "auto" => DeviceRequest::Auto,
            "cpu" => DeviceRequest::Cpu,
            "gpu" => DeviceRequest::Gpu,
            _ => bail!("unknown device {device:?}: use cpu, gpu or nothing for auto"),
        };
        Ok(Candidate { engine, device })
    }
}

/// How a job went.
pub struct Outcome {
    pub report: JobReport,
    /// Its stdout and stderr.
    pub log: PathBuf,
    /// The GPU's clocks, temperature and power over the job, when it could be watched.
    pub gpu: Option<system::GpuSummary>,
}

impl Outcome {
    /// The last lines of the runner's log, for error messages.
    pub fn log_tail(&self, lines: usize) -> String {
        let text = fs::read_to_string(&self.log).unwrap_or_default();
        let all: Vec<&str> = text.lines().collect();
        all[all.len().saturating_sub(lines)..].join("\n")
    }
}

/// The program that runs `candidate`'s jobs.
pub fn runner(candidate: Candidate) -> Result<PathBuf> {
    Ok(match candidate.engine {
        Engine::Reference => reference::ensure()?.exe,
        Engine::Vellum => std::env::current_exe()?,
    })
}

/// Runs `job` on `program` (from [`runner`]) and reads its report. With `watch`, samples the
/// GPU throughout.
pub fn run(program: &Path, job: &Job, watch: bool) -> Result<Outcome> {
    fs::create_dir_all(&job.out_dir)?;
    let job_path = job.out_dir.join("job.json");
    fs::write(&job_path, serde_json::to_vec_pretty(job)?)?;
    let report_path = job.out_dir.join("report.json");
    let _ = fs::remove_file(&report_path);
    let log = job.out_dir.join("log.txt");
    let out = fs::File::create(&log)?;
    let sampler = if watch {
        system::Sampler::start(Duration::from_millis(50))
    } else {
        None
    };
    let mut child = Command::new(program)
        .arg("run-job")
        .arg(&job_path)
        .envs(job.environment())
        .stdin(Stdio::null())
        .stdout(out.try_clone()?)
        .stderr(out)
        .spawn()
        .with_context(|| format!("starting {}", program.display()))?;
    // The runner rewrites its report after each clip: show progress on long jobs.
    let mut shown = 0;
    let started = std::time::Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        std::thread::sleep(Duration::from_millis(250));
        if started.elapsed() > Duration::from_secs(5)
            && let Some(done) = fs::read(&report_path)
                .ok()
                .and_then(|b| serde_json::from_slice::<JobReport>(&b).ok())
                .map(|r| r.clips.len())
            && done != shown
        {
            shown = done;
            eprint!("\r  {done}/{} clips", job.clips.len());
        }
    };
    if shown > 0 {
        eprintln!();
    }
    let gpu = sampler.map(|s| s.stop().summary());
    let mut report: JobReport = match fs::read(&report_path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .with_context(|| format!("reading {}", report_path.display()))?,
        // A crash before the first report: say so, with what the runner printed.
        Err(_) => JobReport {
            error: Some(format!(
                "{} exited with {status} and no report",
                program.file_name().unwrap_or_default().to_string_lossy()
            )),
            ..JobReport::default()
        },
    };
    // The runner reports after every clip, so a crash leaves the clips before it.
    if !status.success() && report.error.is_none() && report.clips.len() < job.clips.len() {
        report.error = Some(format!(
            "crashed ({status}) after {} of {} clips",
            report.clips.len(),
            job.clips.len()
        ));
    }
    Ok(Outcome { report, log, gpu })
}
