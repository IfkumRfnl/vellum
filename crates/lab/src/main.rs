//! The engine lab: Vellum's speech engine against transcribe.cpp on a corpus of recordings, for
//! correctness and speed. See crates/lab/README.md.

mod bench;
mod check;
mod corpus;
mod encoder;
mod engine;
mod goldens;
mod inspect;
mod jobs;
mod kernels;
mod models;
mod paths;
mod probe;
mod reference;
mod report;
mod run;
mod score;
mod sustain;
mod system;
mod whisper;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use corpus::{Corpus, Purpose, Tier};
use run::Candidate;
use std::process::ExitCode;

#[derive(Parser)]
#[command(name = "cargo lab", about = "Vellum's engine against transcribe.cpp")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// The loop: an engine on a tier's clips against the reference and the truth, with gates.
    Check {
        #[arg(long, value_enum, default_value = "smoke")]
        tier: Tier,
        /// Models to check (default all): parakeet, small, turbo, sortformer.
        #[arg(long, value_delimiter = ',')]
        models: Vec<String>,
        /// The engine checked: vellum, vellum-cpu, vellum-gpu, ref, ref-cpu, ref-gpu.
        #[arg(long, default_value = "vellum")]
        engine: Candidate,
        /// The engine it's compared with.
        #[arg(long, default_value = "ref")]
        against: Candidate,
        /// Reference settings as KEY=VALUE (e.g. TRANSCRIBE_NO_FLASH=1).
        #[arg(long = "env", value_parser = key_value)]
        env: Vec<(String, String)>,
        /// Fail on models the engine doesn't support yet.
        #[arg(long)]
        strict: bool,
    },
    /// Paired speed runs: two engines raced live, in alternating order.
    Bench {
        #[arg(long, value_enum, default_value = "smoke")]
        tier: Tier,
        #[arg(long, value_delimiter = ',')]
        models: Vec<String>,
        #[arg(long, default_value = "ref")]
        a: Candidate,
        #[arg(long, default_value = "vellum")]
        b: Candidate,
        /// Jobs per side (pairs per clip).
        #[arg(long, default_value_t = 6)]
        blocks: u32,
        #[arg(long = "env", value_parser = key_value)]
        env: Vec<(String, String)>,
        /// Wait for the GPU to cool below this many °C between jobs.
        #[arg(long)]
        cool: Option<u32>,
    },
    /// One window through both engines, tensor by tensor.
    Probe {
        #[arg(long, default_value = "parakeet")]
        model: String,
        /// The clip (default: the model's first probe clip).
        #[arg(long)]
        clip: Option<String>,
        #[arg(long, default_value = "vellum-cpu")]
        engine: Candidate,
        #[arg(long, default_value = "ref-cpu")]
        against: Candidate,
        /// Blocks whose sub-layers (attention, convolution, feed-forward) are dumped too.
        #[arg(long, value_delimiter = ',')]
        sub_blocks: Vec<u32>,
        /// Checkpoints fed from the reference into the engine (e.g. enc.mel.in).
        #[arg(long, value_delimiter = ',')]
        inject: Vec<String>,
        /// Relative error past which a checkpoint counts as parted (default: from calibration).
        #[arg(long)]
        tolerance: Option<f64>,
        #[arg(long = "env", value_parser = key_value)]
        env: Vec<(String, String)>,
        /// Measure the reference's own GPU-against-CPU spread per checkpoint, which probes
        /// then allow twice of.
        #[arg(long)]
        calibrate: bool,
    },
    /// Makes the reference's results for a tier ahead of time (e.g. overnight for `full`).
    Goldens {
        #[arg(long, value_enum, default_value = "quick")]
        tier: Tier,
        #[arg(long, value_delimiter = ',')]
        models: Vec<String>,
        #[arg(long, default_value = "ref")]
        against: Candidate,
        #[arg(long = "env", value_parser = key_value)]
        env: Vec<(String, String)>,
    },
    /// The GPU kernels on Parakeet's real shapes: error against the CPU, GPU time, TFLOP/s.
    Kernels {
        /// Kernels to run (default all the adapter has): simt, coop, q8.
        #[arg(long, value_delimiter = ',')]
        kernels: Vec<String>,
        /// Rows per call: frames per batch (default 350, 1400, 2800).
        #[arg(long, value_delimiter = ',')]
        rows: Vec<usize>,
        /// Only shapes whose label contains this (e.g. `ff.`).
        #[arg(long)]
        filter: Option<String>,
        /// Timed calls per measurement (the median is reported).
        #[arg(long, default_value_t = 30)]
        runs: u32,
        /// Passes over all measurements, keeping each one's best median.
        #[arg(long, default_value_t = 3)]
        rounds: u32,
        /// The adapter: an index or part of its name (default: `VELLUM_GPU`, else the best).
        #[arg(long)]
        adapter: Option<String>,
        /// Backends to consider: vulkan, dx12, metal (default: `VELLUM_GPU_BACKEND`, else all).
        #[arg(long)]
        backend: Option<String>,
    },
    /// Parakeet's GPU encoder against the CPU oracle, checkpoint by checkpoint, and timed.
    Encoder {
        /// Clips to compare on (default: Parakeet's probe clips).
        #[arg(long = "clip", value_delimiter = ',')]
        clips: Vec<String>,
        /// Windows encoded together for the batch check.
        #[arg(long, default_value_t = 4)]
        batch: usize,
        /// The multiply: simt, coop or q8 (default: tensor cores where the adapter has them).
        #[arg(long)]
        kernel: Option<String>,
        /// Pick the multiply's tile shape by measuring it on this adapter.
        #[arg(long)]
        autotune: bool,
        /// Seconds of audio per timed window.
        #[arg(long, default_value_t = 26.)]
        seconds: f64,
        /// Timed calls per batch size (the median is reported).
        #[arg(long, default_value_t = 5)]
        runs: u32,
        /// The adapter: an index or part of its name (default: `VELLUM_GPU`, else the best).
        #[arg(long)]
        adapter: Option<String>,
        /// Backends to consider: vulkan, dx12, metal.
        #[arg(long)]
        backend: Option<String>,
    },
    /// Whisper's GPU path against the CPU oracle: checkpoints, decisions, timing by batch.
    Whisper {
        /// small or turbo.
        #[arg(long, default_value = "small")]
        model: String,
        /// Clips to compare on (default: the model's probe clips).
        #[arg(long = "clip", value_delimiter = ',')]
        clips: Vec<String>,
        /// Windows encoded together for the batch check.
        #[arg(long, default_value_t = 4)]
        batch: usize,
        /// Timed calls per batch size.
        #[arg(long, default_value_t = 5)]
        runs: u32,
        /// The adapter: an index or part of its name (default: `VELLUM_GPU`, else the best).
        #[arg(long)]
        adapter: Option<String>,
        /// Backends to consider: vulkan, dx12, metal.
        #[arg(long)]
        backend: Option<String>,
        /// Run the clips whole through the CPU oracle and the GPU, windows as the harness
        /// cuts them, and say why each differing window differs.
        #[arg(long)]
        explain: bool,
        /// A long clip for the pipeline check (empty: none).
        #[arg(long, default_value = "youtube/ted-30-days")]
        pipeline: String,
        /// Instead of the checks: the encoder alone, batch after batch, for this many seconds,
        /// each batch's GPU time next to NVML's clocks, memory load and PCIe link (`--batch`
        /// windows; 0 for the encoder's own batch size).
        #[arg(long)]
        sustain: Option<f64>,
    },
    /// Downloads and prepares the recordings a tier needs.
    Fetch {
        #[arg(long, value_enum, default_value = "quick")]
        tier: Tier,
        /// Only these sets (e.g. `librispeech-clean`, `ami`).
        #[arg(long, value_delimiter = ',')]
        sets: Vec<String>,
    },
    /// What's inside a model file: metadata and tensors.
    Inspect {
        /// A model (`parakeet`, `small`, `turbo`, `sortformer`) or a GGUF path.
        model: String,
        /// List every tensor.
        #[arg(long)]
        tensors: bool,
        /// Only keys and tensors containing this.
        #[arg(long)]
        filter: Option<String>,
    },
    /// Runs a job file on Vellum's engine (the lab starts itself this way).
    #[command(hide = true)]
    RunJob { job: std::path::PathBuf },
    /// Lists the corpus: clips and hours per set and tier.
    Corpus {
        #[arg(long, value_enum, default_value = "long")]
        tier: Tier,
        /// Every clip, not just the totals.
        #[arg(long)]
        clips: bool,
    },
}

fn main() -> ExitCode {
    let started = std::time::Instant::now();
    let cli = Cli::parse();
    if let Command::RunJob { job } = &cli.command {
        return vellum_harness::run_job(job, started, engine::load);
    }
    match run(cli) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn key_value(s: &str) -> Result<(String, String)> {
    let (k, v) = s.split_once('=').context("expected KEY=VALUE")?;
    Ok((k.to_owned(), v.to_owned()))
}

fn code(passed: bool) -> ExitCode {
    if passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn run(cli: Cli) -> Result<ExitCode> {
    match cli.command {
        Command::Check {
            tier,
            models,
            engine,
            against,
            env,
            strict,
        } => {
            return Ok(code(check::check(&check::Args {
                tier,
                models,
                engine,
                against,
                env,
                strict,
            })?));
        }
        Command::Bench {
            tier,
            models,
            a,
            b,
            blocks,
            env,
            cool,
        } => {
            return Ok(code(bench::bench(&bench::Args {
                tier,
                models,
                a,
                b,
                blocks,
                env,
                cool,
            })?));
        }
        Command::Probe {
            model,
            clip,
            engine,
            against,
            sub_blocks,
            inject,
            tolerance,
            env,
            calibrate,
        } => {
            return Ok(code(probe::probe(&probe::Args {
                model,
                clip,
                engine,
                against,
                sub_blocks,
                inject,
                tolerance,
                env,
                calibrate,
            })?));
        }
        Command::Kernels {
            kernels,
            rows,
            filter,
            runs,
            rounds,
            adapter,
            backend,
        } => {
            return Ok(code(kernels::kernels(&kernels::Args {
                kernels,
                rows,
                filter,
                runs,
                rounds,
                adapter,
                backend,
            })?));
        }
        Command::Encoder {
            clips,
            batch,
            kernel,
            autotune,
            seconds,
            runs,
            adapter,
            backend,
        } => {
            return Ok(code(encoder::encoder(&encoder::Args {
                clips,
                batch,
                kernel,
                autotune,
                seconds,
                runs,
                adapter,
                backend,
            })?));
        }
        Command::Whisper {
            model,
            clips,
            batch,
            runs,
            adapter,
            backend,
            explain,
            pipeline,
            sustain,
        } => {
            if let Some(seconds) = sustain {
                return Ok(code(sustain::sustain(&sustain::Args {
                    model,
                    seconds,
                    batch,
                    adapter,
                })?));
            }
            return Ok(code(whisper::whisper(&whisper::Args {
                model,
                clips,
                batch,
                runs,
                adapter,
                backend,
                explain,
                pipeline,
            })?));
        }
        Command::Goldens {
            tier,
            models,
            against,
            env,
        } => check::goldens(tier, &models, against, env)?,
        Command::Fetch { tier, sets } => {
            let corpus = corpus::fetch(&paths::corpus()?, tier, &sets)?;
            list(&corpus, tier, false);
        }
        Command::Corpus { tier, clips } => list(&Corpus::load(&paths::corpus()?)?, tier, clips),
        Command::Inspect {
            model,
            tensors,
            filter,
        } => inspect::inspect(&model, tensors, filter.as_deref())?,
        Command::RunJob { .. } => unreachable!(),
    }
    Ok(ExitCode::SUCCESS)
}

fn list(corpus: &Corpus, tier: Tier, clips: bool) {
    use std::collections::BTreeMap;
    let mut sets: BTreeMap<&str, (usize, f64)> = BTreeMap::new();
    for clip in corpus.clips.iter().filter(|c| c.tier <= tier) {
        let entry = sets.entry(&clip.set).or_default();
        entry.0 += 1;
        entry.1 += clip.seconds;
        if clips {
            println!(
                "{:<56} {:>8.1}s {:>6?} {:<3} {:?} {}",
                clip.id,
                clip.seconds,
                clip.tier,
                clip.language,
                clip.purposes,
                clip.tags.join(",")
            );
        }
    }
    let (mut n, mut seconds) = (0, 0.0);
    println!("{:<24} {:>7} {:>9}", "set", "clips", "hours");
    for (set, (count, secs)) in &sets {
        println!("{set:<24} {count:>7} {:>9.2}", secs / 3600.);
        n += count;
        seconds += secs;
    }
    println!(
        "{:<24} {n:>7} {:>9.2}",
        format!("{tier:?} tier"),
        seconds / 3600.
    );
    for purpose in [
        Purpose::Asr,
        Purpose::Diarization,
        Purpose::Probe,
        Purpose::Speed,
        Purpose::Robustness,
    ] {
        let selected = corpus.select(tier, purpose);
        let hours = selected.iter().map(|c| c.seconds).sum::<f64>().max(0.) / 3600.;
        println!("  {purpose:?}: {} clips, {hours:.2} h", selected.len());
    }
    println!("corpus: {}", corpus.root.display());
}
