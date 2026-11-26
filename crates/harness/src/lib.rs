//! What the engine lab and the engines it measures agree on.
//!
//! The lab (`crates/lab`) writes a [`Job`] as JSON and starts an engine's runner process with
//! its path: `vellum-ref` for transcribe.cpp, `lab run-job` for Vellum's own engine. Both
//! runners hand the job to [`run`] with an [`Engine`] adapter, so audio decoding, windowing,
//! repetition, timing and memory measurement are one piece of code for both engines. The
//! runner writes a [`JobReport`] beside the job.

pub mod audio;
pub mod dump;
mod protocol;
mod runner;
mod sys;
pub mod windows;

pub use protocol::*;
pub use runner::{Engine, WindowResult, run_job};
