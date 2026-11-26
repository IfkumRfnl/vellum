//! Where the lab keeps things. Big, slow-to-make data (the corpus, the reference's goldens, run
//! history) lives outside the repository so `cargo clean` doesn't throw it away:
//! `VELLUM_LAB_DIR`, else the user's local data directory (`%LOCALAPPDATA%\vellum\lab` on
//! Windows, `~/.local/share/vellum/lab` on Linux, `~/Library/Application Support/vellum/lab`
//! on macOS). Each run's report goes to `target/lab/` in the repository.

use anyhow::{Context, Result};
use std::path::PathBuf;

pub fn root() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("VELLUM_LAB_DIR") {
        return Ok(dir.into());
    }
    let dirs = directories::ProjectDirs::from("", "", "vellum")
        .context("Could not locate a data directory for the lab")?;
    Ok(dirs.data_local_dir().join("lab"))
}

/// Recordings and their ground truth.
pub fn corpus() -> Result<PathBuf> {
    Ok(root()?.join("corpus"))
}

/// The reference engine's outputs, by build, model, device and clip.
pub fn goldens() -> Result<PathBuf> {
    Ok(root()?.join("goldens"))
}

/// Tensor dumps.
pub fn dumps() -> Result<PathBuf> {
    Ok(root()?.join("dumps"))
}

/// One line per lab run, for trends and regressions.
pub fn history() -> Result<PathBuf> {
    Ok(root()?.join("history.jsonl"))
}

/// Scratch space for jobs; emptied as runs finish.
pub fn scratch() -> Result<PathBuf> {
    Ok(root()?.join("scratch"))
}

/// The repository's `target/lab`, for reports.
pub fn reports() -> PathBuf {
    repo().join("target").join("lab")
}

/// The repository root: two levels above this crate.
pub fn repo() -> PathBuf {
    let crate_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    crate_dir.ancestors().nth(2).unwrap_or(crate_dir).to_owned()
}
