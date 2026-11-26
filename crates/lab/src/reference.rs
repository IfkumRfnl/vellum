//! The reference engine's runner, `vellum-ref`: transcribe.cpp built once, apart from the app,
//! with every CPU instruction this machine has (`GGML_NATIVE`, its best showing) and the
//! validation hooks that let it dump each layer. It lives in `target/reference`, so the app's
//! and the lab's builds never rebuild the C++.

use anyhow::{Context, Result, ensure};
use std::{
    path::PathBuf,
    process::{Command, Stdio},
};

const CMAKE_ARGS: &str = "-DGGML_NATIVE=ON -DTRANSCRIBE_ENABLE_VALIDATION_HOOKS=ON";

pub struct Reference {
    pub exe: PathBuf,
    /// What it was built from and with, as it reports it.
    pub build: String,
    /// SHA-256 of the executable: a rebuild with other flags is another reference.
    pub exe_sha256: String,
}

fn target_dir() -> PathBuf {
    crate::paths::repo().join("target").join("reference")
}

fn exe() -> PathBuf {
    target_dir()
        .join("release")
        .join(format!("vellum-ref{}", std::env::consts::EXE_SUFFIX))
}

/// The newest modification time among the files the reference is built from: its own crate,
/// the harness it links, and the lockfile that pins transcribe.cpp.
fn sources_modified() -> Option<std::time::SystemTime> {
    let repo = crate::paths::repo();
    let mut newest = std::fs::metadata(repo.join("Cargo.lock"))
        .ok()?
        .modified()
        .ok()?;
    for dir in ["crates/reference", "crates/harness"] {
        let mut stack = vec![repo.join(dir)];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).ok()?.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if let Ok(modified) = entry.metadata().and_then(|m| m.modified()) {
                    newest = newest.max(modified);
                }
            }
        }
    }
    Some(newest)
}

/// The reference runner, built first when it's missing or older than its sources. Its build
/// description and hash are kept beside it, so the loop doesn't start Vulkan just to name it.
pub fn ensure() -> Result<Reference> {
    let exe = exe();
    // When cargo finds nothing to rebuild it leaves the executable's date alone, so the last
    // build is marked by a stamp of its own (else every run would "rebuild" after any edit to
    // the lockfile).
    let stamp_file = exe.with_extension("built");
    let built = [&stamp_file, &exe]
        .iter()
        .filter_map(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok())
        .max()
        .filter(|_| exe.is_file());
    let stale = match (built, sources_modified()) {
        (Some(built), Some(sources)) => sources > built,
        _ => true,
    };
    if stale {
        build()?;
        std::fs::write(&stamp_file, "")?;
    }
    let cache = exe.with_extension("lab.json");
    let stamp = std::fs::metadata(&exe)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos().to_string())
        .unwrap_or_default();
    if let Ok(bytes) = std::fs::read(&cache)
        && let Ok(kept) = serde_json::from_slice::<serde_json::Value>(&bytes)
        && kept["stamp"] == stamp.as_str()
    {
        return Ok(Reference {
            build: kept["build"].as_str().unwrap_or_default().to_owned(),
            exe_sha256: kept["sha256"].as_str().unwrap_or_default().to_owned(),
            exe,
        });
    }
    let output = Command::new(&exe)
        .arg("info")
        .stderr(Stdio::null())
        .output()
        .with_context(|| format!("running {}", exe.display()))?;
    ensure!(output.status.success(), "vellum-ref info failed");
    let info: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    let reference = Reference {
        build: info["build"].as_str().unwrap_or_default().to_owned(),
        exe_sha256: crate::corpus::util::sha256_file(&exe)?,
        exe,
    };
    std::fs::write(
        &cache,
        serde_json::to_vec(&serde_json::json!({
            "stamp": stamp,
            "build": reference.build,
            "sha256": reference.exe_sha256,
        }))?,
    )?;
    Ok(reference)
}

fn build() -> Result<()> {
    let mut cargo = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
    cargo
        .current_dir(crate::paths::repo())
        .args([
            "build",
            "--quiet",
            "--release",
            "-p",
            "vellum-ref",
            "--target-dir",
        ])
        .arg(target_dir())
        .env("TRANSCRIBE_CMAKE_ARGS", CMAKE_ARGS)
        .stdin(Stdio::null());
    if std::env::var_os("CMAKE").is_none()
        && let Some(cmake) = find_cmake()
    {
        cargo.env("CMAKE", cmake);
    }
    if !exe().is_file() {
        eprintln!("· building transcribe.cpp for the reference (a few minutes, once)");
    } else {
        eprintln!("· rebuilding the reference runner");
    }
    let status = cargo
        .status()
        .context("running cargo to build vellum-ref")?;
    ensure!(status.success(), "building vellum-ref failed");
    Ok(())
}

/// CMake, when it isn't on the PATH: Visual Studio ships one.
fn find_cmake() -> Option<PathBuf> {
    if Command::new("cmake")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
    {
        return None;
    }
    if !cfg!(windows) {
        return None;
    }
    for base in [
        r"C:\Program Files\Microsoft Visual Studio",
        r"C:\Program Files (x86)\Microsoft Visual Studio",
    ] {
        let Ok(versions) = std::fs::read_dir(base) else {
            continue;
        };
        for version in versions.flatten() {
            let Ok(editions) = std::fs::read_dir(version.path()) else {
                continue;
            };
            for edition in editions.flatten() {
                let cmake = edition
                    .path()
                    .join(r"Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe");
                if cmake.is_file() {
                    return Some(cmake);
                }
            }
        }
    }
    None
}
