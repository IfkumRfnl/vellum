//! The reference engine's results, kept so the inner loop never waits for C++: one file per clip,
//! under a key naming everything that could change them — transcribe.cpp's version and build
//! flags, the model file, the device and its driver, the windowing, the engine's settings and the
//! protocol version (`SCHEMA`, bumped whenever the harness changes what a run produces). The
//! runner binary's own hash isn't in it: editing the harness's code rebuilds `vellum-ref` without
//! changing what transcribe.cpp computes, and redoing 70 hours of audio for that would be waste.
//! Speed is never compared against a kept timing; `cargo lab bench` races both engines live.

use crate::{corpus::Clip, models::Model, paths, run::Candidate};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};
use vellum_harness::{ClipReport, EngineInfo, SCHEMA, Windowing};

/// What a kept result depends on.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Key {
    pub schema: u32,
    pub candidate: String,
    pub build: String,
    pub model: String,
    pub model_sha256: String,
    /// The machine's GPU, driver and OS build, hashed (see `system::Machine::fingerprint`).
    pub machine: String,
    pub windowing: Windowing,
    pub env: Vec<(String, String)>,
}

impl Key {
    pub fn new(
        candidate: Candidate,
        reference: &crate::reference::Reference,
        model: &Model,
        machine: &str,
        windowing: Windowing,
        env: &[(String, String)],
    ) -> Key {
        Key {
            schema: SCHEMA,
            candidate: candidate.to_string(),
            build: reference.build.clone(),
            model: model.id.into(),
            model_sha256: model.digest.into(),
            machine: machine.into(),
            windowing,
            env: env.to_vec(),
        }
    }

    pub fn id(&self) -> String {
        let json = serde_json::to_vec(self).unwrap();
        format!("{:x}", Sha256::digest(json))[..16].to_owned()
    }

    fn dir(&self) -> Result<PathBuf> {
        Ok(paths::goldens()?
            .join(&self.model)
            .join(&self.candidate)
            .join(self.id()))
    }
}

#[derive(Serialize, Deserialize)]
struct Meta {
    key: Key,
    engine: EngineInfo,
}

fn file_name(clip_id: &str) -> String {
    format!("{}.json", clip_id.replace(['/', '\\', ':'], "__"))
}

/// The kept result for `clip`, if there is one for this very audio.
pub fn load(key: &Key, clip: &Clip) -> Result<Option<ClipReport>> {
    let path = key.dir()?.join("clips").join(file_name(&clip.id));
    let Ok(bytes) = fs::read(&path) else {
        return Ok(None);
    };
    let report: ClipReport = serde_json::from_slice(&bytes)?;
    Ok((report.pcm_sha256 == clip.pcm_sha256 && report.error.is_none()).then_some(report))
}

/// The engine the kept results came from, if any were kept.
pub fn engine(key: &Key) -> Result<Option<EngineInfo>> {
    let Ok(bytes) = fs::read(key.dir()?.join("meta.json")) else {
        return Ok(None);
    };
    Ok(Some(serde_json::from_slice::<Meta>(&bytes)?.engine))
}

/// Keeps a job's successful clips.
pub fn save(key: &Key, engine: &EngineInfo, clips: &[ClipReport]) -> Result<()> {
    let dir = key.dir()?;
    fs::create_dir_all(dir.join("clips"))?;
    fs::write(
        dir.join("meta.json"),
        serde_json::to_vec_pretty(&Meta {
            key: key.clone(),
            engine: engine.clone(),
        })?,
    )?;
    for clip in clips.iter().filter(|c| c.error.is_none()) {
        fs::write(
            dir.join("clips").join(file_name(&clip.id)),
            serde_json::to_vec(clip)?,
        )?;
    }
    Ok(())
}
