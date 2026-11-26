//! Tensor dumps in transcribe.cpp's format (`transcribe-debug.h`), written by either engine and
//! read back by the lab: `<dir>/<name>.f32` holds raw little-endian f32 in row-major order and
//! `<dir>/<name>.json` its sidecar `{name, stage?, shape (slowest first), dtype, layout, min,
//! max, mean, source}`. Decoder decisions and other events go to `<dir>/events.jsonl`, one
//! `{"kind": …, …fields}` object per line, in the order they happened.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::SystemTime,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sidecar {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
    pub shape: Vec<usize>,
    #[serde(default)]
    pub dtype: String,
    #[serde(default)]
    pub layout: String,
    #[serde(default)]
    pub min: Option<f64>,
    #[serde(default)]
    pub max: Option<f64>,
    #[serde(default)]
    pub mean: Option<f64>,
    #[serde(default)]
    pub source: serde_json::Value,
}

/// A tensor found in a dump directory.
#[derive(Debug, Clone)]
pub struct Dumped {
    /// The file stem: the tensor's name with any prefix the engine scoped it with.
    pub key: String,
    pub sidecar: Sidecar,
    pub data_path: PathBuf,
    /// When it was written: dumps happen in compute order, so this orders checkpoints.
    pub written: SystemTime,
}

impl Dumped {
    pub fn load(&self) -> Result<Vec<f32>> {
        let bytes = fs::read(&self.data_path)
            .with_context(|| format!("reading {}", self.data_path.display()))?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }
}

/// Writes one tensor. `source` names the engine, e.g. `"vellum"`.
pub fn write_tensor(
    dir: &Path,
    name: &str,
    shape: &[usize],
    data: &[f32],
    source: &str,
) -> Result<()> {
    fs::create_dir_all(dir)?;
    let file = sanitize(name);
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(dir.join(format!("{file}.f32")), bytes)?;
    let (mut min, mut max, mut sum) = (f64::INFINITY, f64::NEG_INFINITY, 0.0);
    for v in data {
        let v = *v as f64;
        min = min.min(v);
        max = max.max(v);
        sum += v;
    }
    let finite = |v: f64| v.is_finite().then_some(v);
    let sidecar = Sidecar {
        name: name.to_owned(),
        stage: None,
        shape: shape.to_vec(),
        dtype: "f32".into(),
        layout: "row-major".into(),
        min: finite(min),
        max: finite(max),
        mean: finite(sum / data.len().max(1) as f64),
        source: serde_json::json!({ "kind": source }),
    };
    fs::write(
        dir.join(format!("{file}.json")),
        serde_json::to_vec_pretty(&sidecar)?,
    )?;
    Ok(())
}

/// Appends one event to `events.jsonl`.
pub fn write_event(dir: &Path, kind: &str, fields: serde_json::Value) -> Result<()> {
    fs::create_dir_all(dir)?;
    let mut line = match fields {
        serde_json::Value::Object(map) => map,
        other => {
            let mut map = serde_json::Map::new();
            map.insert("value".into(), other);
            map
        }
    };
    line.insert("kind".into(), kind.into());
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("events.jsonl"))?;
    writeln!(file, "{}", serde_json::Value::Object(line))?;
    Ok(())
}

/// Every tensor in a dump directory, in the order it was written.
pub fn list(dir: &Path) -> Result<Vec<Dumped>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path();
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        let data_path = path.with_extension("f32");
        if !data_path.is_file() {
            continue;
        }
        let sidecar: Sidecar = serde_json::from_slice(&fs::read(&path)?)
            .with_context(|| format!("reading {}", path.display()))?;
        let written = fs::metadata(&data_path)?.modified()?;
        out.push(Dumped {
            key: path.file_stem().unwrap().to_string_lossy().into_owned(),
            sidecar,
            data_path,
            written,
        });
    }
    out.sort_by(|a, b| a.written.cmp(&b.written).then(a.key.cmp(&b.key)));
    Ok(out)
}

/// Every event in a dump directory, in order.
pub fn events(dir: &Path) -> Result<Vec<serde_json::Value>> {
    let path = dir.join("events.jsonl");
    if !path.is_file() {
        return Ok(Vec::new());
    }
    fs::read_to_string(&path)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| Ok(serde_json::from_str(l)?))
        .collect()
}

fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            c => c,
        })
        .collect()
}
