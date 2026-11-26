//! What a lab run leaves behind: a Markdown report and its JSON beside it in `target/lab/`
//! (`latest.md` is always the last one), and a line in the history for trends and baselines.

use crate::paths;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Write,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    /// `20261126-142501-check-smoke`.
    pub id: String,
    pub command: String,
    /// The commit, with `+dirty` when the tree has changes.
    pub commit: String,
    pub unix_time: u64,
    pub machine: serde_json::Value,
    /// Markdown sections, in order.
    pub sections: Vec<(String, String)>,
    /// The machine-readable results: what later runs compare against.
    pub results: serde_json::Value,
    /// Whether every gate held.
    pub passed: bool,
}

impl Report {
    pub fn new(command: &str, machine: serde_json::Value) -> Report {
        let unix_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or_default();
        Report {
            id: format!("{}-{}", stamp(unix_time), command.replace(' ', "-")),
            command: command.into(),
            commit: commit(),
            unix_time,
            machine,
            sections: Vec::new(),
            results: serde_json::Value::Null,
            passed: true,
        }
    }

    pub fn section(&mut self, title: impl Into<String>, body: impl Into<String>) {
        self.sections.push((title.into(), body.into()));
    }

    pub fn markdown(&self) -> String {
        let mut out = format!(
            "# Lab {}\n\n{} · commit {} · {}\n",
            self.command,
            stamp(self.unix_time),
            self.commit,
            if self.passed {
                "**passed**"
            } else {
                "**FAILED**"
            }
        );
        for (title, body) in &self.sections {
            out.push_str(&format!("\n## {title}\n\n{}\n", body.trim_end()));
        }
        out
    }

    /// Writes the report and its history line; returns the Markdown file.
    pub fn save(&self) -> Result<PathBuf> {
        let dir = paths::reports();
        fs::create_dir_all(&dir)?;
        let md = dir.join(format!("{}.md", self.id));
        fs::write(&md, self.markdown())?;
        fs::write(dir.join("latest.md"), self.markdown())?;
        fs::write(
            dir.join(format!("{}.json", self.id)),
            serde_json::to_vec_pretty(self)?,
        )?;
        let history = paths::history()?;
        fs::create_dir_all(history.parent().unwrap())?;
        let line = serde_json::json!({
            "id": self.id,
            "command": self.command,
            "commit": self.commit,
            "unix_time": self.unix_time,
            "passed": self.passed,
            "results": self.results,
        });
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(history)?;
        writeln!(file, "{line}")?;
        Ok(md)
    }
}

/// The history's runs of `command`, oldest first.
pub fn history(command: &str) -> Result<Vec<serde_json::Value>> {
    let Ok(text) = fs::read_to_string(paths::history()?) else {
        return Ok(Vec::new());
    };
    Ok(text
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter(|v| v["command"] == command)
        .collect())
}

/// `20261126-142501` in UTC.
fn stamp(unix: u64) -> String {
    let (days, secs) = (unix / 86_400, unix % 86_400);
    // Civil date from days since the epoch (Howard Hinnant's algorithm).
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}{month:02}{day:02}-{:02}{:02}{:02}",
        secs / 3600,
        secs / 60 % 60,
        secs % 60
    )
}

fn commit() -> String {
    let git = |args: &[&str]| {
        Command::new("git")
            .current_dir(paths::repo())
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
    };
    let head = git(&["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty = git(&["status", "--porcelain"]).is_some_and(|s| !s.is_empty());
    if dirty { format!("{head}+dirty") } else { head }
}
