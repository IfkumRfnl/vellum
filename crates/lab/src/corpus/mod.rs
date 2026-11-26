//! The recordings the engines are measured on, with whatever ground truth each has.
//!
//! Every clip is 16 kHz mono 16-bit FLAC (or WAV) under the corpus directory, listed in
//! `index.json`. `cargo lab fetch --tier <tier>` downloads and prepares what a tier needs; each
//! source is idempotent and resumable, so a second fetch only adds what's missing.
//!
//! Tiers nest: everything in `micro` is in `smoke`, everything in `smoke` is in `quick`, and so
//! on up to `long`, which is the whole corpus.

mod ami;
mod fleurs;
mod hf;
mod librispeech;
mod synthetic;
pub mod util;
mod voxconverse;
mod youtube;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    clap::ValueEnum,
)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    /// Two or three short clips: a few seconds, for iterating on one kernel.
    Micro,
    /// Under a minute: one probe window per model and a few dozen utterances.
    Smoke,
    /// A few minutes: a broad sample of every set.
    Quick,
    /// Everything but the longest recordings.
    Full,
    /// Everything, including multi-hour recordings.
    Long,
}

/// What a clip is for. A clip can serve several purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum Purpose {
    /// Speech recognition: speech models run on it.
    Asr,
    /// Speaker labelling: the speaker model runs on it.
    Diarization,
    /// Short enough for one window: tensor dumps and stage-by-stage comparison.
    Probe,
    /// In the speed suite: timed in paired runs against the reference.
    Speed,
    /// An edge case: silence, noise, clipping, odd lengths. Checked for agreement and sanity,
    /// not accuracy.
    Robustness,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Clip {
    /// Unique and stable, `<set>/<name>`, e.g. `librispeech-clean/1089-134686-0000`.
    pub id: String,
    /// The source it came from, e.g. `librispeech-clean`, `ami`, `synthetic`.
    pub set: String,
    /// Relative to the corpus directory.
    pub audio: PathBuf,
    pub seconds: f64,
    /// The smallest tier that includes it.
    pub tier: Tier,
    /// ISO 639-1 code of the speech, e.g. `en`; empty when there's none.
    pub language: String,
    pub purposes: Vec<Purpose>,
    pub truth: Truth,
    /// Free-form labels: `read`, `meeting`, `podcast`, `noise:snr5`, `shift:1`, `boundary`…
    pub tags: Vec<String>,
    /// SHA-256 of the decoded samples as 16-bit integers (see `vellum_harness::audio`).
    pub pcm_sha256: String,
}

/// What a clip really contains, as far as anyone knows.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Truth {
    /// The words spoken, as the source transcribed them.
    pub text: Option<String>,
    /// Who spoke when (RTTM-style), times in seconds.
    pub turns: Option<Vec<TruthTurn>>,
    /// The words spoken, line by line, each with the name of whoever spoke it, in order
    /// (BBC scripts, podcast transcripts).
    pub lines: Option<Vec<(String, String)>>,
    /// Whether the text is verbatim (LibriSpeech, FLEURS, AMI) rather than edited (captions,
    /// published scripts), which only bounds accuracy.
    pub verbatim: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TruthTurn {
    pub t0: f64,
    pub t1: f64,
    pub speaker: String,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Corpus {
    #[serde(skip)]
    pub root: PathBuf,
    pub clips: Vec<Clip>,
}

impl Corpus {
    pub fn load(root: &Path) -> Result<Corpus> {
        let path = root.join("index.json");
        let mut corpus: Corpus = if path.is_file() {
            serde_json::from_slice(&fs::read(&path)?)
                .with_context(|| format!("reading {}", path.display()))?
        } else {
            Corpus::default()
        };
        corpus.root = root.to_owned();
        Ok(corpus)
    }

    pub fn save(&self) -> Result<()> {
        fs::create_dir_all(&self.root)?;
        let path = self.root.join("index.json");
        let partial = path.with_extension("json.partial");
        fs::write(&partial, serde_json::to_vec_pretty(self)?)?;
        fs::rename(&partial, &path)?;
        Ok(())
    }

    pub fn audio(&self, clip: &Clip) -> PathBuf {
        self.root.join(&clip.audio)
    }

    /// The clips of `tier` (and smaller tiers) serving `purpose`, in id order.
    pub fn select(&self, tier: Tier, purpose: Purpose) -> Vec<&Clip> {
        let mut clips: Vec<&Clip> = self
            .clips
            .iter()
            .filter(|c| c.tier <= tier && c.purposes.contains(&purpose))
            .collect();
        clips.sort_by(|a, b| a.id.cmp(&b.id));
        clips
    }

    pub fn get(&self, id: &str) -> Option<&Clip> {
        self.clips.iter().find(|c| c.id == id)
    }

    /// Replaces every clip of `set` with `clips`.
    fn replace_set(&mut self, set: &str, clips: Vec<Clip>) {
        self.clips.retain(|c| c.set != set);
        self.clips.extend(clips);
        self.clips.sort_by(|a, b| a.id.cmp(&b.id));
    }
}

/// Every clip a source has, by set.
pub type Found = BTreeMap<String, Vec<Clip>>;

/// A source of clips: one dataset, or the synthetic cases.
pub struct Source {
    /// The sets it provides (its clips' `set`).
    pub sets: &'static [&'static str],
    pub about: &'static str,
    /// Prepares the clips of `tier` and below; returns every clip of its sets it has, by set.
    pub fetch: fn(&mut util::Fetch, Tier) -> Result<Found>,
}

/// In dependency order: synthetic cases are derived from LibriSpeech.
pub const SOURCES: &[Source] = &[
    librispeech::SOURCE,
    synthetic::SOURCE,
    youtube::SOURCE,
    ami::SOURCE,
    fleurs::SOURCE,
    hf::SOURCE,
    voxconverse::SOURCE,
];

/// Prepares `tier`, from every source or only those providing one of `sets`.
pub fn fetch(root: &Path, tier: Tier, sets: &[String]) -> Result<Corpus> {
    let mut corpus = Corpus::load(root)?;
    for source in SOURCES {
        if !sets.is_empty() && !source.sets.iter().any(|s| sets.iter().any(|w| w == s)) {
            continue;
        }
        let mut fetch = util::Fetch::new(root, &corpus)?;
        eprintln!("· {} ({})", source.sets.join(", "), source.about);
        match (source.fetch)(&mut fetch, tier) {
            Ok(found) => {
                // Another fetch may have saved its sets meanwhile: merge into the index as it
                // is on disk now, holding a lock so two saves can't interleave.
                fs::create_dir_all(root)?;
                let lock = fs::File::create(root.join("index.lock"))?;
                lock.lock()?;
                corpus = Corpus::load(root)?;
                for (set, clips) in found {
                    corpus.replace_set(&set, clips);
                }
                corpus.save()?;
                drop(lock);
            }
            // One unreachable source shouldn't keep the others from the loop.
            Err(error) => eprintln!("  {} skipped: {error:#}", source.sets.join(", ")),
        }
    }
    Ok(corpus)
}
