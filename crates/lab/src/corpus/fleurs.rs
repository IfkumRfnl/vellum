//! FLEURS (Google, CC BY 4.0) test utterances in a few languages, for the multilingual Whisper
//! models, from the Hugging Face dataset `google/fleurs`.
//!
//! Each language's `test.tsv` lists the utterances; the audio is one `test.tar.gz` of 32-bit
//! float WAVs sorted by name. The picks are the first 50 by file name with distinct sentences,
//! so streaming the archive can stop after a small part of it. Plus, per language, a ~2 minute
//! clip of quick-tier utterances joined.

use super::{
    Clip, Purpose, Source, Tier, Truth,
    util::{self, Fetch},
};
use anyhow::{Context, Result, bail};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs, io,
    path::Path,
};
use vellum_harness::audio;

pub const SOURCE: Source = Source {
    sets: &["fleurs"],
    about: "multilingual read speech",
    fetch,
};

/// FLEURS' directory name and the ISO 639-1 code of each language.
const LANGUAGES: [(&str, &str); 4] = [
    ("de_de", "de"),
    ("fr_fr", "fr"),
    ("es_419", "es"),
    ("ja_jp", "ja"),
];

const BASE: &str = "https://huggingface.co/datasets/google/fleurs/resolve/main/data";

/// Utterances per language at smoke, quick and full.
const SMOKE: usize = 2;
pub(super) const QUICK: usize = 20;
const FULL: usize = 50;

/// One chosen test utterance.
struct Pick {
    /// The WAV's name without extension.
    stem: String,
    text: String,
    seconds: f64,
    tier: Tier,
}

fn fetch(fetch: &mut Fetch, tier: Tier) -> Result<BTreeMap<String, Vec<Clip>>> {
    let mut clips = Vec::new();
    for (name, code) in LANGUAGES {
        match language(fetch, tier, name, code) {
            Ok(found) => clips.extend(found),
            Err(error) => {
                eprintln!("  fleurs {code} skipped: {error:#}");
                // Keep what an earlier fetch made of it.
                clips.extend(
                    fetch
                        .clips_of("fleurs")
                        .into_iter()
                        .filter(|c| c.language == code && fetch.root.join(&c.audio).is_file())
                        .cloned(),
                );
            }
        }
    }
    clips.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(BTreeMap::from([("fleurs".to_owned(), clips)]))
}

/// The clips of one language.
fn language(fetch: &Fetch, tier: Tier, name: &str, code: &str) -> Result<Vec<Clip>> {
    let dir = fetch.dir(&format!("fleurs/{code}"))?;
    let tsv = dir.join("test.tsv");
    fetch.download(&format!("{BASE}/{name}/test.tsv"), &tsv, None)?;
    let picks = picks(&fs::read_to_string(&tsv)?)?;

    let id = |stem: &str| format!("fleurs/{code}-{stem}");
    let flac = |stem: &str| dir.join(format!("{stem}.flac"));
    let missing: BTreeSet<String> = picks
        .iter()
        .filter(|p| {
            p.tier <= tier && fetch.known(&id(&p.stem)).is_none() && !flac(&p.stem).is_file()
        })
        .map(|p| format!("{}.wav", p.stem))
        .collect();
    if !missing.is_empty() {
        extract(
            fetch,
            &format!("{BASE}/{name}/audio/test.tar.gz"),
            &dir,
            missing,
        )?;
    }

    let truth = |text: String| Truth {
        text: Some(text),
        verbatim: true,
        ..Truth::default()
    };
    // The first smoke utterance that fits one window is a probe.
    let probe = picks
        .iter()
        .find(|p| p.tier <= Tier::Smoke && p.seconds <= 28.0)
        .map(|p| p.stem.clone());
    let mut clips = Vec::new();
    for p in &picks {
        let mut purposes = vec![Purpose::Asr];
        if probe.as_ref() == Some(&p.stem) {
            purposes.push(Purpose::Probe);
        }
        let record = |mut clip: Clip| {
            clip.tier = p.tier;
            clip.purposes = purposes.clone();
            clip.truth = truth(p.text.clone());
            clip
        };
        if let Some(clip) = fetch.known(&id(&p.stem)) {
            clips.push(record(clip));
        } else if p.tier <= tier || flac(&p.stem).is_file() {
            clips.push(fetch.clip(
                id(&p.stem),
                "fleurs",
                &flac(&p.stem),
                p.tier,
                code,
                &purposes,
                truth(p.text.clone()),
                &["read"],
            )?);
        }
    }

    // About two minutes of quick utterances, joined in order.
    let mut parts = Vec::new();
    let mut seconds = 0.0;
    for p in picks.iter().filter(|p| p.tier <= Tier::Quick) {
        if seconds >= 120.0 {
            break;
        }
        parts.push(p);
        seconds += p.seconds;
    }
    let long_id = format!("fleurs/{code}-long");
    let long = dir.join("long.flac");
    let separator = if code == "ja" { "" } else { " " };
    let text = parts
        .iter()
        .map(|p| p.text.as_str())
        .collect::<Vec<_>>()
        .join(separator);
    if let Some(mut clip) = fetch.known(&long_id) {
        clip.truth = truth(text);
        clips.push(clip);
    } else if Tier::Quick <= tier || long.is_file() {
        if !long.is_file() {
            let mut pcm = Vec::new();
            for p in &parts {
                pcm.extend(audio::read(&flac(&p.stem))?);
            }
            util::write_flac(&long, &pcm)?;
        }
        clips.push(fetch.clip(
            long_id,
            "fleurs",
            &long,
            Tier::Quick,
            code,
            &[Purpose::Asr],
            truth(text),
            &["read", "long-form"],
        )?);
    }
    Ok(clips)
}

/// The chosen utterances of a `test.tsv` (id, file, raw transcription, normalized
/// transcription, characters, samples, gender): by file name, the first with each sentence id,
/// up to the full tier's count.
fn picks(tsv: &str) -> Result<Vec<Pick>> {
    let mut rows: Vec<Vec<&str>> = tsv
        .lines()
        .map(|l| l.split('\t').collect::<Vec<_>>())
        .filter(|f| f.len() >= 6)
        .collect();
    rows.sort_by(|a, b| a[1].cmp(b[1]));
    let mut sentences = BTreeSet::new();
    let mut out = Vec::new();
    for row in rows {
        if out.len() == FULL {
            break;
        }
        let Some(stem) = row[1].strip_suffix(".wav") else {
            continue;
        };
        if !stem.chars().all(|c| c.is_ascii_alphanumeric()) || !sentences.insert(row[0]) {
            continue;
        }
        let samples: f64 = row[5]
            .parse()
            .with_context(|| format!("samples of {stem}"))?;
        let tier = match out.len() {
            n if n < SMOKE => Tier::Smoke,
            n if n < QUICK => Tier::Quick,
            _ => Tier::Full,
        };
        out.push(Pick {
            stem: stem.to_owned(),
            text: row[2].trim().to_owned(),
            seconds: samples / audio::RATE as f64,
            tier,
        });
    }
    if out.is_empty() {
        bail!("no utterances in test.tsv");
    }
    Ok(out)
}

/// Streams the archive at `url` until every file of `wanted` is converted into `dir`.
fn extract(fetch: &Fetch, url: &str, dir: &Path, mut wanted: BTreeSet<String>) -> Result<()> {
    eprintln!("  streaming {url} for {} utterances", wanted.len());
    let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(fetch.open(url)?));
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        let Some(name) = path.file_name().and_then(|n| n.to_str()).map(str::to_owned) else {
            continue;
        };
        if !wanted.remove(&name) {
            continue;
        }
        // `name` is one of the picks, so it's a plain alphanumeric name.
        let stem = name.trim_end_matches(".wav");
        let wav = dir.join(format!("{stem}.source.wav"));
        io::copy(&mut entry, &mut fs::File::create(&wav)?)
            .with_context(|| format!("extracting {name}"))?;
        let converted = util::to_flac(&wav, &dir.join(format!("{stem}.flac")), None, None);
        let _ = fs::remove_file(&wav);
        converted?;
        if wanted.is_empty() {
            return Ok(());
        }
    }
    bail!(
        "{url} lacks {}",
        wanted.into_iter().collect::<Vec<_>>().join(", ")
    )
}
