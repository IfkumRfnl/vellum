//! LibriSpeech test-clean and test-other (OpenSLR 12, CC BY 4.0): read speech with verbatim
//! transcripts.
//!
//! Each split is extracted whole, once, straight from the download stream (the FLAC files are
//! already 16 kHz mono 16-bit, so they're kept as they are); a `<split>.done` marker holding
//! the archive's SHA-256 skips the download afterwards. Every utterance is in the corpus, the
//! smaller tiers taking a fixed sample stratified across speakers and lengths. A few whole
//! chapters, their utterances joined, give long-form clips with verbatim text.

use super::{Clip, Purpose, Source, Tier, Truth, util::Fetch};
use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};
use vellum_harness::audio;

pub const SOURCE: Source = Source {
    sets: &["librispeech-clean", "librispeech-other"],
    about: "read audiobooks, verbatim text",
    fetch,
};

/// Each set and the split it comes from.
const SPLITS: [(&str, &str); 2] = [
    ("librispeech-clean", "test-clean"),
    ("librispeech-other", "test-other"),
];

const MIRRORS: [&str; 2] = [
    "https://www.openslr.org/resources/12",
    "https://us.openslr.org/resources/12",
];

/// One utterance of a split, as its transcript and FLAC header describe it.
struct Utterance {
    /// `<speaker>-<chapter>-<utterance>`.
    name: String,
    speaker: String,
    chapter: String,
    text: String,
    seconds: f64,
}

/// What a clip of the split is for, once chosen.
struct Plan {
    tier: Tier,
    purposes: Vec<Purpose>,
}

fn fetch(fetch: &mut Fetch, tier: Tier) -> Result<BTreeMap<String, Vec<Clip>>> {
    let mut found = BTreeMap::new();
    for (set, split) in SPLITS {
        let dir = fetch.dir(&format!("librispeech/{split}"))?;
        extract(fetch, split, &dir)?;
        let utterances = utterances(&dir)?;
        let clips = records(fetch, set, &dir, &utterances, tier)
            .with_context(|| format!("preparing {set}"))?;
        found.insert(set.to_owned(), clips);
    }
    Ok(found)
}

/// Downloads and extracts `split` into `dir` unless its marker says that's done.
fn extract(fetch: &Fetch, split: &str, dir: &Path) -> Result<()> {
    let marker = dir.with_extension("done");
    if marker.is_file() {
        return Ok(());
    }
    let mut errors = Vec::new();
    for base in MIRRORS {
        let url = format!("{base}/{split}.tar.gz");
        match extract_from(fetch, &url, dir) {
            Ok(sha256) => {
                fs::write(&marker, format!("{sha256}  {url}\n"))?;
                return Ok(());
            }
            Err(error) => {
                eprintln!("  {url}: {error:#}");
                errors.push(format!("{error:#}"));
            }
        }
    }
    bail!("could not fetch {split}: {}", errors.join("; "))
}

/// Streams the archive at `url`, writing its FLAC files and transcripts flat into `dir`, and
/// returns the archive's SHA-256.
fn extract_from(fetch: &Fetch, url: &str, dir: &Path) -> Result<String> {
    eprintln!("  downloading and extracting {url}");
    let reader = Hashing {
        inner: fetch.open(url)?,
        hash: Sha256::new(),
    };
    let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(reader));
    let mut count = 0;
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let wanted = name.ends_with(".flac") || name.ends_with(".trans.txt");
        // Only plain names land on disk, whatever the archive says.
        let plain = name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.');
        if !wanted || !plain {
            continue;
        }
        let dest = dir.join(name);
        let partial = dir.join(format!("{name}.partial"));
        io::copy(&mut entry, &mut fs::File::create(&partial)?)
            .with_context(|| format!("extracting {name}"))?;
        fs::rename(&partial, &dest)?;
        count += 1;
        if count % 500 == 0 {
            eprintln!("  {count} files");
        }
    }
    // Read whatever follows the tar's end so the hash covers the whole download.
    let mut rest = archive.into_inner().into_inner();
    io::copy(&mut rest, &mut io::sink())?;
    if count == 0 {
        bail!("{url} held no FLAC files");
    }
    eprintln!("  {count} files");
    Ok(format!("{:x}", rest.hash.finalize()))
}

/// A reader that hashes what passes through it.
struct Hashing<R> {
    inner: R,
    hash: Sha256,
}

impl<R: Read> Read for Hashing<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.hash.update(&buf[..n]);
        Ok(n)
    }
}

/// Every utterance extracted into `dir`, in id order, with its transcript and length.
fn utterances(dir: &Path) -> Result<Vec<Utterance>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        let is_transcript = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with(".trans.txt"));
        if !is_transcript {
            continue;
        }
        for line in fs::read_to_string(&path)?.lines() {
            let Some((name, text)) = line.trim().split_once(' ') else {
                continue;
            };
            let mut parts = name.split('-');
            let (Some(speaker), Some(chapter)) = (parts.next(), parts.next()) else {
                continue;
            };
            let flac = dir.join(format!("{name}.flac"));
            let reader = claxon::FlacReader::open(&flac)
                .with_context(|| format!("opening {}", flac.display()))?;
            let samples = reader.streaminfo().samples.unwrap_or(0);
            out.push(Utterance {
                name: name.to_owned(),
                speaker: speaker.to_owned(),
                chapter: chapter.to_owned(),
                text: text.trim().to_owned(),
                seconds: samples as f64 / audio::RATE as f64,
            });
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    if out.is_empty() {
        bail!("no transcripts in {}", dir.display());
    }
    Ok(out)
}

/// The tier and purposes of every utterance of a split, by index.
///
/// micro: one 5–12 s utterance (clean only); smoke: 10, one of them a 15–25 s probe and five
/// of them timed; quick: 200, with a second probe; full: the rest. Picks beyond the fixed ones
/// are stratified across speakers and lengths.
fn plan_utterances(set: &str, utterances: &[Utterance]) -> Vec<Plan> {
    let mut plans: Vec<Option<Plan>> = utterances.iter().map(|_| None).collect();
    let take = |i: usize, tier: Tier, plans: &mut Vec<Option<Plan>>| {
        plans[i] = Some(Plan {
            tier,
            purposes: vec![Purpose::Asr],
        });
    };
    if set == "librispeech-clean"
        && let Some(i) = utterances
            .iter()
            .position(|u| (5.0..=12.0).contains(&u.seconds))
    {
        take(i, Tier::Micro, &mut plans);
    }
    // Two probes of 15–25 s from different speakers.
    let probe = |u: &Utterance| (15.0..=25.0).contains(&u.seconds);
    let first = utterances
        .iter()
        .enumerate()
        .position(|(i, u)| probe(u) && plans[i].is_none());
    let second = first.and_then(|f| {
        utterances
            .iter()
            .enumerate()
            .position(|(i, u)| probe(u) && plans[i].is_none() && u.speaker != utterances[f].speaker)
    });
    // The first is in micro too: one window for iterating on a kernel.
    let micro = if set == "librispeech-clean" {
        Tier::Micro
    } else {
        Tier::Smoke
    };
    for (i, tier) in [(first, micro), (second, Tier::Quick)] {
        if let Some(i) = i {
            take(i, tier, &mut plans);
            plans[i].as_mut().unwrap().purposes.push(Purpose::Probe);
        }
    }
    for (tier, total) in [(Tier::Smoke, 10usize), (Tier::Quick, 200)] {
        let have = plans.iter().flatten().filter(|p| p.tier <= tier).count();
        for i in stratified(utterances, &plans, total.saturating_sub(have)) {
            take(i, tier, &mut plans);
        }
    }
    // Five smoke utterances, in id order, are timed.
    let mut timed = 0;
    for plan in plans.iter_mut().flatten() {
        if plan.tier <= Tier::Smoke && timed < 5 {
            plan.purposes.push(Purpose::Speed);
            timed += 1;
        }
    }
    plans
        .into_iter()
        .map(|p| {
            p.unwrap_or(Plan {
                tier: Tier::Full,
                purposes: vec![Purpose::Asr],
            })
        })
        .collect()
}

/// `count` utterances not yet planned, spread across speakers and, within the picks, across
/// lengths: with fewer picks than speakers, evenly spaced speakers each give the utterance at
/// a rising length quantile; otherwise each speaker gives its share at evenly spaced lengths.
fn stratified(utterances: &[Utterance], plans: &[Option<Plan>], count: usize) -> Vec<usize> {
    let mut by_speaker: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, u) in utterances.iter().enumerate() {
        if plans[i].is_none() {
            by_speaker.entry(&u.speaker).or_default().push(i);
        }
    }
    for list in by_speaker.values_mut() {
        list.sort_by(|&a, &b| utterances[a].seconds.total_cmp(&utterances[b].seconds));
    }
    let speakers: Vec<&Vec<usize>> = by_speaker.values().collect();
    let n = speakers.len();
    if count == 0 || n == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    if count < n {
        for k in 0..count {
            let list = speakers[k * n / count];
            out.push(list[(2 * k + 1) * list.len() / (2 * count)]);
        }
    } else {
        for (s, list) in speakers.iter().enumerate() {
            let share = (count / n + usize::from(s < count % n)).min(list.len());
            for j in 0..share {
                out.push(list[(2 * j + 1) * list.len() / (2 * share)]);
            }
        }
    }
    out
}

/// The chapters joined into long-form clips: of those running 5–15 minutes, six evenly
/// spaced, two at quick (the first timed) and four at full. Keys are `<speaker>-<chapter>`.
fn plan_chapters(utterances: &[Utterance]) -> Vec<(String, Plan)> {
    let mut chapters: BTreeMap<String, f64> = BTreeMap::new();
    for u in utterances {
        *chapters
            .entry(format!("{}-{}", u.speaker, u.chapter))
            .or_default() += u.seconds;
    }
    let candidates: Vec<&String> = chapters
        .iter()
        .filter(|(_, s)| (300.0..=900.0).contains(*s))
        .map(|(k, _)| k)
        .collect();
    let n = candidates.len();
    let mut out: Vec<(String, Plan)> = Vec::new();
    for i in 0..6.min(n) {
        let key = candidates[i * n / 6.min(n)];
        let tier = if i == 0 || i == 3 {
            Tier::Quick
        } else {
            Tier::Full
        };
        let mut purposes = vec![Purpose::Asr];
        if i == 0 {
            purposes.push(Purpose::Speed);
        }
        out.push((key.clone(), Plan { tier, purposes }));
    }
    out
}

/// What a clip record needs before its audio is measured.
struct Pending {
    id: String,
    audio: PathBuf,
    plan: Plan,
    text: String,
    tags: &'static [&'static str],
    /// For a chapter, the utterances to join when its file is missing.
    parts: Vec<PathBuf>,
}

/// The clip records of the split: every utterance and chapter of `tier` and below, plus any
/// a previous, bigger fetch recorded.
fn records(
    fetch: &Fetch,
    set: &str,
    dir: &Path,
    utterances: &[Utterance],
    tier: Tier,
) -> Result<Vec<Clip>> {
    let mut pending = Vec::new();
    for (u, plan) in utterances.iter().zip(plan_utterances(set, utterances)) {
        pending.push(Pending {
            id: format!("{set}/{}", u.name),
            audio: dir.join(format!("{}.flac", u.name)),
            plan,
            text: u.text.clone(),
            tags: &["read"],
            parts: Vec::new(),
        });
    }
    for (key, plan) in plan_chapters(utterances) {
        let parts: Vec<&Utterance> = utterances
            .iter()
            .filter(|u| format!("{}-{}", u.speaker, u.chapter) == key)
            .collect();
        pending.push(Pending {
            id: format!("{set}/chapter-{key}"),
            audio: dir.join(format!("chapter-{key}.flac")),
            plan,
            text: parts
                .iter()
                .map(|u| u.text.as_str())
                .collect::<Vec<_>>()
                .join(" "),
            tags: &["read", "long-form"],
            parts: parts
                .iter()
                .map(|u| dir.join(format!("{}.flac", u.name)))
                .collect(),
        });
    }

    let mut clips = Vec::new();
    let mut measure = Vec::new();
    for p in pending {
        let truth = Truth {
            text: Some(p.text.clone()),
            verbatim: true,
            ..Truth::default()
        };
        if let Some(mut clip) = fetch.known(&p.id) {
            // Keep the measured audio, refresh what the plan says.
            clip.tier = p.plan.tier;
            clip.purposes = p.plan.purposes;
            clip.truth = truth;
            clip.tags = p.tags.iter().map(|t| t.to_string()).collect();
            clips.push(clip);
        } else if p.plan.tier <= tier || (p.audio.is_file() && !p.parts.is_empty()) {
            measure.push((p, truth));
        }
    }

    // Joining chapters and hashing thousands of files: spread over the cores.
    let next = AtomicUsize::new(0);
    let made = Mutex::new(Vec::new());
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some((p, truth)) = measure.get(i) else {
                        break;
                    };
                    let clip = make(fetch, set, p, truth.clone())
                        .with_context(|| format!("preparing {}", p.id));
                    made.lock().unwrap().push(clip);
                }
            });
        }
    });
    for clip in made.into_inner().unwrap() {
        clips.push(clip?);
    }
    clips.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(clips)
}

/// Joins a missing chapter, then measures the clip.
fn make(fetch: &Fetch, set: &str, p: &Pending, truth: Truth) -> Result<Clip> {
    if !p.audio.is_file() {
        if p.parts.is_empty() {
            bail!("{} is missing", p.audio.display());
        }
        eprintln!("  joining {}", p.id);
        let mut pcm = Vec::new();
        for part in &p.parts {
            pcm.extend(audio::read(part)?);
        }
        super::util::write_flac(&p.audio, &pcm)?;
    }
    fetch.clip(
        p.id.clone(),
        set,
        &p.audio,
        p.plan.tier,
        "en",
        &p.plan.purposes,
        truth,
        p.tags,
    )
}
