//! Open ASR Leaderboard test sets and long-form sets from the Hugging Face Hub, with references.
//!
//! Short sets (the leaderboard's own test data, `hf-audio/open-asr-leaderboard`, each sorted by
//! length): `earnings` (Earnings-22 earnings calls, CC BY-SA 4.0), `voxpopuli` (European
//! Parliament speeches, CC0; the leaderboard's re-transcribed "cleaned" subset) and `ami-ihm`
//! (AMI meeting utterances on headset microphones, CC BY 4.0, cleaned). A hundred utterances of
//! each, evenly spaced through the sorted split so every length is represented. Left out: the
//! leaderboard's TED-LIUM (its files are gone from the repository; the long-form talks below
//! cover TED), its uncleaned VoxPopuli (the datasets-server can't serve its rows), and the gated
//! or restrictively licensed sets (GigaSpeech, SPGISpeech).
//!
//! `longform`: whole recordings with their full references, where windowing and long-audio
//! behaviour show. TED talks from `distil-whisper/tedlium-long-form`, and Earnings-21 calls from
//! Rev's own repository (MP3 and references, CC BY-SA 4.0; the Hub copy only comes as parquet).
//! Rev16 is left out: it exists only as parquet, and its podcast audio has no clear licence.
//!
//! Rows come from the Hub's datasets-server, which hands out each row's audio as a WAV link, so
//! nothing needs a parquet reader.

use super::ami::{rttm, speaker_count};
use super::{Clip, Purpose, Source, Tier, Truth, util::Fetch};
use anyhow::{Context, Result, ensure};
use std::{collections::BTreeMap, fs, path::Path, time::Duration};

pub const SOURCE: Source = Source {
    sets: &["earnings", "voxpopuli", "ami-ihm", "longform"],
    about: "leaderboard and long-form test sets",
    fetch,
};

const SERVER: &str = "https://datasets-server.huggingface.co";
const LEADERBOARD: &str = "hf-audio/open-asr-leaderboard";
/// (set, leaderboard config).
const SHORT: &[(&str, &str)] = &[
    ("earnings", "earnings22"),
    ("voxpopuli", "voxpopuli_cleaned_aa"),
    ("ami-ihm", "ami_cleaned"),
];
const PICKS: usize = 100;

/// (row, speaker id, tier) of `distil-whisper/tedlium-long-form`'s test split: four talks of 6 to
/// 16 minutes.
const TED_TALKS: &[(usize, &str, Tier)] = &[
    (6, "GaryFlake", Tier::Quick),
    (2, "DanBarber", Tier::Full),
    (5, "EricMead_2009P_EricMead", Tier::Full),
    (10, "RobertGupta", Tier::Full),
];
/// revdotcom/speech-datasets, pinned.
const EARNINGS21: &str = "https://raw.githubusercontent.com/revdotcom/speech-datasets/c05ab6fd8b4b627d123c922a22a39e993dd37635/earnings21";
/// The four shortest Earnings-21 calls (18 to 25 minutes, five to eight speakers); the first is
/// also timed.
const CALLS: &[(&str, Tier)] = &[
    ("4386541", Tier::Quick),
    ("4387332", Tier::Full),
    ("4392809", Tier::Full),
    ("4384683", Tier::Full),
];

fn fetch(fetch: &mut Fetch, tier: Tier) -> Result<BTreeMap<String, Vec<Clip>>> {
    let mut sets = BTreeMap::new();
    for &(set, config) in SHORT {
        let clips = leaderboard(fetch, tier, set, config).unwrap_or_else(|error| {
            eprintln!("  {set} skipped: {error:#}");
            fetch.clips_of(set).into_iter().cloned().collect()
        });
        sets.insert(set.to_owned(), clips);
    }
    sets.insert("longform".to_owned(), long_form(fetch, tier));
    Ok(sets)
}

/// The tier of the `k`th of the hundred picks: 3 in smoke, 30 in quick. Picks run from the
/// longest utterance to the shortest, so smoke's are short.
fn pick_tier(k: usize) -> Tier {
    match k {
        30 | 60 | 90 => Tier::Smoke,
        _ if [0, 3, 6].contains(&(k % 10)) => Tier::Quick,
        _ => Tier::Full,
    }
}

fn rows_url(dataset: &str, config: &str, split: &str, offset: usize) -> String {
    format!(
        "{SERVER}/rows?dataset={dataset}&config={config}&split={split}&offset={offset}&length=1"
    )
}

fn leaderboard(fetch: &Fetch, tier: Tier, set: &str, config: &str) -> Result<Vec<Clip>> {
    let size = patiently(|| {
        fetch.get_json(&format!(
            "{SERVER}/size?dataset={LEADERBOARD}&config={config}"
        ))
    })?;
    let rows = size["size"]["splits"]
        .as_array()
        .and_then(|s| s.iter().find(|s| s["split"] == "test"))
        .and_then(|s| s["num_rows"].as_u64())
        .with_context(|| format!("no test split size for {config}"))? as usize;
    ensure!(rows >= PICKS, "{config} has only {rows} rows");
    let dir = fetch.dir(&format!("hf/{set}"))?;
    let mut clips = Vec::new();
    for k in 0..PICKS {
        // The middle of each hundredth of the split.
        let row = (2 * k + 1) * rows / (2 * PICKS);
        let id = format!("{set}/{row:04}");
        let clip_tier = pick_tier(k);
        let flac = dir.join(format!("{row:04}.flac"));
        if let Some(mut clip) = fetch.known(&id) {
            clip.tier = clip_tier;
            clips.push(clip);
            continue;
        }
        if clip_tier > tier {
            continue;
        }
        let made = (|| -> Result<Clip> {
            let row = row_of(fetch, &rows_url(LEADERBOARD, config, "test", row))?;
            let text = row["text"]
                .as_str()
                .context("row without text")?
                .trim()
                .to_owned();
            let utterance = row["id"].as_str().unwrap_or_default().to_owned();
            wav_to_flac(fetch, &row, &flac)?;
            let utterance = format!("utt:{utterance}");
            fetch.clip(
                id.clone(),
                set,
                &flac,
                clip_tier,
                "en",
                &[Purpose::Asr],
                Truth {
                    text: Some(text),
                    verbatim: true,
                    ..Truth::default()
                },
                &[&format!("esb:{config}"), &utterance],
            )
        })();
        match made {
            Ok(clip) => clips.push(clip),
            Err(error) => eprintln!("  {id} skipped: {error:#}"),
        }
    }
    Ok(clips)
}

/// Runs `request` again, after a growing wait, while the server says too many requests (429)
/// or that it's busy (502, 503): the datasets-server allows each client only so many a minute,
/// and the AMI mirror sheds load. Downloads resume where they stopped.
pub(super) fn patiently<T>(mut request: impl FnMut() -> Result<T>) -> Result<T> {
    let mut wait = 5;
    loop {
        match request() {
            Err(error) if wait <= 160 && passing(&error) => {
                eprintln!("  server busy; waiting {wait} s");
                std::thread::sleep(Duration::from_secs(wait));
                wait *= 2;
            }
            result => return result,
        }
    }
}

fn passing(error: &anyhow::Error) -> bool {
    error.chain().any(|e| {
        matches!(
            e.downcast_ref::<ureq::Error>(),
            Some(ureq::Error::Status(429 | 502 | 503, _))
        )
    })
}

/// The one row a datasets-server `rows` request for one row returns.
fn row_of(fetch: &Fetch, url: &str) -> Result<serde_json::Value> {
    let mut page = patiently(|| fetch.get_json(url))?;
    if let Some(error) = page["error"].as_str() {
        anyhow::bail!("{url}: {error}");
    }
    Ok(page["rows"][0]["row"].take())
}

/// Downloads a row's audio (a signed WAV link that expires) and converts it.
fn wav_to_flac(fetch: &Fetch, row: &serde_json::Value, flac: &Path) -> Result<()> {
    let src = row["audio"][0]["src"]
        .as_str()
        .context("row without audio")?;
    let wav = flac.with_extension("wav");
    patiently(|| fetch.download(src, &wav, None))?;
    let result = super::util::to_flac(&wav, flac, None, None);
    fs::remove_file(&wav)?;
    result
}

fn long_form(fetch: &Fetch, tier: Tier) -> Vec<Clip> {
    let mut clips = Vec::new();
    let Ok(dir) = fetch.dir("hf/longform") else {
        return clips;
    };
    for &(row, speaker, clip_tier) in TED_TALKS {
        let id = format!("longform/ted-{speaker}");
        let flac = dir.join(format!("ted-{speaker}.flac"));
        clips.extend(long_clip(fetch, tier, &id, clip_tier, || {
            let url = rows_url("distil-whisper/tedlium-long-form", "default", "test", row);
            let row = row_of(fetch, &url)?;
            ensure!(
                row["speaker_id"] == speaker,
                "tedlium-long-form row isn't {speaker}'s talk"
            );
            let text = row["text"]
                .as_str()
                .context("row without text")?
                .trim()
                .to_owned();
            if !flac.is_file() {
                wav_to_flac(fetch, &row, &flac)?;
            }
            fetch.clip(
                id.clone(),
                "longform",
                &flac,
                clip_tier,
                "en",
                &[Purpose::Asr],
                Truth {
                    text: Some(text),
                    verbatim: true,
                    ..Truth::default()
                },
                &["long-form", "talk", "tedlium-long-form"],
            )
        }));
    }
    for (i, &(call, clip_tier)) in CALLS.iter().enumerate() {
        let id = format!("longform/earnings21-{call}");
        let flac = dir.join(format!("earnings21-{call}.flac"));
        clips.extend(long_clip(fetch, tier, &id, clip_tier, || {
            if !flac.is_file() {
                let mp3 = flac.with_extension("mp3");
                fetch.download(&format!("{EARNINGS21}/media/{call}.mp3"), &mp3, None)?;
                let result = super::util::to_flac(&mp3, &flac, None, None);
                fs::remove_file(&mp3)?;
                result?;
            }
            let text = nlp_text(&fetch.get_text(&format!(
                "{EARNINGS21}/transcripts/nlp_references/{call}.nlp"
            ))?)?;
            let turns = rttm(&fetch.get_text(&format!("{EARNINGS21}/rttms/{call}.rttm"))?)?;
            let speakers = format!("speakers:{}", speaker_count(&turns));
            let purposes: &[Purpose] = if i == 0 {
                &[Purpose::Asr, Purpose::Speed]
            } else {
                &[Purpose::Asr]
            };
            fetch.clip(
                id.clone(),
                "longform",
                &flac,
                clip_tier,
                "en",
                purposes,
                Truth {
                    text: Some(text),
                    turns: Some(turns),
                    verbatim: true,
                    ..Truth::default()
                },
                &["long-form", "call", "earnings21", &speakers],
            )
        }));
    }
    clips
}

/// A long-form clip: the known record, or a new one when `tier` wants it.
fn long_clip(
    fetch: &Fetch,
    tier: Tier,
    id: &str,
    clip_tier: Tier,
    make: impl FnOnce() -> Result<Clip>,
) -> Option<Clip> {
    if let Some(mut clip) = fetch.known(id) {
        clip.tier = clip_tier;
        return Some(clip);
    }
    if clip_tier > tier {
        return None;
    }
    eprintln!("  {id}");
    make()
        .inspect_err(|error| eprintln!("  {id} skipped: {error:#}"))
        .ok()
}

/// The text of a Rev `.nlp` reference: one token per line, `token|speaker|ts|endTs|punctuation|…`.
fn nlp_text(nlp: &str) -> Result<String> {
    let mut words = Vec::new();
    for line in nlp.lines().skip(1) {
        let fields: Vec<&str> = line.split('|').collect();
        ensure!(fields.len() >= 5, "short .nlp line {line:?}");
        words.push(format!("{}{}", fields[0], fields[4]));
    }
    ensure!(!words.is_empty(), "empty .nlp reference");
    Ok(words.join(" "))
}
