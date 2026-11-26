//! VoxConverse test recordings with four or fewer speakers (CC BY 4.0): YouTube-style audio with RTTM.
//!
//! News segments and debates from YouTube, the closest public match to the app's speaker
//! labelling. The RTTMs are the authors' (v0.3, pinned). The authors' audio archive now needs a
//! password, so the audio comes from a CC BY 4.0 mirror on the Hugging Face Hub that keeps one
//! 16 kHz WAV per recording (pinned too), which lets us take only the recordings we use.

use super::ami::{rttm, speaker_count};
use super::{Clip, Purpose, Source, Tier, Truth, util::Fetch};
use anyhow::{Context, Result, ensure};
use std::{collections::BTreeMap, fs, io::Read};

pub const SOURCE: Source = Source {
    sets: &["voxconverse"],
    about: "YouTube-style speaker turns",
    fetch,
};

const RTTMS: &str = "https://codeload.github.com/joonson/voxconverse/tar.gz/24bf60be297701cd7e4ef18550c6d390c1b87365";
const MIRROR: &str = "https://huggingface.co/datasets/ggfox00000/dia-voxconverse-test";
const REVISION: &str = "7e3ea2807400b9ed79451e20513f7a50ac4fd846";

/// What a recording must be to be chosen: Streaming Sortformer labels up to four speakers, and
/// one speaker says nothing about telling them apart.
const SPEAKERS: std::ops::RangeInclusive<usize> = 2..=4;
const SECONDS: std::ops::RangeInclusive<f64> = 60.0..=900.0;
const FULL: usize = 10;
/// The picks in quick: eguui, laoyl and oubab, with three, two and four speakers.
const QUICK: &[usize] = &[2, 4, 7];

fn fetch(fetch: &mut Fetch, tier: Tier) -> Result<BTreeMap<String, Vec<Clip>>> {
    let rttms = test_rttms(fetch)?;
    // Lengths from the mirror's file sizes: 16 kHz 16-bit mono after a short header.
    let listing = fetch.get_json(&format!(
        "{}/tree/{REVISION}/audio/test",
        MIRROR.replace("/datasets/", "/api/datasets/")
    ))?;
    let mut sizes = BTreeMap::new();
    for entry in listing.as_array().context("mirror listing isn't a list")? {
        let (Some(path), Some(size)) = (entry["path"].as_str(), entry["size"].as_u64()) else {
            continue;
        };
        if let Some(name) = path
            .strip_prefix("audio/test/")
            .and_then(|p| p.strip_suffix(".wav"))
        {
            sizes.insert(name.to_owned(), size);
        }
    }

    // Every eligible recording in name order (the names are random), then evenly spaced picks.
    let eligible: Vec<(&String, usize)> = rttms
        .iter()
        .filter_map(|(name, text)| {
            let speakers = speaker_count(&rttm(text).ok()?);
            let seconds = (*sizes.get(name)? as f64 - 44.0) / 32000.0;
            (SPEAKERS.contains(&speakers) && SECONDS.contains(&seconds)).then_some((name, speakers))
        })
        .collect();
    ensure!(
        eligible.len() >= FULL,
        "only {} eligible VoxConverse recordings",
        eligible.len()
    );
    let chosen: Vec<_> = (0..FULL)
        .map(|i| eligible[i * eligible.len() / FULL])
        .collect();

    let dir = fetch.dir("voxconverse")?;
    let mut clips = Vec::new();
    for (i, (name, speakers)) in chosen.into_iter().enumerate() {
        let id = format!("voxconverse/{name}");
        let clip_tier = if QUICK.contains(&i) {
            Tier::Quick
        } else {
            Tier::Full
        };
        let flac = dir.join(format!("{name}.flac"));
        if let Some(mut clip) = fetch.known(&id) {
            clip.tier = clip_tier;
            clips.push(clip);
            continue;
        }
        if clip_tier > tier && !flac.is_file() {
            continue;
        }
        let made = (|| -> Result<Clip> {
            if !flac.is_file() {
                eprintln!("  {id}");
                let wav = dir.join(format!("{name}.wav"));
                fetch.download(
                    &format!("{MIRROR}/resolve/{REVISION}/audio/test/{name}.wav"),
                    &wav,
                    None,
                )?;
                let result = super::util::to_flac(&wav, &flac, None, None);
                fs::remove_file(&wav)?;
                result?;
            }
            let turns = rttm(&rttms[name])?;
            let speakers = format!("speakers:{speakers}");
            fetch.clip(
                id.clone(),
                "voxconverse",
                &flac,
                clip_tier,
                "en",
                &[Purpose::Diarization],
                Truth {
                    turns: Some(turns),
                    verbatim: true,
                    ..Truth::default()
                },
                &["broadcast", &speakers],
            )
        })();
        match made {
            Ok(clip) => clips.push(clip),
            Err(error) => eprintln!("  {id} skipped: {error:#}"),
        }
    }
    Ok(BTreeMap::from([("voxconverse".to_owned(), clips)]))
}

/// The test set's RTTMs by recording name, from the repository's tarball (a quarter of a MB).
fn test_rttms(fetch: &Fetch) -> Result<BTreeMap<String, String>> {
    let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(fetch.open(RTTMS)?));
    let mut rttms = BTreeMap::new();
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.to_string_lossy().into_owned();
        let Some(name) = path
            .split_once("/test/")
            .and_then(|(_, file)| file.strip_suffix(".rttm"))
            .map(str::to_owned)
        else {
            continue;
        };
        let mut text = String::new();
        entry.read_to_string(&mut text)?;
        rttms.insert(name, text);
    }
    ensure!(
        rttms.len() > 200,
        "only {} test RTTMs in {RTTMS}",
        rttms.len()
    );
    Ok(rttms)
}
