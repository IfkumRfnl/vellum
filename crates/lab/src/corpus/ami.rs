//! AMI Meeting Corpus test meetings (CC BY 4.0), headset mix, with pyannote's RTTM speaker turns.
//!
//! The meeting list, speaker turns (word-aligned, "only_words") and scoring regions come from
//! pyannote's AMI-diarization-setup, the split most diarization papers report on; the audio from
//! the University of Edinburgh's AMI mirror. Each meeting has four speakers (EN2002c has three),
//! within Streaming Sortformer's limit.

use super::hf::patiently;
use super::{Clip, Purpose, Source, Tier, Truth, TruthTurn, util::Fetch};
use anyhow::{Context, Result, ensure};
use std::{collections::BTreeMap, fs, path::Path};

pub const SOURCE: Source = Source {
    sets: &["ami"],
    about: "meetings with speaker turns",
    fetch,
};

/// pyannote/AMI-diarization-setup, pinned so the turns don't change under the goldens.
const SETUP: &str = "https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/67c2d539286e89f68952d5dcf83912bd9f01dfae";
const AUDIO: &str = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus";

/// The first session of three of the four test series: the shortest whole meetings (14–25 min).
const QUICK: &[&str] = &["IS1009a", "ES2004a", "TS3003a"];
/// The smoke crop: three minutes of `QUICK[0]`, from five minutes in, past the introductions.
const CROP: (f64, f64) = (300.0, 180.0);
/// The quick meeting also heard through the table's distant microphone array.
const ARRAY: &str = "ES2004a";

fn fetch(fetch: &mut Fetch, tier: Tier) -> Result<BTreeMap<String, Vec<Clip>>> {
    let meetings: Vec<String> = fetch
        .get_text(&format!("{SETUP}/lists/test.meetings.txt"))?
        .lines()
        .map(|l| l.trim().to_owned())
        .filter(|l| !l.is_empty())
        .collect();
    ensure!(
        meetings.len() == 16,
        "expected 16 AMI test meetings, got {}",
        meetings.len()
    );
    let dir = fetch.dir("ami")?;

    let mut plan = vec![Recording {
        id: format!("ami/{}-3min", QUICK[0]),
        meeting: QUICK[0].to_owned(),
        channel: "Mix-Headset",
        crop: Some(CROP),
        tier: Tier::Smoke,
        purposes: vec![Purpose::Diarization],
        tags: &["meeting", "headset-mix", "crop"],
    }];
    for meeting in meetings {
        let quick = QUICK.contains(&meeting.as_str());
        let mut purposes = vec![Purpose::Diarization];
        if meeting == QUICK[0] {
            purposes.push(Purpose::Speed);
        }
        plan.push(Recording {
            id: format!("ami/{meeting}"),
            meeting,
            channel: "Mix-Headset",
            crop: None,
            tier: if quick { Tier::Quick } else { Tier::Full },
            purposes,
            tags: &["meeting", "headset-mix"],
        });
    }
    plan.push(Recording {
        id: format!("ami/{ARRAY}-array"),
        meeting: ARRAY.to_owned(),
        channel: "Array1-01",
        crop: None,
        tier: Tier::Quick,
        purposes: vec![Purpose::Diarization],
        tags: &["meeting", "far-field"],
    });

    let mut clips = Vec::new();
    for recording in plan {
        let Recording {
            id,
            meeting,
            channel,
            crop,
            ..
        } = &recording;
        let flac = match crop {
            Some(_) => dir.join(format!("{meeting}-3min.flac")),
            None if *channel == "Mix-Headset" => dir.join(format!("{meeting}.flac")),
            None => dir.join(format!("{meeting}.{channel}.flac")),
        };
        if let Some(mut clip) = fetch.known(id) {
            (clip.tier, clip.purposes) = (recording.tier, recording.purposes.clone());
            clips.push(clip);
            continue;
        }
        if recording.tier > tier && !flac.is_file() {
            continue;
        }
        let made = (|| -> Result<Clip> {
            if !flac.is_file() {
                eprintln!("  {id}");
                let whole = dir.join(format!("{meeting}.flac"));
                if crop.is_some() && whole.is_file() {
                    super::util::to_flac(&whole, &flac, Some(CROP.0), Some(CROP.1))?;
                } else {
                    let wav = dir.join(format!("{meeting}.{channel}.wav"));
                    let url = format!("{AUDIO}/{meeting}/audio/{meeting}.{channel}.wav");
                    patiently(|| fetch.download(&url, &wav, None))?;
                    let result =
                        super::util::to_flac(&wav, &flac, crop.map(|c| c.0), crop.map(|c| c.1));
                    fs::remove_file(&wav)?;
                    result?;
                }
            }
            meeting_clip(fetch, &recording, &flac)
        })();
        match made {
            Ok(clip) => clips.push(clip),
            Err(error) => eprintln!("  {id} skipped: {error:#}"),
        }
    }
    Ok(BTreeMap::from([("ami".to_owned(), clips)]))
}

/// One clip to make: a meeting, a microphone, and maybe a crop (start, seconds).
struct Recording {
    id: String,
    meeting: String,
    /// `Mix-Headset` or `Array1-01`, as in the mirror's file names.
    channel: &'static str,
    crop: Option<(f64, f64)>,
    tier: Tier,
    purposes: Vec<Purpose>,
    tags: &'static [&'static str],
}

/// The clip record of one meeting's audio, with its turns limited to the scoring region (and to
/// the crop, shifted so the crop starts at zero).
fn meeting_clip(fetch: &Fetch, recording: &Recording, flac: &Path) -> Result<Clip> {
    let meeting = &recording.meeting;
    let turns = rttm(&fetch.get_text(&format!("{SETUP}/only_words/rttms/test/{meeting}.rttm"))?)?;
    let uem = fetch.get_text(&format!("{SETUP}/uems/test/{meeting}.uem"))?;
    let fields: Vec<&str> = uem.split_whitespace().collect();
    ensure!(fields.len() >= 4, "{meeting}.uem: unexpected {uem:?}");
    let (from, to): (f64, f64) = (fields[2].parse()?, fields[3].parse()?);
    let mut turns = clip_turns(&turns, from, to, 0.0);
    let (start, seconds) = recording.crop.unwrap_or((0.0, f64::INFINITY));
    turns = clip_turns(&turns, start, start + seconds, start);
    let speakers = speaker_count(&turns);
    let mut clip = fetch.clip(
        recording.id.clone(),
        "ami",
        flac,
        recording.tier,
        "en",
        &recording.purposes,
        Truth {
            turns: Some(turns),
            verbatim: true,
            ..Truth::default()
        },
        recording.tags,
    )?;
    // Say when the scoring region isn't the whole clip, so scoring can ignore the rest.
    let (from, to) = ((from - start).max(0.0), (to - start).min(clip.seconds));
    if from > 0.01 || to < clip.seconds - 0.5 {
        clip.tags.push(format!("uem:{from:.3}-{to:.3}"));
    }
    clip.tags.push(format!("speakers:{speakers}"));
    Ok(clip)
}

/// Speaker turns from RTTM `SPEAKER <file> <channel> <start> <duration> … <speaker> …` lines,
/// in time order.
pub(super) fn rttm(text: &str) -> Result<Vec<TruthTurn>> {
    let mut turns = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.first() != Some(&"SPEAKER") {
            continue;
        }
        ensure!(f.len() >= 8, "short RTTM line {line:?}");
        let t0: f64 = f[3]
            .parse()
            .with_context(|| format!("RTTM line {line:?}"))?;
        let duration: f64 = f[4]
            .parse()
            .with_context(|| format!("RTTM line {line:?}"))?;
        turns.push(TruthTurn {
            t0,
            t1: ((t0 + duration) * 1000.0).round() / 1000.0,
            speaker: f[7].to_owned(),
        });
    }
    ensure!(!turns.is_empty(), "RTTM without turns");
    turns.sort_by(|a, b| a.t0.total_cmp(&b.t0));
    Ok(turns)
}

/// The parts of `turns` within `from..to`, moved `shift` seconds earlier (to the millisecond,
/// as RTTMs are, rather than carrying float noise into the index).
pub(super) fn clip_turns(turns: &[TruthTurn], from: f64, to: f64, shift: f64) -> Vec<TruthTurn> {
    let ms = |t: f64| (t * 1000.0).round() / 1000.0;
    turns
        .iter()
        .filter(|t| t.t1 > from && t.t0 < to)
        .map(|t| TruthTurn {
            t0: ms(t.t0.max(from) - shift),
            t1: ms(t.t1.min(to) - shift),
            speaker: t.speaker.clone(),
        })
        .filter(|t| t.t1 - t.t0 > 1e-3)
        .collect()
}

pub(super) fn speaker_count(turns: &[TruthTurn]) -> usize {
    let mut names: Vec<&str> = turns.iter().map(|t| t.speaker.as_str()).collect();
    names.sort();
    names.dedup();
    names.len()
}
