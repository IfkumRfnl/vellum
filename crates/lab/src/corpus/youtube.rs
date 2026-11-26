//! The app's own live-test videos: a TED talk, two BBC 6 Minute English episodes with scripts, a 2.5-hour podcast with a transcript.
//!
//! The same videos `tests/app/live` checks the app on, fetched the way the app fetches them
//! (yt-dlp, best audio). References are edited, not verbatim: the uploader's captions for the
//! talk, the published scripts and transcript for the rest, which also name each line's speaker.
//! YouTube can't be relied on to serve a video forever, so the FLAC is kept once made.

use super::{Clip, Purpose, Source, Tier, Truth, util::Fetch};
use anyhow::{Context, Result, bail, ensure};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

pub const SOURCE: Source = Source {
    sets: &["youtube"],
    about: "the app's YouTube cases",
    fetch,
};

/// Where a video's reference comes from.
enum Reference {
    /// The uploader's English captions (not YouTube's automatic ones).
    Captions,
    /// A BBC Learning English script page.
    Bbc(&'static str),
    /// A Lex Fridman Podcast transcript page.
    Lex(&'static str),
}

struct Video {
    name: &'static str,
    url: &'static str,
    tier: Tier,
    purposes: &'static [Purpose],
    tags: &'static [&'static str],
    reference: Reference,
}

const VIDEOS: &[Video] = &[
    Video {
        name: "ted-30-days",
        url: "https://youtu.be/JnfBXjWm7hc",
        tier: Tier::Quick,
        purposes: &[Purpose::Asr, Purpose::Speed],
        tags: &["talk", "captions"],
        reference: Reference::Captions,
    },
    Video {
        name: "bbc-food-waste",
        url: "https://youtu.be/Z51Q29u4CWc",
        tier: Tier::Quick,
        purposes: &[Purpose::Asr, Purpose::Diarization],
        tags: &["conversation", "script"],
        reference: Reference::Bbc(
            "https://www.bbc.co.uk/learningenglish/english/features/6-minute-english_2022/ep-221229",
        ),
    },
    Video {
        name: "bbc-doing-nothing",
        url: "https://youtu.be/Y681hXWwhQY",
        tier: Tier::Quick,
        purposes: &[Purpose::Asr, Purpose::Diarization],
        tags: &["conversation", "script"],
        reference: Reference::Bbc(
            "https://www.bbc.co.uk/learningenglish/english/features/6-minute-english_2023/ep-230615",
        ),
    },
    Video {
        name: "lex-jensen-huang",
        url: "https://www.youtube.com/watch?v=vif8NQcjVf0",
        tier: Tier::Long,
        purposes: &[Purpose::Asr, Purpose::Diarization, Purpose::Speed],
        tags: &["podcast", "transcript"],
        reference: Reference::Lex("https://lexfridman.com/jensen-huang-transcript/"),
    },
];

/// The talk's one-window probe: this long, from the middle, cut at caption boundaries.
const PROBE: (&str, f64) = ("ted-30-days-probe", 25.0);

fn fetch(fetch: &mut Fetch, tier: Tier) -> Result<BTreeMap<String, Vec<Clip>>> {
    let dir = fetch.dir("youtube")?;
    let mut clips = Vec::new();
    for video in VIDEOS {
        let id = format!("youtube/{}", video.name);
        let flac = dir.join(format!("{}.flac", video.name));
        let probe = matches!(video.reference, Reference::Captions).then(|| {
            (
                format!("youtube/{}", PROBE.0),
                dir.join(format!("{}.flac", PROBE.0)),
            )
        });
        let known = fetch.known(&id);
        let probe_known = probe.as_ref().and_then(|(id, _)| fetch.known(id));
        let wanted = video.tier <= tier || (probe.is_some() && Tier::Smoke <= tier);
        let complete = known.is_some() && (probe.is_none() || probe_known.is_some());
        if complete || (!wanted && !flac.is_file()) {
            clips.extend(known.map(|mut c| {
                (c.tier, c.purposes) = (video.tier, video.purposes.to_vec());
                c
            }));
            clips.extend(probe_known);
            continue;
        }
        eprintln!("  {id}");
        match make(fetch, video, &id, &flac, probe.as_ref(), known.clone()) {
            Ok(made) => clips.extend(made),
            Err(error) => {
                eprintln!("  {id} skipped: {error:#}");
                // A failed probe mustn't lose the talk.
                clips.extend(known);
            }
        }
    }
    Ok(BTreeMap::from([("youtube".to_owned(), clips)]))
}

/// The video's clip (and the talk's probe), downloading and converting its audio if need be.
fn make(
    fetch: &Fetch,
    video: &Video,
    id: &str,
    flac: &Path,
    probe: Option<&(String, PathBuf)>,
    known: Option<Clip>,
) -> Result<Vec<Clip>> {
    let dir = flac.parent().context("no directory")?;
    if !flac.is_file() {
        download_audio(dir, video.name, video.url, flac)?;
    }
    let mut clips = Vec::new();
    let captions = match video.reference {
        Reference::Captions => Some(captions(dir, video.name, video.url)?),
        _ => None,
    };
    let truth = match &video.reference {
        Reference::Captions => {
            let text = join(captions.iter().flatten().map(|c| c.2.as_str()));
            Truth {
                text: Some(text),
                ..Truth::default()
            }
        }
        Reference::Bbc(url) => named_lines(bbc_script(fetch, url)?),
        Reference::Lex(url) => named_lines(lex_transcript(fetch, url)?),
    };
    let mut tags = video.tags.to_vec();
    let speakers = truth.lines.as_ref().map(|lines| {
        let mut names: Vec<&str> = lines.iter().map(|l| l.0.as_str()).collect();
        names.sort();
        names.dedup();
        format!("speakers:{}", names.len())
    });
    tags.extend(speakers.as_deref());
    let clip = match known {
        Some(clip) => clip,
        None => fetch.clip(
            id.to_owned(),
            "youtube",
            flac,
            video.tier,
            "en",
            video.purposes,
            truth,
            &tags,
        )?,
    };
    let seconds = clip.seconds;
    clips.push(clip);

    if let (Some((probe_id, probe_flac)), Some(captions)) = (probe, captions) {
        // The first caption from a little before the middle, and those that end within the
        // window after it.
        let first = captions
            .iter()
            .position(|c| c.0 >= seconds / 2.0 - PROBE.1 / 2.0)
            .context("no captions past the middle")?;
        let start = captions[first].0;
        let window: Vec<_> = captions[first..]
            .iter()
            .take_while(|c| c.1 <= start + PROBE.1)
            .collect();
        ensure!(!window.is_empty(), "no caption fits the probe window");
        let end = window.last().map_or(start, |c| c.1);
        super::util::to_flac(flac, probe_flac, Some(start), Some(end - start))?;
        let text = join(window.iter().map(|c| c.2.as_str()));
        let from = format!("from:{start:.3}");
        clips.push(fetch.clip(
            probe_id.clone(),
            "youtube",
            probe_flac,
            Tier::Smoke,
            "en",
            &[Purpose::Probe, Purpose::Asr],
            Truth {
                text: Some(text),
                ..Truth::default()
            },
            &["talk", "captions", "excerpt", &from],
        )?);
    }
    Ok(clips)
}

fn named_lines(lines: Vec<(String, String)>) -> Truth {
    Truth {
        text: Some(join(lines.iter().map(|l| l.1.as_str()))),
        lines: Some(lines),
        ..Truth::default()
    }
}

fn join<'a>(parts: impl Iterator<Item = &'a str>) -> String {
    parts
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// yt-dlp as the app runs it (`src/media/youtube.rs`), in `dir`.
fn ytdlp(dir: &Path) -> Command {
    let mut cmd = Command::new("yt-dlp");
    cmd.current_dir(dir).stdin(Stdio::null()).args([
        "--ignore-config",
        "--no-playlist",
        "--no-warnings",
        "--no-progress",
        "--no-cache-dir",
        "--socket-timeout",
        "25",
        "--retries",
        "2",
        "--js-runtimes",
        "node",
    ]);
    cmd
}

fn run(cmd: &mut Command) -> Result<()> {
    let output = cmd
        .output()
        .context("running yt-dlp (is it on the PATH?)")?;
    if !output.status.success() {
        bail!(
            "yt-dlp failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

/// The files yt-dlp wrote whose names start with `prefix`.
fn written(dir: &Path, prefix: &str) -> Result<Vec<PathBuf>> {
    Ok(fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with(prefix))
        })
        .collect())
}

fn download_audio(dir: &Path, name: &str, url: &str, flac: &Path) -> Result<()> {
    let prefix = format!("{name}.source.");
    let result = (|| {
        run(ytdlp(dir).args([
            "-f",
            "bestaudio[ext=m4a]/bestaudio/best",
            "-o",
            &format!("{prefix}%(ext)s"),
            "--",
            url,
        ]))?;
        let media = written(dir, &prefix)?
            .into_iter()
            .find(|p| p.extension().is_some_and(|e| e != "part"))
            .context("yt-dlp wrote no audio file")?;
        super::util::to_flac(&media, flac, None, None)
    })();
    for leftover in written(dir, &prefix)? {
        let _ = fs::remove_file(leftover);
    }
    result
}

/// The uploader's English captions as (start, end, text) in seconds, with sound cues such as
/// "(Laughter)" removed.
fn captions(dir: &Path, name: &str, url: &str) -> Result<Vec<(f64, f64, String)>> {
    let prefix = format!("{name}.captions");
    let file = dir.join(format!("{prefix}.en.json3"));
    let result = (|| {
        run(ytdlp(dir).args([
            "--skip-download",
            "--write-subs",
            "--sub-langs",
            "en",
            "--sub-format",
            "json3",
            "-o",
            &prefix,
            "--",
            url,
        ]))?;
        ensure!(
            file.is_file(),
            "{url} has no English captions from the uploader"
        );
        let json: serde_json::Value = serde_json::from_slice(&fs::read(&file)?)?;
        let mut captions = Vec::new();
        for event in json["events"]
            .as_array()
            .context("captions without events")?
        {
            let Some(segs) = event["segs"].as_array() else {
                continue;
            };
            let raw: String = segs.iter().filter_map(|s| s["utf8"].as_str()).collect();
            let text = without_cues(&raw);
            if text.is_empty() {
                continue;
            }
            let start = event["tStartMs"].as_f64().unwrap_or_default() / 1000.0;
            let end = start + event["dDurationMs"].as_f64().unwrap_or_default() / 1000.0;
            captions.push((start, end, text));
        }
        ensure!(captions.len() >= 10, "only {} captions", captions.len());
        Ok(captions)
    })();
    for leftover in written(dir, &prefix)? {
        let _ = fs::remove_file(leftover);
    }
    result
}

/// Text without bracketed or parenthesised cues, on one line.
fn without_cues(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut depth = 0usize;
    for c in text.chars() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth = depth.saturating_sub(1),
            '♪' => {}
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Markup as plain text: tags become spaces, entities characters.
fn plain(html: &str) -> String {
    let mut text = String::with_capacity(html.len());
    let mut in_tag = false;
    for c in html.chars() {
        match c {
            '<' => in_tag = true,
            '>' if in_tag => {
                in_tag = false;
                text.push(' ');
            }
            _ if !in_tag => text.push(c),
            _ => {}
        }
    }
    let text = html_escape::decode_html_entities(&text);
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A BBC Learning English episode's script: each line under its speaker's bold name. Vocabulary
/// is bold too, mid-sentence; a name stands alone before a line break. (As in
/// `tests/app/live/reference.rs`.)
fn bbc_script(fetch: &Fetch, url: &str) -> Result<Vec<(String, String)>> {
    let page = fetch.get_text(url)?;
    let start = page
        .find(">TRANSCRIPT<")
        .context("BBC page without a transcript")?;
    let end = page[start..]
        .find("</div>")
        .map_or(page.len(), |e| start + e);
    let mut rest = &page[start..end];
    let mut turns: Vec<(String, String)> = Vec::new();
    let mut name: Option<String> = None;
    let mut text = String::new();
    while let Some(at) = rest.find("<strong>") {
        text.push_str(&rest[..at]);
        let after = &rest[at + "<strong>".len()..];
        let close = after.find("</strong>").context("unclosed <strong>")?;
        let (inner, tail) = (&after[..close], &after[close + "</strong>".len()..]);
        let label = plain(inner);
        let alone = inner.trim_end().ends_with("/>") || tail.trim_start().starts_with("<br");
        if alone && !label.is_empty() && label.chars().count() <= 40 {
            if let Some(name) = name.take() {
                turns.push((name, plain(&text)));
            }
            name = Some(label);
            text.clear();
        } else {
            text.push_str(inner);
        }
        rest = tail;
    }
    text.push_str(rest);
    turns.extend(name.map(|name| (name, plain(&text))));
    ensure!(
        turns.len() >= 20,
        "{url}: only {} script lines",
        turns.len()
    );
    Ok(turns)
}

/// A Lex Fridman Podcast transcript: segments each with the speaker's name. (As in
/// `tests/app/live/reference.rs`.)
fn lex_transcript(fetch: &Fetch, url: &str) -> Result<Vec<(String, String)>> {
    let page = fetch.get_text(url)?;
    let mut turns: Vec<(String, String)> = Vec::new();
    for segment in page.split("<div class=\"ts-segment\">").skip(1) {
        let segment = &segment[..segment.find("</div>").unwrap_or(segment.len())];
        let field = |class: &str| {
            let open = format!("<span class=\"{class}\">");
            segment.find(&open).map(|at| {
                let body = &segment[at + open.len()..];
                plain(&body[..body.find("</span>").unwrap_or(body.len())])
            })
        };
        let text = field("ts-text").unwrap_or_default();
        let name = field("ts-name")
            .filter(|n| !n.is_empty())
            .or_else(|| turns.last().map(|t| t.0.clone()))
            .context("transcript starts without a speaker")?;
        turns.push((name, text));
    }
    ensure!(turns.len() >= 100, "{url}: only {} segments", turns.len());
    Ok(turns)
}
