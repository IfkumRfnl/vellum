use super::Cue;
use crate::{job::Progress, pipeline::Stage};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Duration,
};
use url::Url;

pub fn video_url(input: &str) -> Result<String> {
    let input = input.trim();
    let qualified = if input.contains("://") {
        input.to_owned()
    } else {
        format!("https://{input}")
    };
    let url = Url::parse(&qualified).context("Paste a link to a YouTube video.")?;
    let path: Vec<_> = url.path().trim_matches('/').split('/').collect();
    let id = match (url.host_str(), path.as_slice()) {
        (Some("youtu.be"), [id]) => Some(id.to_string()),
        (
            Some("youtube.com" | "www.youtube.com" | "m.youtube.com" | "music.youtube.com"),
            ["watch"],
        ) => url
            .query_pairs()
            .find(|(k, _)| k == "v")
            .map(|(_, v)| v.into_owned()),
        (
            Some("youtube.com" | "www.youtube.com" | "m.youtube.com" | "music.youtube.com"),
            ["embed" | "shorts" | "live", id],
        ) => Some(id.to_string()),
        _ => None,
    };
    let id = id
        .filter(|s| {
            matches!(url.scheme(), "http" | "https")
                && s.len() == 11
                && s.bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
        })
        .context("Paste a link to a YouTube video.")?;
    Ok(format!("https://www.youtube.com/watch?v={id}"))
}

#[derive(Deserialize)]
struct Track {
    ext: String,
    url: String,
}
type Tracks = BTreeMap<String, Vec<Track>>;

#[derive(Deserialize)]
pub struct Video {
    pub title: String,
    pub uploader: Option<String>,
    pub duration: Option<f64>,
    language: Option<String>,
    is_live: Option<bool>,
    #[serde(default)]
    subtitles: Tracks,
    #[serde(default)]
    automatic_captions: Tracks,
    #[serde(default)]
    http_headers: BTreeMap<String, String>,
}

fn command(name: &str) -> Command {
    let exe_name = format!("{name}{}", std::env::consts::EXE_SUFFIX);
    let sibling = std::env::current_exe()
        .ok()
        .and_then(|p| Some(p.parent()?.join(&exe_name)));
    let mut cmd = Command::new(
        sibling
            .filter(|p| p.is_file())
            .unwrap_or_else(|| exe_name.into()),
    );
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000);
    }
    #[cfg(target_os = "macos")]
    {
        // Apps launched from Finder do not inherit an interactive shell's PATH.
        // Pass Homebrew's locations through to yt-dlp's JS runtime lookup too.
        let path = std::env::var_os("PATH").unwrap_or_else(|| "/usr/bin:/bin".into());
        let dirs = std::env::split_paths(&path).chain([
            PathBuf::from("/opt/homebrew/bin"),
            PathBuf::from("/usr/local/bin"),
        ]);
        if let Ok(path) = std::env::join_paths(dirs) {
            cmd.env("PATH", path);
        }
    }
    cmd.stdin(Stdio::null());
    cmd
}

fn ytdlp(dir: &Path) -> Command {
    let mut cmd = command("yt-dlp");
    cmd.current_dir(dir).args([
        "--ignore-config",
        "--no-playlist",
        "--no-warnings",
        "--no-cache-dir",
        "--no-colors",
        "--socket-timeout",
        "25",
        "--retries",
        "2",
        "--js-runtimes",
        "node",
        "--extractor-args",
        "youtube:skip=translated_subs",
        "-f",
        "bestaudio[ext=m4a]/bestaudio/best",
    ]);
    cmd
}

pub fn open(dir: &Path, input: &str, progress: &Progress) -> Result<Video> {
    let json = progress.command(
        ytdlp(dir).args(["--no-progress", "--dump-single-json", "--", input]),
        "yt-dlp",
    )?;
    let video: Video = serde_json::from_slice(&json).context("Could not read YouTube metadata")?;
    ensure!(
        video.is_live != Some(true),
        "This video is live. Open it again after the stream finishes."
    );
    fs::write(dir.join("video.json"), json)?;
    Ok(video)
}

/// Asks yt-dlp to print one parseable progress line per update.
fn with_progress(cmd: &mut Command) -> &mut Command {
    cmd.args([
        "--progress",
        "--newline",
        "--progress-template",
        "download:VELLUM %(progress.downloaded_bytes)s %(progress.total_bytes)s %(progress.total_bytes_estimate)s",
    ])
}

/// The download fraction in one of `with_progress`'s lines.
fn download_fraction(line: &str) -> Option<f32> {
    let mut fields = line.strip_prefix("VELLUM ")?.split_whitespace();
    let done: f64 = fields.next()?.parse().ok()?;
    let total = fields
        .filter_map(|v| v.parse::<f64>().ok())
        .find(|v| *v > 0.0)?;
    Some((done / total).clamp(0.0, 1.0) as f32)
}

pub fn audio(
    dir: &Path,
    link: &str,
    duration: Option<f64>,
    progress: &Progress,
) -> Result<PathBuf> {
    progress.stage(Stage::Download)?;
    let report = |line: &str| {
        if let Some(f) = download_fraction(line) {
            let _ = progress.fraction(f);
        }
    };
    let download = progress.watch(
        with_progress(&mut ytdlp(dir)).args([
            "--load-info-json",
            "video.json",
            "--no-simulate",
            "-o",
            "audio.%(ext)s",
        ]),
        "yt-dlp",
        report,
    );
    if let Err(error) = download {
        let message = error.to_string();
        if !message.contains("HTTP Error 403") && !message.contains("HTTP Error 410") {
            return Err(error);
        }
        // The signed media URL can expire between extraction and download.
        // Re-extract once and let yt-dlp test that the selected format is accessible.
        progress.note("Refreshing the audio link…")?;
        progress.watch(
            with_progress(&mut ytdlp(dir)).args([
                "--check-formats",
                "--no-continue",
                "-o",
                "audio.%(ext)s",
                "--",
                link,
            ]),
            "yt-dlp",
            report,
        )?;
    }
    let media = fs::read_dir(dir)?
        .filter_map(|r| r.ok())
        .map(|e| e.path())
        .find(|p| p.file_stem().is_some_and(|v| v == "audio"))
        .context("yt-dlp returned no audio file")?;
    progress.stage(Stage::Convert)?;
    let wav = dir.join("speech.wav");
    progress.watch(
        command("ffmpeg")
            .args([
                "-nostdin",
                "-v",
                "error",
                "-nostats",
                "-progress",
                "pipe:1",
                "-i",
            ])
            .arg(media)
            .args(["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"])
            .arg(&wav),
        "FFmpeg",
        |line| {
            let decoded = line
                .strip_prefix("out_time_us=")
                .and_then(|v| v.parse::<f64>().ok());
            if let (Some(us), Some(total)) = (decoded, duration.filter(|d| *d > 0.0)) {
                let _ = progress.fraction((us / 1e6 / total) as f32);
            }
        },
    )?;
    Ok(wav)
}

#[derive(Deserialize)]
struct CaptionData {
    events: Vec<CaptionEvent>,
}
#[derive(Deserialize)]
struct CaptionEvent {
    #[serde(rename = "tStartMs", default)]
    start: f64,
    #[serde(rename = "dDurationMs", default)]
    duration: f64,
    #[serde(default)]
    segs: Vec<CaptionFragment>,
}
#[derive(Deserialize)]
struct CaptionFragment {
    #[serde(default)]
    utf8: String,
}

fn cues(data: CaptionData) -> Vec<Cue> {
    data.events
        .into_iter()
        .filter_map(|event| {
            let raw: String = event.segs.into_iter().map(|s| s.utf8).collect();
            let text = html_escape::decode_html_entities(&raw)
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            (!text.is_empty()).then(|| {
                Cue::new(
                    event.start / 1000.,
                    (event.start + event.duration) / 1000.,
                    text,
                )
            })
        })
        .collect()
}

impl Video {
    pub fn captions(&self, progress: &Progress) -> Result<Option<(Vec<Cue>, String, String)>> {
        let agent = ureq::AgentBuilder::new()
            .try_proxy_from_env(true)
            .timeout(Duration::from_secs(25))
            .build();
        let mut problem = None;
        for (tracks, source) in [
            (&self.subtitles, "YouTube captions"),
            (&self.automatic_captions, "YouTube auto-captions"),
        ] {
            let mut langs: Vec<_> = tracks.keys().filter(|lang| *lang != "live_chat").collect();
            langs.sort_by_key(|lang| {
                let base = lang.trim_end_matches("-orig");
                (
                    Some(base) != self.language.as_deref(),
                    !lang.ends_with("-orig"),
                    base != "en",
                    lang.as_str(),
                )
            });
            for lang in langs {
                progress.check()?;
                let Some(track) = tracks[lang].iter().find(|t| {
                    t.ext == "json3"
                        && Url::parse(&t.url)
                            .is_ok_and(|u| !u.query_pairs().any(|(k, _)| k == "tlang"))
                }) else {
                    continue;
                };
                let mut request = agent.get(&track.url);
                for (name, value) in &self.http_headers {
                    request = request.set(name, value);
                }
                let fetch = || -> Result<Vec<Cue>> {
                    let data = serde_json::from_reader(request.call()?.into_reader())?;
                    Ok(cues(data))
                };
                match fetch() {
                    Ok(cues) if !cues.is_empty() => {
                        return Ok(Some((
                            cues,
                            source.to_string(),
                            lang.trim_end_matches("-orig").to_owned(),
                        )));
                    }
                    Err(error) => {
                        problem = Some(error);
                        break;
                    }
                    _ => {}
                }
            }
        }
        match problem {
            Some(e) => Err(e),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalizes_video_links() {
        for url in [
            "youtu.be/jNQXAC9IVRw?t=7",
            "https://m.youtube.com/shorts/jNQXAC9IVRw",
            "youtube.com/watch?v=jNQXAC9IVRw&list=1",
        ] {
            assert_eq!(
                video_url(url).unwrap(),
                "https://www.youtube.com/watch?v=jNQXAC9IVRw"
            );
        }
        for url in [
            "https://youtube.com.example/watch?v=jNQXAC9IVRw",
            "youtube.com/playlist?list=abc",
            "",
        ] {
            assert!(video_url(url).is_err());
        }
    }
    #[test]
    fn reads_download_progress() {
        assert_eq!(download_fraction("VELLUM 50 NA 200"), Some(0.25));
        assert_eq!(download_fraction("VELLUM 10 40 NA"), Some(0.25));
        assert_eq!(download_fraction("VELLUM 10 NA NA"), None);
        assert_eq!(download_fraction("[download] Destination: audio.m4a"), None);
    }

    #[test]
    fn reads_caption_fragments() {
        let json = r#"{"events":[{"tStartMs":500,"dDurationMs":1000,"segs":[{"utf8":"One "},{"utf8":"&amp; two."}]},{"tStartMs":1600},{"segs":[{"utf8":"&gt;&gt; Three."}]}]}"#;
        let parsed = cues(serde_json::from_str(json).unwrap());
        assert_eq!(parsed.len(), 2);
        assert_eq!(
            (parsed[0].start, parsed[0].end, parsed[0].text.as_str()),
            (0.5, 1.5, "One & two.")
        );
        assert_eq!(parsed[1].text, ">> Three.");
    }
}
