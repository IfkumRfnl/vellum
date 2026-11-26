//! What every source needs: downloads that resume and verify, FFmpeg, and clip records.

use super::{Clip, Corpus, Purpose, Tier, Truth};
use anyhow::{Context, Result, bail, ensure};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::Command,
    time::Duration,
};
use vellum_harness::audio;

/// A source's working context.
pub struct Fetch {
    /// The corpus directory.
    pub root: PathBuf,
    /// The index as it stood before this source ran (earlier sources' clips included).
    pub existing: Vec<Clip>,
    agent: ureq::Agent,
}

impl Fetch {
    pub fn new(root: &Path, corpus: &Corpus) -> Result<Fetch> {
        fs::create_dir_all(root)?;
        Ok(Fetch {
            root: root.to_owned(),
            existing: corpus.clips.clone(),
            agent: ureq::AgentBuilder::new()
                .try_proxy_from_env(true)
                .timeout_connect(Duration::from_secs(30))
                .timeout_read(Duration::from_secs(120))
                .user_agent("vellum-lab (speech engine test corpus)")
                .build(),
        })
    }

    /// A directory under the corpus, created.
    pub fn dir(&self, relative: &str) -> Result<PathBuf> {
        let dir = self.root.join(relative);
        fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    /// The clips an earlier source made of `set`.
    pub fn clips_of(&self, set: &str) -> Vec<&Clip> {
        self.existing.iter().filter(|c| c.set == set).collect()
    }

    pub fn get_text(&self, url: &str) -> Result<String> {
        self.agent
            .get(url)
            .call()
            .with_context(|| format!("GET {url}"))?
            .into_string()
            .with_context(|| format!("reading {url}"))
    }

    pub fn get_json(&self, url: &str) -> Result<serde_json::Value> {
        serde_json::from_str(&self.get_text(url)?).with_context(|| format!("parsing {url}"))
    }

    /// A streaming reader of `url`'s body, for archives too big to keep.
    pub fn open(&self, url: &str) -> Result<Box<dyn Read + Send + Sync>> {
        Ok(self
            .agent
            .get(url)
            .call()
            .with_context(|| format!("GET {url}"))?
            .into_reader())
    }

    /// Downloads `url` to `dest` unless it's already there, resuming a partial download, and
    /// returns its SHA-256. With `sha256`, a mismatch is an error and the file is removed.
    pub fn download(&self, url: &str, dest: &Path, sha256: Option<&str>) -> Result<String> {
        if dest.is_file() {
            let have = sha256_file(dest)?;
            match sha256 {
                Some(want) if want != have => fs::remove_file(dest)?,
                _ => return Ok(have),
            }
        }
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        let partial = dest.with_extension(format!(
            "{}.partial",
            dest.extension()
                .and_then(|e| e.to_str())
                .unwrap_or_default()
        ));
        let have = partial.metadata().map(|m| m.len()).unwrap_or(0);
        let mut request = self.agent.get(url);
        if have > 0 {
            request = request.set("Range", &format!("bytes={have}-"));
        }
        let response = request.call().with_context(|| format!("GET {url}"))?;
        let resumed = have > 0 && response.status() == 206;
        let total = response
            .header("Content-Length")
            .and_then(|v| v.parse::<u64>().ok())
            .map(|n| if resumed { n + have } else { n });
        let mut file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(resumed)
            .truncate(!resumed)
            .open(&partial)?;
        let mut input = response.into_reader();
        let mut buffer = vec![0; 1 << 20];
        let mut done = if resumed { have } else { 0 };
        let mut shown = 0;
        loop {
            let n = input.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            file.write_all(&buffer[..n])?;
            done += n as u64;
            if done - shown > 50 << 20 {
                shown = done;
                match total {
                    Some(t) => eprintln!("  {} {} / {} MB", name(dest), done >> 20, t >> 20),
                    None => eprintln!("  {} {} MB", name(dest), done >> 20),
                }
            }
        }
        file.sync_all()?;
        drop(file);
        if let Some(total) = total {
            ensure!(done == total, "{url}: got {done} of {total} bytes");
        }
        let got = sha256_file(&partial)?;
        if let Some(want) = sha256
            && want != got
        {
            fs::remove_file(&partial)?;
            bail!("{url}: SHA-256 {got}, expected {want}");
        }
        fs::rename(&partial, dest)?;
        Ok(got)
    }

    /// A clip record for audio already at `audio` (absolute, under the corpus), with its length
    /// and PCM hash measured.
    #[allow(clippy::too_many_arguments)]
    pub fn clip(
        &self,
        id: String,
        set: &str,
        audio_path: &Path,
        tier: Tier,
        language: &str,
        purposes: &[Purpose],
        truth: Truth,
        tags: &[&str],
    ) -> Result<Clip> {
        let pcm = audio::read(audio_path)?;
        Ok(Clip {
            id,
            set: set.to_owned(),
            audio: audio_path
                .strip_prefix(&self.root)
                .with_context(|| format!("{} is outside the corpus", audio_path.display()))?
                .to_owned(),
            seconds: pcm.len() as f64 / audio::RATE as f64,
            tier,
            language: language.to_owned(),
            purposes: purposes.to_vec(),
            truth,
            tags: tags.iter().map(|t| t.to_string()).collect(),
            pcm_sha256: audio::pcm_sha256(&pcm),
        })
    }

    /// The record of a clip a previous fetch made, if its audio is still in place: re-reading
    /// thousands of files to hash them again would make every fetch slow.
    pub fn known(&self, id: &str) -> Option<Clip> {
        self.existing
            .iter()
            .find(|c| c.id == id && self.root.join(&c.audio).is_file())
            .cloned()
    }
}

fn name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1 << 20];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

/// FFmpeg, from `VELLUM_FFMPEG` or the PATH.
pub fn ffmpeg() -> Command {
    let mut cmd =
        Command::new(std::env::var_os("VELLUM_FFMPEG").unwrap_or_else(|| "ffmpeg".into()));
    cmd.args(["-nostdin", "-hide_banner", "-loglevel", "error", "-y"]);
    cmd
}

/// Converts any audio FFmpeg reads to 16 kHz mono 16-bit FLAC, optionally only `seconds` of it
/// from `start` seconds in.
pub fn to_flac(
    input: &Path,
    output: &Path,
    start: Option<f64>,
    seconds: Option<f64>,
) -> Result<()> {
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let partial = output.with_extension("partial.flac");
    let mut cmd = ffmpeg();
    if let Some(start) = start {
        cmd.args(["-ss", &format!("{start:.3}")]);
    }
    cmd.arg("-i").arg(input);
    if let Some(seconds) = seconds {
        cmd.args(["-t", &format!("{seconds:.3}")]);
    }
    cmd.args([
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        "-c:a",
        "flac",
    ])
    .arg(&partial);
    let status = cmd
        .status()
        .context("running FFmpeg (set VELLUM_FFMPEG or add it to the PATH)")?;
    ensure!(status.success(), "FFmpeg failed on {}", input.display());
    fs::rename(&partial, output)?;
    Ok(())
}

/// Writes samples as 16 kHz mono 16-bit FLAC (through FFmpeg).
pub fn write_flac(output: &Path, pcm: &[f32]) -> Result<()> {
    let wav = output.with_extension("tmp.wav");
    audio::write_wav(&wav, pcm)?;
    let result = to_flac(&wav, output, None, None);
    let _ = fs::remove_file(&wav);
    result
}
