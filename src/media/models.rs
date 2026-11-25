use crate::job::Progress;
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

pub struct Artifact {
    repo: &'static str,
    revision: &'static str,
    filename: &'static str,
    bytes: u64,
    digest: &'static str,
}

pub struct SpeechModel {
    pub id: &'static str,
    pub name: &'static str,
    pub language: &'static str,
    pub artifact: Artifact,
}

pub const SPEECH: [SpeechModel; 3] = [
    SpeechModel {
        id: "parakeet_unified_en",
        name: "Parakeet Unified English",
        language: "English",
        artifact: Artifact {
            repo: "handy-computer/parakeet-unified-en-0.6b-gguf",
            revision: "7e948f21b7bdbac698d3318db9d350f1096f3b6c",
            filename: "parakeet-unified-en-0.6b-Q8_0.gguf",
            bytes: 731357568,
            digest: "4b50b6dd862bf6e346929aaf4f5eaacec003bfa3f56462d6c874b41ef2f38795",
        },
    },
    SpeechModel {
        id: "small",
        name: "Whisper Small",
        language: "Multilingual",
        artifact: Artifact {
            repo: "handy-computer/whisper-small-gguf",
            revision: "4afdc7942ecf1997f359deae48a63506b062eac4",
            filename: "whisper-small-Q8_0.gguf",
            bytes: 269751136,
            digest: "9b9c8811bbcc82a7766f0fb0925614bdacb0923b2cc630daeac17108b655b860",
        },
    },
    SpeechModel {
        id: "turbo",
        name: "Whisper Large v3 Turbo",
        language: "Multilingual",
        artifact: Artifact {
            repo: "handy-computer/whisper-large-v3-turbo-gguf",
            revision: "d222c9f621c1128299248f2ded4d8a1820519780",
            filename: "whisper-large-v3-turbo-Q8_0.gguf",
            bytes: 886381824,
            digest: "d5e65f2b0828802ae2c231673d31982cebe3a778c95d9494a9f3efee6bd17448",
        },
    },
];

/// NVIDIA's Streaming Sortformer: who spoke when, for up to four speakers, on the same
/// devices as the speech models. Q8_0 labels 99% of speech as F16 does at 60% of the size.
pub const SPEAKERS: Artifact = Artifact {
    repo: "handy-computer/diar_streaming_sortformer_4spk-v2.1-gguf",
    revision: "60ebab3ef7cae9869be0abcde13360347e96ba03",
    filename: "diar_streaming_sortformer_4spk-v2.1-Q8_0.gguf",
    bytes: 139310336,
    digest: "a5dacdc650790266c7a362e54e6bf51952015487edaa606c4e11632bc32442a9",
};

pub fn speech(id: &str) -> Result<&'static SpeechModel> {
    SPEECH
        .iter()
        .find(|m| m.id == id)
        .context("Unknown speech model")
}

pub fn cache_root() -> Result<PathBuf> {
    if let Some(dir) = env::var_os("HF_HUB_CACHE").or_else(|| env::var_os("HUGGINGFACE_HUB_CACHE"))
    {
        return Ok(dir.into());
    }
    if let Some(home) = env::var_os("HF_HOME") {
        return Ok(PathBuf::from(home).join("hub"));
    }
    let base = env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| directories::BaseDirs::new().map(|dirs| dirs.home_dir().join(".cache")))
        .context("Could not locate the model cache")?;
    Ok(base.join("huggingface").join("hub"))
}

impl Artifact {
    pub fn megabytes(&self) -> u64 {
        (self.bytes + 500_000) / 1_000_000
    }

    fn path(&self, root: &Path) -> PathBuf {
        root.join(format!("models--{}", self.repo.replace('/', "--")))
            .join("snapshots")
            .join(self.revision)
            .join(self.filename)
    }

    fn present(&self, path: &Path) -> bool {
        path.metadata()
            .is_ok_and(|m| m.is_file() && m.len() == self.bytes)
    }

    pub fn get(&self, root: &Path, progress: &Progress) -> Result<PathBuf> {
        let path = self.path(root);
        if self.present(&path) {
            return Ok(path);
        }
        ensure!(
            !env::var("HF_HUB_OFFLINE").is_ok_and(|v| matches!(
                v.to_ascii_uppercase().as_str(),
                "1" | "ON" | "YES" | "TRUE"
            )),
            "The model is missing from the offline cache"
        );
        fs::create_dir_all(path.parent().unwrap())?;
        let lock = fs::File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path.with_extension("lock"))?;
        loop {
            progress.check()?;
            match lock.try_lock() {
                Ok(()) => break,
                Err(std::fs::TryLockError::WouldBlock) => {
                    std::thread::sleep(Duration::from_millis(50))
                }
                Err(std::fs::TryLockError::Error(e)) => return Err(e.into()),
            }
        }
        if self.present(&path) {
            return Ok(path);
        }
        let endpoint = env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into());
        let url = format!(
            "{}/{}/resolve/{}/{}",
            endpoint.trim_end_matches('/'),
            self.repo,
            self.revision,
            self.filename
        );
        let mut input = ureq::AgentBuilder::new()
            .try_proxy_from_env(true)
            .timeout_connect(Duration::from_secs(25))
            .timeout_read(Duration::from_secs(90))
            .build()
            .get(&url)
            .call()
            .context("Could not download model")?
            .into_reader();
        let partial = tempfile::TempPath::try_from_path(path.with_extension("partial"))?;
        let mut output = fs::File::create(&partial)?;
        let mut hash = Sha256::new();
        let mut buffer = vec![0; 256 * 1024];
        let mut bytes = 0;
        loop {
            progress.check()?;
            let n = input.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            output.write_all(&buffer[..n])?;
            hash.update(&buffer[..n]);
            bytes += n as u64;
            progress.fraction(bytes as f32 / self.bytes as f32)?;
        }
        ensure!(
            bytes == self.bytes && format!("{:x}", hash.finalize()) == self.digest,
            "Incomplete or incorrect model download"
        );
        output.sync_all()?;
        drop(output);
        partial.persist(&path)?;
        Ok(path)
    }
}

pub fn installed(root: &Path) -> Vec<String> {
    SPEECH
        .iter()
        .filter(|m| m.artifact.present(&m.artifact.path(root)))
        .map(|m| m.id.into())
        .collect()
}
