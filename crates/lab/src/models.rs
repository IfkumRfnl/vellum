//! The models the lab measures: the app's, pinned to the same files (`src/media/models.rs`), in
//! the same Hugging Face cache, so both engines read exactly what users download.

use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    io::{Read, Write},
    path::PathBuf,
    time::Duration,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Speech,
    Speakers,
}

#[derive(Debug)]
pub struct Model {
    /// The lab's short name: `parakeet`, `small`, `turbo`, `sortformer`.
    pub id: &'static str,
    /// The app's id, where it has one.
    pub app_id: &'static str,
    pub name: &'static str,
    pub kind: Kind,
    /// The one language it transcribes, or `None` for any.
    pub language: Option<&'static str>,
    repo: &'static str,
    revision: &'static str,
    filename: &'static str,
    bytes: u64,
    pub digest: &'static str,
}

pub const MODELS: [Model; 4] = [
    Model {
        id: "parakeet",
        app_id: "parakeet_unified_en",
        name: "Parakeet Unified English",
        kind: Kind::Speech,
        language: Some("en"),
        repo: "handy-computer/parakeet-unified-en-0.6b-gguf",
        revision: "7e948f21b7bdbac698d3318db9d350f1096f3b6c",
        filename: "parakeet-unified-en-0.6b-Q8_0.gguf",
        bytes: 731357568,
        digest: "4b50b6dd862bf6e346929aaf4f5eaacec003bfa3f56462d6c874b41ef2f38795",
    },
    Model {
        id: "small",
        app_id: "small",
        name: "Whisper Small",
        kind: Kind::Speech,
        language: None,
        repo: "handy-computer/whisper-small-gguf",
        revision: "4afdc7942ecf1997f359deae48a63506b062eac4",
        filename: "whisper-small-Q8_0.gguf",
        bytes: 269751136,
        digest: "9b9c8811bbcc82a7766f0fb0925614bdacb0923b2cc630daeac17108b655b860",
    },
    Model {
        id: "turbo",
        app_id: "turbo",
        name: "Whisper Large v3 Turbo",
        kind: Kind::Speech,
        language: None,
        repo: "handy-computer/whisper-large-v3-turbo-gguf",
        revision: "d222c9f621c1128299248f2ded4d8a1820519780",
        filename: "whisper-large-v3-turbo-Q8_0.gguf",
        bytes: 886381824,
        digest: "d5e65f2b0828802ae2c231673d31982cebe3a778c95d9494a9f3efee6bd17448",
    },
    Model {
        id: "sortformer",
        app_id: "",
        name: "Streaming Sortformer 4spk v2.1",
        kind: Kind::Speakers,
        language: None,
        repo: "handy-computer/diar_streaming_sortformer_4spk-v2.1-gguf",
        revision: "60ebab3ef7cae9869be0abcde13360347e96ba03",
        filename: "diar_streaming_sortformer_4spk-v2.1-Q8_0.gguf",
        bytes: 139310336,
        digest: "a5dacdc650790266c7a362e54e6bf51952015487edaa606c4e11632bc32442a9",
    },
];

pub fn get(id: &str) -> Result<&'static Model> {
    MODELS
        .iter()
        .find(|m| m.id == id || (!m.app_id.is_empty() && m.app_id == id))
        .with_context(|| {
            let ids: Vec<&str> = MODELS.iter().map(|m| m.id).collect();
            format!("unknown model {id:?}; the lab knows {}", ids.join(", "))
        })
}

/// The Hugging Face hub cache, found as the app finds it.
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

impl Model {
    pub fn path(&self) -> Result<PathBuf> {
        Ok(cache_root()?
            .join(format!("models--{}", self.repo.replace('/', "--")))
            .join("snapshots")
            .join(self.revision)
            .join(self.filename))
    }

    /// Whether this model transcribes clips in `language`.
    pub fn speaks(&self, language: &str) -> bool {
        self.kind == Kind::Speech
            && !language.is_empty()
            && self.language.is_none_or(|l| l == language)
    }

    /// The model file, downloaded and verified first if it isn't in the cache.
    pub fn ensure(&self) -> Result<PathBuf> {
        let path = self.path()?;
        if path.metadata().is_ok_and(|m| m.len() == self.bytes) {
            return Ok(path);
        }
        eprintln!("· downloading {} ({} MB)", self.name, self.bytes >> 20);
        fs::create_dir_all(path.parent().unwrap())?;
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
        let partial = path.with_extension("partial");
        let mut output = fs::File::create(&partial)?;
        let mut hash = Sha256::new();
        let mut buffer = vec![0; 1 << 20];
        let mut bytes = 0;
        loop {
            let n = input.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            output.write_all(&buffer[..n])?;
            hash.update(&buffer[..n]);
            bytes += n as u64;
        }
        ensure!(
            bytes == self.bytes && format!("{:x}", hash.finalize()) == self.digest,
            "Incomplete or incorrect model download"
        );
        output.sync_all()?;
        drop(output);
        fs::rename(&partial, &path)?;
        Ok(path)
    }
}
