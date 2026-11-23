//! User preferences and their background writer.

use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Write,
    path::PathBuf,
    sync::mpsc::{self, Receiver, Sender},
    thread::{self, JoinHandle},
};

/// Average silent reading speed for English non-fiction, used until Vellum has measured yours.
pub const DEFAULT_WPM: f32 = 238.0;

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub configured: bool,
    pub model: String,
    pub captions_first: bool,
    pub speakers: bool,
    pub font: usize,
    pub size: u8,
    pub reading_width: ReadingWidth,
    pub timestamps: bool,
    pub spacious: bool,
    /// Your measured reading speed, carried from one transcript to the next.
    pub reading_wpm: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            configured: false,
            model: "parakeet_unified_en".into(),
            captions_first: false,
            speakers: false,
            font: 0,
            size: 21,
            reading_width: ReadingWidth::default(),
            timestamps: true,
            spacious: true,
            reading_wpm: DEFAULT_WPM,
        }
    }
}

impl Settings {
    /// Reads saved preferences, repairing values a newer or hand-edited file could hold.
    pub fn load(path: Option<&PathBuf>) -> Self {
        let mut settings: Settings = path
            .and_then(|p| fs::read(p).ok())
            .and_then(|data| serde_json::from_slice(&data).ok())
            .unwrap_or_default();
        settings.font = settings.font.min(2);
        settings.size = settings.size.clamp(16, 32);
        if !settings.reading_wpm.is_finite() {
            settings.reading_wpm = DEFAULT_WPM;
        }
        settings.reading_wpm = settings.reading_wpm.clamp(80.0, 900.0);
        if (settings.model != "captions" && crate::media::models::speech(&settings.model).is_err())
            || (!settings.configured && settings.model == "captions")
        {
            settings.model = Settings::default().model;
        }
        settings
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum ReadingWidth {
    Narrow,
    #[default]
    Standard,
    Wide,
}

impl ReadingWidth {
    pub const ALL: [(&'static str, ReadingWidth); 3] = [
        ("Narrow", ReadingWidth::Narrow),
        ("Standard", ReadingWidth::Standard),
        ("Wide", ReadingWidth::Wide),
    ];

    pub fn points(self) -> f32 {
        match self {
            ReadingWidth::Narrow => 500.0,
            ReadingWidth::Standard => 660.0,
            ReadingWidth::Wide => 900.0,
        }
    }
}

pub fn settings_path() -> Option<PathBuf> {
    std::env::var_os("VELLUM_CONFIG_DIR")
        .map(PathBuf::from)
        .or_else(|| ProjectDirs::from("", "", "Vellum").map(|p| p.config_dir().to_owned()))
        .map(|dir| dir.join("settings.json"))
}

/// Writes settings on a background thread, atomically, newest last. Dropping it flushes.
pub struct Store {
    saves: Option<Sender<Settings>>,
    writer: Option<JoinHandle<()>>,
    pub errors: Receiver<String>,
}

impl Store {
    pub fn new(path: Option<PathBuf>) -> Self {
        let (saves, recv) = mpsc::channel::<Settings>();
        let (report, errors) = mpsc::channel();
        let writer = thread::spawn(move || {
            for value in recv {
                let save = || -> std::io::Result<()> {
                    let file = path
                        .as_ref()
                        .ok_or_else(|| std::io::Error::other("No config directory"))?;
                    let dir = file.parent().unwrap();
                    fs::create_dir_all(dir)?;
                    let mut tmp = tempfile::NamedTempFile::new_in(dir)?;
                    tmp.write_all(&serde_json::to_vec_pretty(&value)?)?;
                    tmp.persist(file)?;
                    Ok(())
                };
                if let Err(e) = save() {
                    let _ = report.send(format!("Could not save preferences. {e}"));
                }
            }
        });
        Self {
            saves: Some(saves),
            writer: Some(writer),
            errors,
        }
    }

    pub fn save(&self, settings: &Settings) {
        if let Some(sender) = &self.saves {
            let _ = sender.send(settings.clone());
        }
    }
}

impl Drop for Store {
    fn drop(&mut self) {
        self.saves.take();
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
    }
}
