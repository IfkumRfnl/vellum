use super::*;
use crate::pipeline::{Stage, Update};
use std::{fmt::Write as _, io::Write as _, sync::Mutex};

mod accuracy;
mod reference;
mod score;

/// One video read as the app reads it, and how long each stage took.
struct Run {
    doc: Document,
    /// The video's length, in seconds.
    duration: f64,
    /// The device that recognized the speech, as the progress screen names it.
    device: String,
    stages: Vec<(Stage, Duration)>,
    total: Duration,
}

impl Run {
    fn time(&self, stage: Stage) -> Option<Duration> {
        self.stages
            .iter()
            .find(|(s, _)| *s == stage)
            .map(|(_, took)| *took)
    }

    fn on_gpu(&self) -> bool {
        !self.device.is_empty() && self.device != "CPU"
    }

    /// How many seconds of audio `stage` got through per second.
    fn speed(&self, stage: Stage) -> Option<f64> {
        self.time(stage)
            .map(|took| self.duration / took.as_secs_f64().max(1e-3))
    }

    /// "Vulkan · RTX 4060 | 6:12 of audio | transcribe 0:05 (74× real time) · … | total 0:31"
    fn timing(&self) -> String {
        let mut stages = String::new();
        for (stage, took) in &self.stages {
            if !stages.is_empty() {
                stages.push_str(" · ");
            }
            let _ = write!(
                stages,
                "{} {}",
                format!("{stage:?}").to_lowercase(),
                clock(*took)
            );
            if matches!(stage, Stage::Transcribe | Stage::Speakers) {
                let _ = write!(stages, " ({:.0}× real time)", self.speed(*stage).unwrap());
            }
        }
        format!(
            "{} | {} of audio | {stages} | total {}",
            if self.device.is_empty() {
                "no speech model"
            } else {
                &self.device
            },
            clock(Duration::from_secs_f64(self.duration)),
            clock(self.total)
        )
    }
}

/// 1:05:09, 4:07
fn clock(d: Duration) -> String {
    crate::document::timestamp(d.as_secs_f64().round())
}

/// Prints a test's results and keeps them in target/live-report.md, apart from the engine's logs.
fn report(name: &str, lines: &[String]) {
    let text = format!(
        "## {name}

{}

",
        lines.join(
            "
"
        )
    );
    eprint!(
        "
{text}"
    );
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/live-report.md");
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = file.write_all(text.as_bytes());
    }
}

fn read_video(url: &'static str, model: &str, official_captions: bool, speakers: bool) -> Run {
    // Each live case exercises the whole video. Give the local inference engine
    // one job at a time even when Cargo runs test functions concurrently.
    static SLOT: Mutex<()> = Mutex::new(());
    let _slot = SLOT.lock().unwrap_or_else(|e| e.into_inner());
    let settings = Settings {
        model: model.into(),
        official_captions,
        speakers,
        ..Settings::default()
    };
    let job = Job::start(Arc::new(|| {}), move |progress| {
        media::read_video(url, &settings, progress).map(Event::Result)
    });
    let start = Instant::now();
    // Enough to download and read a three-hour podcast on the CPU.
    let budget = Duration::from_secs(3600);
    let mut duration = 0.0;
    let mut device = String::new();
    let mut began: Vec<(Stage, Instant)> = Vec::new();
    loop {
        match job
            .events
            .recv_timeout(budget.saturating_sub(start.elapsed()))
            .unwrap_or_else(|e| panic!("{url} ({model}): {e}"))
        {
            Event::Update(Update::Video { duration: d, .. }) => duration = d.unwrap_or_default(),
            Event::Update(Update::Device(d)) if device.is_empty() => device = d,
            Event::Update(Update::Stage(stage)) => began.push((stage, Instant::now())),
            Event::Update(Update::Note(note)) => eprintln!("{model}: {note}"),
            Event::Update(_) => {}
            Event::Result(doc) => {
                let end = Instant::now();
                let stages = began
                    .iter()
                    .enumerate()
                    .map(|(i, (stage, at))| {
                        let until = began.get(i + 1).map_or(end, |next| next.1);
                        (*stage, until - *at)
                    })
                    .collect();
                return Run {
                    doc,
                    duration,
                    device,
                    stages,
                    total: end - start,
                };
            }
            Event::Error(error) => panic!("{url} ({model}): {error}"),
            Event::Cached(_) => panic!("unexpected model preparation result"),
        }
    }
}

#[test]
#[ignore = "downloads captions from a public YouTube video"]
fn youtube_link_opens_reader_and_remembers_type() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("settings.json");
    let mut s = load_state(Some(path.clone()));
    s.wake = Some(Arc::new(|| {}));
    let mut rt = Runtime::new(
        s,
        ui::view,
        (960.0, 760.0),
        Env {
            system_dark: true,
            scale: 1.0,
        },
    );
    rt.poll = Some(poll);
    rt.frame(0.0);
    tap(&mut rt, "Skip");
    tap(&mut rt, "url");
    rt.paste("https://www.youtube.com/watch?v=jNQXAC9IVRw");
    rt.frame(0.02);
    tap(&mut rt, "Transcribe");
    let limit = Instant::now() + Duration::from_secs(180);
    while matches!(rt.state.screen, Screen::Working(_)) && Instant::now() < limit {
        thread::sleep(Duration::from_millis(100));
        rt.frame(0.1);
    }
    assert!(
        matches!(rt.state.screen, Screen::Reader),
        "{}",
        rt.state.message
    );
    let doc = rt.state.doc.as_ref().unwrap();
    assert_eq!(doc.title, "Me at the zoo");
    assert!(!doc.paragraphs.is_empty());
    tap(&mut rt, "Aa");
    tap(&mut rt, "Sans");
    tap(&mut rt, "+");
    drop(rt);
    let saved = load_state(Some(path));
    assert_eq!(saved.settings.font, 1);
    assert_eq!(saved.settings.size, 22);
    assert!(matches!(saved.screen, Screen::Home));
}

#[test]
#[ignore = "downloads a captioned video and labels the existing caption text"]
fn labels_the_publishers_captions_without_running_speech_recognition() {
    let doc = read_video("https://youtu.be/jNQXAC9IVRw", "captions", false, true).doc;
    assert_eq!(doc.source, "YouTube captions");
    assert_eq!(doc.speaker_note, "Estimated speakers");
    assert!(
        doc.paragraphs
            .iter()
            .any(|p| p.speaker.is_some() && p.text.to_lowercase().contains("elephant"))
    );
}
