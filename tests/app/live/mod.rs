use super::*;
use crate::pipeline::Update;
use std::sync::Mutex;

mod accuracy;

fn read_video(url: &'static str, model: &str, captions_first: bool, speakers: bool) -> Document {
    // Each live case exercises the whole video. Give the local inference engine
    // one job at a time even when Cargo runs test functions concurrently.
    static SLOT: Mutex<()> = Mutex::new(());
    let _slot = SLOT.lock().unwrap_or_else(|e| e.into_inner());
    let settings = Settings {
        model: model.into(),
        captions_first,
        speakers,
        ..Settings::default()
    };
    let job = Job::start(Arc::new(|| {}), move |progress| {
        media::read_video(url, &settings, progress).map(Event::Result)
    });
    let start = Instant::now();
    let budget = Duration::from_secs(900);
    let mut seen_device = false;
    loop {
        match job
            .events
            .recv_timeout(budget.saturating_sub(start.elapsed()))
            .unwrap_or_else(|e| panic!("{url} ({model}): {e}"))
        {
            Event::Update(Update::Device(device)) if !seen_device => {
                eprintln!("{model}: {device}");
                seen_device = true;
            }
            Event::Update(Update::Stage(stage)) => eprintln!("{model}: {stage:?}"),
            Event::Update(Update::Note(note)) => eprintln!("{model}: {note}"),
            Event::Update(_) => {}
            Event::Result(doc) => return doc,
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
    let doc = read_video("https://youtu.be/jNQXAC9IVRw", "captions", false, true);
    assert_eq!(doc.source, "YouTube captions");
    assert_eq!(doc.speaker_note, "Estimated speakers");
    assert!(
        doc.paragraphs
            .iter()
            .any(|p| p.speaker.is_some() && p.text.to_lowercase().contains("elephant"))
    );
}

#[test]
#[ignore = "downloads a public video's audio and runs native speech and speaker inference"]
fn transcribes_video_audio_and_labels_speech() {
    let doc = read_video(
        "https://youtu.be/jNQXAC9IVRw",
        "parakeet_unified_en",
        false,
        true,
    );
    assert_eq!(doc.source, "Parakeet Unified English");
    assert!(
        doc.paragraphs
            .iter()
            .any(|p| p.text.to_lowercase().contains("elephant"))
    );
    assert!(
        doc.paragraphs.iter().any(|p| p.speaker.is_some()),
        "{}",
        doc.speaker_note
    );
}
