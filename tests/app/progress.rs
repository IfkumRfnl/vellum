use super::*;
use crate::{
    app::Working,
    pipeline::{self, Stage, Tracker, Update},
};

fn long_document() -> Document {
    Document {
        title: "A long talk".into(),
        channel: "Channel".into(),
        link: "https://www.youtube.com/watch?v=jNQXAC9IVRw".into(),
        source: "Captions".into(),
        language: "en".into(),
        speaker_note: String::new(),
        paragraphs: (0..80)
            .map(|i| Paragraph {
                start: i as f64 * 20.0,
                speaker: None,
                text: "Words to read on a long and patient page. ".repeat(8),
            })
            .collect(),
    }
}

#[test]
fn paging_through_the_reader_moves_its_progress() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = load_state(Some(dir.path().join("settings.json")));
    s.settings.configured = true;
    s.open_reader(long_document());
    let mut rt = Runtime::new(s, ui::view, (980.0, 780.0), Env::default());
    rt.on_key = Some(ui::on_key);
    rt.frame(0.0);
    assert_eq!(rt.state.reading_fraction(), 0.0);
    for _ in 0..3 {
        rt.key(rune::KeyEvent {
            key: rune::Key::Text(" ".into()),
            mods: Default::default(),
        });
        for _ in 0..40 {
            rt.frame(0.02);
        }
    }
    let after = rt.state.reading_fraction();
    assert!(after > 0.0 && after < 1.0, "fraction {after}");
    rt.key(rune::KeyEvent {
        key: rune::Key::End,
        mods: Default::default(),
    });
    for _ in 0..120 {
        rt.frame(0.02);
    }
    assert!(rt.state.reading_fraction() > 0.99);
    assert!(rt.find("reading-pill").is_some());
}

#[test]
fn the_progress_screen_celebrates_then_opens_the_transcript() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = load_state(Some(dir.path().join("settings.json")));
    s.settings.configured = true;
    let mut tracker = Tracker::new(pipeline::plan(&s.settings));
    tracker.apply(Update::Stage(Stage::Fetch), 0.0);
    tracker.apply(Update::Stage(Stage::Transcribe), 0.5);
    tracker.apply(Update::Fraction(0.5), 0.6);
    let (send, events) = std::sync::mpsc::channel();
    s.screen = Screen::Working(Box::new(Working {
        job: Job::start(Arc::new(|| {}), move |progress| {
            // Finish once the test says so.
            let _: () = events.recv()?;
            progress.check()?;
            Ok(Event::Result(long_document()))
        }),
        tracker,
        result: None,
    }));
    let mut rt = Runtime::new(s, ui::view, (980.0, 780.0), Env::default());
    rt.poll = Some(poll);
    rt.frame(0.0);
    assert!(rt.find("plate").is_some());
    for _ in 0..30 {
        rt.frame(0.02);
    }
    let Screen::Working(w) = &rt.state.screen else {
        panic!("left the progress screen early");
    };
    assert!(w.tracker.shown > 0.1, "meter at {}", w.tracker.shown);
    send.send(()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while matches!(rt.state.screen, Screen::Working(_)) && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(10));
        rt.frame(0.05);
    }
    assert!(matches!(rt.state.screen, Screen::Reader));
    assert_eq!(rt.state.doc.as_ref().unwrap().title, "A long talk");
}

#[test]
fn escape_cancels_a_transcription() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = load_state(Some(dir.path().join("settings.json")));
    s.settings.configured = true;
    s.screen = Screen::Working(Box::new(Working {
        job: Job::start(Arc::new(|| {}), |progress| {
            while progress.check().is_ok() {
                thread::sleep(Duration::from_millis(10));
            }
            Err(anyhow::anyhow!("Cancelled"))
        }),
        tracker: Tracker::new(vec![Stage::Fetch]),
        result: None,
    }));
    let mut rt = Runtime::new(s, ui::view, (980.0, 780.0), Env::default());
    rt.on_key = Some(ui::on_key);
    rt.frame(0.0);
    rt.key(rune::KeyEvent {
        key: rune::Key::Escape,
        mods: Default::default(),
    });
    rt.frame(0.02);
    assert!(matches!(rt.state.screen, Screen::Home));
}
