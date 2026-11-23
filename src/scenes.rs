//! Demo states for Rune's headless tooling, used for screenshots and visual checks:
//!
//! ```sh
//! cargo run -- --scene working --snap working.png --at 2
//! cargo run -- --scene demo --realtime --video demo.mp4 --duration 14
//! ```

use crate::{
    Document, Paragraph, Screen, State,
    app::Working,
    job::{Event, Job, Progress},
    pipeline::{self, Stage, Tracker, Update},
};
use std::{sync::Arc, thread, time::Duration};

/// A scene's name and how it sets up the window's state.
type Scene = (&'static str, fn(&mut State));

pub const ALL: &[Scene] = &[
    ("demo", demo),
    ("home", home),
    ("working", working),
    ("finished", finished),
    ("reader", reader),
];

fn configured(s: &mut State) {
    s.settings.configured = true;
    s.settings.speakers = true;
    s.settings.model = "parakeet_unified_en".into();
    s.ready_models = vec!["parakeet_unified_en".into()];
}

fn home(s: &mut State) {
    configured(s);
    s.screen = Screen::Home;
}

/// A job that runs until it's dropped, so the scene's meter keeps its pose.
fn idle_job() -> Job {
    Job::start(Arc::new(|| {}), |progress| {
        while progress.check().is_ok() {
            thread::sleep(Duration::from_millis(50));
        }
        Err(anyhow::anyhow!("Cancelled"))
    })
}

const HEARD: [&str; 3] = [
    "So the first thing I tell people is that the transcript isn't the talk. It's a map of it.",
    "You skim the map, you find the part where the argument turns, and then you go back and listen to just that.",
    "And honestly most of the value is in about four minutes of a forty minute conversation.",
];

fn tracker(now: f64) -> Tracker {
    let mut t = Tracker::new(pipeline::plan(&crate::Settings {
        model: "parakeet_unified_en".into(),
        speakers: true,
        ..Default::default()
    }));
    let steps = [
        (Stage::Fetch, -64.0),
        (Stage::Download, -58.0),
        (Stage::Convert, -41.0),
        (Stage::Model, -37.5),
        (Stage::Transcribe, -33.0),
    ];
    t.apply(
        Update::Video {
            title: "How to read a forty minute conversation in four".into(),
            duration: Some(2412.0),
        },
        now - 62.0,
    );
    for (stage, at) in steps {
        t.apply(Update::Stage(stage), now + at);
    }
    t.apply(
        Update::Device("Vulkan · NVIDIA GeForce RTX 4060 Laptop GPU".into()),
        now - 33.0,
    );
    for (i, text) in HEARD.iter().enumerate() {
        t.apply(
            Update::Fraction(0.40 + 0.03 * i as f32),
            now - 4.0 + i as f64 * 2.0,
        );
        t.apply(
            Update::Heard {
                text: (*text).into(),
                words: 640 + i * 61,
                until: 1000.0 + i as f64 * 30.0,
            },
            now - 4.0 + i as f64 * 2.0,
        );
    }
    t.shown = t.target(now);
    t
}

/// The whole pipeline in about nine seconds, through the real progress plumbing.
fn demo(s: &mut State) {
    configured(s);
    let wake = s.wake.clone().unwrap_or_else(|| Arc::new(|| {}));
    let plan = pipeline::plan(&s.settings);
    s.screen = Screen::Working(Box::new(Working {
        job: Job::start(wake, play),
        tracker: Tracker::new(plan),
        result: None,
    }));
}

fn play(p: &Progress) -> anyhow::Result<Event> {
    let pause = |seconds: f64| thread::sleep(Duration::from_secs_f64(seconds));
    let ramp = |seconds: f64, steps: u32| -> anyhow::Result<()> {
        for i in 1..=steps {
            pause(seconds / f64::from(steps));
            p.fraction(i as f32 / steps as f32)?;
        }
        Ok(())
    };
    pause(0.6);
    p.stage(Stage::Fetch)?;
    pause(0.9);
    p.update(Update::Video {
        title: "How to read a forty minute conversation in four".into(),
        duration: Some(2412.0),
    })?;
    p.stage(Stage::Download)?;
    ramp(1.3, 26)?;
    p.stage(Stage::Convert)?;
    ramp(0.5, 10)?;
    p.stage(Stage::Model)?;
    pause(0.7);
    p.update(Update::Device(
        "Vulkan · NVIDIA GeForce RTX 4060 Laptop GPU".into(),
    ))?;
    p.stage(Stage::Transcribe)?;
    let lines = SAMPLE.iter().map(|(_, _, text)| *text).collect::<Vec<_>>();
    for (i, text) in lines.iter().enumerate() {
        pause(0.55);
        let done = (i + 1) as f32 / lines.len() as f32;
        p.fraction(done)?;
        p.update(Update::Heard {
            text: (*text).into(),
            words: crate::document::word_count(text) * 9,
            until: f64::from(done) * 2412.0,
        })?;
    }
    p.stage(Stage::Speakers)?;
    pause(1.2);
    Ok(Event::Result(sample()))
}

fn working(s: &mut State) {
    configured(s);
    s.screen = Screen::Working(Box::new(Working {
        job: idle_job(),
        tracker: tracker(s.clock),
        result: None,
    }));
}

fn finished(s: &mut State) {
    configured(s);
    let mut t = tracker(s.clock);
    t.apply(Update::Stage(Stage::Speakers), s.clock - 6.0);
    t.finish(s.clock);
    t.shown = 1.0;
    s.screen = Screen::Working(Box::new(Working {
        job: idle_job(),
        tracker: t,
        result: Some(sample()),
    }));
}

fn reader(s: &mut State) {
    configured(s);
    s.open_reader(sample());
}

const SAMPLE: [(f64, &str, &str); 8] = [
    (
        0.0,
        "Speaker 1",
        "Welcome back. Today I want to talk about something that sounds small but changed how I spend my week: reading conversations instead of watching them.",
    ),
    (
        14.0,
        "Speaker 2",
        "I was skeptical when you first mentioned it. A two hour interview is a two hour interview. What does reading it actually buy you?",
    ),
    (
        26.5,
        "Speaker 1",
        "Speed, mostly, but not in the way people expect. Most of us read around two hundred and fifty words a minute, and most people speak at about one fifty. So the transcript is already faster. The bigger win is that you can see the shape of the whole thing at once. You notice where the argument turns. You notice the part where everyone suddenly starts interrupting each other, which is usually where the interesting disagreement is.",
    ),
    (71.0, "Speaker 2", "So you skim first."),
    (
        73.5,
        "Speaker 1",
        "I skim first, then I slow down. And when a passage really matters, I click its timestamp and listen to just that minute. Tone carries a lot. Somebody saying \u{201c}that's fine\u{201d} can mean three different things.",
    ),
    (
        98.0,
        "Speaker 2",
        "Do you worry about losing something? There's a kind of patience you build by sitting with a long conversation.",
    ),
    (
        109.0,
        "Speaker 1",
        "Sometimes. But I think the patience moves somewhere else. Instead of sitting through the parts I don't need, I spend it rereading the parts I do. I end up quoting people more accurately, too, because I have the words in front of me rather than my memory of them.",
    ),
    (
        140.0,
        "Speaker 2",
        "That's probably the best argument for it I've heard. Accuracy, not speed.",
    ),
];

fn sample() -> Document {
    Document {
        title: "How to read a forty minute conversation in four".into(),
        channel: "The Long Form".into(),
        link: "https://www.youtube.com/watch?v=jNQXAC9IVRw".into(),
        source: "Parakeet Unified English".into(),
        language: "en".into(),
        speaker_note: "Estimated speakers".into(),
        paragraphs: SAMPLE
            .iter()
            .map(|(start, speaker, text)| Paragraph {
                start: *start,
                speaker: Some((*speaker).into()),
                text: (*text).into(),
            })
            .collect(),
    }
}
