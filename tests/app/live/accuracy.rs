//! Whole videos read as the app reads them, scored against published references: word error
//! rate against the uploader's captions or transcript, speaker accuracy against scripts that
//! name who said each line, and each stage's speed against the length of the audio.

use super::{Run, read_video, reference, report, score};
use crate::{Document, pipeline::Stage};
use std::sync::OnceLock;

struct Video {
    url: &'static str,
    channel: &'static str,
    topic: &'static str,
    min_last_paragraph: f64,
}

const TALK: Video = Video {
    url: "https://youtu.be/JnfBXjWm7hc",
    channel: "TED",
    topic: "30 days",
    min_last_paragraph: 150.0,
};
const FOOD: Video = Video {
    url: "https://youtu.be/Z51Q29u4CWc",
    channel: "BBC Learning English",
    topic: "food waste",
    min_last_paragraph: 300.0,
};
const FOOD_SCRIPT: &str =
    "https://www.bbc.co.uk/learningenglish/english/features/6-minute-english_2022/ep-221229";
const REST: Video = Video {
    url: "https://youtu.be/Y681hXWwhQY",
    channel: "BBC Learning English",
    topic: "doing nothing",
    min_last_paragraph: 300.0,
};
const REST_SCRIPT: &str =
    "https://www.bbc.co.uk/learningenglish/english/features/6-minute-english_2023/ep-230615";
/// Two and a half hours of two voices: where long audio once took half an hour and crashed.
const PODCAST: Video = Video {
    url: "https://www.youtube.com/watch?v=vif8NQcjVf0",
    channel: "Lex Fridman",
    topic: "jensen huang",
    min_last_paragraph: 8400.0,
};
const PODCAST_TRANSCRIPT: &str = "https://lexfridman.com/jensen-huang-transcript/";

fn check_document(doc: &Document, video: &Video) {
    assert_eq!(doc.channel, video.channel);
    assert!(
        doc.title.to_lowercase().contains(video.topic),
        "unexpected title: {}",
        doc.title
    );
    assert!(
        doc.paragraphs
            .iter()
            .all(|p| !p.text.trim().is_empty() && p.start.is_finite() && p.start >= 0.0)
    );
    assert!(doc.paragraphs.windows(2).all(|p| p[0].start <= p[1].start));
    assert!(
        doc.paragraphs
            .last()
            .is_some_and(|p| p.start >= video.min_last_paragraph),
        "missing the end of {}",
        doc.title
    );
}

/// The uploader's own captions, read through the app with a speech model selected: they
/// must win over transcription.
fn creator_reference(video: &Video, model: &str) -> Vec<String> {
    let doc = read_video(video.url, model, true, false).doc;
    assert_eq!(
        doc.source, "YouTube captions",
        "reference must be publisher-supplied"
    );
    assert!(
        doc.language == "en" || doc.language.starts_with("en-"),
        "reference language: {}",
        doc.language
    );
    check_document(&doc, video);
    let words = score::labelled(&doc).0;
    assert!(
        words.len() >= 300,
        "short caption reference: {}",
        words.len()
    );
    words
}

struct Scored {
    run: Run,
    wer: f64,
    speakers: Option<score::SpeakerScore>,
}

/// Reads `video` with `model`, labelling speakers when there's a `script` to score them by.
fn measure(
    test: &str,
    video: &Video,
    model: &str,
    reference: &[String],
    script: Option<&[(String, String)]>,
) -> Scored {
    let run = read_video(video.url, model, false, script.is_some());
    let doc = &run.doc;
    assert_eq!(
        doc.source,
        crate::media::models::speech(model).unwrap().name
    );
    check_document(doc, video);
    let (words, labels) = score::labelled(doc);
    let wer = score::align(reference, &words).wer(reference.len());
    let mut lines = vec![
        format!(
            "{} | {} | WER {:.1}% against {} reference words, {} recognized",
            doc.title,
            doc.source,
            100.0 * wer,
            reference.len(),
            words.len()
        ),
        run.timing(),
    ];
    let speakers = match script {
        Some(turns) => {
            assert_eq!(doc.speaker_note, "Estimated speakers");
            let (said, names) = score::named(turns);
            let speakers = score::speakers(&score::align(&said, &words), &names, &labels);
            lines.push(speakers.summary());
            Some(speakers)
        }
        None => {
            assert!(doc.paragraphs.iter().all(|p| p.speaker.is_none()));
            None
        }
    };
    report(test, &lines);
    Scored { run, wer, speakers }
}

fn at_most_wer(scored: &Scored, max: f64) {
    assert!(
        scored.wer <= max,
        "WER {:.1}% exceeds {:.0}%",
        100.0 * scored.wer,
        100.0 * max
    );
}

fn at_least_speakers(scored: &Scored, min: f64, people: &[(&str, f64)]) {
    let speakers = scored.speakers.as_ref().unwrap();
    assert!(
        speakers.accuracy() >= min,
        "speaker accuracy {:.1}% under {:.0}%: {}",
        100.0 * speakers.accuracy(),
        100.0 * min,
        speakers.summary()
    );
    for (name, min) in people {
        assert!(
            speakers.recall(name) >= *min,
            "{name}'s words {:.1}% labelled theirs, under {:.0}%: {}",
            100.0 * speakers.recall(name),
            100.0 * min,
            speakers.summary()
        );
    }
}

/// On a GPU, at least `min` seconds of audio per second in `stage`. A CPU is only reported.
fn at_least_speed(scored: &Scored, stage: Stage, min: f64) {
    let run = &scored.run;
    if !run.on_gpu() {
        return;
    }
    let speed = run.speed(stage).unwrap();
    assert!(
        speed >= min,
        "{stage:?} at {speed:.0}× real time on {}, under {min:.0}×",
        run.device
    );
}

fn ted(test: &str, model: &str, max_wer: f64) {
    static REFERENCE: OnceLock<Vec<String>> = OnceLock::new();
    let reference = REFERENCE.get_or_init(|| creator_reference(&TALK, "captions"));
    at_most_wer(&measure(test, &TALK, model, reference, None), max_wer);
}

#[test]
#[ignore = "live TED captions and full local transcription"]
fn ted_parakeet() {
    // Measured: 2.5%.
    ted("ted_parakeet", "parakeet_unified_en", 0.05);
}

#[test]
#[ignore = "live TED captions and full local transcription"]
fn ted_whisper_small() {
    // Measured: 10.3%.
    ted("ted_whisper_small", "small", 0.13);
}

#[test]
#[ignore = "live TED captions and full local transcription"]
fn ted_whisper_turbo() {
    // Measured: 2.9%.
    ted("ted_whisper_turbo", "turbo", 0.05);
}

// Each BBC episode has two hosts and three guests heard in recorded clips. The speaker model
// tells at most four voices apart, so the guests may share a label, costing their words.

#[test]
#[ignore = "live BBC podcast and script, Parakeet and speaker analysis"]
fn bbc_food_waste_parakeet_with_speakers() {
    let reference = creator_reference(&FOOD, "parakeet_unified_en");
    let script = reference::bbc_script(FOOD_SCRIPT);
    let scored = measure(
        "bbc_food_waste_parakeet_with_speakers",
        &FOOD,
        "parakeet_unified_en",
        &reference,
        Some(&script),
    );
    // Measured: WER 5.5%; 92.1% of words to their speaker, both hosts 100%.
    at_most_wer(&scored, 0.08);
    at_least_speakers(&scored, 0.88, &[("Neil", 0.95), ("Sam", 0.95)]);
}

#[test]
#[ignore = "live BBC podcast and script, Whisper and speaker analysis"]
fn bbc_doing_nothing_whisper_with_speakers() {
    let reference = creator_reference(&REST, "turbo");
    let script = reference::bbc_script(REST_SCRIPT);
    let scored = measure(
        "bbc_doing_nothing_whisper_with_speakers",
        &REST,
        "turbo",
        &reference,
        Some(&script),
    );
    // Measured: WER 2.7%; 95.4% of words to their speaker, the hosts 100% and 97%.
    at_most_wer(&scored, 0.05);
    at_least_speakers(&scored, 0.9, &[("Neil", 0.95), ("Beth", 0.93)]);
}

#[test]
#[ignore = "live 2½-hour podcast and its transcript, Parakeet and speaker analysis"]
fn long_podcast_parakeet_with_speakers() {
    let transcript = reference::lex_transcript(PODCAST_TRANSCRIPT);
    let reference = score::named(&transcript).0;
    let scored = measure(
        "long_podcast_parakeet_with_speakers",
        &PODCAST,
        "parakeet_unified_en",
        &reference,
        Some(&transcript),
    );
    // Measured: WER 7.0%; 99.7% of words to their speaker. On an RTX 4060 over Vulkan,
    // transcription at 138× real time and speakers at 397×.
    at_most_wer(&scored, 0.09);
    at_least_speakers(
        &scored,
        0.97,
        &[("Lex Fridman", 0.97), ("Jensen Huang", 0.97)],
    );
    at_least_speed(&scored, Stage::Transcribe, 40.0);
    at_least_speed(&scored, Stage::Speakers, 80.0);
}
