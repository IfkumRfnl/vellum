use super::{Document, read_video};
use std::collections::HashMap;
use std::sync::OnceLock;

struct Video {
    url: &'static str,
    channel: &'static str,
    topic: &'static str,
    min_words: usize,
    min_last_paragraph: f64,
}

const TALK: Video = Video {
    url: "https://youtu.be/JnfBXjWm7hc",
    channel: "TED",
    topic: "30 days",
    min_words: 340,
    min_last_paragraph: 150.0,
};
const FOOD: Video = Video {
    url: "https://youtu.be/Z51Q29u4CWc",
    channel: "BBC Learning English",
    topic: "food waste",
    min_words: 600,
    min_last_paragraph: 300.0,
};
const REST: Video = Video {
    url: "https://youtu.be/Y681hXWwhQY",
    channel: "BBC Learning English",
    topic: "doing nothing",
    min_words: 600,
    min_last_paragraph: 300.0,
};

fn words(doc: &Document) -> Vec<String> {
    doc.paragraphs
        .iter()
        .flat_map(|p| p.text.split_whitespace())
        .filter_map(|s| {
            let word: String = s
                .chars()
                .filter(|c| c.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect();
            (!word.is_empty()).then_some(word)
        })
        .collect()
}

// Word error rate counts substitutions, insertions and deletions. Only two
// rows of the alignment are needed, even for a complete podcast episode.
fn word_errors(reference: &[String], actual: &[String]) -> usize {
    let mut previous: Vec<usize> = (0..=actual.len()).collect();
    let mut current = vec![0; previous.len()];
    for (i, expected) in reference.iter().enumerate() {
        current[0] = i + 1;
        for (j, got) in actual.iter().enumerate() {
            current[j + 1] = (previous[j] + usize::from(expected != got))
                .min(previous[j + 1] + 1)
                .min(current[j] + 1);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[actual.len()]
}

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

fn creator_reference(video: &Video, model: &str) -> Vec<String> {
    // With a speech model selected this also exercises captions-first routing.
    let doc = read_video(video.url, model, true, false);
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
    let reference = words(&doc);
    assert!(
        reference.len() >= video.min_words,
        "short caption reference: {} words",
        reference.len()
    );
    reference
}

fn compare(
    video: &Video,
    model: &str,
    reference: &[String],
    label_speakers: bool,
    max_wer: f64,
) -> Document {
    let doc = read_video(video.url, model, false, label_speakers);
    assert_eq!(
        doc.source,
        crate::media::models::speech(model).unwrap().name
    );
    check_document(&doc, video);
    let hypothesis = words(&doc);
    let wer = word_errors(reference, &hypothesis) as f64 / reference.len() as f64;
    eprintln!(
        "{} | {} | {} reference words | {} recognized words | WER {:.1}%",
        doc.title,
        doc.source,
        reference.len(),
        hypothesis.len(),
        100.0 * wer
    );
    assert!(
        wer <= max_wer,
        "{} / {}: WER {:.1}% exceeds {:.0}%",
        doc.title,
        doc.source,
        100.0 * wer,
        100.0 * max_wer
    );
    if label_speakers {
        assert_eq!(doc.speaker_note, "Estimated speakers");
        let mut counts = HashMap::new();
        for p in &doc.paragraphs {
            if let Some(speaker) = &p.speaker {
                *counts.entry(speaker).or_insert(0usize) += p.text.split_whitespace().count();
            }
        }
        eprintln!("{} | words assigned per speaker: {counts:?}", doc.title);
        assert!(
            counts.values().filter(|n| **n >= 20).count() >= 2,
            "expected substantial turns for at least two voices"
        );
        assert!(
            counts.values().sum::<usize>() * 5 >= hypothesis.len() * 4,
            "fewer than 80% of words have a speaker label"
        );
    } else {
        assert!(doc.paragraphs.iter().all(|p| p.speaker.is_none()));
    }
    doc
}

fn ted(model: &str, max_wer: f64) {
    static REFERENCE: OnceLock<Vec<String>> = OnceLock::new();
    let reference = REFERENCE.get_or_init(|| creator_reference(&TALK, "captions"));
    compare(&TALK, model, reference, false, max_wer);
}

#[test]
#[ignore = "live TED captions and full local transcription"]
fn ted_parakeet() {
    ted("parakeet_unified_en", 0.08);
}

#[test]
#[ignore = "live TED captions and full local transcription"]
fn ted_whisper_small() {
    ted("small", 0.16);
}

#[test]
#[ignore = "live TED captions and full local transcription"]
fn ted_whisper_turbo() {
    ted("turbo", 0.08);
}

#[test]
#[ignore = "live BBC podcast, captions-first, Parakeet and speaker analysis"]
fn bbc_food_waste_parakeet_with_speakers() {
    let reference = creator_reference(&FOOD, "parakeet_unified_en");
    let doc = compare(&FOOD, "parakeet_unified_en", &reference, true, 0.12);
    // The BBC script identifies Neil and Sam independently of the arbitrary
    // speaker numbers assigned by the model. Both hosts return after a turn.
    let speaker = |phrase: &str| {
        doc.paragraphs
            .iter()
            .find(|p| p.text.to_lowercase().contains(phrase))
            .and_then(|p| p.speaker.as_deref())
            .unwrap_or_else(|| panic!("missing labelled phrase: {phrase}"))
    };
    let neil = speaker("kitchen refrigerator");
    let sam = speaker("guilty of wasting food");
    assert_ne!(neil, sam, "Neil and Sam were merged");
    assert_eq!(
        sam,
        speaker("import food"),
        "Sam was split across two labels"
    );
    assert_eq!(
        neil,
        speaker("local area"),
        "Neil was split across two labels"
    );
    // The reporter and two interviewees would make five voices: the speaker model tells
    // apart at most four, so they may share a label, but never a host's.
    for phrase in ["giant avocados", "big truck", "wrong temperature"] {
        let voice = speaker(phrase);
        assert!(
            voice != neil && voice != sam,
            "{phrase:?} was given to a host: {voice}"
        );
    }
}

#[test]
#[ignore = "live BBC podcast, captions-first, Whisper and speaker analysis"]
fn bbc_doing_nothing_whisper_with_speakers() {
    let reference = creator_reference(&REST, "turbo");
    compare(&REST, "turbo", &reference, true, 0.12);
}
