pub mod models;
mod speech;
mod youtube;

pub use youtube::video_url;

use crate::{
    Document, Paragraph, Settings,
    job::Progress,
    pipeline::{Stage, Update},
};
use anyhow::{Result, bail, ensure};

struct Cue {
    start: f64,
    end: f64,
    text: String,
    speaker: Option<String>,
}
impl Cue {
    fn new(start: f64, end: f64, text: String) -> Self {
        Self {
            start,
            end,
            text,
            speaker: None,
        }
    }
}

pub fn read_video(link: &str, settings: &Settings, progress: &Progress) -> Result<Document> {
    let dir = progress.media_dir()?;
    let folder = dir.as_path();
    let cache = models::cache_root()?;
    progress.stage(Stage::Fetch)?;
    let link = youtube::video_url(link)?;
    let video = youtube::open(folder, &link, progress)?;
    progress.update(Update::Video {
        title: video.title.clone(),
        duration: video.duration,
    })?;
    let mut transcript = None;
    if settings.model == "captions" || settings.official_captions {
        progress.stage(Stage::Captions)?;
        // Before a speech model, only the uploader's captions are worth taking: YouTube's
        // automatic ones are usually less accurate than the model.
        transcript = match video.captions(progress, settings.model != "captions") {
            Ok(track) => track,
            Err(_) if settings.model == "captions" => bail!(
                "YouTube could not deliver this caption track. Choose a speech model to transcribe its audio."
            ),
            Err(_) => None,
        };
    }
    let mut audio = None;
    let (mut cues, source, language) = if let Some(transcript) = transcript {
        progress.update(caption_preview(&transcript.0))?;
        transcript
    } else {
        ensure!(
            settings.model != "captions",
            "This video has no usable YouTube captions. Choose a speech model to transcribe its audio."
        );
        let wav = youtube::audio(folder, &link, video.duration, progress)?;
        let result = speech::transcribe(&wav, &cache, models::speech(&settings.model)?, progress)?;
        audio = Some(wav);
        result
    };
    let mut speaker_note = String::new();
    if settings.speakers && !cues.is_empty() {
        let analyze = || -> Result<()> {
            let wav = match audio {
                Some(path) => path,
                None => youtube::audio(folder, &link, video.duration, progress)?,
            };
            speech::label_speakers(&wav, &cache, &mut cues, progress)
        };
        speaker_note = match analyze() {
            Ok(()) if cues.iter().any(|c| c.speaker.is_some()) => "Estimated speakers".into(),
            Ok(()) => "No distinct voices detected".into(),
            Err(e) => format!("Speaker analysis unavailable ({e:#})."),
        };
    }
    let paragraphs = paragraphs(cues);
    ensure!(!paragraphs.is_empty(), "No speech was found in this video.");
    Ok(Document {
        title: video.title,
        channel: video.uploader.unwrap_or_default(),
        link,
        source,
        language,
        speaker_note,
        paragraphs,
    })
}

/// A caption track as recognized speech: all of its words, its last few on show.
fn caption_preview(cues: &[Cue]) -> Update {
    let words = cues
        .iter()
        .map(|c| crate::document::word_count(&c.text))
        .sum();
    let start = cues.len().saturating_sub(12);
    Update::Heard {
        text: join_words(cues[start..].iter().map(|c| c.text.as_str())),
        words,
        until: cues.last().map_or(0.0, |c| c.end),
    }
}

fn is_cjk(c: char) -> bool {
    matches!(c, '\u{3040}'..='\u{30ff}' | '\u{3400}'..='\u{9fff}')
}

/// Appends a recognized piece to running text, with a space unless punctuation or CJK script
/// joins them directly. Returns the characters added.
fn append(text: &mut String, piece: &str) -> usize {
    let join = text.is_empty()
        || piece.starts_with(['、', '。', '，', '！', '？', ',', '.', '!', '?', ';', ':'])
        || (text.chars().next_back().is_some_and(is_cjk)
            && piece.chars().next().is_some_and(is_cjk));
    if !join {
        text.push(' ');
    }
    text.push_str(piece);
    usize::from(!join) + piece.chars().count()
}

/// Recognized pieces as one run of text.
pub(super) fn join_words<'a>(pieces: impl Iterator<Item = &'a str>) -> String {
    let mut text = String::new();
    for piece in pieces.map(str::trim).filter(|p| !p.is_empty()) {
        append(&mut text, piece.strip_prefix(">>").unwrap_or(piece).trim());
    }
    text
}

fn ends_sentence(text: &str) -> bool {
    text.trim_end()
        .trim_end_matches(['”', '"', '’', ')'])
        .ends_with(['.', '!', '?', '…', '。', '！', '？'])
}

fn paragraphs(cues: Vec<Cue>) -> Vec<Paragraph> {
    let mut result: Vec<Paragraph> = Vec::new();
    let mut end = 0.;
    let mut length = 0;
    for cue in cues {
        let text = cue.text.trim();
        let turn = text.starts_with(">>");
        let text = text.strip_prefix(">>").unwrap_or(text).trim();
        if text.is_empty() {
            continue;
        }
        let new_paragraph = result.last().is_none_or(|p| {
            let sentence_end = ends_sentence(&p.text);
            turn || p.speaker != cue.speaker
                || cue.start - end >= 2.2
                || length >= 650
                || (length >= 250 && sentence_end)
        });
        if new_paragraph {
            result.push(Paragraph {
                start: cue.start,
                speaker: cue.speaker,
                text: text.into(),
            });
            length = text.chars().count();
        } else if let Some(p) = result.last_mut() {
            length += append(&mut p.text, text);
        }
        end = cue.end;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn separates_turns_pauses_and_cjk_text() {
        let cues = [
            (0., 1., "One", "A"),
            (1., 2., "Two", "B"),
            (5., 6., "你好", "B"),
            (6., 7., "世界", "B"),
            (7., 8., ">> Next", "B"),
        ]
        .into_iter()
        .map(|(start, end, text, who)| Cue {
            speaker: Some(who.into()),
            ..Cue::new(start, end, text.into())
        })
        .collect();
        let paragraphs = paragraphs(cues);
        assert_eq!(
            paragraphs
                .iter()
                .map(|p| p.text.as_str())
                .collect::<Vec<_>>(),
            ["One", "Two", "你好世界", "Next"]
        );
    }

    #[test]
    fn joins_recognized_pieces() {
        let pieces = [" Hello", ",", "world", ">> again", "你", "好"];
        assert_eq!(join_words(pieces.into_iter()), "Hello, world again 你好");
    }
}
