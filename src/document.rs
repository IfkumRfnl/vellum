//! A finished transcript.

pub struct Paragraph {
    pub start: f64,
    pub speaker: Option<String>,
    pub text: String,
}

pub struct Document {
    pub title: String,
    pub channel: String,
    /// The canonical watch link, for jumping to a paragraph's moment in the video.
    pub link: String,
    pub source: String,
    pub language: String,
    pub speaker_note: String,
    pub paragraphs: Vec<Paragraph>,
}

/// Words as a reader counts them: whitespace-separated runs, and each CJK character.
pub fn word_count(text: &str) -> usize {
    let cjk = |c: char| matches!(c, '\u{3040}'..='\u{30ff}' | '\u{3400}'..='\u{9fff}');
    text.split_whitespace()
        .map(|w| {
            let ideographs = w.chars().filter(|c| cjk(*c)).count();
            if ideographs > 0 {
                ideographs + usize::from(w.chars().any(|c| !cjk(c) && c.is_alphanumeric()))
            } else {
                1
            }
        })
        .sum()
}

impl Document {
    pub fn words(&self) -> usize {
        self.paragraphs.iter().map(|p| word_count(&p.text)).sum()
    }

    /// The whole transcript as plain text, each speaker's name on its own line before their turn.
    pub fn plain_text(&self) -> String {
        self.paragraphs
            .iter()
            .map(|p| match &p.speaker {
                Some(speaker) => format!("{speaker}\n{}", p.text),
                None => p.text.clone(),
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// The video at `seconds`, for a timestamp's link.
    pub fn link_at(&self, seconds: f64) -> Option<String> {
        (!self.link.is_empty()).then(|| format!("{}&t={}s", self.link, seconds.max(0.0) as u64))
    }
}

pub fn timestamp(seconds: f64) -> String {
    let n = seconds.max(0.0) as u64;
    if n >= 3600 {
        format!("{}:{:02}:{:02}", n / 3600, n / 60 % 60, n % 60)
    } else {
        format!("{}:{:02}", n / 60, n % 60)
    }
}

/// "1 min", "12 min", "1 h 05 min": a reading time.
pub fn minutes(seconds: f64) -> String {
    let m = (seconds / 60.0).round().max(1.0) as u64;
    if m >= 60 {
        format!("{} h {:02} min", m / 60, m % 60)
    } else {
        format!("{m} min")
    }
}

/// "12 min", "1h 5m": a compact reading time for tight spaces.
pub fn short_minutes(seconds: f64) -> String {
    let m = (seconds / 60.0).round().max(1.0) as u64;
    if m >= 60 {
        format!("{}h {}m", m / 60, m % 60)
    } else {
        format!("{m} min")
    }
}

/// 12,480
pub fn grouped(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_words_and_ideographs() {
        assert_eq!(word_count("I read the  wrong word."), 5);
        assert_eq!(word_count("你好世界"), 4);
        assert_eq!(word_count(""), 0);
    }

    #[test]
    fn copies_speakers_on_their_own_lines() {
        let doc = Document {
            title: String::new(),
            channel: String::new(),
            link: String::new(),
            source: String::new(),
            language: String::new(),
            speaker_note: String::new(),
            paragraphs: vec![
                Paragraph {
                    start: 0.0,
                    speaker: Some("Speaker 1".into()),
                    text: "Hello.".into(),
                },
                Paragraph {
                    start: 2.0,
                    speaker: None,
                    text: "Bye.".into(),
                },
            ],
        };
        assert_eq!(doc.plain_text(), "Speaker 1\nHello.\n\nBye.");
    }

    #[test]
    fn formats_numbers_and_times() {
        assert_eq!(grouped(0), "0");
        assert_eq!(grouped(1234567), "1,234,567");
        assert_eq!(timestamp(3725.0), "1:02:05");
        assert_eq!(minutes(20.0), "1 min");
        assert_eq!(minutes(3900.0), "1 h 05 min");
        assert_eq!(short_minutes(3900.0), "1h 5m");
    }
}
