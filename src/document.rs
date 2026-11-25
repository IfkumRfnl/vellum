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
        let mut previous = None;
        self.paragraphs
            .iter()
            .map(|p| {
                let turn = previous != Some(&p.speaker);
                previous = Some(&p.speaker);
                match &p.speaker {
                    Some(speaker) if turn => format!("{speaker}\n{}", p.text),
                    _ => p.text.clone(),
                }
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
