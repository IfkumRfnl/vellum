//! Text as the words that are compared.
//!
//! Two modes. `Plain` is for one engine against another: lowercase letters, digits and marks,
//! punctuation dropped, nothing else changed, so any difference in what the engines wrote shows
//! (tests/app/live/score.rs's `words`, which also drops marks).
//! `Asr` is for an engine against ground truth: Whisper's normalizers as the Open ASR
//! Leaderboard applies them, so "Mr. Smith's 2nd colour" and "mister smith is second color"
//! score the same, and a WER is comparable with published ones.
//!
//! Languages written without spaces (Japanese, Chinese, Thai…) are split into characters in
//! both modes, which makes their error rate a character error rate.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::OnceLock};
use unicode_normalization::{UnicodeNormalization, char::is_combining_mark};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Plain,
    Asr,
}

/// `text` as words to compare. `language` is an ISO 639-1 code (`en`, `ja`…); empty or `en`
/// takes the English normalizer in `Asr` mode.
pub fn words(text: &str, mode: Mode, language: &str) -> Vec<String> {
    let language = language
        .split(['-', '_'])
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    let normalized;
    let text = match mode {
        Mode::Plain => text,
        Mode::Asr if language.is_empty() || language == "en" => {
            normalized = english(text);
            &normalized
        }
        Mode::Asr => {
            normalized = basic(text);
            &normalized
        }
    };
    let by_character = CHARACTER_LANGUAGES.contains(&language.as_str());
    let mut out = Vec::new();
    for token in text.split_whitespace() {
        let word: String = match mode {
            // Marks too: a Hindi virama or a Thai tone mark is part of the word.
            Mode::Plain => token
                .chars()
                .filter(|c| c.is_alphanumeric() || is_combining_mark(*c))
                .flat_map(char::to_lowercase)
                .collect(),
            Mode::Asr => token.to_owned(),
        };
        if word.is_empty() {
            continue;
        }
        if by_character || word.chars().any(unspaced) {
            split_characters(&word, &mut out);
        } else {
            out.push(word);
        }
    }
    out
}

/// Languages scored by character: written without spaces between words.
const CHARACTER_LANGUAGES: &[&str] = &["ja", "zh", "yue", "th", "lo", "my", "km", "bo"];

/// Scripts written without spaces: a run of them is split into characters whatever the language.
fn unspaced(c: char) -> bool {
    matches!(c,
        '\u{3040}'..='\u{30ff}' // kana
        | '\u{3400}'..='\u{4dbf}' | '\u{4e00}'..='\u{9fff}' | '\u{f900}'..='\u{faff}' // Han
        | '\u{ff66}'..='\u{ff9f}' // half-width katakana
        | '\u{0e00}'..='\u{0eff}' // Thai, Lao
        | '\u{1000}'..='\u{109f}' // Myanmar
        | '\u{1780}'..='\u{17ff}' // Khmer
        | '\u{0f00}'..='\u{0fff}' // Tibetan
    )
}

/// Each character with the marks that follow it (Thai vowels, accents) as one unit.
fn split_characters(word: &str, out: &mut Vec<String>) {
    let mut current = String::new();
    for c in word.chars() {
        if !(is_combining_mark(c) && !current.is_empty()) && !current.is_empty() {
            out.push(std::mem::take(&mut current));
        }
        current.push(c);
    }
    if !current.is_empty() {
        out.push(current);
    }
}

fn re(pattern: &str) -> Regex {
    Regex::new(pattern).unwrap()
}

/// Whisper's `BasicTextNormalizer`, keeping marks: lowercase, bracketed and parenthesized
/// asides removed, symbols and punctuation turned to spaces. Whisper's turns marks to spaces
/// too, which splits Hindi or Tamil words at every vowel sign.
pub fn basic(text: &str) -> String {
    static BRACKETS: OnceLock<(Regex, Regex)> = OnceLock::new();
    let (brackets, parens) =
        BRACKETS.get_or_init(|| (re(r"[<\[][^>\]]*[>\]]"), re(r"\(([^)]+?)\)")));
    let s = text.to_lowercase();
    let s = brackets.replace_all(&s, "");
    let s = parens.replace_all(&s, "");
    let s: String = s
        .nfkc()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() || is_combining_mark(c) {
                c
            } else {
                ' '
            }
        })
        .collect();
    s.to_lowercase()
}

struct English {
    brackets: Regex,
    parens: Regex,
    fillers: Regex,
    apostrophe: Regex,
    replacers: Vec<(Regex, &'static str)>,
    digit_commas: Regex,
    periods: Regex,
}

/// Whisper's `EnglishTextNormalizer` contraction and title rules, in its order.
const REPLACERS: &[(&str, &str)] = &[
    // Not Whisper's: its "o" in "five o clock" reads as a zero, "50 clock".
    (r"\bo\s*'\s*clock\b", "oclock"),
    (r"\bwon't\b", "will not"),
    (r"\bcan't\b", "can not"),
    (r"\blet's\b", "let us"),
    (r"\bain't\b", "aint"),
    (r"\by'all\b", "you all"),
    (r"\bwanna\b", "want to"),
    (r"\bgotta\b", "got to"),
    (r"\bgonna\b", "going to"),
    (r"\bi'ma\b", "i am going to"),
    (r"\bimma\b", "i am going to"),
    (r"\bwoulda\b", "would have"),
    (r"\bcoulda\b", "could have"),
    (r"\bshoulda\b", "should have"),
    (r"\bma'am\b", "madam"),
    (r"\bmr\b", "mister "),
    (r"\bmrs\b", "missus "),
    (r"\bst\b", "saint "),
    (r"\bdr\b", "doctor "),
    (r"\bprof\b", "professor "),
    (r"\bcapt\b", "captain "),
    (r"\bgov\b", "governor "),
    (r"\bald\b", "alderman "),
    (r"\bgen\b", "general "),
    (r"\bsen\b", "senator "),
    (r"\brep\b", "representative "),
    (r"\bpres\b", "president "),
    (r"\brev\b", "reverend "),
    (r"\bhon\b", "honorable "),
    (r"\basst\b", "assistant "),
    (r"\bassoc\b", "associate "),
    (r"\blt\b", "lieutenant "),
    (r"\bcol\b", "colonel "),
    (r"\bjr\b", "junior "),
    (r"\bsr\b", "senior "),
    (r"\besq\b", "esquire "),
    (r"'d been\b", " had been"),
    (r"'s been\b", " has been"),
    (r"'d gone\b", " had gone"),
    (r"'s gone\b", " has gone"),
    (r"'d done\b", " had done"),
    (r"'s got\b", " has got"),
    (r"n't\b", " not"),
    (r"'re\b", " are"),
    (r"'s\b", " is"),
    (r"'d\b", " would"),
    (r"'ll\b", " will"),
    (r"'t\b", " not"),
    (r"'ve\b", " have"),
    (r"'m\b", " am"),
];

/// Whisper's `EnglishTextNormalizer`: lowercase; asides in brackets and parentheses, and
/// hesitations, removed; contractions and titles spelled out; numbers as digits; British
/// spellings as American; punctuation and symbols removed except where they belong to a number.
pub fn english(text: &str) -> String {
    static RULES: OnceLock<English> = OnceLock::new();
    let rules = RULES.get_or_init(|| English {
        brackets: re(r"[<\[][^>\]]*[>\]]"),
        parens: re(r"\(([^)]+?)\)"),
        // Whisper's hesitations, and "er", "erm" and "uhm" as British transcripts spell them.
        fillers: re(r"\b(hmm|mm|mhm|mmm|mhmm|uh|um|uhm|er|erm)\b"),
        apostrophe: re(r"\s+'"),
        replacers: REPLACERS.iter().map(|(p, r)| (re(p), *r)).collect(),
        digit_commas: re(r"(\d),(\d)"),
        periods: re(r"\.([^0-9]|$)"),
    });
    // Curly apostrophes as straight ones, so "don’t" is "do not" like "don't".
    let s = text.to_lowercase().replace(['’', '‘', 'ʼ'], "'");
    let s = rules.brackets.replace_all(&s, "");
    let s = rules.parens.replace_all(&s, "");
    let s = rules.fillers.replace_all(&s, "");
    let mut s = rules.apostrophe.replace_all(&s, "'").into_owned();
    for (pattern, replacement) in &rules.replacers {
        if let std::borrow::Cow::Owned(replaced) = pattern.replace_all(&s, *replacement) {
            s = replaced;
        }
    }
    let s = rules.digit_commas.replace_all(&s, "${1}${2}");
    let s = rules.periods.replace_all(&s, " ${1}");
    let s = remove_symbols_and_diacritics(&s);
    let s = numbers(&s);
    let s = spellings(&s);
    // Symbols that aren't part of a number go: "$" not before a digit, "%" not after one.
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    for (i, &c) in chars.iter().enumerate() {
        let digit_after = chars.get(i + 1).is_some_and(|c| c.is_ascii_digit());
        let digit_before = i > 0 && chars[i - 1].is_ascii_digit();
        match c {
            '.' | '$' | '¢' | '€' | '£' if !digit_after => out.push(' '),
            '%' if !digit_before => out.push(' '),
            c => out.push(c),
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// NFKD, accents dropped, letters like "æ" and "ß" spelled out, symbols and punctuation as
/// spaces, except the ones numbers keep (`.%$¢€£`).
fn remove_symbols_and_diacritics(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.nfkd() {
        match c {
            '.' | '%' | '$' | '¢' | '€' | '£' => out.push(c),
            'œ' => out.push_str("oe"),
            'Œ' => out.push_str("OE"),
            'ø' => out.push('o'),
            'Ø' => out.push('O'),
            'æ' => out.push_str("ae"),
            'Æ' => out.push_str("AE"),
            'ß' => out.push_str("ss"),
            'ẞ' => out.push_str("SS"),
            'đ' | 'ð' => out.push('d'),
            'Đ' | 'Ð' => out.push('D'),
            'þ' | 'Þ' => out.push_str("th"),
            'ł' => out.push('l'),
            'Ł' => out.push('L'),
            c if is_combining_mark(c) => {}
            c if c.is_alphanumeric() || c.is_whitespace() => out.push(c),
            _ => out.push(' '),
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Numbers: Whisper's `EnglishNumberNormalizer`.

const ONES: &[&str] = &[
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const TENS: &[(&str, i128)] = &[
    ("twenty", 20),
    ("thirty", 30),
    ("forty", 40),
    ("fifty", 50),
    ("sixty", 60),
    ("seventy", 70),
    ("eighty", 80),
    ("ninety", 90),
];
const MULTIPLIERS: &[&str] = &[
    "hundred",
    "thousand",
    "million",
    "billion",
    "trillion",
    "quadrillion",
    "quintillion",
    "sextillion",
    "septillion",
    "octillion",
    "nonillion",
    "decillion",
];

/// What a number word is.
#[derive(Clone, Copy)]
enum Kind {
    Zero,
    One(i128),
    OneSuffixed(i128, &'static str),
    Ten(i128),
    TenSuffixed(i128, &'static str),
    Multiplier(i128),
    MultiplierSuffixed(i128, &'static str),
    /// "minus", "plus": a sign before a number.
    Preceding(&'static str),
    /// "dollars", "pounds", "cents": a currency after a number.
    Following(&'static str),
    /// "percent", or "per" before "cent".
    Percent,
    Per,
    And,
    Double,
    Triple,
    Point,
}

fn number_words() -> &'static HashMap<String, Kind> {
    static WORDS: OnceLock<HashMap<String, Kind>> = OnceLock::new();
    WORDS.get_or_init(|| {
        let mut m = HashMap::new();
        for z in ["o", "oh", "zero"] {
            m.insert(z.to_owned(), Kind::Zero);
        }
        for (i, name) in ONES.iter().enumerate() {
            let value = i as i128 + 1;
            m.insert(name.to_string(), Kind::One(value));
            let plural = if *name == "six" {
                "sixes".to_owned()
            } else {
                format!("{name}s")
            };
            m.insert(plural, Kind::OneSuffixed(value, "s"));
            let ordinal = match value {
                1 => "first".to_owned(),
                2 => "second".to_owned(),
                3 => "third".to_owned(),
                5 => "fifth".to_owned(),
                12 => "twelfth".to_owned(),
                _ if name.ends_with('t') => format!("{name}h"),
                _ => format!("{name}th"),
            };
            let suffix = match value {
                1 => "st",
                2 => "nd",
                3 => "rd",
                _ => "th",
            };
            m.insert(ordinal, Kind::OneSuffixed(value, suffix));
            // Whisper's table spells it "nineth", so "ninth" stayed a word.
            if value == 9 {
                m.insert("ninth".to_owned(), Kind::OneSuffixed(9, "th"));
            }
        }
        m.insert("zeroth".to_owned(), Kind::OneSuffixed(0, "th"));
        for (name, value) in TENS {
            m.insert(name.to_string(), Kind::Ten(*value));
            m.insert(name.replace('y', "ies"), Kind::TenSuffixed(*value, "s"));
            m.insert(name.replace('y', "ieth"), Kind::TenSuffixed(*value, "th"));
        }
        for (i, name) in MULTIPLIERS.iter().enumerate() {
            let value = if i == 0 { 100 } else { 1000i128.pow(i as u32) };
            m.insert(name.to_string(), Kind::Multiplier(value));
            m.insert(format!("{name}s"), Kind::MultiplierSuffixed(value, "s"));
            m.insert(format!("{name}th"), Kind::MultiplierSuffixed(value, "th"));
        }
        for (name, sign) in [
            ("minus", "-"),
            ("negative", "-"),
            ("plus", "+"),
            ("positive", "+"),
        ] {
            m.insert(name.to_owned(), Kind::Preceding(sign));
        }
        for (name, symbol) in [
            ("pound", "£"),
            ("pounds", "£"),
            ("euro", "€"),
            ("euros", "€"),
            ("dollar", "$"),
            ("dollars", "$"),
            ("cent", "¢"),
            ("cents", "¢"),
        ] {
            m.insert(name.to_owned(), Kind::Following(symbol));
        }
        m.insert("percent".to_owned(), Kind::Percent);
        m.insert("per".to_owned(), Kind::Per);
        m.insert("and".to_owned(), Kind::And);
        m.insert("double".to_owned(), Kind::Double);
        m.insert("triple".to_owned(), Kind::Triple);
        m.insert("point".to_owned(), Kind::Point);
        m
    })
}

/// A number being read: an integer, or digits as written ("1990", "3.5", "007").
#[derive(Clone, Debug)]
enum Value {
    Int(i128),
    Text(String),
}

impl Value {
    fn text(&self) -> String {
        match self {
            Value::Int(v) => v.to_string(),
            Value::Text(s) => s.clone(),
        }
    }
}

/// `s` as an exact decimal `numerator / 10^scale`, if it's digits with at most one point.
fn decimal(s: &str) -> Option<(i128, u32)> {
    let (whole, fraction) = s.split_once('.').unwrap_or((s, ""));
    if whole.is_empty() && fraction.is_empty()
        || !whole
            .chars()
            .chain(fraction.chars())
            .all(|c| c.is_ascii_digit())
        || fraction.len() > 30
    {
        return None;
    }
    let mut n: i128 = 0;
    for c in whole.chars().chain(fraction.chars()) {
        n = n.checked_mul(10)?.checked_add((c as u8 - b'0') as i128)?;
    }
    Some((n, fraction.len() as u32))
}

/// `value × multiplier` if it's a whole number.
fn times(value: &Value, multiplier: i128) -> Option<i128> {
    let (n, scale) = match value {
        Value::Int(v) => (*v, 0),
        Value::Text(s) => decimal(s)?,
    };
    let product = n.checked_mul(multiplier)?;
    let divisor = 10i128.checked_pow(scale)?;
    (product % divisor == 0).then(|| product / divisor)
}

fn is_arabic(s: &str) -> bool {
    let (whole, fraction) = match s.split_once('.') {
        Some((w, f)) => (w, Some(f)),
        None => (s, None),
    };
    !whole.is_empty()
        && whole.chars().all(|c| c.is_ascii_digit())
        && fraction.is_none_or(|f| !f.is_empty() && f.chars().all(|c| c.is_ascii_digit()))
}

/// Spelled-out numbers as digits: "two thousand and five" is "2005", "nineteen ninety" is
/// "1990", "twenty first" is "21st", "five percent" is "5%", "ten dollars" is "$10", "one oh
/// one" is "101", "three point one four" is "3.14", "two and a half" is "2.5".
pub fn numbers(s: &str) -> String {
    let s = preprocess(s);
    let words: Vec<&str> = s.split_whitespace().collect();
    let out = process(&words);
    postprocess(&out.join(" "))
}

fn preprocess(s: &str) -> String {
    static RULES: OnceLock<[Regex; 4]> = OnceLock::new();
    let [half, letter_digit, digit_letter, suffix] = RULES.get_or_init(|| {
        [
            re(r"\band\s+a\s+half\b"),
            re(r"([a-z])([0-9])"),
            re(r"([0-9])([a-z])"),
            re(r"([0-9])\s+(st|nd|rd|th|s)\b"),
        ]
    });
    let words = number_words();
    // "<number> and a half" as "<number> point five".
    let segments: Vec<&str> = half.split(s).collect();
    let mut results: Vec<&str> = Vec::new();
    for (i, segment) in segments.iter().enumerate() {
        if i == segments.len() - 1 {
            results.push(segment);
            continue;
        }
        results.push(segment);
        let last = segment.split_whitespace().next_back().unwrap_or("");
        let numeric = matches!(
            words.get(last),
            Some(Kind::Zero | Kind::One(_) | Kind::Ten(_) | Kind::Multiplier(_))
        );
        results.push(if numeric { "point five" } else { "and a half" });
    }
    let s = results.join(" ");
    let s = letter_digit.replace_all(&s, "${1} ${2}");
    let s = digit_letter.replace_all(&s, "${1} ${2}");
    suffix.replace_all(&s, "${1}${2}").into_owned()
}

fn process(words: &[&str]) -> Vec<String> {
    let table = number_words();
    let mut out: Vec<String> = Vec::new();
    let mut prefix: Option<String> = None;
    let mut value: Option<Value> = None;
    let mut skip = false;
    // Emits `result` with any pending sign or currency before it, and clears the number.
    let output = |result: String, prefix: &mut Option<String>, value: &mut Option<Value>| {
        let result = match prefix.take() {
            Some(p) => p + &result,
            None => result,
        };
        *value = None;
        result
    };
    for index in 0..words.len() {
        if skip {
            skip = false;
            continue;
        }
        let prev = index.checked_sub(1).map(|i| words[i]);
        let current = words[index];
        let next = words.get(index + 1).copied();
        let next_is_numeric = next.is_some_and(is_arabic);
        let kind_of = |w: Option<&str>| w.and_then(|w| table.get(w)).copied();
        let next_kind = kind_of(next);
        let prev_kind = kind_of(prev);
        let prev_is_one = matches!(prev_kind, Some(Kind::One(_)));
        let prev_is_ten = matches!(prev_kind, Some(Kind::Ten(_)));
        let mut kind = table.get(current).copied();
        // Whisper reads every "oh" as a zero ("oh my god" is "0 my god"); here only after
        // another number: "one oh one", "nineteen oh five".
        if matches!(kind, Some(Kind::Zero)) && current != "zero" && value.is_none() {
            kind = None;
        }
        let first = current.chars().next().unwrap();
        let has_prefix = matches!(first, '-' | '+' | '£' | '€' | '$' | '¢');
        let without_prefix = if has_prefix {
            &current[first.len_utf8()..]
        } else {
            current
        };
        if is_arabic(without_prefix) {
            if let Some(v) = &value {
                if let Value::Text(t) = v
                    && t.ends_with('.')
                {
                    // Decimals and IP-address parts.
                    value = Some(Value::Text(format!("{t}{current}")));
                    continue;
                }
                let text = v.text();
                out.push(output(text, &mut prefix, &mut value));
            }
            if has_prefix {
                prefix = Some(first.to_string());
            }
            value = Some(match decimal(without_prefix) {
                Some((n, scale)) if n % 10i128.pow(scale) == 0 => Value::Int(n / 10i128.pow(scale)),
                _ => Value::Text(without_prefix.to_owned()),
            });
            continue;
        }
        let Some(kind) = kind else {
            if let Some(v) = &value {
                let text = v.text();
                out.push(output(text, &mut prefix, &mut value));
            }
            out.push(output(current.to_owned(), &mut prefix, &mut value));
            continue;
        };
        // Python's `str(value or "")`: nothing, or a zero, reads as no digits yet.
        let digits = |v: &Option<Value>| match v {
            None | Some(Value::Int(0)) => String::new(),
            Some(v) => v.text(),
        };
        match kind {
            Kind::Zero => value = Some(Value::Text(digits(&value) + "0")),
            Kind::One(ones) | Kind::OneSuffixed(ones, _) => {
                let combined = match &value {
                    None => Value::Int(ones),
                    Some(Value::Text(t)) if prev_is_ten && ones < 10 && t.ends_with('0') => {
                        Value::Text(format!("{}{ones}", &t[..t.len() - 1]))
                    }
                    Some(v @ Value::Text(_)) => Value::Text(format!("{}{ones}", v.text())),
                    Some(Value::Int(v)) if prev_is_one => Value::Text(format!("{v}{ones}")),
                    Some(Value::Int(v)) if ones < 10 && v % 10 == 0 => Value::Int(v + ones),
                    Some(Value::Int(v)) if ones >= 10 && v % 100 == 0 => Value::Int(v + ones),
                    Some(Value::Int(v)) => Value::Text(format!("{v}{ones}")),
                };
                if let Kind::OneSuffixed(_, suffix) = kind {
                    out.push(output(combined.text() + suffix, &mut prefix, &mut value));
                } else {
                    value = Some(combined);
                }
            }
            Kind::Ten(tens) | Kind::TenSuffixed(tens, _) => {
                let combined = match &value {
                    None => Value::Int(tens),
                    Some(v @ Value::Text(_)) => Value::Text(format!("{}{tens}", v.text())),
                    Some(Value::Int(v)) if v % 100 == 0 => Value::Int(v + tens),
                    Some(Value::Int(v)) => Value::Text(format!("{v}{tens}")),
                };
                if let Kind::TenSuffixed(_, suffix) = kind {
                    out.push(output(combined.text() + suffix, &mut prefix, &mut value));
                } else {
                    value = Some(combined);
                }
            }
            Kind::Multiplier(multiplier) => match &value {
                None => value = Some(Value::Int(multiplier)),
                Some(v @ (Value::Text(_) | Value::Int(0))) => match times(v, multiplier) {
                    Some(p) => value = Some(Value::Int(p)),
                    None => {
                        let text = v.text();
                        out.push(output(text, &mut prefix, &mut value));
                        value = Some(Value::Int(multiplier));
                    }
                },
                Some(Value::Int(v)) => {
                    let (before, residual) = (v / 1000 * 1000, v % 1000);
                    value = Some(match residual.checked_mul(multiplier) {
                        Some(r) => Value::Int(before + r),
                        None => Value::Text(format!("{v}{multiplier}")),
                    });
                }
            },
            Kind::MultiplierSuffixed(multiplier, suffix) => {
                match &value {
                    None => out.push(output(
                        format!("{multiplier}{suffix}"),
                        &mut prefix,
                        &mut value,
                    )),
                    Some(v @ Value::Text(_)) => match times(v, multiplier) {
                        Some(p) => {
                            out.push(output(format!("{p}{suffix}"), &mut prefix, &mut value))
                        }
                        None => {
                            let text = v.text();
                            out.push(output(text, &mut prefix, &mut value));
                            out.push(output(
                                format!("{multiplier}{suffix}"),
                                &mut prefix,
                                &mut value,
                            ));
                        }
                    },
                    Some(Value::Int(v)) => {
                        let (before, residual) = (v / 1000 * 1000, v % 1000);
                        let total = residual.checked_mul(multiplier).map_or_else(
                            || format!("{v}{multiplier}"),
                            |r| (before + r).to_string(),
                        );
                        out.push(output(total + suffix, &mut prefix, &mut value));
                    }
                }
                value = None;
            }
            Kind::Preceding(sign) => {
                if let Some(v) = &value {
                    let text = v.text();
                    out.push(output(text, &mut prefix, &mut value));
                }
                if next_kind.is_some() || next_is_numeric {
                    prefix = Some(sign.to_owned());
                } else {
                    out.push(output(current.to_owned(), &mut prefix, &mut value));
                }
            }
            Kind::Following(symbol) => match &value {
                Some(v) => {
                    let text = v.text();
                    prefix = Some(symbol.to_owned());
                    out.push(output(text, &mut prefix, &mut value));
                }
                None => out.push(output(current.to_owned(), &mut prefix, &mut value)),
            },
            Kind::Percent | Kind::Per => match &value {
                Some(v) if matches!(kind, Kind::Percent) => {
                    let text = v.text() + "%";
                    out.push(output(text, &mut prefix, &mut value));
                }
                Some(v) if next == Some("cent") => {
                    let text = v.text() + "%";
                    out.push(output(text, &mut prefix, &mut value));
                    skip = true;
                }
                Some(v) => {
                    let text = v.text();
                    out.push(output(text, &mut prefix, &mut value));
                    out.push(output(current.to_owned(), &mut prefix, &mut value));
                }
                None => out.push(output(current.to_owned(), &mut prefix, &mut value)),
            },
            Kind::And | Kind::Double | Kind::Triple | Kind::Point => {
                // Only when the next word can be part of a number.
                if !(next_is_numeric || next_kind.is_some()) {
                    if let Some(v) = &value {
                        let text = v.text();
                        out.push(output(text, &mut prefix, &mut value));
                    }
                    out.push(output(current.to_owned(), &mut prefix, &mut value));
                    continue;
                }
                let repeats = if matches!(kind, Kind::Double) { 2 } else { 3 };
                match (kind, next_kind) {
                    // "and" after "hundred", "thousand"… is part of the number.
                    (Kind::And, _) if matches!(prev_kind, Some(Kind::Multiplier(_))) => {}
                    (Kind::Double | Kind::Triple, Some(Kind::One(ones))) => {
                        value = Some(Value::Text(
                            digits(&value) + &ones.to_string().repeat(repeats),
                        ));
                        skip = true;
                    }
                    (Kind::Double | Kind::Triple, Some(Kind::Zero)) => {
                        value = Some(Value::Text(digits(&value) + &"0".repeat(repeats)));
                        skip = true;
                    }
                    (Kind::Point, Some(Kind::Zero | Kind::One(_) | Kind::Ten(_))) => {
                        value = Some(Value::Text(digits(&value) + "."));
                    }
                    (Kind::Point, _) if next_is_numeric => {
                        value = Some(Value::Text(digits(&value) + "."));
                    }
                    // Whisper drops a "point" that starts no decimal; here it stays a word.
                    _ => {
                        if let Some(v) = &value {
                            let text = v.text();
                            out.push(output(text, &mut prefix, &mut value));
                        }
                        out.push(output(current.to_owned(), &mut prefix, &mut value));
                    }
                }
            }
        }
    }
    if let Some(v) = &value {
        let text = v.text();
        out.push(output(text, &mut prefix, &mut value));
    }
    out
}

fn postprocess(s: &str) -> String {
    static RULES: OnceLock<(Regex, Regex)> = OnceLock::new();
    let (cents, only_cents) = RULES.get_or_init(|| {
        (
            re(r"([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})\b"),
            re(r"[€£$]0\.([0-9]{1,2})\b"),
        )
    });
    // "$2 and ¢7" is "$2.07"; "$0.07" is "¢7".
    let s = cents.replace_all(s, |c: &regex::Captures| {
        let n: u32 = c[3].parse().unwrap();
        format!("{}{}.{n:02}", &c[1], &c[2])
    });
    let s = only_cents.replace_all(&s, |c: &regex::Captures| {
        format!("¢{}", c[1].parse::<u32>().unwrap())
    });
    // "one" and "ones" stay words: "the one", "little ones".
    s.split(' ')
        .map(|w| match w {
            "1" => "one",
            "1s" => "ones",
            w => w,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

// ---------------------------------------------------------------------------------------------
// Spellings: British as American, a modest table in the spirit of Whisper's `english.json`.

/// Words ending "our" in British spelling ("colour"), with their inflections generated.
const OUR: &[&str] = &[
    "arbour",
    "ardour",
    "armour",
    "behaviour",
    "candour",
    "clamour",
    "colour",
    "demeanour",
    "endeavour",
    "favour",
    "fervour",
    "flavour",
    "harbour",
    "honour",
    "humour",
    "labour",
    "neighbour",
    "odour",
    "parlour",
    "rancour",
    "rigour",
    "rumour",
    "saviour",
    "savour",
    "splendour",
    "succour",
    "tumour",
    "valour",
    "vapour",
    "vigour",
];
/// Words ending "ise" that American spelling writes "ize".
const ISE: &[&str] = &[
    "agonise",
    "apologise",
    "authorise",
    "capitalise",
    "categorise",
    "centralise",
    "characterise",
    "civilise",
    "criticise",
    "customise",
    "democratise",
    "dramatise",
    "economise",
    "emphasise",
    "energise",
    "equalise",
    "familiarise",
    "finalise",
    "formalise",
    "fossilise",
    "generalise",
    "globalise",
    "harmonise",
    "hospitalise",
    "idealise",
    "immunise",
    "italicise",
    "itemise",
    "jeopardise",
    "legalise",
    "legitimise",
    "liberalise",
    "localise",
    "maximise",
    "memorise",
    "minimise",
    "mobilise",
    "modernise",
    "monetise",
    "moralise",
    "nationalise",
    "neutralise",
    "normalise",
    "optimise",
    "organise",
    "patronise",
    "penalise",
    "personalise",
    "polarise",
    "popularise",
    "prioritise",
    "privatise",
    "publicise",
    "realise",
    "recognise",
    "revolutionise",
    "romanticise",
    "sanitise",
    "scrutinise",
    "sensitise",
    "socialise",
    "specialise",
    "stabilise",
    "standardise",
    "sterilise",
    "stigmatise",
    "subsidise",
    "summarise",
    "symbolise",
    "sympathise",
    "synchronise",
    "synthesise",
    "terrorise",
    "theorise",
    "tranquillise",
    "trivialise",
    "urbanise",
    "utilise",
    "vaporise",
    "verbalise",
    "victimise",
    "visualise",
    "vocalise",
];
/// Words ending "re" that American spelling writes "er".
const RE: &[&str] = &[
    "calibre",
    "centimetre",
    "centre",
    "fibre",
    "kilometre",
    "litre",
    "lustre",
    "manoeuvre",
    "meagre",
    "metre",
    "millimetre",
    "sabre",
    "sceptre",
    "sombre",
    "spectre",
    "theatre",
];
/// Everything else, word for word.
const SPELLINGS: &[(&str, &str)] = &[
    ("aeroplane", "airplane"),
    ("aeroplanes", "airplanes"),
    ("ageing", "aging"),
    ("aluminium", "aluminum"),
    ("anaemia", "anemia"),
    ("anaesthetic", "anesthetic"),
    ("analogue", "analog"),
    ("analyse", "analyze"),
    ("analysed", "analyzed"),
    ("analyses", "analyzes"),
    ("analysing", "analyzing"),
    ("acknowledgement", "acknowledgment"),
    ("acknowledgements", "acknowledgments"),
    ("artefact", "artifact"),
    ("artefacts", "artifacts"),
    ("cancelled", "canceled"),
    ("cancelling", "canceling"),
    ("catalogue", "catalog"),
    ("catalogues", "catalogs"),
    ("channelled", "channeled"),
    ("cheque", "check"),
    ("cheques", "checks"),
    ("cosy", "cozy"),
    ("counselling", "counseling"),
    ("counsellor", "counselor"),
    ("counsellors", "counselors"),
    ("defence", "defense"),
    ("defences", "defenses"),
    ("dialled", "dialed"),
    ("dialling", "dialing"),
    ("dialogue", "dialog"),
    ("dialogues", "dialogs"),
    ("draught", "draft"),
    ("draughts", "drafts"),
    ("encyclopaedia", "encyclopedia"),
    ("enrol", "enroll"),
    ("enrolment", "enrollment"),
    ("foetus", "fetus"),
    ("fuelled", "fueled"),
    ("fuelling", "fueling"),
    ("fulfil", "fulfill"),
    ("fulfilment", "fulfillment"),
    ("grey", "gray"),
    ("greys", "grays"),
    ("haemorrhage", "hemorrhage"),
    ("instalment", "installment"),
    ("instalments", "installments"),
    ("jewellery", "jewelry"),
    ("judgement", "judgment"),
    ("judgements", "judgments"),
    ("kerb", "curb"),
    ("labelled", "labeled"),
    ("labelling", "labeling"),
    ("leukaemia", "leukemia"),
    ("levelled", "leveled"),
    ("levelling", "leveling"),
    ("licence", "license"),
    ("licences", "licenses"),
    ("marvellous", "marvelous"),
    ("modelled", "modeled"),
    ("modelling", "modeling"),
    ("monologue", "monolog"),
    ("mould", "mold"),
    ("moulds", "molds"),
    ("mouldy", "moldy"),
    ("moustache", "mustache"),
    ("oestrogen", "estrogen"),
    ("offence", "offense"),
    ("offences", "offenses"),
    ("orthopaedic", "orthopedic"),
    ("paediatric", "pediatric"),
    ("paediatrics", "pediatrics"),
    ("paralyse", "paralyze"),
    ("paralysed", "paralyzed"),
    ("plough", "plow"),
    ("practise", "practice"),
    ("practised", "practiced"),
    ("practising", "practicing"),
    ("pretence", "pretense"),
    ("programme", "program"),
    ("programmes", "programs"),
    ("pyjamas", "pajamas"),
    ("quarrelled", "quarreled"),
    ("sceptic", "skeptic"),
    ("sceptical", "skeptical"),
    ("scepticism", "skepticism"),
    ("signalled", "signaled"),
    ("signalling", "signaling"),
    ("skilful", "skillful"),
    ("storey", "story"),
    ("storeys", "stories"),
    ("sulphur", "sulfur"),
    ("totalled", "totaled"),
    ("travelled", "traveled"),
    ("traveller", "traveler"),
    ("travellers", "travelers"),
    ("travelling", "traveling"),
    ("tunnelled", "tunneled"),
    ("tyre", "tire"),
    ("tyres", "tires"),
    ("wilful", "willful"),
];

fn spelling_table() -> &'static HashMap<String, String> {
    static TABLE: OnceLock<HashMap<String, String>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut m: HashMap<String, String> = HashMap::new();
        for word in OUR {
            let stem = &word[..word.len() - 3];
            for suffix in [
                "", "s", "ed", "ing", "ful", "fully", "able", "ably", "ite", "ites", "er", "ers",
                "hood", "hoods", "ly", "less",
            ] {
                m.insert(format!("{word}{suffix}"), format!("{stem}or{suffix}"));
            }
        }
        for word in ISE {
            let stem = &word[..word.len() - 3];
            for suffix in ["e", "es", "ed", "ing", "er", "ers", "ation", "ations"] {
                m.insert(format!("{stem}is{suffix}"), format!("{stem}iz{suffix}"));
            }
        }
        for word in RE {
            let stem = &word[..word.len() - 2];
            for suffix in ["", "s", "d"] {
                m.insert(format!("{stem}re{suffix}"), format!("{stem}er{suffix}"));
            }
        }
        m.insert("manoeuvre".into(), "maneuver".into());
        m.insert("manoeuvres".into(), "maneuvers".into());
        m.insert("manoeuvred".into(), "maneuvered".into());
        for (british, american) in SPELLINGS {
            m.insert(british.to_string(), american.to_string());
        }
        m
    })
}

fn spellings(s: &str) -> String {
    let table = spelling_table();
    s.split_whitespace()
        .map(|w| table.get(w).map_or(w, String::as_str))
        .collect::<Vec<_>>()
        .join(" ")
}
