//! Whisper's vocabulary: GPT-2 byte-level BPE as GGUF files carry it (`tokenizer.ggml.tokens`,
//! `tokenizer.ggml.model = "gpt2"`), decoded as transcribe.cpp decodes it
//! (`decode_gpt2_bytes` in `src/transcribe-tokenizer.cpp` there): each piece's characters are
//! GPT-2's printable stand-ins for bytes, mapped back to the bytes they stand for. The engine
//! never encodes text (the app gives Whisper no prompt), so there are no merges here.

use crate::{Error, Result, gguf::Gguf, gguf::Value};
use std::collections::HashMap;

pub struct Tokenizer {
    /// Each id's piece, as the vocabulary spells it.
    tokens: Vec<String>,
    /// Ids of the special pieces (`<|…|>`), by their text.
    specials: HashMap<String, i32>,
    /// GPT-2's `bytes_to_unicode`, reversed: the byte each stand-in character means.
    bytes: HashMap<char, u8>,
}

impl Tokenizer {
    pub fn from_gguf(file: &Gguf) -> Result<Tokenizer> {
        let Some(Value::Array(items)) = file.metadata.get("tokenizer.ggml.tokens") else {
            return Err(Error::Gguf("no tokenizer.ggml.tokens".into()));
        };
        let tokens: Vec<String> = items
            .iter()
            .map(|v| v.as_str().unwrap_or_default().to_owned())
            .collect();
        let specials = tokens
            .iter()
            .enumerate()
            .filter(|(_, t)| t.starts_with("<|") && t.ends_with("|>"))
            .map(|(id, t)| (t.clone(), id as i32))
            .collect();
        Ok(Tokenizer {
            tokens,
            specials,
            bytes: byte_stand_ins(),
        })
    }

    /// The id of a special piece such as `<|en|>`.
    pub fn special(&self, piece: &str) -> Option<i32> {
        self.specials.get(piece).copied()
    }

    /// The bytes of a run of tokens: every character of each piece that stands for a byte
    /// becomes that byte; any other character is passed through as its UTF-8 (as the reference
    /// does for a special piece that reaches it). Ids outside the vocabulary are skipped.
    pub fn decode_bytes(&self, ids: impl IntoIterator<Item = i32>) -> Vec<u8> {
        let mut out = Vec::new();
        for id in ids {
            let Some(piece) = usize::try_from(id).ok().and_then(|i| self.tokens.get(i)) else {
                continue;
            };
            for c in piece.chars() {
                match self.bytes.get(&c) {
                    Some(&b) => out.push(b),
                    None => out.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes()),
                }
            }
        }
        out
    }
}

/// GPT-2's `bytes_to_unicode` reversed. The printable bytes `!`–`~`, `¡`–`¬` and `®`–`ÿ` stand
/// for themselves; the other 68 bytes, in order, for U+0100 onwards.
fn byte_stand_ins() -> HashMap<char, u8> {
    let printable = |b: u8| matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
    let mut map = HashMap::with_capacity(256);
    let mut next = 256u32;
    for b in 0..=255u8 {
        let c = if printable(b) {
            b as u32
        } else {
            next += 1;
            next - 1
        };
        map.insert(char::from_u32(c).expect("a valid stand-in"), b);
    }
    map
}

/// `s` without leading and trailing ASCII spaces, tabs, newlines and carriage returns (the
/// reference's `trim_ascii`).
pub fn trim(s: &[u8]) -> &[u8] {
    let blank = |b: &u8| matches!(b, b' ' | b'\t' | b'\n' | b'\r');
    let start = s.iter().position(|b| !blank(b)).unwrap_or(s.len());
    let end = s.iter().rposition(|b| !blank(b)).map_or(start, |e| e + 1);
    &s[start..end]
}
