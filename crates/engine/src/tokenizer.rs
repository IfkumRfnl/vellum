//! SentencePiece vocabularies as GGUF files carry them (`tokenizer.ggml.*`): token ids to text,
//! as transcribe.cpp detokenizes them (`src/transcribe-tokenizer.cpp` there).

use crate::{Result, gguf::Gguf, gguf::Value};

/// SentencePiece's word-boundary marker, U+2581 ("▁"): a piece starting with it begins a word,
/// and it reads as a space.
pub const WORD_MARKER: &str = "\u{2581}";

/// Token types, by llama.cpp's convention (`tokenizer.ggml.token_type`).
const TYPE_UNKNOWN: i32 = 2;
const TYPE_CONTROL: i32 = 3;

pub struct Tokenizer {
    /// Each id's piece, as the vocabulary spells it (with `▁` for spaces).
    pub tokens: Vec<String>,
    /// Each id's type (1 normal, 2 unknown, 3 control…); empty when the file has none.
    pub types: Vec<i32>,
    /// The `<unk>` id, if the file names one.
    pub unknown: Option<usize>,
}

impl Tokenizer {
    pub fn from_gguf(file: &Gguf) -> Result<Tokenizer> {
        let array = |key: &str| match file.metadata.get(key) {
            Some(Value::Array(items)) => items.as_slice(),
            _ => &[],
        };
        let tokens = array("tokenizer.ggml.tokens")
            .iter()
            .map(|v| v.as_str().unwrap_or_default().to_owned())
            .collect();
        let types = array("tokenizer.ggml.token_type")
            .iter()
            .map(|v| match *v {
                Value::I32(t) => t,
                ref other => other.as_u64().map_or(1, |t| t as i32),
            })
            .collect();
        let unknown = file
            .metadata_u64("tokenizer.ggml.unknown_token_id")
            .map(|v| v as usize);
        Ok(Tokenizer {
            tokens,
            types,
            unknown,
        })
    }

    /// The piece for `id`, or "" for an id outside the vocabulary.
    pub fn piece(&self, id: usize) -> &str {
        self.tokens.get(id).map_or("", String::as_str)
    }

    /// Whether `id` is a control or unknown piece, which the reference leaves out of results.
    pub fn is_special(&self, id: usize) -> bool {
        matches!(self.types.get(id), Some(&(TYPE_UNKNOWN | TYPE_CONTROL)))
            || self.unknown == Some(id)
            || self.piece(id) == "<unk>"
    }

    /// The text of a run of tokens: pieces joined, `▁` read as a space, and byte-fallback
    /// pieces (`<0xHH>`) as the bytes they stand for. Nothing is trimmed.
    pub fn decode(&self, ids: impl IntoIterator<Item = usize>) -> String {
        let mut out = Vec::new();
        for id in ids {
            let Some(piece) = self.tokens.get(id) else {
                continue;
            };
            match byte_fallback(piece) {
                Some(byte) => out.push(byte),
                None => out.extend_from_slice(piece.replace(WORD_MARKER, " ").as_bytes()),
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }
}

/// The byte a SentencePiece byte-fallback piece (`<0xE1>`) stands for.
fn byte_fallback(piece: &str) -> Option<u8> {
    let hex = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}
