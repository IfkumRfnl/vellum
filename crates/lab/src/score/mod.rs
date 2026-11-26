//! How close results are: words against words, speakers against speakers, tensors against
//! tensors, and timings against timings.
//!
//! - `text`: transcripts as the words compared (`Plain` for engine against engine, `Asr`,
//!   Whisper's normalizers, for engine against ground truth).
//! - `wer`: word alignment and error rate.
//! - `parity`: one engine's output against another's on the same clip.
//! - `der`: diarization error rate.
//! - `speakers`: speaker labels on words, as the app gives them, and their accuracy.
//! - `tensors`: dumped tensors, checkpoint by checkpoint.
//! - `stats`: robust statistics and paired speed comparison.
//! - `table`: Markdown tables for reports.

// A library for the flows: not every flow uses every part.
#![allow(dead_code, unused_imports)]

pub mod der;
pub mod parity;
pub mod speakers;
pub mod stats;
pub mod table;
pub mod tensors;
pub mod text;
pub mod wer;

pub use der::{DerOptions, DerScore, Segment, der};
pub use parity::{DiarizationParity, TranscriptParity, diarizations, transcripts};
pub use speakers::{SpeakerReport, SpeakerScore, label};
pub use stats::{PairedSpeed, mad, median, noise_floor, paired, quantile};
pub use table::Table;
pub use tensors::{DumpComparison, LogitDiff, TensorDiff, compare, compare_dirs, compare_logits};
pub use text::{Mode, words};
pub use wer::{Alignment, WerScore, align};
