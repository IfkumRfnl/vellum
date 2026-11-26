//! Recordings in, samples out: 16 kHz mono, 16-bit WAV or FLAC, scaled as the app scales them
//! (`sample / 32768`).

use anyhow::{Context, Result, bail, ensure};
use sha2::{Digest, Sha256};
use std::path::Path;

pub const RATE: u32 = 16_000;

pub fn read(path: &Path) -> Result<Vec<f32>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let read = match ext.as_str() {
        "wav" => read_wav(path),
        "flac" => read_flac(path),
        _ => bail!("not a WAV or FLAC file"),
    };
    read.with_context(|| format!("reading {}", path.display()))
}

fn read_wav(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    ensure!(
        spec.sample_rate == RATE && spec.channels == 1 && spec.bits_per_sample == 16,
        "needs 16 kHz mono 16-bit audio, found {} Hz, {} channels, {} bits",
        spec.sample_rate,
        spec.channels,
        spec.bits_per_sample
    );
    reader
        .samples::<i16>()
        .map(|s| Ok(s? as f32 / 32768.))
        .collect()
}

fn read_flac(path: &Path) -> Result<Vec<f32>> {
    let mut reader = claxon::FlacReader::open(path)?;
    let info = reader.streaminfo();
    ensure!(
        info.sample_rate == RATE && info.channels == 1 && info.bits_per_sample == 16,
        "needs 16 kHz mono 16-bit audio, found {} Hz, {} channels, {} bits",
        info.sample_rate,
        info.channels,
        info.bits_per_sample
    );
    let mut out = Vec::with_capacity(info.samples.unwrap_or(0) as usize);
    for s in reader.samples() {
        out.push(s? as f32 / 32768.);
    }
    Ok(out)
}

/// Writes 16-bit WAV, the samples clamped to [-1, 1).
pub fn write_wav(path: &Path, pcm: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for s in pcm {
        writer.write_sample(quantize(*s))?;
    }
    writer.finalize()?;
    Ok(())
}

pub fn quantize(s: f32) -> i16 {
    (s * 32768.).round().clamp(-32768., 32767.) as i16
}

/// SHA-256 of the samples as 16-bit little-endian integers, whatever file they came from.
pub fn pcm_sha256(pcm: &[f32]) -> String {
    let mut hash = Sha256::new();
    for s in pcm {
        hash.update(quantize(*s).to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}
