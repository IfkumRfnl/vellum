//! Deterministic edge cases made from LibriSpeech and from nothing: silence, noise, tones,
//! clipping, odd lengths, shifts, window boundaries, a language switch.
//!
//! Every case is computed from LibriSpeech clean utterances an earlier source recorded (smoke
//! cases from the smoke utterances, the rest from the quick ones) or from a fixed-seed
//! generator, and written once as FLAC. A case whose words all survive keeps the utterances'
//! text and is scored for accuracy too; the others are only checked for sanity.

use super::{
    Clip, Purpose, Source, Tier, Truth,
    util::{self, Fetch},
};
use anyhow::{Context, Result, bail};
use std::{cell::OnceCell, collections::BTreeMap, f64::consts::PI, path::PathBuf};
use vellum_harness::audio;

pub const SOURCE: Source = Source {
    sets: &["synthetic"],
    about: "edge cases derived from LibriSpeech",
    fetch,
};

const RATE: usize = audio::RATE as usize;

/// A clip to make: what it's called and tagged, and how to compute its samples and the words
/// they still hold (none when there are no words, or not all of them).
struct Case<'a> {
    name: String,
    tier: Tier,
    tags: Vec<String>,
    language: &'static str,
    make: Make<'a>,
}

/// A case's samples and, when every word survives, the text.
type Made = (Vec<f32>, Option<String>);
type Make<'a> = Box<dyn Fn() -> Result<Made> + 'a>;

/// A recorded utterance (or FLEURS clip), its samples read on first use.
struct Utt<'a> {
    clip: &'a Clip,
    path: PathBuf,
    pcm: OnceCell<Vec<f32>>,
}

impl Utt<'_> {
    fn pcm(&self) -> Result<&[f32]> {
        if let Some(pcm) = self.pcm.get() {
            return Ok(pcm);
        }
        let pcm = audio::read(&self.path)?;
        Ok(self.pcm.get_or_init(|| pcm))
    }

    fn text(&self) -> &str {
        self.clip.truth.text.as_deref().unwrap_or_default()
    }
}

fn fetch(fetch: &mut Fetch, tier: Tier) -> Result<BTreeMap<String, Vec<Clip>>> {
    let fetch: &Fetch = fetch;
    let dir = fetch.dir("synthetic")?;
    // A pool is complete only once its tier was fetched: the picks then never change.
    let complete = |pool: &[Utt], tier: Tier| pool.iter().any(|u| u.clip.tier == tier);
    let smoke = pool(fetch, "librispeech-clean", "en", Tier::Smoke);
    let quick = pool(fetch, "librispeech-clean", "en", Tier::Quick);
    let german = pool(fetch, "fleurs", "de", Tier::Quick);
    if !complete(&smoke, Tier::Smoke) {
        bail!("needs the LibriSpeech clean smoke tier first (fetch librispeech-clean)");
    }
    let quick_ok = complete(&quick, Tier::Quick);
    if !quick_ok && tier >= Tier::Quick {
        eprintln!("  the quick cases need the LibriSpeech clean quick tier first");
    }
    let german_ok = german.len() >= super::fleurs::QUICK;

    let mut cases = Vec::new();
    cases.extend(from_nothing());
    cases.extend(from_utterance(base(&smoke)));
    if quick_ok {
        cases.extend(from_pool(&quick));
        if german_ok {
            cases.push(language_switch(&german, &quick));
        }
    }

    let mut clips = Vec::new();
    for case in cases {
        let id = format!("synthetic/{}", case.name);
        let path = dir.join(format!("{}.flac", case.name));
        if let Some(mut clip) = fetch.known(&id) {
            clip.tier = case.tier;
            clip.tags = case.tags;
            clips.push(clip);
            continue;
        }
        if case.tier > tier && !path.is_file() {
            continue;
        }
        let (pcm, text) = (case.make)().with_context(|| format!("making {id}"))?;
        if !path.is_file() {
            eprintln!("  making {id}");
            util::write_flac(&path, &pcm)?;
        }
        let mut purposes = vec![Purpose::Robustness];
        if text.is_some() {
            purposes.insert(0, Purpose::Asr);
        }
        let truth = Truth {
            verbatim: text.is_some(),
            text,
            ..Truth::default()
        };
        let tags: Vec<&str> = case.tags.iter().map(String::as_str).collect();
        clips.push(fetch.clip(
            id,
            "synthetic",
            &path,
            case.tier,
            case.language,
            &purposes,
            truth,
            &tags,
        )?);
    }
    Ok(BTreeMap::from([("synthetic".to_owned(), clips)]))
}

/// The utterances of `set` in `language` up to `upto`, in id order, long-form clips aside.
fn pool<'a>(fetch: &'a Fetch, set: &str, language: &str, upto: Tier) -> Vec<Utt<'a>> {
    let mut clips: Vec<&Clip> = fetch
        .clips_of(set)
        .into_iter()
        .filter(|c| {
            c.tier <= upto
                && c.language == language
                && !c.tags.iter().any(|t| t == "long-form")
                && c.truth.text.is_some()
        })
        .collect();
    clips.sort_by(|a, b| a.id.cmp(&b.id));
    clips
        .into_iter()
        .map(|clip| Utt {
            clip,
            path: fetch.root.join(&clip.audio),
            pcm: OnceCell::new(),
        })
        .collect()
}

fn case<'a>(
    name: &str,
    tier: Tier,
    tags: &[&str],
    language: &'static str,
    make: impl Fn() -> Result<Made> + 'a,
) -> Case<'a> {
    Case {
        name: name.to_owned(),
        tier,
        tags: tags.iter().map(|t| t.to_string()).collect(),
        language,
        make: Box::new(make),
    }
}

/// Silence, noise, a tone and chords.
fn from_nothing() -> Vec<Case<'static>> {
    let none = |pcm: Vec<f32>| -> Result<Made> { Ok((pcm, None)) };
    vec![
        case(
            "silence-10s",
            Tier::Smoke,
            &["case:silence"],
            "",
            move || none(vec![0.0; 10 * RATE]),
        ),
        // The app skips windows that are exactly silent; this one isn't.
        case(
            "silence-10s-one-sample",
            Tier::Micro,
            &["case:near-silence"],
            "",
            move || {
                let mut pcm = vec![0.0; 10 * RATE];
                pcm[5 * RATE] = 1.0 / 32768.0;
                none(pcm)
            },
        ),
        case(
            "white-noise-30s",
            Tier::Smoke,
            &["case:noise", "noise:white", "level:-30dbfs"],
            "",
            move || none(scaled(white(30 * RATE, 1), db(-30.0))),
        ),
        case(
            "silence-60s",
            Tier::Quick,
            &["case:silence"],
            "",
            move || none(vec![0.0; 60 * RATE]),
        ),
        case(
            "pink-noise-30s",
            Tier::Quick,
            &["case:noise", "noise:pink", "level:-30dbfs"],
            "",
            move || none(scaled(pink(30 * RATE, 2), db(-30.0))),
        ),
        case(
            "tone-440hz-10s",
            Tier::Quick,
            &["case:tone"],
            "",
            move || none(tone(440.0, 10 * RATE)),
        ),
        case(
            "music-chords-30s",
            Tier::Quick,
            &["case:music"],
            "",
            move || none(chords(30 * RATE)),
        ),
    ]
}

/// The utterance most single-utterance cases start from: the first smoke one of 8–20 s.
fn base<'u, 'a>(smoke: &'u [Utt<'a>]) -> &'u Utt<'a> {
    smoke
        .iter()
        .find(|u| (8.0..=20.0).contains(&u.clip.seconds))
        .unwrap_or(&smoke[0])
}

/// Cases made from one utterance: noise, gain, clipping, band limits, shifts, cuts.
fn from_utterance<'u>(u: &'u Utt<'u>) -> Vec<Case<'u>> {
    let base_tag = format!("base:{}", u.clip.id);
    let tags = |extra: &[&str]| -> Vec<String> {
        extra
            .iter()
            .map(|t| t.to_string())
            .chain([base_tag.clone()])
            .collect()
    };
    let words = move |pcm: Vec<f32>| Ok((pcm, Some(u.text().to_owned())));
    let mut cases = Vec::new();
    let mut add = |name: String, tier: Tier, tags: Vec<String>, make: Make<'u>| {
        cases.push(Case {
            name,
            tier,
            tags,
            language: "en",
            make,
        });
    };

    for (seconds, tier) in [(0.3, Tier::Smoke), (1.0, Tier::Quick)] {
        add(
            format!("speech-{seconds:.1}s"),
            tier,
            tags(&["case:short-speech"]),
            Box::new(move || {
                let pcm = u.pcm()?;
                let start = speech_start(pcm);
                let n = (seconds * RATE as f64) as usize;
                Ok((pcm[start..(start + n).min(pcm.len())].to_vec(), None))
            }),
        );
    }
    for (snr, tier) in [
        (5, Tier::Smoke),
        (0, Tier::Quick),
        (10, Tier::Quick),
        (20, Tier::Quick),
    ] {
        add(
            format!("speech-white-snr{snr}"),
            tier,
            tags(&["case:noise-mix", "noise:white", &format!("noise:snr{snr}")]),
            Box::new(move || {
                let pcm = u.pcm()?;
                let noise = white(pcm.len(), 100 + snr as u64);
                words(mix(pcm, &noise, snr as f64))
            }),
        );
    }
    add(
        "speech-gain-minus30db".into(),
        Tier::Quick,
        tags(&["case:gain", "gain:-30db"]),
        Box::new(move || words(scaled_by(u.pcm()?, db(-30.0) as f32))),
    );
    add(
        "speech-clipped".into(),
        Tier::Quick,
        tags(&["case:clipping", "gain:x8"]),
        Box::new(move || {
            words(
                u.pcm()?
                    .iter()
                    .map(|s| (s * 8.0).clamp(-1.0, 1.0))
                    .collect(),
            )
        }),
    );
    add(
        "speech-lowpass-4khz".into(),
        Tier::Quick,
        tags(&["case:bandlimit", "lowpass:4khz"]),
        Box::new(move || words(lowpass(u.pcm()?, 4000.0))),
    );
    add(
        "speech-dc-offset".into(),
        Tier::Quick,
        tags(&["case:dc-offset", "dc:0.1"]),
        Box::new(move || words(u.pcm()?.iter().map(|s| s + 0.1).collect())),
    );
    for shift in [1, 80, 160] {
        add(
            format!("speech-shift-{shift}"),
            Tier::Quick,
            tags(&["case:shift", &format!("shift:{shift}")]),
            Box::new(move || {
                let mut pcm = vec![0.0; shift];
                pcm.extend_from_slice(u.pcm()?);
                words(pcm)
            }),
        );
    }
    add(
        "speech-trailing-zeros-2s".into(),
        Tier::Quick,
        tags(&["case:trailing-zeros"]),
        Box::new(move || {
            let mut pcm = u.pcm()?.to_vec();
            pcm.extend(std::iter::repeat_n(0.0, 2 * RATE));
            words(pcm)
        }),
    );
    cases
}

/// Cases made from several quick utterances: gaps, window boundaries, long noisy speech.
fn from_pool<'u>(pool: &'u [Utt<'u>]) -> Vec<Case<'u>> {
    let mut cases = Vec::new();
    cases.push(case(
        "speech-gaps-5s",
        Tier::Quick,
        &["case:gaps"],
        "en",
        move || {
            let parts = take(pool, 60.0);
            let mut pcm = Vec::new();
            for (i, u) in parts.iter().enumerate() {
                if i > 0 {
                    pcm.extend(std::iter::repeat_n(0.0, 5 * RATE));
                }
                pcm.extend_from_slice(u.pcm()?);
            }
            Ok((pcm, Some(text(&parts))))
        },
    ));
    // The app's windows: 28 s at most, cut at the quietest 40 ms of each one's last 3 s.
    let window = 28 * RATE;
    for (name, len) in [
        ("boundary-28s", window),
        ("boundary-28s-plus-1", window + 1),
        ("boundary-27.999s", window - RATE / 1000),
    ] {
        cases.push(case(
            name,
            Tier::Quick,
            &["boundary", "case:window-length"],
            "en",
            // All three come from the same utterances, whose speech is over by 27.999 s.
            move || straddle(pool, window - RATE / 1000, window + 1, len),
        ));
    }
    cases.push(case(
        "boundary-56s",
        Tier::Quick,
        &["boundary", "case:window-length"],
        "en",
        move || straddle(pool, 2 * window, 2 * window, 2 * window),
    ));
    cases.push(case(
        "boundary-no-pauses-90s",
        Tier::Quick,
        &["boundary", "case:no-pauses"],
        "en",
        move || {
            let mut parts = Vec::new();
            let mut pcm = Vec::new();
            // Whole utterances until about 90 s, skipping any that would run far past it.
            for u in pool {
                let squeezed = squeeze(u.pcm()?);
                if pcm.len() + squeezed.len() > 95 * RATE {
                    if pcm.len() >= 85 * RATE {
                        break;
                    }
                    continue;
                }
                pcm.extend(squeezed);
                parts.push(u);
            }
            Ok((pcm, Some(text(&parts))))
        },
    ));
    cases.push(case(
        "speech-shaped-snr10-10min",
        Tier::Full,
        &[
            "case:noise-mix",
            "noise:speech-shaped",
            "noise:snr10",
            "long-form",
        ],
        "en",
        move || {
            let parts = take(pool, 600.0);
            let speech = join(&parts)?;
            let noise = speech_shaped(&speech, 3);
            Ok((mix(&speech, &noise, 10.0), Some(text(&parts))))
        },
    ));
    cases
}

/// ~30 s of German followed by ~30 s of English: does a model that latched onto the first
/// language follow the switch?
fn language_switch<'u>(german: &'u [Utt<'u>], english: &'u [Utt<'u>]) -> Case<'u> {
    case(
        "language-switch-de-en",
        Tier::Quick,
        &["language-switch", "case:language-switch"],
        "de",
        move || {
            let de = take(german, 30.0);
            let en = take(english, 30.0);
            let mut pcm = join(&de)?;
            pcm.extend(join(&en)?);
            Ok((pcm, Some(format!("{} {}", text(&de), text(&en)))))
        },
    )
}

/// The first utterances of `pool` adding up to at least `seconds`.
fn take<'u, 'a>(pool: &'u [Utt<'a>], seconds: f64) -> Vec<&'u Utt<'a>> {
    let mut out = Vec::new();
    let mut total = 0.0;
    for u in pool {
        if total >= seconds {
            break;
        }
        out.push(u);
        total += u.clip.seconds;
    }
    out
}

fn join(parts: &[&Utt]) -> Result<Vec<f32>> {
    let mut pcm = Vec::new();
    for u in parts {
        pcm.extend_from_slice(u.pcm()?);
    }
    Ok(pcm)
}

fn text(parts: &[&Utt]) -> String {
    parts.iter().map(|u| u.text()).collect::<Vec<_>>().join(" ")
}

/// Whole utterances of `pool`, joined and cut to `len` samples, chosen so that every word is
/// over by `quiet_from` samples while the audio runs on to at least `min_len`: the last
/// utterance's trailing quiet spans the cut, so no word is lost.
fn straddle(pool: &[Utt], quiet_from: usize, min_len: usize, len: usize) -> Result<Made> {
    // Fill to a second or more short of the cut, then look for a last utterance that fits,
    // backing off one utterance at a time.
    let mut prefix = Vec::new();
    let mut total = 0;
    for (i, u) in pool.iter().enumerate() {
        let n = u.pcm()?.len();
        if total + n + RATE > quiet_from {
            break;
        }
        prefix.push(i);
        total += n;
    }
    loop {
        for (j, u) in pool.iter().enumerate() {
            if prefix.contains(&j) {
                continue;
            }
            let pcm = u.pcm()?;
            if total + speech_end(pcm) <= quiet_from && total + pcm.len() >= min_len {
                let parts: Vec<&Utt> = prefix.iter().map(|&i| &pool[i]).chain([u]).collect();
                let mut out = join(&parts)?;
                out.truncate(len);
                return Ok((out, Some(text(&parts))));
            }
        }
        let Some(i) = prefix.pop() else {
            bail!("no utterances end in quiet at {quiet_from} samples");
        };
        total -= pool[i].pcm()?.len();
    }
}

/// Energies of 10 ms frames.
fn frames(pcm: &[f32]) -> Vec<f32> {
    pcm.chunks(RATE / 100)
        .map(|f| f.iter().map(|s| s * s).sum::<f32>() / f.len() as f32)
        .collect()
}

/// Which 10 ms frames are pauses: within 10 dB of the recording's floor (its 10th-percentile
/// frame) and at least 20 dB under its mean.
fn quiet(pcm: &[f32]) -> Vec<bool> {
    let energy = frames(pcm);
    let mut sorted = energy.clone();
    sorted.sort_by(f32::total_cmp);
    let floor = sorted[sorted.len() / 10];
    let mean = energy.iter().sum::<f32>() / energy.len().max(1) as f32;
    let threshold = (floor * 10.0).min(mean * 0.01);
    energy.iter().map(|&e| e <= threshold).collect()
}

/// Where the speech starts: the first frame that isn't a pause.
fn speech_start(pcm: &[f32]) -> usize {
    quiet(pcm).iter().position(|q| !q).unwrap_or(0) * (RATE / 100)
}

/// Where the speech ends: after the last frame that isn't a pause.
fn speech_end(pcm: &[f32]) -> usize {
    let quiet = quiet(pcm);
    let last = quiet.iter().rposition(|q| !q).map_or(0, |i| i + 1);
    (last * (RATE / 100)).min(pcm.len())
}

/// The utterance with every pause shortened to 20 ms, and the pauses at its ends to 10 ms, so
/// joined utterances have no 40 ms of quiet anywhere.
fn squeeze(pcm: &[f32]) -> Vec<f32> {
    let frame = RATE / 100;
    let quiet = quiet(pcm);
    let mut out = Vec::with_capacity(pcm.len());
    let mut i = 0;
    while i < quiet.len() {
        let mut j = i;
        while j < quiet.len() && quiet[j] == quiet[i] {
            j += 1;
        }
        let keep = if !quiet[i] {
            j - i
        } else if i == 0 || j == quiet.len() {
            1.min(j - i)
        } else {
            2.min(j - i)
        };
        let start = i * frame;
        let end = ((i + keep) * frame).min(pcm.len());
        out.extend_from_slice(&pcm[start..end]);
        i = j;
    }
    out
}

/// A fixed-seed generator (SplitMix64).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1].
    fn uniform(&mut self) -> f64 {
        ((self.next() >> 11) + 1) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal (Box–Muller).
    fn normal(&mut self) -> f64 {
        let (a, b) = (self.uniform(), self.uniform());
        (-2.0 * a.ln()).sqrt() * (2.0 * PI * b).cos()
    }
}

fn db(level: f64) -> f64 {
    10f64.powf(level / 20.0)
}

fn rms(pcm: &[f32]) -> f64 {
    (pcm.iter().map(|&s| s as f64 * s as f64).sum::<f64>() / pcm.len().max(1) as f64).sqrt()
}

/// `pcm` scaled to an RMS of `level` (linear).
fn scaled(pcm: Vec<f32>, level: f64) -> Vec<f32> {
    let gain = level / rms(&pcm).max(1e-12);
    scaled_by(&pcm, gain as f32)
}

fn scaled_by(pcm: &[f32], gain: f32) -> Vec<f32> {
    pcm.iter().map(|s| s * gain).collect()
}

/// `speech` plus `noise` scaled to `snr` dB below the speech's power.
fn mix(speech: &[f32], noise: &[f32], snr: f64) -> Vec<f32> {
    let gain = (rms(speech) * db(-snr) / rms(noise).max(1e-12)) as f32;
    speech
        .iter()
        .zip(noise)
        .map(|(s, n)| s + n * gain)
        .collect()
}

fn white(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..n).map(|_| rng.normal() as f32).collect()
}

/// Pink noise: white noise through Paul Kellet's -3 dB/octave filter.
fn pink(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    let mut b = [0f64; 7];
    (0..n)
        .map(|_| {
            let w = rng.normal();
            b[0] = 0.99886 * b[0] + w * 0.0555179;
            b[1] = 0.99332 * b[1] + w * 0.0750759;
            b[2] = 0.96900 * b[2] + w * 0.1538520;
            b[3] = 0.86650 * b[3] + w * 0.3104856;
            b[4] = 0.55000 * b[4] + w * 0.5329522;
            b[5] = -0.7616 * b[5] - w * 0.0168980;
            let out = b.iter().sum::<f64>() + w * 0.5362;
            b[6] = w * 0.115926;
            out as f32
        })
        .collect()
}

/// A sine at -6 dBFS with 10 ms fades.
fn tone(hz: f64, n: usize) -> Vec<f32> {
    let fade = RATE / 100;
    (0..n)
        .map(|i| {
            let edge = (i.min(n - 1 - i) as f64 / fade as f64).min(1.0);
            (0.5 * edge * (2.0 * PI * hz * i as f64 / RATE as f64).sin()) as f32
        })
        .collect()
}

/// Something music-like: C–G–Am–F, two seconds a chord, each note with six harmonics and a
/// plucked envelope, over the root an octave down; peaks at -6 dBFS.
fn chords(n: usize) -> Vec<f32> {
    const PROGRESSION: [[f64; 3]; 4] = [
        [261.63, 329.63, 392.00],
        [196.00, 246.94, 293.66],
        [220.00, 261.63, 329.63],
        [174.61, 220.00, 261.63],
    ];
    let chord = 2 * RATE;
    let mut pcm: Vec<f32> = (0..n)
        .map(|i| {
            let notes = PROGRESSION[(i / chord) % PROGRESSION.len()];
            let t = (i % chord) as f64 / RATE as f64;
            let envelope = (t / 0.01).min(1.0) * (-t / 0.8).exp();
            let time = i as f64 / RATE as f64;
            let mut sum = 0.0;
            for hz in notes.into_iter().chain([notes[0] / 2.0]) {
                for k in 1..=6 {
                    sum += 0.6f64.powi(k) * (2.0 * PI * hz * k as f64 * time).sin();
                }
            }
            (sum * envelope) as f32
        })
        .collect();
    let peak = pcm.iter().fold(0f32, |m, s| m.max(s.abs())).max(1e-9);
    for s in &mut pcm {
        *s *= 0.5 / peak;
    }
    pcm
}

/// A windowed-sinc (Blackman, 129 taps) low-pass at `cutoff` Hz, without delay.
fn lowpass(pcm: &[f32], cutoff: f64) -> Vec<f32> {
    const TAPS: usize = 129;
    let m = (TAPS / 2) as isize;
    let fc = cutoff / RATE as f64;
    let mut h: Vec<f64> = (0..TAPS)
        .map(|i| {
            let x = i as f64 - m as f64;
            let sinc = if x == 0.0 {
                2.0 * fc
            } else {
                (2.0 * PI * fc * x).sin() / (PI * x)
            };
            let w = 0.42 - 0.5 * (2.0 * PI * i as f64 / (TAPS - 1) as f64).cos()
                + 0.08 * (4.0 * PI * i as f64 / (TAPS - 1) as f64).cos();
            sinc * w
        })
        .collect();
    let sum: f64 = h.iter().sum();
    for v in &mut h {
        *v /= sum;
    }
    (0..pcm.len() as isize)
        .map(|i| {
            let mut acc = 0.0;
            for (k, &v) in h.iter().enumerate() {
                let j = i + k as isize - m;
                if j >= 0 && (j as usize) < pcm.len() {
                    acc += v * pcm[j as usize] as f64;
                }
            }
            acc as f32
        })
        .collect()
}

/// Noise with speech's long-term spectrum: 32 copies of `speech`, each rotated by a random
/// offset and given a random sign, summed. So many overlapping talkers aren't intelligible.
fn speech_shaped(speech: &[f32], seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    let n = speech.len();
    let mut out = vec![0f32; n];
    for _ in 0..32 {
        let offset = (rng.next() % n.max(1) as u64) as usize;
        let sign = if rng.next() & 1 == 0 { 1.0 } else { -1.0 };
        for (i, o) in out.iter_mut().enumerate() {
            *o += sign * speech[(i + offset) % n];
        }
    }
    out
}
