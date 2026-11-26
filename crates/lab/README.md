# The engine lab

Vellum's own speech engine (`crates/engine`) is measured against transcribe.cpp, the C++ engine the app
ships today, on a corpus of recordings: for correctness at every level (tensors, decoder steps,
transcripts, speaker turns) and for speed. The goal is an engine that is faster and at least as
accurate; the lab is how we know.

```sh
cargo lab fetch --tier full               # once: download and prepare the recordings (~1.6 GB)
cargo lab goldens --tier long             # once, or after a reference change: its results for everything
cargo lab check                           # the loop: our engine against the reference, smoke tier
cargo lab check --tier micro --models parakeet   # three clips, under a second plus the engine's time
cargo lab probe --calibrate --model parakeet     # once per model: the reference's own CPU/GPU spread
cargo lab probe --model parakeet          # stage by stage: where do the tensors part ways?
cargo lab bench --tier quick --models parakeet   # paired speed runs, both engines live
cargo lab inspect parakeet --tensors      # what's in a model file
cargo lab kernels                         # GPU matrix multiplies on Parakeet's shapes: error and TFLOP/s
```

`kernels` runs each GPU matrix-multiply kernel (portable and tensor-core) on Parakeet's real weights
and shapes (M = 350, 1400, 2800 frames), checks it against the CPU with exactly dequantized weights
next to the rounding floor of f16 weights, and times it with GPU timestamps. `--adapter` and
`--backend` pick another GPU or API (e.g. the Intel iGPU, DX12). On the RTX 4060 the tensor-core
kernel reaches 20–24 TFLOP/s, about 90% of the card's f32-accumulate peak.

Every command takes `--engine`/`--against` (or `--a`/`--b` for `bench`): `vellum`, `vellum-cpu`,
`vellum-gpu`, `ref` (the reference as the app runs it), `ref-cpu`, `ref-gpu`. `--env KEY=VALUE`
passes settings to the reference, e.g. `--env TRANSCRIBE_NO_FLASH=1`.

`cargo lab` builds with the `lab` profile (release speed, incremental) and never links C++, so it
rebuilds in seconds. The reference runs in its own process, `vellum-ref`, built once in
`target/reference` with `GGML_NATIVE` (its best showing on this CPU) and transcribe.cpp's validation
hooks (so it can dump every layer). The lab builds it when it's missing; that takes a few minutes.

## Tiers

| Tier | For | Size |
| --- | --- | --- |
| `micro` | iterating on one kernel | a few clips, seconds |
| `smoke` | every change | one probe window per model, a few dozen utterances, edge cases; under a minute |
| `quick` | before a commit | a broad sample of every set; minutes |
| `full` | before a release or a claim | everything but multi-hour recordings |
| `long` | long-audio behaviour | everything, including a 2½-hour podcast |

Tiers nest: `quick` includes `smoke`, and so on. `cargo lab corpus` lists what's there.

## The corpus

Every clip is 16 kHz mono 16-bit FLAC in the lab's data directory (`VELLUM_LAB_DIR`, else
`%LOCALAPPDATA%\vellum\data\lab` on Windows), with its ground truth in `corpus/index.json`:

- **Read speech with verbatim text:** LibriSpeech test-clean and test-other, utterances and whole
  chapters.
- **The app's own cases:** a TED talk, two BBC 6 Minute English episodes with scripts naming each
  speaker, and a 2½-hour podcast with a speaker-named transcript.
- **Talks, calls and meetings:** Open ASR Leaderboard test sets, long-form sets, AMI meetings and
  VoxConverse recordings with speaker turns.
- **Other languages,** for Whisper: FLEURS.
- **Edge cases,** generated deterministically: silence (exact and almost), noise, tones, music-like
  sound, clipping, low gain, telephone bandwidth, noise at several SNRs, DC offset, shifts of 1, 80
  and 160 samples, lengths either side of the 28-second window, speech without pauses for the window
  cutter, and a language switch.

## What `check` compares

Both engines get exactly the windows the app would cut (`crates/harness/src/windows.rs`). For each
model and clip:

- **Against the reference:** identical text, word error rate between the two transcripts, token
  agreement, the first token where they part, word-timing differences; speaker-turn agreement.
- **Against the truth, for both engines:** word error rate with the Open ASR Leaderboard's text
  normalization, diarization error rate, and the share of words labelled with the right speaker.
- **Edge cases:** no crash, no words invented from silence, noise or tones.

The reference's results are kept (`goldens/`), keyed by everything that could change them: its
build, the model file, the machine and driver, the windowing and its settings. The loop never waits
for C++ unless something changed. Speed is never compared against kept timings.

Gates: no clip fails where the reference worked; word error rate at most 0.2 points above the
reference's, diarization error at most 0.5 points above, speaker accuracy at most 0.5 points below;
no more words invented from no speech than the reference invents. A model the engine doesn't
support yet shows as "not implemented yet" and passes unless `--strict`. Each report also says how
the numbers moved since the last run of the same check, so a regression inside the gates shows.

Bit-for-bit parity with ggml isn't the goal: ggml's GPU path is itself an approximation. On this
corpus the reference's CPU and GPU paths give the same Parakeet text on every smoke clip, Whisper
differs only in punctuation on a few, and Sortformer's DER moves by half a point.

## What `bench` measures

Both engines live, in alternating order (ABBA), on the speed clips, each in a fresh process with its
model loaded once:

- **Warm speed:** per clip, from samples in memory to results in memory.
- **Cold costs:** process start to model ready, and the first run with its kernel compilation.
- **Memory:** peak resident memory and GPU memory.
- **The GPU's clocks, temperature, power and throttle reasons** throughout. A pair where either
  side was throttled doesn't count. Run on AC power with the default power plan; `bench` warns
  otherwise.

Speedups are reported as the median of per-pair ratios with a bootstrap interval over clips. Running
the same engine on both sides (`--a ref --b ref`) measures the laptop's own noise; a speedup inside
it isn't one.

A job's GPU counts as slowed when its median clock is more than 5% under the best job's, or the
driver reports a hardware slowdown; its pairs are dropped. The driver's other reasons don't decide:
this laptop reports "software thermal slowdown" alongside every power cap, at full boost clocks.

## What `probe` shows

Both engines dump their tensors at the same checkpoints (`enc.mel.in`, `enc.pre_encode.out`,
`enc.block.<L>.out`, `enc.final`, `dec.joint.0`, …) for one window, and their decoder decisions step
by step. `probe` lines them up: relative error, cosine similarity, SNR, worst element, top-1
agreement and margins for logits, and the first checkpoint past tolerance. `--sub-blocks 0,23`
adds those blocks' attention, convolution and feed-forward outputs. `--inject enc.mel.in` feeds a
checkpoint of the reference into our engine, so one stage can be tested on exactly the input the
reference gave it.

Tolerances come from `--calibrate`: the reference's GPU against its CPU over the model's probe
clips, per checkpoint; a probe allows twice that (at least 1e-5). For Parakeet the spread is
4–7e-3 relative error through the encoder and 1e-3 at the joint output; mel, positional
embeddings and the decoder's first state are exact. The intended use: our CPU path is the oracle,
checked against the reference's CPU path, and our GPU path is held to the calibrated spread
against it.

## Where the reference stands

On the RTX 4060 Laptop GPU (driver 616.56), quick tier, warm runs:

- Parakeet: 158–178× real time on recordings of 3–18 minutes; about 34 ms per window even for
  2 seconds of audio, the fixed cost of a run. Model load 0.84 s, peak memory 808 MB, 2.8 GB of GPU
  memory in use.
- The first run after installing a build compiles Vulkan pipelines for about 12 s; later
  processes find them in the driver's cache (about 0.1 s).
- Word error rates (smoke): Parakeet 1.3%, Whisper Small 4.9%, Whisper Turbo 3.7%. Sortformer:
  DER 25% on AMI meetings, 2.4% on VoxConverse, and 94.3% of words labelled with the right speaker
  on the BBC episodes (the app measured 92–95%).
- Noise: the reference against itself on the quick tier's speed clips measures 0.997 (95% interval
  0.987–1.014), so speedups beyond about ±1.5% are real.

## Files

- `target/lab/<run>.md` and `.json`: each run's report; `target/lab/latest.md` is the last one.
- `<lab dir>/history.jsonl`: every run's results, for trends and regressions.
- `<lab dir>/goldens/`, `<lab dir>/dumps/`: the reference's kept results and tensors.
