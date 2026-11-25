# Changelog

Every release of Vellum. Downloads are on the [releases page](https://github.com/IfkumRfnl/vellum/releases).

## [0.2.0] — 2026-11-25

### Speaker labels, rebuilt for the GPU

Vellum now tells voices apart with NVIDIA's [Streaming Sortformer](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1), on the same device that transcribes: Vulkan on Windows and Linux, Metal on macOS, the CPU elsewhere. It follows each voice through the whole video in one pass, so a two-hour podcast keeps the same two speakers from the first word to the last.

| A 2½-hour, two-person podcast on an RTX 4060 laptop | 0.1.0 | 0.2.0 |
| --- | --- | --- |
| Speaker analysis | Crashed the app | **22 s** (397× real time) |
| The whole video, link to finished transcript | — | **1 min 56 s** |
| Words labelled with the right speaker | — | **99.7%** |

Transcription is as fast as before (138× real time with Parakeet). On BBC Learning English episodes, with two hosts and three guests in recorded clips, 92–95% of words go to the right speaker and the hosts 97–100%.

### Fixed

- **Long videos no longer crash Vellum.** On a two-hour video, speaker analysis ran out of memory and closed the app.
- **A question's last word stays with whoever asked it.** A change of speaker that lands mid-sentence moves to the nearest sentence end, so "So what do you ship?" is no longer split between the host and the guest.
- **Talking over someone doesn't split their sentence.** A listener's "yeah" or laugh over a word leaves it with the speaker, and a word in a brief pause keeps its neighbours' speaker instead of losing its label.
- **Each speaker is named once per turn**, where it begins, instead of above every paragraph, in the reader and in copied transcripts.

### Changed

- **Up to four voices.** The speaker model tells at most four people apart; in a video with more, some of them share a label.
- **New speaker model download: 139 MB**, in place of the pyannote and TitaNet models (46 MB). The old ones can be deleted from the Hugging Face cache (`models--csukuangfj--sherpa-onnx-pyannote-segmentation-3-0` and `models--csukuangfj--speaker-embedding-models`).
- **One speech engine.** sherpa-onnx and ONNX Runtime are gone: speech and speakers both run on transcribe.cpp, and Windows no longer ships an 18 MB ONNX Runtime beside Vellum.

### For developers

- **Live tests measure, not just pass.** Each reads whole videos as the app does and reports word error rate, speaker accuracy against scripts that name who said each line, and every stage's time as a multiple of real time, to `target/live-report.md`. A 2½-hour podcast covers long audio. See [tests/README.md](https://github.com/IfkumRfnl/vellum/blob/main/tests/README.md).
- Unit tests of internal logic are gone in favour of those.
- CI keeps each desktop's built speech engine and compiled dependencies between runs, and builds every desktop only for a version tag or by hand.

## [0.1.0] — 2026-11-23

The first release.

- **Local speech recognition.** Parakeet Unified English, Whisper Small and Whisper Large v3 Turbo through transcribe.cpp, on Vulkan, Metal or the CPU. No Python, nothing uploaded.
- **Speaker labels.** Vellum can tell voices apart and label each speaker's turns.
- **One progress indicator.** A plate of dots fills as the transcript is made, with the percentage, the time left and the words as they're recognized.
- **Reading progress.** A hairline along the top fills as you read, and the toolbar shows how far you are and the time left at your reading pace.
- **Made for reading.** One continuous page; serif, sans or mono; adjustable size, width and spacing; timestamps open the video at that moment.

[0.2.0]: https://github.com/IfkumRfnl/vellum/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/IfkumRfnl/vellum/releases/tag/v0.1.0
