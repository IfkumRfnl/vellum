# Changelog

Every release of Vellum. Downloads are on the [releases page](https://github.com/IfkumRfnl/vellum/releases).

## [Unreleased]

### Vellum's own speech engine

Speech and speaker labels now run on Vellum's own engine, written in Rust: Parakeet Unified English, Whisper Small, Whisper Large v3 Turbo and Streaming Sortformer, ported from transcribe.cpp and checked against it step by step. It reads the same model files, so there's nothing new to download. It runs on the GPU through the same graphics stack as the interface (Vulkan on Windows and Linux, Metal on macOS), on the CPU when there's no discrete GPU, and moves the rest of a video to the CPU if the GPU fails mid-way.

| A 2½-hour, two-person podcast on an RTX 4060 laptop | transcribe.cpp | Vellum's engine |
| --- | --- | --- |
| Transcription (Parakeet) | 54 s (163× real time) | **9 s** (926× real time) |
| Speaker analysis | 21 s (412× real time) | **9 s** (949× real time) |
| The whole video, link to finished transcript | 1 min 35 s | **39 s** |
| Word error rate | 7.0% | 7.0% |
| Words labelled with the right speaker | 99.7% | 99.7% |

Shorter videos gain too. A 6-minute BBC episode transcribes at 726–743× real time with Parakeet (146× before) and 134–149× with Whisper Large v3 Turbo (66–68× before), with the same speaker accuracy (92.0% and 95.4%); a 3½-minute TED talk at 632× with Parakeet (152×), 240× with Whisper Small (68×) and 145× with Turbo (60×). Word error rates are the same throughout, but for one word in the BBC episode with Parakeet (5.6% against 5.5%).

The words are transcribe.cpp's. In the engine lab, on a broad sample of read speech, meetings, calls, talks and edge cases, Parakeet gives its exact transcript on 521 of 527 recordings with the same word error rate, Whisper Small on 570 and Turbo on 589 of 612. Where the Whisper port differs, the lab traced each case to a near-tie (two words, or two languages, the model rates almost equally) or to a window that falls back to random sampling, which transcribe.cpp seeds differently on every run. Speaker turns match within transcribe.cpp's own spread between its CPU and GPU (DER 22.29% against 22.34%, 94.26% of words to the right speaker either way). Paired against it on the same laptop, the median recording is faster everywhere:

| Median recording, against transcribe.cpp | GPU (RTX 4060 laptop) | CPU (Core i5-13450HX) |
| --- | --- | --- |
| Parakeet | **4.0×** | **2.6×** |
| Speaker labels | **3.6×** | **2.3×** |
| Whisper Small | **2.45×** | 1.07× (level on long recordings) |
| Whisper Large v3 Turbo | **1.7×** | **2.3×** |

On the GPU, every recording in the speed set is faster than transcribe.cpp. Parakeet's GPU encoder runs on int8 tensor cores, and the CPU uses AVX-VNNI where the processor has it.

### Changed

- **Smaller download, one file.** Vellum is a single executable: no speech engine DLLs beside `vellum.exe` on Windows, no `vellum-libs/` on Linux, no libraries in the Mac app. The Windows x64 download goes from 28 MB to 11 MB (80 MB to 28 MB unpacked).
- **Cancelling stops at once.** On the CPU, a transcription or speaker analysis stops within about a tenth of a second instead of finishing the window it's on.
- **The progress screen names the device** every model runs on, as "Vulkan · NVIDIA GeForce RTX 4060 Laptop GPU" or "CPU".
- **Integrated GPUs are left alone.** Speech runs on a discrete GPU or Apple's, otherwise on the CPU, which is often faster than an integrated GPU.
- **`VELLUM_DEVICE=cpu`** keeps speech on the CPU, in place of `TRANSCRIBE_BACKENDS=cpu`.
- **Downloads include `THIRD_PARTY_NOTICES.md`,** with the MIT licenses of transcribe.cpp and ggml, which the engine is ported from.

### For developers

- **The engine lab** (`cargo lab`, see [crates/lab/README.md](https://github.com/IfkumRfnl/vellum/blob/main/crates/lab/README.md)) runs Vellum's engine against transcribe.cpp on a corpus of recordings with known text and speakers: `check` compares transcripts and speaker turns with the reference's and the truth, `probe` compares every intermediate tensor on one window, `bench` measures paired speed. transcribe.cpp is now only the lab's yardstick, built apart in `target/reference`; the app never compiles it.
- **Building needs no C++.** No CMake, C++ compiler, Vulkan SDK or Metal toolchain: Rust and a C compiler for a few dependencies. CI no longer builds or caches a native speech engine.

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

[Unreleased]: https://github.com/IfkumRfnl/vellum/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/IfkumRfnl/vellum/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/IfkumRfnl/vellum/releases/tag/v0.1.0
