<p align="center">
  <img src="assets/app-icon.svg" width="120" alt="Vellum">
</p>

<h1 align="center">Vellum</h1>

<p align="center">
  <b>Read any video.</b> Paste a YouTube link and get a clean, readable transcript, transcribed on your own machine.
</p>

<p align="center">
  <img src="docs/working.jpg" width="49%" alt="Transcription in progress">
  <img src="docs/reader.jpg" width="49%" alt="The reader">
</p>

Vellum is a native desktop app written in Rust with [Rune](https://github.com/anomalyco/rune). It reads a video's captions, or downloads its audio and transcribes it locally with Parakeet or Whisper, optionally labelling speakers.

- **Local speech recognition.** Parakeet Unified English, Whisper Small and Whisper Large v3 Turbo through [transcribe.cpp](https://crates.io/crates/transcribe-cpp), on Vulkan, Metal or the CPU. No Python, nothing uploaded.
- **Speaker labels** from sherpa-onnx's pyannote segmentation and TitaNet embeddings.
- **One progress indicator.** A plate of dots fills as the transcript is made, with the percentage, the time left and the words as they're recognized.
- **Reading progress.** A hairline along the top and a small pill show how far you are, how long is left and your reading pace, which Vellum learns from how you scroll and remembers.
- **Made for reading.** One continuous page; serif, sans or mono; adjustable size, width and spacing; timestamps open the video at that moment; corrections in place.

## Install and run

Vellum uses the standalone [yt-dlp](https://github.com/yt-dlp/yt-dlp/releases) executable with Node.js 22+ or Deno for YouTube extraction, and FFmpeg for audio. It finds yt-dlp and FFmpeg beside its executable or on `PATH`. On macOS it also checks the Homebrew directories when launched from Finder.

Building from source requires Rust 1.99.1 or newer, CMake and a C++ compiler:

- **Windows:** Visual Studio C++ build tools and the Windows SDK.
- **Windows and Linux:** Vulkan headers and shader tools, for the optional GPU backend.
- **macOS:** Xcode with the Metal toolchain.

```sh
cargo run --release --locked
```

When you move a build, keep the native libraries with it: the DLLs beside `vellum.exe` on Windows, or the `vellum-libs/` directory beside `vellum` on Linux.

| Desktop | Speech acceleration | Display |
| --- | --- | --- |
| Windows | Vulkan or CPU | Direct3D 12 |
| Linux | Vulkan or CPU | A graphical session with a Vulkan driver, hardware or software |
| macOS | Metal or CPU | A Metal-capable Mac |

## Using Vellum

On first launch, choose a speech model. Parakeet Unified English (731 MB) is the English default. Whisper Small (270 MB) and Whisper Large v3 Turbo (886 MB) handle many languages. The model starts downloading in the background while you continue. **Skip** reads YouTube captions instead.

Paste a link and press **Transcribe**. The model button in the top right switches models, chooses YouTube captions, or turns on **Use YouTube captions first**. **Label speakers** adds speaker turns.

In the reader, **Aa** changes the typeface, size, reading width, line spacing and timestamps. **Edit** lets you correct paragraphs. **Copy transcript** at the end of the page copies everything, including your corrections and speaker labels.

| Keys | Action |
| --- | --- |
| Space / Shift+Space | Page down / up |
| Home / End | Glide to the top / end |
| Ctrl + mouse wheel | Text size |
| Esc | Cancel a transcription, or close **Aa** |

Speech recognition picks the GPU automatically and falls back to the CPU on a device error, keeping the work already done. Set `TRANSCRIBE_BACKENDS=cpu` to force the CPU. Speaker analysis always runs on the CPU. Audio is temporary. Model weights are checked against their SHA-256 digests and kept in the Hugging Face cache (`HF_HOME`, `HF_HUB_CACHE`). Settings live in your platform's config directory, or in `VELLUM_CONFIG_DIR` if set.

## How it's built

```
src/
├── main.rs          Window setup
├── app.rs           State, screens, actions and the per-frame poll
├── settings.rs      Preferences and their background writer
├── document.rs      A transcript, word counts, time formats
├── pipeline.rs      Worker updates, and the Tracker behind the progress indicator
├── reading.rs       Reading progress and your measured pace
├── job.rs           Worker threads, cancellation, tool processes
├── scenes.rs        Demo states for headless screenshots
├── media/           YouTube, speech recognition, speaker labels, model downloads
└── ui/              Theme, widgets, home, model picker, progress screen, reader,
                     and fx/ (the compositor that draws the progress dots)
```

The worker reports updates (stage, fraction, recognized speech, device) over a channel. `pipeline::Tracker` folds them into one smoothed fraction that only moves forward: it glides between sparse reports and estimates the steps that report nothing. The dots are an `El::effect` slot drawn by `ui/fx/dots.wgsl`. Scrolling the reader never re-runs the view; the scroll handler writes Rune live values.

## Development

```sh
cargo test --locked                                      # offline: logic and UI
cargo test --release --locked -- --include-ignored --test-threads=1 --nocapture   # plus live videos
```

The live suite downloads public videos and compares full local transcriptions with the publisher's captions. See [the test cases and acceptance criteria](tests/README.md).

Debug builds include Rune's headless tooling, so you can render any screen without a window:

```sh
cargo run -- --scene working --snap working.png --at 2      # also: home, finished, reader
cargo run -- --scene demo --realtime --video demo.mp4 --duration 14
```

## License

[MIT](LICENSE)
