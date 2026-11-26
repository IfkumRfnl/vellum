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

- **Local speech recognition.** Parakeet Unified English, Whisper Small and Whisper Large v3 Turbo on Vellum's own speech engine, written in Rust, on the GPU or the CPU. No Python, nothing uploaded.
- **Speaker labels.** Vellum can tell up to four voices apart and label each speaker's turns, on the GPU when there is one.
- **One progress indicator.** A plate of dots fills as the transcript is made, with the percentage, the time left and the words as they're recognized.
- **Reading progress.** A hairline along the top fills as you read, and the toolbar shows how far you are and the time left at your reading pace, which Vellum learns from how you scroll and remembers.
- **Made for reading.** One continuous page; serif, sans or mono; adjustable size, width and spacing; timestamps open the video at that moment.

## Install

Download Vellum for your system from the [latest release](https://github.com/IfkumRfnl/vellum/releases/latest): Windows (x64, ARM64), macOS (Apple Silicon, Intel) or Linux (x64, ARM64). Unpack it and run `vellum`, or open `Vellum.app` on a Mac. The Mac app isn't notarized, so the first time, right-click it and choose **Open**. What's new in each release is in the [changelog](CHANGELOG.md).

Vellum also needs [yt-dlp](https://github.com/yt-dlp/yt-dlp/releases), with Node.js 22+ or Deno, and [FFmpeg](https://ffmpeg.org/download.html). It finds them beside its executable or on your `PATH`, and on a Mac in Homebrew's directories too.

### Build from source

You need Rust 1.99.1 or newer and a C compiler for a few dependencies (the Visual Studio build tools Rust uses on Windows, Xcode's command line tools on macOS). Linux also needs the xkbcommon, fontconfig and Wayland development packages.

```sh
cargo run --release --locked
```

A build is a single executable, `target/release/vellum`, with no libraries to keep beside it.

| Desktop | Speech acceleration | Display |
| --- | --- | --- |
| Windows x64 | A discrete GPU (Vulkan, else Direct3D 12) or the CPU | Direct3D 12 |
| Windows ARM64 | CPU | Direct3D 12 |
| Linux | A discrete GPU (Vulkan) or the CPU | A graphical session with a Vulkan driver, hardware or software |
| macOS | Metal or CPU | A Metal-capable Mac |

## Using Vellum

On first launch, choose a speech model. Parakeet Unified English (731 MB) is the English default. Whisper Small (270 MB) and Whisper Large v3 Turbo (886 MB) handle many languages. The model starts downloading in the background while you continue. **Skip** reads YouTube captions instead.

Paste a link and press **Transcribe**. When the video has its uploader's own captions, Vellum uses them, which takes seconds; otherwise it transcribes the audio with your speech model. The model button in the top right switches models or chooses YouTube captions, automatic ones included. **Label speakers** adds speaker turns.

In the reader, **Copy** copies the whole transcript, and **Aa** changes the typeface, size, reading width, line spacing and timestamps. The toolbar shows how far you are and how long is left at your reading pace.

| Keys | Action |
| --- | --- |
| Space / Shift+Space | Page down / up |
| Home / End | Glide to the top / end |
| Ctrl + mouse wheel | Text size |
| Esc | Cancel a transcription, or close **Aa** |

Speech recognition uses your GPU when it can and falls back to the CPU otherwise; set `VELLUM_DEVICE=cpu` to always use the CPU. Downloaded audio is deleted once the transcript is ready. Models are kept in the Hugging Face cache, so other apps that use it can share them.

## Development

```sh
cargo test --locked
```

See [tests/README.md](tests/README.md) for the optional tests that transcribe real videos, and [crates/lab/README.md](crates/lab/README.md) for the engine lab, which measures Vellum's speech engine against transcribe.cpp.

## Credits

Vellum stands on [Rune](https://github.com/anomalyco/rune) for its interface, [transcribe.cpp](https://crates.io/crates/transcribe-cpp), from which its speech engine is ported, NVIDIA's [Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) (CC-BY-4.0) and OpenAI's [Whisper](https://github.com/openai/whisper) (MIT) models, NVIDIA's [Streaming Sortformer](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) speaker model (NVIDIA Open Model License), [yt-dlp](https://github.com/yt-dlp/yt-dlp), [FFmpeg](https://ffmpeg.org) and the [Inter](https://rsms.me/inter/) typeface.

## License

[MIT](LICENSE)
