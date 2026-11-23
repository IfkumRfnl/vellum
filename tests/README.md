# Vellum tests

```sh
cargo test --locked
```

runs offline: the transcript, progress and reading logic, and the app's screens through Rune's headless UI harness.

```sh
cargo test --release --locked tests::live -- --ignored --test-threads=1 --nocapture
```

also downloads a few public videos and transcribes them with each speech model, checking the results against the videos' own captions. It needs yt-dlp, FFmpeg and Node.js or Deno, and downloads any speech models that aren't cached yet. Set `HF_HUB_OFFLINE=1` to use only cached models, or `TRANSCRIBE_BACKENDS=cpu` to test on the CPU.
