# Vellum tests

```sh
cargo test --locked
```

runs offline: the app's screens through Rune's headless UI harness (onboarding, progress, reader).

```sh
cargo test --release --locked tests::live -- --ignored --test-threads=1
```

reads whole public videos as the app does and measures the results:

- **Words:** word error rate against the uploader's own captions (a TED talk with each speech model, two BBC 6 Minute English episodes) or a published transcript (a 2½-hour Lex Fridman Podcast episode).
- **Speakers:** the share of words labelled with the right speaker, against the BBC scripts and the podcast transcript, which name who said each line. Labels pair with people one to one, so splitting a voice or merging two costs words.
- **Speed:** each stage's time, and transcription and speaker analysis as multiples of real time. On a GPU, the long podcast must stay above a speed floor.

Each test prints its results and appends them to `target/live-report.md`. Thresholds sit just under the measured values noted beside them.

The live tests need yt-dlp, FFmpeg and Node.js or Deno, and download any speech models that aren't cached yet. Set `HF_HUB_OFFLINE=1` to use only cached models, or `VELLUM_DEVICE=cpu` to test on the CPU.
