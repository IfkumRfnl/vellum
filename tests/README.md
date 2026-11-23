# Vellum tests

The app tests live here so the Rune UI harness can exercise the same private state and event handlers as the desktop window. Cargo runs them with the Vellum binary test target.

| Video | What runs |
| --- | --- |
| [Me at the zoo](https://youtu.be/jNQXAC9IVRw) | First launch → Skip → paste link → transcribe → reader. Also labels speakers independently on the existing caption track and on forced Parakeet output. |
| [Try something new for 30 days](https://youtu.be/JnfBXjWm7hc) | Captions only, then Parakeet Unified English, Whisper Small and Whisper Large v3 Turbo on the same full talk. |
| [United against food waste](https://youtu.be/Z51Q29u4CWc) | Captions first with Parakeet selected, then forced Parakeet transcription with speaker labels. |
| [The benefits of doing nothing](https://youtu.be/Y681hXWwhQY) | Captions first with Whisper Large v3 Turbo selected, then forced Turbo transcription with speaker labels. |

Each accuracy case checks the channel, title, reference language, chronological paragraphs and coverage near the end of the recording. It compares the full transcript with the publisher's English YouTube captions. Word error rate counts missing, extra and substituted words, ignoring case and punctuation.

| Recording | Model | Measured word error rate¹ | Test fails above |
| --- | --- | ---: | ---: |
| TED talk | Parakeet Unified English | 2.5% | 8% |
| TED talk | Whisper Small | 10.3% | 16% |
| TED talk | Whisper Large v3 Turbo | 2.9% | 8% |
| BBC, food waste | Parakeet Unified English | 5.5% | 12% |
| BBC, doing nothing | Whisper Large v3 Turbo | 2.7% | 12% |

¹ Measured on Windows with Vulkan on an RTX 4060 Laptop GPU. A separate CPU-only Parakeet run on the TED talk also measured 2.5%.

The BBC cases require at least two speakers with 20 or more words each and labels covering at least 80% of the recognized words. The food-waste case also checks that the two hosts, reporter and two interviewees receive distinct labels on known lines, and that both hosts keep their labels when they return. Those checks passed, but the run produced eight labels overall, including one covering a single word. [BBC's speaker-labelled script](https://downloads.bbc.co.uk/learningenglish/features/6min/221229_6_minute_english_united_against_food_waste.pdf)

```sh
cargo test --release --locked
cargo test --release --locked tests::live -- --ignored --test-threads=1 --nocapture
```

The live suite uses yt-dlp, FFmpeg, a supported JavaScript runtime and the same native inference engines as the app. Videos and caption URLs are fetched afresh. Cached models are reused. An explicitly enabled live case can download missing weights. Set `HF_HUB_OFFLINE=1` to require cached weights while still fetching videos from YouTube. Set `TRANSCRIBE_BACKENDS=cpu` before starting the test process to exercise CPU inference.

The offline UI cases cover model selection, reading width, zoom, saved preferences, the progress screen's finish and cancel, and reading progress while paging through a long transcript. Model preparation has its own opt-in case. To run that alongside the live suite, use the full verification command in the project README.
