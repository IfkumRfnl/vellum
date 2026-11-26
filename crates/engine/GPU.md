# The GPU path: plan

What the engine's GPU path is built to do, and why, from the lab's measurements of the reference and
three independent design reviews. Numbers marked "est." are estimates to be replaced by `cargo lab
bench` results.

## Where the reference loses time

transcribe.cpp on the RTX 4060 Laptop GPU, over 17.9 hours of audio (`cargo lab goldens`):

| Stage | Where | Share | Per 26 s window, 18-minute call |
| --- | --- | ---: | ---: |
| mel | CPU | 3.7% | 10 ms |
| encoder | GPU, tensor cores (coopmat2) | 55.3% | 79 ms |
| RNN-T decoding | CPU, small ggml graphs per step | 25.4% | 70 ms |
| everything else | graph setup, scheduling, copies | 15.5% | 26 ms |

The stages run one after another, one window at a time. A 2.2 s clip takes 42 ms: a fixed cost of
roughly 15–35 ms per call. The encoder reaches about 5 TFLOP/s.

## Where we are

RTX 4060 Laptop GPU, quick tier, paired against the reference (`cargo lab bench --tier quick
--models parakeet --a ref --b vellum`): median speedup 3.17× (95% interval 2.97–4.65); the
18-minute earnings call in 1.20 s (911× real time, the reference 6.96 s); 2–5 s clips in 13–18 ms
(the reference 34–49 ms); model ready in 0.87–0.95 s (0.83 s); peak memory 1.0 GB (0.8 GB). 524 of
527 clips give the reference's transcript, with equal WER. Long recordings are now bound by the GPU
encoder alone (about 26 ms per 28 s window in batches of 8, back to back); short ones by its
fixed ~11.5 ms per call (streaming 1.25 GB of f16 weights at low occupancy).

Learned on the way:

- Windows 11 power-throttles a background process onto efficiency cores at low clocks (decoding
  7× slower). The engine opts its process out when it loads a model (`keep_full_speed`); the
  reference measures the same either way.
- The encoder stage keeps two batches in flight (`submit` returns the batch before), and batches
  shrink toward the end of a recording (…8, 5, 2, 1) so the last windows' decoding overlaps too.
- A single window (short clips, a recording's tail) decodes on a team of up to 6 threads that
  split each LSTM step and joint evaluation by rows (`parakeet/team.rs`), bit-identical to one
  thread and 4× faster; the CPU budget is fixed (4 mel threads, 1 GPU feeder, the rest decoders).

## The execution model

1. **A recording at a time.** The harness hands the engine every window of a recording at once
   (`Engine::transcribe_all`); the windows are the app's, so results stay comparable.
2. **Ragged batches.** B windows (8 on the 4060, measured per adapter) packed as one `[ΣT, C]`
   sequence with an offsets table. Linear layers, LayerNorm and feed-forward need no masks and waste
   nothing on padding; only attention, the depthwise convolution and the pre-encode stop at window
   edges. A window's result must not depend on its batch: the lab checks B=1 against B=8.
3. **Three stages in flight.** CPU workers make mel features, one GPU thread encodes batches, CPU
   workers decode windows (one each). Two batches in flight so the GPU never waits on a readback.
4. **The joint network's encoder projection (1024→640) runs on the GPU**, and only its output comes
   back: 0.9 MB per window, through a ring of mapped buffers.
5. **A fixed plan.** Persistent buffers and bind groups made at load, shapes in push constants, about
   14 dispatches per block, a submit every few blocks (each submit well under 200 ms, so a slow GPU
   never hits Windows' 2 s timeout).
6. **The engine's own `wgpu::Device`**, apart from the UI's: nothing is shared between them, and a
   lost device or a long queue then can't stall the window.

Est.: 50–65 ms per window without tensor cores (400–500× real time), 18–25 ms with them (~1000×).
Short clips under 10 ms.

## The encoder

- **Weights dequantized to f16 once at load** (about 1.25 GB), in one buffer per block (drivers cap
  storage bindings at 128 MiB). The residual stream, every accumulator, LayerNorm statistics and
  attention softmax stay in f32: activations reach 1,381 after block 0's first feed-forward, and a sum
  of squares over 1024 channels overflows f16 at an RMS near 8.
- **Matrix multiplies on tensor cores where the adapter has them.** Rune's wgpu-hal (which the app
  builds against) enables `EXPERIMENTAL_COOPERATIVE_MATRIX` for any configuration the driver reports;
  the 4060 reports f16 16×16×16 with f32 accumulation, and naga 30 compiles `coop_mat16x16`. The
  reference already uses tensor cores here, so this kernel (`gemm_f16_cm16`, after ggml's `mul_mm`
  shape: 64×64 tiles, K steps of 16, staging in workgroup memory) decides whether we win the encoder.
  A portable SIMT kernel (`gemm_f16_simt`, register tiles, autotuned at first run) covers other
  adapters; `dot4I8Packed` with 8-bit activations is a specialization for adapters where it measures
  faster, at about one reference spread of extra error.
- **Fusions, by payoff:** bias, activation, the ½ scale and the residual add in every multiply's
  epilogue; `W_pos·pos_emb` precomputed per layer for every relative offset (34 MB, 11% of the FLOPs,
  exact); q, k, v as one 1024→3072 multiply; attention scores computed as
  `(q+u)·kⱼ + (q+v)·p[i−j]` by index, never materializing the shifted matrix; GLU inside pointwise-1
  (halves interleaved at load); batch norm folded into the depthwise weights at load, with SiLU, in one
  kernel; residual + LayerNorm in one kernel.
- **Shader hygiene:** trusted shader modules without runtime bounds checks once verified, no workgroup
  zero-init, about 20 pipelines in all (never specialized by T or B), a persisted pipeline cache.

## RNN-T decoding

On the CPU, in f32, one window per worker; a GPU decoder needs hundreds of round trips per window for
nothing while the cores idle. Exact speedups over the reference's loop:

- the predictor runs only when a token is emitted, and its joint projection is cached between;
- layer 0's input product `W_ih·emb[token] + b` is a lookup table (1025×2560, 10.5 MB);
- across a run of blanks the joint is evaluated for the next several frames as one small multiply,
  stopping at the first token;
- LSTM weights stay int8 with scales in cache (6.5 MB), converted as they're read;
- argmax keeps the lowest index on ties; `max_symbols` and confidence as the reference computes them.

## Order of work

1. CPU oracle (in progress): mel, encoder, decoder, token-exact against the reference.
2. Fast CPU decoder (the list above) and the three-stage pipeline with the reference-speed encoder
   still on the CPU — the pipeline shape is testable before any GPU code.
3. GPU encoder, SIMT kernels first, checkpoint by checkpoint against the oracle within the
   calibrated spread; then batching; then the tensor-core multiply.
4. Profile mode: GPU timestamps per kernel family (one pass per kernel, measurement only).

## Portability

Everywhere: ragged batching, the fusions, f16 storage with f32 math, the CPU decoder. NVIDIA and AMD
on Vulkan get the 16×16 tensor-core multiply; Apple gets Metal's 8×8 simdgroup matrices; DX12 and
Intel the SIMT or int8 kernels. The device is chosen by a short benchmark, not by vendor: this
laptop's own Intel iGPU is slower than its CPU.

## Whisper (Small, Large v3 Turbo)

`whisper/gpu.rs` and `whisper/gpu/`, against the CPU oracle by `cargo lab whisper` (checkpoints,
decisions with their margins, batch independence, timing by batch size; `--explain` runs whole
clips through both and classifies each window that differs). Same rules as Parakeet's: f16
weights for the tensor cores, everything else f32; a window's result never depends on its batch.

- **Encoder**, a batch of 30 s windows at once: the two convolutions as multiplies over their
  receptive fields (`im2col.wgsl`, weights reordered tap-major at load), GELU in the epilogue, the
  second one accumulating into the learned positions; per block a layer norm, the fused q/k/v
  multiply, flash attention on tensor cores (`attend.wgsl`: 64 queries × 64-key blocks, f16
  operands, f32 scores, statistics and output; the rescale skipped when no row's maximum rose,
  the next block's keys and values loaded while this one multiplies), the output projection and
  the GELU feed-forward added into the residual stream by their epilogues; a final layer norm.
  Then every decoder layer's cross-attention keys and values in one stacked multiply, rounded to
  f16 into the window's decoder slot (the oracle's cache rounding).
- **Decoder**, every window with a slot in lockstep, a token per step: weights as the file's
  Q8_0 (`gemv.wgsl`: exact `scale · q`, half f16's bytes; the layer norm before a layer fused into
  its multiply), self-attention keys and values cached in f16, cross-attention split over six
  parts of the encoder positions so one window still fills the GPU, and the choice on the GPU
  (`choose.wgsl`: suppression, the timestamp rules, argmax with its log-probability and margin).
  Each slot's state lives on the GPU, so greedy steps go 8 (one window) or 16 (the pipeline) per
  submit, two submits in flight, and only the state and choices come back. A fallback attempt
  (temperature > 0) reads its slot's logits back each step and samples on the CPU with the
  oracle's own code and seeded generator; its prompt's keys and values are reused from the first
  attempt, and language detection's step is the first attempt's first.
- **Pipeline** (`whisper/pipeline.rs`, `run_each`): mel features on their own thread, encoder
  batches between decoder runs, eight decoder slots; a window keeps its slot through its
  passes and attempts. Without a language only the first unfinished window decodes until one has
  text (the app's rule), so results are exactly `run`'s window by window: the quick tier's 612
  clips per model give the same tokens, probabilities and times either way
  (`VELLUM_WHISPER_SEQUENTIAL=1` for the window-by-window path).
- **Memory, allocated once per model**: encoder arenas, decoder slots and caches, the decoder's
  three readback buffers; a run allocates nothing but wgpu's command buffers and upload staging.
  An encoder batch costs the same per window at any size, so its activations get a budget (192
  MiB: 4 windows for Small, 2 for Turbo). Uploads are flushed every 128 MB during a load (wgpu
  otherwise keeps every buffer's staging copy in RAM until the first submit: the whole f16
  encoder twice over). A load's out-of-memory is an error (`Gpu::allocating`, an error scope);
  any later one, or a lost device, is kept by the device and returned by the next wait or
  readback (`Gpu::check`) instead of only printed.
- **Mel**: frames of the 30 s padding are silence, power 0 in every bin, so they take the clamp's
  value without an FFT (the same bits; a 2 s window's mel goes from 7 ms to under 1).

On Windows the NVIDIA driver charges the device memory a process allocates to the system's
commit: Turbo's 2.5 GB on the GPU shows as 2.7 GB of private bytes. With the commit nearly
exhausted by other programs, either engine can fail to allocate (a 66 MB host allocation in a
load, out of memory for the cross cache), which once failed every clip of a Turbo check.
Over 612 recordings in one process: GPU memory in use 4.56 → 4.54 → 4.45 GB (start, middle,
end), private bytes 2676 → 2725 → 2733 MB (the harness keeping each clip's output).

Measured on the RTX 4060 Laptop GPU (`cargo lab whisper`, GPU timestamps):

| | Small | Turbo |
| --- | ---: | ---: |
| encoder, ms per 30 s window (any batch) | 20 | 118 |
| — of it matrix multiplies, attention | 13, 5.5 | 89, 27 |
| decoder step, 1 window | 1.10 ms | 1.15 ms (before the gemv change) |
| decoder step, 8 windows (per window) | 3.8 ms (0.47) | 2.7 ms (0.33) |
| a step past every window's end | 0.09 ms | 0.04 ms |

What bounds it: the encoder's multiplies run at 21–22 TFLOP/s, about 92% of the 4060's f16
tensor-core rate with f32 accumulation (24 SMs × 512 FLOP/clock × ~1.9 GHz). The reference gets
its encoder speed differently: it quantizes activations to 8 bits and multiplies on int8 tensor
cores, which wgpu doesn't expose (its cooperative-matrix types are f32, f16, i32 and u32); f16
accumulation (F16→F16) measured no faster. Encoding less than 30 s isn't exact: the encoder's
attention is global, so every position depends on the padding, and the reference always encodes
all 3000 frames. A one-window decoder step streams Small's 139 MB of Q8 weights and 55 MB of
encoder keys and values (a 0.86 ms floor at ~225 GB/s; 1.10 ms measured). The gemv kernel is
compiled per window count (1, 2, 4, 8), so one window keeps 4 accumulators per lane, not 32:
`gemv/head` went from ~45 to ~225 GB/s, with each output summed exactly as before. Eight
windows in lockstep are bound by their encoder keys and values (cross-attention 1.85 ms of a
3.8 ms Small step) and by the gemv re-reading x per row.

Against the reference (`cargo lab bench --tier quick --models small,turbo --a ref --b vellum
--blocks 4`, whole recordings through `run_each`): median speedup 1.91× for Small (95% interval
1.85–2.33; every clip faster; short clips 1.69–1.94×, long recordings 2.47–2.72×), 1.45× for
Turbo (1.43–1.49; short clips 1.39–1.52×, two chapters and the TED talk 1.62–1.92×, the
18-minute earnings call 0.93×). Turbo's short clips are its 118 ms encoder pass, which the
reference does in int8: they can't reach 2× without a faster multiply.

The encoder's slow spells: running Turbo's encoder alone, back to back on the same input, its
tensor-core kernels (the multiplies, sometimes attention) slow 10–19× in a fixed cycle after
6–70 s of sustained load (seven batches fast, eight slow), clocks, P-state and power (the 80 W
cap) unchanged, no hardware or thermal slowdown reported, memory controller 81% busy against
29%. Nothing else uses the GPU then (per-process engine counters), nothing is demoted to shared
memory, and it isn't the decoder, the pipeline, uploads, submit size or the tile shape (coop
64×64×32 too; SIMT only later, at 40% of the speed). Small with batches of 8 does it too;
Small's own batches (47 ms of work) almost never. In the bench it costs the long recordings:
up to 40% of a Turbo job's encoder batches, the earnings call above. Unexplained below the API;
the reference, lighter on the GPU, doesn't trigger it.
