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
