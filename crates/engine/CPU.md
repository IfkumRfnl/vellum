# The CPU path

What the engine's CPU path computes, why, and what bounds it. The CPU path is both the lab's
oracle and the app's fallback where there is no usable GPU; it has two modes.

- **The int8 path** (the default on x86-64 with AVX2, where a model was measured to stay inside
  the lab's gates with it): the products, and for Parakeet and Sortformer the attention, as the
  reference's CPU path (ggml) computes them. Parakeet, Sortformer and Whisper Turbo's encoder take
  it; Whisper Small and Whisper's decoder don't (below).
- **The f32 oracle** (`VELLUM_CPU_GEMM=f32`, and the default where there is no int8 vector kernel):
  every layer in f32 against its exactly dequantized weights. `VELLUM_GGML_ROUNDING=1` adds the
  rounding experiment on top of it (plain scalar code: the int8 kernels' reference).

## Int8 products (`cpu/int8.rs`)

ggml's CPU build (the reference's, `GGML_NATIVE` on MSVC: AVX2, FMA, F16C, llamafile's sgemm)
multiplies a Q8_0 weight by its input quantized the same way: per 32 values an f16 scale
`d = max|x| / 127` and `round(x · 127 / max|x|)` as int8, ties to even (`quantize_row_q8_0`),
then per block the exact integer dot product, `(d_w · d_x) · Σ` added in f32
(`tinyBLAS_Q0_AVX`, `ggml_vec_dot_q8_0_q8_0`). `Q8Matrix` does the same arithmetic: the same int8
input, each block's integer sum exact, then one fused multiply-add per block, block after block.
ggml spreads a block over eight f32 lanes and adds them at the end; that order is the only
difference (about 1e-7). The kernels are bit-identical to a scalar implementation of that
arithmetic (checked on Parakeet's real weights at 1–1500 rows).

- **Layout.** Weights are repacked at load: groups of eight output rows, interleaved four inputs at a
  time, so one 32-byte vector holds the same four inputs of eight rows. One `vpdpbusd` then adds 32
  products into eight row sums, the input's four bytes broadcast. A block's sums for eight rows are
  one vector: converted, scaled by `d_w · d_x` and accumulated in one instruction each. Tiles of 3
  input rows × 2 groups (1 × 4 for a single row) keep 12 integer sums and 12 accumulators in the 16
  registers. The repacked matrix is exactly the file's size (34 bytes per 32 weights).
- **Kernels, chosen at run time:** AVX-VNNI (`vpdpbusd`, VEX: Alder Lake and later, Zen 5) and
  AVX-512 VNNI's 256-bit form (Ice Lake, Sapphire Rapids, Zen 4) take the weights offset by 128 as
  unsigned bytes and take each input block's `128 · Σx` off again (exact); plain AVX2 keeps them
  signed and uses `maddubs` with the signs moved onto the weights, as ggml does; a portable loop
  computes the same sums anywhere else. `VELLUM_CPU_INT8=avx2|portable` forces a slower one (same
  bits). The AVX-512 VNNI kernel is the AVX-VNNI one with the EVEX intrinsic; there is no such CPU
  here, so it is untested. There is no NEON kernel: on aarch64 the f32 oracle stays the default
  (`VELLUM_CPU_GEMM=int8` forces the portable loop).
- **F16 weights** (Parakeet's and Sortformer's pointwise convolutions) multiply as ggml does: the
  input rounded to f16, products and sums in f32 (`F16Matrix`, kept as f16: half the f32 copy).
- **F32 weights** stay exact (Whisper's convolutions: the reference keeps them f32 too).

On the i5-13450HX (16 threads), Parakeet's shapes, against the f32 path:

| shape | rows | f32 | int8 (AVX-VNNI) | int8 (AVX2) |
| --- | ---: | ---: | ---: | ---: |
| ff.linear1 1024→4096 | 28 | 0.96 ms | 0.18 ms | 0.33 ms |
| ff.linear1 1024→4096 | 375 | 5.45 ms (289 GMAC/s) | 2.14 ms (735) | 4.04 ms (390) |
| ff.linear2 4096→1024 | 1500 | 33.3 ms (189) | 6.92 ms (909) | 14.3 ms (441) |
| attn.q 1024→1024 | 375 | 1.50 ms | 0.67 ms | 1.15 ms |

## The reference's other roundings, reproduced on the int8 path

- **Flash attention** (`cpu/flash.rs`). The conformer (Parakeet, Sortformer) runs ggml's CPU
  flash attention. With 64 queries or more it is the tiled kernel: keys in tiles of
  64, each score a chain of fused multiply-adds over the head dimension, scaled, the mask added, an
  online softmax over the tiles with ggml's vector `expf`, values summed in f32. With fewer (Parakeet
  windows under ~5 s) it is the per-query kernel: the query rounded to the keys' type, the softmax
  updated key by key, the values summed in an f16 accumulator when they are f16. The conformer's
  relative-position scores are its mask, scaled and rounded to f16; Parakeet's keys and values are
  f16 (`kv_type` AUTO), streaming Sortformer's f32. Whisper keeps the oracle's exact attention: the
  same emulation of its encoder's kernel (f32 keys and values) and of its decoder's f16 caches
  probed further from the reference, not closer (encoder output 9.2e-3 vs 7.8e-3).
- **SiLU** as ggml's `ggml_vec_silu_f32` (`x / (1 + e^−x)` with its vector `expf`), **GELU** as its
  `ggml_vec_gelu_erf_f32` (the C library's `erff`), the subsampling's pointwise convolutions on f16
  input and kernel (ggml's f16 im2col).

Probing Parakeet against `ref-cpu` (`cargo lab probe --model parakeet`), relative L2:

| checkpoint | f32 oracle | int8 path |
| --- | ---: | ---: |
| enc.pre_encode.out | 3.2e-3 | 2.4e-4 |
| enc.block.0.out | 3.4e-3 | 8.3e-4 |
| enc.final | 5.4e-3 | 4.7e-3 |
| dec.joint.0 | 6.8e-4 | 1.9e-4 |

Fed the reference's input (`--inject`), each sub-step of a block lands at 2–6e-5; what remains end
to end is the quantized products amplifying small upstream differences (a value that rounds the
other way moves by a whole quantization step).

## Elsewhere on the CPU path

- `f32::mul_add` is a library call per element on a baseline x86-64 build; the attention's value
  sums, the depthwise and subsampling convolutions now use `axpy`/`fma_into` (AVX2+FMA at run time,
  same bits). Attention scores go through the tiled `products` (each still exactly `dot`).
- **Memory.** Converted layers keep their own copy only; `Gguf::release` drops the file's pages from
  the working set once a layer has its copy (Windows `VirtualUnlock`, `madvise(DONTNEED)` elsewhere),
  so a load never holds the file and the converted weights at once.

## Which model takes what

Quick tier against `ref-cpu` (`cargo lab check --tier quick --engine vellum-cpu --against ref-cpu`):

| model | path | identical | our WER (ref) | parity WER | DER (ref) | speakers (ref) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Parakeet | f32 oracle | 518/527 | 2.97% (2.96%) | 0.01% | | |
| Parakeet | int8 + flash | **520/527** | 2.98% | 0.01% | | |
| Sortformer | f32 oracle | | | | 22.30% (22.06%) | 94.26% (94.26%) |
| Sortformer | int8, exact attention | | | | 22.32% | 92.11% (fails) |
| Sortformer | int8 + flash | | | | **22.28%** | **94.26%** |
| Small | f32 oracle (**kept**) | 542/612 | 8.48% (8.29%) | 1.76% | | |
| Small | int8 enc + dec, flash emulation | 540/612 | 8.87% (fails) | 2.45% | | |
| Small | int8 enc + dec, exact attention | 542/612 | 8.56% (fails) | 1.99% | | |
| Small | int8 encoder, f32 decoder | 546/612 | 8.70% (fails) | 1.75% | | |
| Turbo | f32 oracle | 564/612 | 4.21% (4.09%) | 0.52% | | |
| Turbo | int8 encoder, f32 decoder (**kept**) | **573/612** | 4.21% | 1.07% | | |
| (reference GPU vs CPU) | | 543/612 (Small), 565/612 (Turbo) | 8.28%, 4.15% | 1.68%, 0.75% | | |

Small's WER is set by a handful of windows where greedy decoding with temperature fallback
reaches a near-tie: the excess over the reference's errors swung from 51 to 155 words across the
configurations above, on the same clips (a Tagalog-detected LibriSpeech utterance the reference's
CPU path happens to decode as English, VoxPopuli hallucinations, a repetition loop in the earnings
call, an early end of text in FLEURS French), while identity stayed 540–546. Every int8 variant
landed above the 0.2-point gate, so Small keeps the f32 path; its int8 encoder would make a window
about 1.7× faster (2.2 s clip: 1.23 → 0.76 s). The decoder stays f32 for both sizes: a step streams
the weights once for a few rows, which the f32 path already reads as Q8_0, and the exact decoder
kept the transcripts closest.

Pipelines stay bit-identical to window-by-window runs on the int8 path (tokens, probabilities and
times: Parakeet 9/9 windows of a TED talk, Turbo 5/5): every int8 output is one fixed sum per row.

## Speed

`cargo lab bench --tier quick --a ref-cpu --b vellum-cpu --blocks 2` on the i5-13450HX (AC power,
nothing else busy, ~5 GB of RAM free):

| model | median speedup (95% interval) | speed clips | model ready | peak memory |
| --- | ---: | --- | ---: | ---: |
| Parakeet | 2.62× (2.49–2.79) | all faster, 1.70–2.85×; long recordings 37× real time (ref 13×) | 286 ms (381) | 982 MB (1339) |
| Sortformer | 2.27× | 14 min of AMI in 15.6 s (35.5 s) | 73 ms (189) | 359 MB (529) |
| Turbo | 2.27× | all faster, 2.09–2.37× | 187–205 ms (257–413) | 1082 MB (1075) |
| Small (f32) | 1.07× | short clips 1.05–1.18×, long recordings 0.99–1.01× | 80–159 ms (198–238) | 547 MB (497) |

## What bounds it now

- **Parakeet, short windows:** a window costs ~33 ms even at 0.5 s: streaming all 0.73 GB of
  weights once (about 22 GB/s here), then ~23 ms per second of audio. Per block at 2 s the f16
  pointwise convolutions (f32 products on f16 weights, as the reference computes them) cost more
  than all the int8 products together while holding an eighth of the weights.
- **Parakeet, long windows:** the int8 products run at 700–900 GMAC/s on 16 threads (VNNI
  throughput bound); the rest of a block (layer norms, SiLU, attention, allocation of fresh
  buffers) is still about a third of its time.
- **Whisper:** Turbo's encoder (int8) and the f32 decoder per token; Small is the f32 products.
