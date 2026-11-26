// The subsampling's first two convolutions in one pass: conv.0 (3×3, stride 2, from the mel
// spectrogram's one channel to C) with its ReLU, then conv.2 (depthwise 3×3, stride 2) over
// that. Each conv.0 output a conv.2 tap needs is recomputed from the mel rather than stored:
// 81 multiply-adds per output instead of writing and reading back the largest activation of
// the encoder (a quarter of the mel's points × C channels).
//
// Both are 3×3 with one row/column of zero padding; channel-last out, one row of C values per
// (time, frequency) point. One thread per (output point, 4 channels); every sum in the CPU
// oracle's order (Δt outer, Δf inner, skipping the padding) with fused multiply-adds.
//
// Prepended by the host: const C4 (channels / 4).

struct Params {
    // The window's mel: `mels` rows of `frames` values, starting at `offset`.
    frames: u32,
    mels: u32,
    offset: u32,
    out_time: u32,
    out_freq: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read> mel: array<f32>;
// [tap][C4], tap = 3·Δt + Δf, for conv.0 and conv.2.
@group(0) @binding(1) var<storage, read> weight0: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bias0: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> weight2: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> bias2: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> out: array<vec4<f32>>;

const POINTS: u32 = 256u / C4;

@compute @workgroup_size(C4, POINTS)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let c = lid.x;
    let point = wg.x * POINTS + lid.y;
    if point >= p.out_time * p.out_freq {
        return;
    }
    // conv.0's output grid.
    let t1 = i32((p.frames + 1u) / 2u);
    let f1 = i32((p.mels + 1u) / 2u);
    let t = i32(point / p.out_freq);
    let f = i32(point % p.out_freq);
    var y = bias2[c];
    for (var dt = 0; dt < 3; dt++) {
        let ti = 2 * t + dt - 1;
        if ti < 0 || ti >= t1 {
            continue;
        }
        for (var df = 0; df < 3; df++) {
            let fi = 2 * f + df - 1;
            if fi < 0 || fi >= f1 {
                continue;
            }
            // conv.0 + ReLU at (ti, fi).
            var z = bias0[c];
            for (var et = 0; et < 3; et++) {
                let mt = 2 * ti + et - 1;
                if mt < 0 || mt >= i32(p.frames) {
                    continue;
                }
                for (var ef = 0; ef < 3; ef++) {
                    let mf = 2 * fi + ef - 1;
                    if mf < 0 || mf >= i32(p.mels) {
                        continue;
                    }
                    let v = mel[p.offset + u32(mf) * p.frames + u32(mt)];
                    z = fma(weight0[u32(et * 3 + ef) * C4 + c], vec4<f32>(v), z);
                }
            }
            z = max(z, vec4<f32>(0.0));
            y = fma(weight2[u32(dt * 3 + df) * C4 + c], z, y);
        }
    }
    out[point * C4 + c] = y;
}
