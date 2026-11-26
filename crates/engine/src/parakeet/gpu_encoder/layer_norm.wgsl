// Layer normalization of each row (d_model features), with learned scale and shift. The
// statistics are f32, two passes over values held in registers (mean, then the mean square
// deviation from it), so nothing overflows or cancels at the residual stream's magnitudes
// (up to ~1,400).
//
// One workgroup of 256 threads per row; each thread holds up to ⌈D4 / 256⌉ vec4s (thread t
// the ones at t, t + 256, …).
//
// Prepended by the host: const D4 (the width / 4).

struct Params {
    rows: u32,
    eps: f32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> weight: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out: array<vec4<f32>>;

const PER: u32 = (D4 + 255u) / 256u;

var<workgroup> partial: array<f32, 256>;

fn total(tid: u32, v: f32) -> f32 {
    partial[tid] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if tid < s {
            partial[tid] += partial[tid + s];
        }
        workgroupBarrier();
    }
    let t = partial[0];
    workgroupBarrier();
    return t;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let row = wg.x;
    if row >= p.rows {
        return;
    }
    let base = row * D4;
    var v: array<vec4<f32>, PER>;
    var sum = 0.0;
    for (var i = 0u; i < PER; i++) {
        v[i] = vec4<f32>(0.0);
        if tid + i * 256u < D4 {
            v[i] = x[base + tid + i * 256u];
            sum += (v[i].x + v[i].y) + (v[i].z + v[i].w);
        }
    }
    let n = f32(D4 * 4u);
    let mean = total(tid, sum) / n;
    var sq = 0.0;
    for (var i = 0u; i < PER; i++) {
        if tid + i * 256u < D4 {
            let d = v[i] - mean;
            sq += dot(d, d);
        }
    }
    let inv = 1.0 / sqrt(total(tid, sq) / n + p.eps);
    for (var i = 0u; i < PER; i++) {
        let at = tid + i * 256u;
        if at < D4 {
            out[base + at] = (v[i] - mean) * inv * weight[at] + bias[at];
        }
    }
}
