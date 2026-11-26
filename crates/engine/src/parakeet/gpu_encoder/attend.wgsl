// Softmax and the attention-weighted values in one pass: for every window and head of a
// batch, context[i, head h] = Σ_j softmax(S[b, h, i, ·])_j · v_j, written into the context
// rows (packed like the input) at head h's columns, ready for the output projection.
//
// A workgroup takes 64 queries × the head's 128 features. First each of its rows' maximum and
// sum of exp(s − max), in f32 (a warp per row at a time, lanes striding over the keys, the
// lanes then reduced in workgroup memory); then the keys in slices of 16, the probabilities
// exp(s − max) / sum staged beside the values. Each of the 256 threads owns 4 queries
// (ty + 16·a) × 2 vec4s of features (tx and tx + 16), summing over keys in order with fused
// multiply-adds, as the CPU oracle does.
//
// Prepended by the host: const HEADS, D_HEAD (a multiple of 128), D_MODEL.

struct Params {
    tm: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read> qkv: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> scores: array<f32>;
@group(0) @binding(2) var<storage, read> windows: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> context: array<vec4<f32>>;

const ROW4: u32 = 3u * D_MODEL / 4u;
const M4: u32 = D_MODEL / 4u;

// 64 queries × 16 keys, padded by one against bank conflicts; 16 keys × 32 vec4s.
var<workgroup> as_: array<f32, 64u * 17u>;
var<workgroup> vs: array<vec4<f32>, 512>;
// Per-lane partial results for 8 rows at a time, then each row's maximum and 1 / sum.
var<workgroup> partial: array<f32, 2048>;
var<workgroup> row_max: array<f32, 64>;
var<workgroup> row_inv: array<f32, 64>;
var<workgroup> window: vec2<u32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let b = wg.z / HEADS;
    let h = wg.z % HEADS;
    if tid == 0u {
        window = windows[b];
    }
    let w = workgroupUniformLoad(&window);
    let start = w.x;
    let frames = w.y;
    let i0 = wg.y * 64u;
    if i0 >= frames {
        return;
    }
    let base = wg.z * p.tm;

    // Row statistics: warp `slot` takes rows slot, slot + 8, …; its lanes stride over keys.
    let slot = tid / 32u;
    let lane = tid % 32u;
    for (var k = 0u; k < 8u; k++) {
        let r = slot + 8u * k;
        var m = -3.402823e38;
        if i0 + r < frames {
            let row = (base + i0 + r) * p.tm;
            for (var j = lane; j < frames; j += 32u) {
                m = max(m, scores[row + j]);
            }
        }
        partial[k * 256u + tid] = m;
    }
    workgroupBarrier();
    for (var s = 16u; s > 0u; s >>= 1u) {
        if lane < s {
            for (var k = 0u; k < 8u; k++) {
                let at = k * 256u + tid;
                partial[at] = max(partial[at], partial[at + s]);
            }
        }
        workgroupBarrier();
    }
    if lane == 0u {
        for (var k = 0u; k < 8u; k++) {
            row_max[slot + 8u * k] = partial[k * 256u + tid];
        }
    }
    workgroupBarrier();
    for (var k = 0u; k < 8u; k++) {
        let r = slot + 8u * k;
        var sum = 0.0;
        if i0 + r < frames {
            let row = (base + i0 + r) * p.tm;
            let m = row_max[r];
            for (var j = lane; j < frames; j += 32u) {
                sum += exp(scores[row + j] - m);
            }
        }
        partial[k * 256u + tid] = sum;
    }
    workgroupBarrier();
    for (var s = 16u; s > 0u; s >>= 1u) {
        if lane < s {
            for (var k = 0u; k < 8u; k++) {
                let at = k * 256u + tid;
                partial[at] += partial[at + s];
            }
        }
        workgroupBarrier();
    }
    if lane == 0u {
        for (var k = 0u; k < 8u; k++) {
            row_inv[slot + 8u * k] = 1.0 / partial[k * 256u + tid];
        }
    }

    // This tile's 32 vec4s of the head's features.
    let f4 = h * (D_HEAD / 4u) + wg.x * 32u;
    let tx = tid % 16u;
    let ty = tid / 16u;
    var acc: array<vec4<f32>, 8>;
    for (var a = 0u; a < 8u; a++) {
        acc[a] = vec4<f32>(0.0);
    }
    for (var j0 = 0u; j0 < frames; j0 += 16u) {
        workgroupBarrier();
        for (var s = 0u; s < 4u; s++) {
            let e = tid + s * 256u;
            let r = e / 16u;
            let c = e % 16u;
            var v = 0.0;
            if i0 + r < frames && j0 + c < frames {
                v = exp(scores[(base + i0 + r) * p.tm + j0 + c] - row_max[r]) * row_inv[r];
            }
            as_[r * 17u + c] = v;
        }
        for (var s = 0u; s < 2u; s++) {
            let e = tid + s * 256u;
            let r = e / 32u;
            let c = e % 32u;
            var v = vec4<f32>(0.0);
            if j0 + r < frames {
                v = qkv[(start + j0 + r) * ROW4 + 2u * M4 + f4 + c];
            }
            vs[e] = v;
        }
        workgroupBarrier();
        let n = min(16u, frames - j0);
        for (var jj = 0u; jj < n; jj++) {
            let v0 = vs[jj * 32u + tx];
            let v1 = vs[jj * 32u + tx + 16u];
            for (var a = 0u; a < 4u; a++) {
                let s = vec4<f32>(as_[(ty + 16u * a) * 17u + jj]);
                acc[2u * a] = fma(s, v0, acc[2u * a]);
                acc[2u * a + 1u] = fma(s, v1, acc[2u * a + 1u]);
            }
        }
    }
    for (var a = 0u; a < 4u; a++) {
        let i = i0 + ty + 16u * a;
        if i < frames {
            context[(start + i) * M4 + f4 + tx] = acc[2u * a];
            context[(start + i) * M4 + f4 + tx + 16u] = acc[2u * a + 1u];
        }
    }
}
