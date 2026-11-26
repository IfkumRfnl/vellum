enable f16;

// A decoder step's cross-attention, split over the encoder positions: for every decoding window
// (workgroup z), head (x) and SPLITS-th of the positions (y), the query, rounded to f16 as the
// CPU oracle rounds it against f16 keys, over that part of the window's encoder keys and
// values (made once per window, f16). Each part leaves its maximum score, its sum of
// exp(score − maximum) and its exp-weighted values (f32) for `cross_combine.wgsl`, which
// joins the parts. Splitting puts enough workgroups on the GPU to stream the keys and values
// at its bandwidth when only a window or two decode.
//
// The cache: [layer][key, value][window][FRAMES][D_MODEL] f16, bound twice (as vec4<u32>, eight
// features, for the keys; as vec2<u32>, four, for the values). The parts: [window][head]
// [split] of 2 + 64 floats.
//
// Prepended by the host: const HEADS, D_MODEL, FRAMES, SPLITS, SLOT_WORDS, MODE.

struct Params {
    layer: u32,
    slots: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> keys: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> values: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> state: array<u32>;
@group(0) @binding(4) var<storage, read_write> parts: array<f32>;

const CHUNK: u32 = (FRAMES + SPLITS - 1u) / SPLITS;

var<workgroup> qs: array<vec4<f32>, 16>;
var<workgroup> scores: array<f32, CHUNK>;
var<workgroup> red: array<f32, 256>;
var<workgroup> part: array<vec4<f32>, 256>;
var<workgroup> mode: u32;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let h = wg.x;
    let split = wg.y;
    let s = wg.z;
    if tid == 0u {
        mode = state[1u + s * SLOT_WORDS + MODE];
    }
    if workgroupUniformLoad(&mode) == 0u {
        return;
    }
    if tid < 16u {
        let at = s * D_MODEL + h * 64u + 4u * tid;
        let v = vec4<f32>(q[at], q[at + 1u], q[at + 2u], q[at + 3u]);
        qs[tid] = vec4<f32>(vec4<f16>(v));
    }
    workgroupBarrier();
    let t0 = split * CHUNK;
    let n = min(CHUNK, FRAMES - t0);

    // Keys: a thread per position at a time, eight vec4<u32> (64 f16) each.
    let kbase = ((p.layer * 2u) * p.slots + s) * FRAMES * (D_MODEL / 8u) + h * 8u;
    var m = -3.0e38;
    for (var t = tid; t < n; t += 256u) {
        var acc = 0.0;
        for (var c = 0u; c < 8u; c++) {
            let k = keys[kbase + (t0 + t) * (D_MODEL / 8u) + c];
            let lo = vec4<f32>(unpack2x16float(k.x), unpack2x16float(k.y));
            let hi = vec4<f32>(unpack2x16float(k.z), unpack2x16float(k.w));
            acc += dot(qs[2u * c], lo) + dot(qs[2u * c + 1u], hi);
        }
        let v = acc * 0.125;
        scores[t] = v;
        m = max(m, v);
    }
    red[tid] = m;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st >>= 1u) {
        if tid < st {
            red[tid] = max(red[tid], red[tid + st]);
        }
        workgroupBarrier();
    }
    let mx = red[0];
    workgroupBarrier();
    var sum = 0.0;
    for (var t = tid; t < n; t += 256u) {
        let e = exp(scores[t] - mx);
        scores[t] = e;
        sum += e;
    }
    red[tid] = sum;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st >>= 1u) {
        if tid < st {
            red[tid] += red[tid + st];
        }
        workgroupBarrier();
    }

    // Values: sixteen groups of positions, each thread four features of the head.
    let g = tid / 16u;
    let c = tid % 16u;
    let vbase = ((p.layer * 2u + 1u) * p.slots + s) * FRAMES * (D_MODEL / 4u) + h * 16u + c;
    var acc = vec4<f32>(0.0);
    for (var t = g; t < n; t += 16u) {
        let v = values[vbase + (t0 + t) * (D_MODEL / 4u)];
        acc += scores[t] * vec4<f32>(unpack2x16float(v.x), unpack2x16float(v.y));
    }
    part[tid] = acc;
    workgroupBarrier();
    let out = ((s * HEADS + h) * SPLITS + split) * 66u;
    if tid < 16u {
        var total = vec4<f32>(0.0);
        for (var k = 0u; k < 16u; k++) {
            total += part[k * 16u + tid];
        }
        parts[out + 2u + 4u * tid] = total.x;
        parts[out + 3u + 4u * tid] = total.y;
        parts[out + 4u + 4u * tid] = total.z;
        parts[out + 5u + 4u * tid] = total.w;
    }
    if tid == 0u {
        parts[out] = mx;
        parts[out + 1u] = red[0];
    }
}
