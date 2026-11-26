// A decoder step's causal self-attention, for every decoding window (workgroup y) and head
// (x): the new key and value (from the fused q/k/v product) rounded to f16 into the cache at
// the window's position, then the query, rounded to f16 as the CPU oracle rounds it against
// f16 keys, over every cached position up to and including it. Scores and their softmax are
// f32, the output the probability-weighted values divided by their sum.
//
// The caches: [layer][key, value][window][CTX][D_MODEL] f16, as u32 pairs.
//
// Prepended by the host: const HEADS, D_MODEL, CTX, SLOT_WORDS, MODE, LEN.

struct Params {
    layer: u32,
    slots: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> cache: array<u32>;
@group(0) @binding(2) var<storage, read> state: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const D2: u32 = D_MODEL / 2u;

var<workgroup> q: array<f32, 64>;
var<workgroup> scores: array<f32, CTX>;
var<workgroup> red: array<f32, 64>;
var<workgroup> part: array<vec2<f32>, 64>;
var<workgroup> info: vec2<u32>;

fn rounded(v: f32) -> f32 {
    return unpack2x16float(pack2x16float(vec2<f32>(v, 0.0))).x;
}

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let h = wg.x;
    let s = wg.y;
    if tid == 0u {
        let at = 1u + s * SLOT_WORDS;
        info = vec2<u32>(state[at + MODE], state[at + LEN]);
    }
    let w = workgroupUniformLoad(&info);
    if w.x == 0u {
        return;
    }
    let len = w.y;
    let row = s * 3u * D_MODEL;
    let kbase = ((p.layer * 2u) * p.slots + s) * CTX * D2 + h * 32u;
    let vbase = ((p.layer * 2u + 1u) * p.slots + s) * CTX * D2 + h * 32u;
    // The new key and value: threads 0–31 a pair of the key's features each, 32–63 the value's.
    let i = tid % 32u;
    if tid < 32u {
        let at = row + D_MODEL + h * 64u + 2u * i;
        cache[kbase + len * D2 + i] = pack2x16float(vec2<f32>(qkv[at], qkv[at + 1u]));
    } else {
        let at = row + 2u * D_MODEL + h * 64u + 2u * i;
        cache[vbase + len * D2 + i] = pack2x16float(vec2<f32>(qkv[at], qkv[at + 1u]));
    }
    q[tid] = rounded(qkv[row + h * 64u + tid]);
    storageBarrier();
    workgroupBarrier();

    let n = len + 1u;
    var m = -3.0e38;
    for (var j = tid; j < n; j += 64u) {
        var acc = 0.0;
        for (var c = 0u; c < 32u; c++) {
            let k = unpack2x16float(cache[kbase + j * D2 + c]);
            acc = fma(q[2u * c], k.x, acc);
            acc = fma(q[2u * c + 1u], k.y, acc);
        }
        let v = acc * 0.125;
        scores[j] = v;
        m = max(m, v);
    }
    red[tid] = m;
    workgroupBarrier();
    for (var st = 32u; st > 0u; st >>= 1u) {
        if tid < st {
            red[tid] = max(red[tid], red[tid + st]);
        }
        workgroupBarrier();
    }
    let mx = red[0];
    workgroupBarrier();
    var sum = 0.0;
    for (var j = tid; j < n; j += 64u) {
        let e = exp(scores[j] - mx);
        scores[j] = e;
        sum += e;
    }
    red[tid] = sum;
    workgroupBarrier();
    for (var st = 32u; st > 0u; st >>= 1u) {
        if tid < st {
            red[tid] += red[tid + st];
        }
        workgroupBarrier();
    }
    let inv = 1.0 / red[0];

    // Weighted values: threads 0–31 the even positions, 32–63 the odd, a feature pair each.
    var acc = vec2<f32>(0.0);
    for (var j = tid / 32u; j < n; j += 2u) {
        acc += scores[j] * unpack2x16float(cache[vbase + j * D2 + i]);
    }
    part[tid] = acc;
    workgroupBarrier();
    if tid < 32u {
        let v = (part[tid] + part[tid + 32u]) * inv;
        let at = s * D_MODEL + h * 64u + 2u * tid;
        out[at] = v.x;
        out[at + 1u] = v.y;
    }
}
