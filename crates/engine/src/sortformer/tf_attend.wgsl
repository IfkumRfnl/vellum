// Plain multi-head self-attention over one window (Sortformer's transformer: no positional
// terms, no mask): context_i = Σ_j softmax_j((q_i·s)·(k_j·s)) · v_j per head, NeMo scaling
// queries and keys each by s = d_head^(-1/4).
//
// A workgroup (one warp) takes one head and 32 queries, a thread per query holding it and its
// output in registers. The keys (then keys and values) are staged 32 at a time in workgroup
// memory, read from global memory in order so neighbouring threads load neighbouring values;
// every thread then reads the same staged value at once (a broadcast). Two passes over the
// keys, as the CPU oracle computes it: each query's maximum score, then Σ exp(s − max) and
// Σ exp(s − max)·v with fused multiply-adds, keys in order, divided by the sum at the end.
//
// q, k, v are one row of 3·D_MODEL per frame (the fused product); context is D_MODEL per frame.
//
// Prepended by the host: const HEADS, D_HEAD (at most 64), D_MODEL.

struct Params {
    frames: u32,
    scale: f32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> context: array<f32>;

const ROW: u32 = 3u * D_MODEL;
// Queries per workgroup (a thread each) and keys per staged slice.
const TILE: u32 = 32u;

var<workgroup> ks: array<f32, TILE * D_HEAD>;
var<workgroup> vs: array<f32, TILE * D_HEAD>;

// Keys (and, with `values`, values) j0 … j0 + TILE − 1 of head `head` into workgroup memory.
fn stage(tid: u32, j0: u32, head: u32, values: bool) {
    for (var e = tid; e < TILE * D_HEAD; e += TILE) {
        let r = e / D_HEAD;
        let c = e % D_HEAD;
        var k = 0.0;
        var v = 0.0;
        if j0 + r < p.frames {
            let at = (j0 + r) * ROW + head + c;
            k = qkv[at + D_MODEL] * p.scale;
            if values {
                v = qkv[at + 2u * D_MODEL];
            }
        }
        ks[e] = k;
        vs[e] = v;
    }
}

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let i = wg.x * TILE + tid;
    let head = wg.y * D_HEAD;
    let n = p.frames;
    let live = i < n;
    var q: array<f32, D_HEAD>;
    var acc: array<f32, D_HEAD>;
    for (var d = 0u; d < D_HEAD; d++) {
        q[d] = 0.0;
        acc[d] = 0.0;
        if live {
            q[d] = qkv[i * ROW + head + d] * p.scale;
        }
    }

    var top = -3.402823e38;
    for (var j0 = 0u; j0 < n; j0 += TILE) {
        workgroupBarrier();
        stage(tid, j0, head, false);
        workgroupBarrier();
        let m = min(TILE, n - j0);
        for (var jj = 0u; jj < m; jj++) {
            var s = 0.0;
            for (var d = 0u; d < D_HEAD; d++) {
                s = fma(q[d], ks[jj * D_HEAD + d], s);
            }
            top = max(top, s);
        }
    }

    var sum = 0.0;
    for (var j0 = 0u; j0 < n; j0 += TILE) {
        workgroupBarrier();
        stage(tid, j0, head, true);
        workgroupBarrier();
        let m = min(TILE, n - j0);
        for (var jj = 0u; jj < m; jj++) {
            var s = 0.0;
            for (var d = 0u; d < D_HEAD; d++) {
                s = fma(q[d], ks[jj * D_HEAD + d], s);
            }
            let e = exp(s - top);
            sum += e;
            for (var d = 0u; d < D_HEAD; d++) {
                acc[d] = fma(e, vs[jj * D_HEAD + d], acc[d]);
            }
        }
    }

    if live {
        let inv = 1.0 / sum;
        for (var d = 0u; d < D_HEAD; d++) {
            context[i * D_MODEL + head + d] = acc[d] * inv;
        }
    }
}
