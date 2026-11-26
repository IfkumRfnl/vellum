// Attention scores with relative positions, for every window and head of a batch:
//
//     S[b, h, i, j] = ((q_i + u)·k_j + (q_i + v)·p[i − j]) · scale
//
// where q, k come from the fused q/k/v product of window b's rows, u and v are the layer's
// learned biases, and p[r] is the layer's projected positional encoding of offset r, from a
// table made at load: row R − 1 − r holds offset r, for every offset a window of up to R
// frames has. No shifted matrix is ever made: the offset is just index arithmetic.
//
// A workgroup computes a 64×64 tile of one window's scores for one head, over D_HEAD in
// slices of 16 staged in workgroup memory: the 64 queries with each bias, the 64 keys, and
// the 127 offsets the tile spans (query i0 + a against key j0 + b needs offset row
// R − 1 − i0 − 63 + j0 + (63 − a + b)). Each of the 64 threads owns an 8×8 block of
// consecutive queries and keys, so the offsets it needs are 15 consecutive staged rows per
// step (8 + 8 − 1), not one per output.
//
// Windows are packed one after another in the rows of qkv (`windows[b]` = first row,
// frames). S is [windows][heads][tm][tm], tm the longest window in the batch.
//
// Prepended by the host: const HEADS, D_HEAD (a multiple of 16), D_MODEL.

struct Params {
    tm: u32,
    // R: the positional table covers offsets R − 1 down to −(R − 1).
    max_frames: u32,
    scale: f32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read> qkv: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bias_u: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> bias_v: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> windows: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read_write> scores: array<f32>;

const ROW4: u32 = 3u * D_MODEL / 4u;
const M4: u32 = D_MODEL / 4u;
// A staged row: 16 values as 4 vec4s, plus one of padding, and one more every 8 rows: a
// thread reads rows 8 apart from its neighbours', which would otherwise share banks.
fn at(row: u32, c4: u32) -> u32 {
    return row * 5u + (row >> 3u) + c4;
}

var<workgroup> qu: array<vec4<f32>, 64u * 5u + 8u>;
var<workgroup> qv: array<vec4<f32>, 64u * 5u + 8u>;
var<workgroup> ks: array<vec4<f32>, 64u * 5u + 8u>;
var<workgroup> ps: array<vec4<f32>, 128u * 5u + 16u>;
var<workgroup> window: vec2<u32>;

fn dot4(a: vec4<f32>, b: vec4<f32>, c: f32) -> f32 {
    return fma(a.w, b.w, fma(a.z, b.z, fma(a.y, b.y, fma(a.x, b.x, c))));
}

@compute @workgroup_size(64)
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
    let j0 = wg.x * 64u;
    if i0 >= frames || j0 >= frames {
        return;
    }
    let tx = tid % 8u;
    let ty = tid / 8u;
    // The offset row of staged row 0: R − 1 − i0 − 63 + j0 (may be negative for tiles past
    // the window's end; those scores are never stored).
    let p0 = i32(p.max_frames) - 1 - i32(i0) - 63 + i32(j0);
    let p_rows = i32(2u * p.max_frames - 1u);
    // Query ty·8 + a against key tx·8 + c reads staged offset row 63 − 8·ty − a + 8·tx + c:
    // rows d0 + (c − a + 7) for d0 = 56 + 8·(tx − ty).
    let d0 = 56u + 8u * tx - 8u * ty;

    var acc: array<f32, 64>;
    for (var e = 0u; e < 64u; e++) {
        acc[e] = 0.0;
    }

    for (var k0 = 0u; k0 < D_HEAD / 4u; k0 += 4u) {
        workgroupBarrier();
        // 64 rows × 4 vec4s of queries and keys, 128 × 4 of offsets: 4 and 8 per thread.
        for (var s = 0u; s < 4u; s++) {
            let e = tid + s * 64u;
            let r = e / 4u;
            let c4 = e % 4u;
            let col = h * (D_HEAD / 4u) + k0 + c4;
            var q = vec4<f32>(0.0);
            var u = vec4<f32>(0.0);
            var v = vec4<f32>(0.0);
            if i0 + r < frames {
                q = qkv[(start + i0 + r) * ROW4 + col];
                u = q + bias_u[col];
                v = q + bias_v[col];
            }
            qu[at(r, c4)] = u;
            qv[at(r, c4)] = v;
            var k = vec4<f32>(0.0);
            if j0 + r < frames {
                k = qkv[(start + j0 + r) * ROW4 + M4 + col];
            }
            ks[at(r, c4)] = k;
        }
        for (var s = 0u; s < 8u; s++) {
            let e = tid + s * 64u;
            let l = e / 4u;
            let c4 = e % 4u;
            let row = p0 + i32(l);
            var v = vec4<f32>(0.0);
            if l < 127u && row >= 0 && row < p_rows {
                v = pos[u32(row) * M4 + h * (D_HEAD / 4u) + k0 + c4];
            }
            ps[at(l, c4)] = v;
        }
        workgroupBarrier();

        for (var kk = 0u; kk < 4u; kk++) {
            // Content: (q + u)·k, a query at a time against the thread's 8 keys.
            var y: array<vec4<f32>, 8>;
            for (var c = 0u; c < 8u; c++) {
                y[c] = ks[at(tx * 8u + c, kk)];
            }
            for (var a = 0u; a < 8u; a++) {
                let x = qu[at(ty * 8u + a, kk)];
                for (var c = 0u; c < 8u; c++) {
                    acc[a * 8u + c] = dot4(x, y[c], acc[a * 8u + c]);
                }
            }
            // Position: (q + v)·p[i − j], over the thread's 15 offsets.
            var o: array<vec4<f32>, 15>;
            for (var d = 0u; d < 15u; d++) {
                o[d] = ps[at(d0 + d, kk)];
            }
            for (var a = 0u; a < 8u; a++) {
                let x = qv[at(ty * 8u + a, kk)];
                for (var c = 0u; c < 8u; c++) {
                    acc[a * 8u + c] = dot4(x, o[c + 7u - a], acc[a * 8u + c]);
                }
            }
        }
    }

    let base = (b * HEADS + h) * p.tm;
    for (var a = 0u; a < 8u; a++) {
        let i = i0 + ty * 8u + a;
        if i >= frames {
            continue;
        }
        for (var c = 0u; c < 8u; c++) {
            let j = j0 + tx * 8u + c;
            if j < frames {
                scores[(base + i) * p.tm + j] = acc[a * 8u + c] * p.scale;
            }
        }
    }
}
