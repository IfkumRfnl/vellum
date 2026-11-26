// A decoder step's linear layer: y[b] = epilogue(W · x[b] + bias) for the few rows of a step
// (one per decoding window, at most MAX_B), the weights kept as the file has them, Q8_0: an
// f16 scale per 32 values and the values as int8, so each is exactly `scale · q` (as the CPU
// oracle multiplies) at half the bytes of f16. A step streams every weight once; this is a
// memory-bound kernel.
//
// With NORM, x is layer-normed on the way in (the layer norm before the layer, fused): each
// workgroup first takes every row's mean and inverse deviation (subgroup b row b, two passes
// in f32, as the layer-norm kernel does), then normalizes each value as it loads it.
//
// Each subgroup of 32 takes ROWS output rows: its lanes stride over K sixteen values at a time
// (one vec4<u32> of int8 per row), each value against every row of x, the lanes' partial sums
// then added across the subgroup. Lane r·MAX_B + b writes row r's output for x row b.
//
// The step's state (word 0: whether any window is still decoding) lets a step recorded ahead
// of time cost nothing once every window is done.
//
// Prepended by the host: const MAX_B (8), ROWS (1, 2 or 4), NORM (0 or 1).

struct Params {
    n: u32,
    k: u32,
    batch: u32,
    // Bit 0: add the bias. Bits 1–2: activation (0 none, 3 GELU). Bit 3: accumulate.
    flags: u32,
    eps: f32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> qs: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> scales: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> state: array<u32>;
@group(0) @binding(5) var<storage, read> norm_weight: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> norm_bias: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> y: array<f32>;

const OUTS: u32 = ROWS * MAX_B;

var<workgroup> stats: array<vec2<f32>, MAX_B>;

fn erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.5 * abs(x));
    let y = 1.0 - t * exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196
        + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398
        + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return select(-y, y, x >= 0.0);
}

fn dot16(w: array<vec4<f32>, 4>, x0: vec4<f32>, x1: vec4<f32>, x2: vec4<f32>, x3: vec4<f32>) -> f32 {
    return (dot(w[0], x0) + dot(w[1], x1)) + (dot(w[2], x2) + dot(w[3], x3));
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(subgroup_id) sg: u32,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    if state[0] == 0u {
        return;
    }
    let k16 = p.k / 16u;
    let k4 = p.k / 4u;
    let k64 = p.k / 64u;

    if NORM == 1u {
        // Row b's statistics, by subgroup b.
        let b = sg;
        if b < p.batch {
            var sum = 0.0;
            for (var i = lane; i < k4; i += 32u) {
                let v = x[b * k4 + i];
                sum += (v.x + v.y) + (v.z + v.w);
            }
            let mean = subgroupAdd(sum) / f32(p.k);
            var sq = 0.0;
            for (var i = lane; i < k4; i += 32u) {
                let d = x[b * k4 + i] - mean;
                sq += dot(d, d);
            }
            let inv = 1.0 / sqrt(subgroupAdd(sq) / f32(p.k) + p.eps);
            if lane == 0u {
                stats[b] = vec2<f32>(mean, inv);
            }
        }
        workgroupBarrier();
    }

    let row0 = (wg.x * 8u + sg) * ROWS;
    var acc: array<f32, OUTS>;
    for (var i = 0u; i < OUTS; i++) {
        acc[i] = 0.0;
    }
    for (var c = lane; c < k16; c += 32u) {
        var w: array<array<vec4<f32>, 4>, ROWS>;
        for (var r = 0u; r < ROWS; r++) {
            let n = min(row0 + r, p.n - 1u);
            let q = qs[n * k16 + c];
            let pair = unpack2x16float(scales[n * k64 + c / 4u]);
            let s = select(pair.x, pair.y, ((c / 2u) & 1u) == 1u);
            w[r][0] = vec4<f32>(unpack4xI8(q.x)) * s;
            w[r][1] = vec4<f32>(unpack4xI8(q.y)) * s;
            w[r][2] = vec4<f32>(unpack4xI8(q.z)) * s;
            w[r][3] = vec4<f32>(unpack4xI8(q.w)) * s;
        }
        var g: array<vec4<f32>, 4>;
        var h: array<vec4<f32>, 4>;
        if NORM == 1u {
            for (var j = 0u; j < 4u; j++) {
                g[j] = norm_weight[c * 4u + j];
                h[j] = norm_bias[c * 4u + j];
            }
        }
        for (var b = 0u; b < MAX_B; b++) {
            if b < p.batch {
                let at = b * k4 + c * 4u;
                var xs: array<vec4<f32>, 4>;
                for (var j = 0u; j < 4u; j++) {
                    xs[j] = x[at + j];
                    if NORM == 1u {
                        let st = stats[b];
                        xs[j] = (xs[j] - st.x) * st.y * g[j] + h[j];
                    }
                }
                for (var r = 0u; r < ROWS; r++) {
                    acc[r * MAX_B + b] += dot16(w[r], xs[0], xs[1], xs[2], xs[3]);
                }
            }
        }
    }
    var mine = 0.0;
    for (var i = 0u; i < OUTS; i++) {
        let total = subgroupAdd(acc[i]);
        if i == lane {
            mine = total;
        }
    }
    let r = lane / MAX_B;
    let b = lane % MAX_B;
    let n = row0 + r;
    if lane < OUTS && n < p.n && b < p.batch {
        var v = mine;
        if (p.flags & 1u) != 0u {
            v += bias[n];
        }
        if ((p.flags >> 1u) & 3u) == 3u {
            v = 0.5 * v * (1.0 + erf(v * 0.70710678));
        }
        let at = b * p.n + n;
        if (p.flags & 8u) != 0u {
            v += y[at];
        }
        y[at] = v;
    }
}
