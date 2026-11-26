// The convolution module's middle: GLU of the first pointwise product, the depthwise
// convolution over time (centred, zero-padded at the window's edges) with batch norm folded
// into its weights and bias at load, then SiLU. In: the pointwise product, 2·C per row (the
// GLU's value half, then its gate half); out: C per row.
//
// One workgroup per row, a thread per 4 channels; each tap's GLU is recomputed (K sigmoids
// per output, cheaper than a pass over memory). Taps are summed in the CPU oracle's order.
//
// Prepended by the host: const C4 (channels / 4, a multiple of 64), K (kernel, odd).

struct Params {
    rows: u32,
    windows: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
// [tap][C4]
@group(0) @binding(1) var<storage, read> weight: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> windows: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> out: array<vec4<f32>>;

const PER: u32 = C4 / 64u;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let row = wg.x;
    if row >= p.rows {
        return;
    }
    var start = 0u;
    var frames = 0u;
    for (var b = 0u; b < p.windows; b++) {
        let w = windows[b];
        if row >= w.x && row < w.x + w.y {
            start = w.x;
            frames = w.y;
        }
    }
    let t = i32(row - start);
    for (var i = 0u; i < PER; i++) {
        let c = tid + i * 64u;
        var y = bias[c];
        for (var k = 0; k < i32(K); k++) {
            let ti = t + k - i32(K / 2u);
            if ti < 0 || ti >= i32(frames) {
                continue;
            }
            let at = (start + u32(ti)) * 2u * C4 + c;
            let a = x[at];
            let g = x[at + C4];
            let glu = a * (1.0 / (1.0 + exp(-g)));
            y = fma(weight[u32(k) * C4 + c], glu, y);
        }
        out[row * C4 + c] = y * (1.0 / (1.0 + exp(-y)));
    }
}
