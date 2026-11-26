// Pre-encode conv.2 and conv.5: depthwise 3×3 convolutions with stride 2 and zero padding,
// channel-last in and out (each channel its own kernel). No activation: the ReLU comes after
// the pointwise product that follows.
//
// One thread per (output point, 4 channels), taps summed in the CPU oracle's order.
//
// Prepended by the host: const C4 (channels / 4).

struct Params {
    in_time: u32,
    in_freq: u32,
    out_time: u32,
    out_freq: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> weight: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out: array<vec4<f32>>;

const POINTS: u32 = 256u / C4;

@compute @workgroup_size(C4, POINTS)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let c = lid.x;
    let point = wg.x * POINTS + lid.y;
    if point >= p.out_time * p.out_freq {
        return;
    }
    let t = i32(point / p.out_freq);
    let f = i32(point % p.out_freq);
    var y = bias[c];
    for (var dt = 0; dt < 3; dt++) {
        let ti = 2 * t + dt - 1;
        if ti < 0 || ti >= i32(p.in_time) {
            continue;
        }
        for (var df = 0; df < 3; df++) {
            let fi = 2 * f + df - 1;
            if fi < 0 || fi >= i32(p.in_freq) {
                continue;
            }
            let v = x[(u32(ti) * p.in_freq + u32(fi)) * C4 + c];
            y = fma(weight[u32(dt * 3 + df) * C4 + c], v, y);
        }
    }
    out[point * C4 + c] = y;
}
