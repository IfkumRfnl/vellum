// The parts of a decoder step's cross-attention (`cross_attend.wgsl`) joined, for every
// decoding window (workgroup y) and head (x): each part's sum and weighted values scaled from
// its own maximum score to the overall one, added, and the values divided by the sum.
//
// Prepended by the host: const HEADS, D_MODEL, SPLITS, SLOT_WORDS, MODE.

@group(0) @binding(0) var<storage, read> parts: array<f32>;
@group(0) @binding(1) var<storage, read> state: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) c: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let h = wg.x;
    let s = wg.y;
    if state[1u + s * SLOT_WORDS + MODE] == 0u {
        return;
    }
    let base = (s * HEADS + h) * SPLITS * 66u;
    var m = -3.0e38;
    for (var k = 0u; k < SPLITS; k++) {
        m = max(m, parts[base + k * 66u]);
    }
    var sum = 0.0;
    var acc = 0.0;
    for (var k = 0u; k < SPLITS; k++) {
        let at = base + k * 66u;
        let scale = exp(parts[at] - m);
        sum += parts[at + 1u] * scale;
        acc += parts[at + 2u + c] * scale;
    }
    out[s * D_MODEL + h * 64u + c] = acc / sum;
}
