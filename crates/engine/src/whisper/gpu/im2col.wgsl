// The rows a width-3, padding-1 convolution of `stride` multiplies, for every window of a
// batch: output frame t of a window is input frames stride·t − 1, stride·t, stride·t + 1 side
// by side (zeros outside the window), tap-major, then zeros up to the padded width (the
// weights are reordered to match at load), ready for the matrix multiply.
//
// Prepended by the host: nothing. Shapes are immediates; C and the padded width are multiples
// of 4.

struct Params {
    // Input frames per window, channels / 4, output frames per window, stride, padded width / 4.
    frames_in: u32,
    c4: u32,
    frames_out: u32,
    stride: u32,
    width4: u32,
    rows: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let at = id.x + id.y * groups.x * 256u;
    if at >= p.rows * p.width4 {
        return;
    }
    let r = at / p.width4;
    let j = at % p.width4;
    let tap = j / p.c4;
    let c = j % p.c4;
    var v = vec4<f32>(0.0);
    if tap < 3u {
        let window = r / p.frames_out;
        let t = r % p.frames_out;
        let src = i32(p.stride * t + tap) - 1;
        if src >= 0 && u32(src) < p.frames_in {
            v = x[(window * p.frames_in + u32(src)) * p.c4 + c];
        }
    }
    out[at] = v;
}
