// ReLU in place over the first `len` values (the speaker head's input).

struct Params {
    len: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < p.len {
        x[id.x] = max(x[id.x], 0.0);
    }
}
