// One window's encoder keys and values for every decoder layer, from the stacked product
// (rows of [layer 0 keys, layer 0 values, layer 1 keys, …], f32) into its slot of the cross
// cache ([layer][key, value][window][FRAMES][D_MODEL] f16, as u32 pairs), rounded to f16 as
// the CPU oracle's cache rounds them.
//
// Prepended by the host: const D_MODEL, LAYERS, FRAMES.

struct Params {
    slot: u32,
    slots: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read> stacked: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> cache: array<u32>;

const D2: u32 = D_MODEL / 2u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let at = id.x + id.y * groups.x * 256u;
    if at >= LAYERS * 2u * FRAMES * D2 {
        return;
    }
    let c2 = at % D2;
    let t = (at / D2) % FRAMES;
    let lk = at / (D2 * FRAMES);
    let v = stacked[t * (LAYERS * 2u * D2) + lk * D2 + c2];
    cache[((lk * p.slots + p.slot) * FRAMES + t) * D2 + c2] = pack2x16float(v);
}
