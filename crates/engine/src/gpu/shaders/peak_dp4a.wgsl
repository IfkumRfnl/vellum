// Peak rate of the packed int8 dot product (`dot4I8Packed`, dp4a): 16 independent i32
// accumulators per thread, ITERS rounds; the second operand changes each round so nothing
// is hoisted. Measurement only (`cargo lab kernels`).

@group(0) @binding(0) var<storage, read_write> out: array<i32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    var x: array<u32, 16>;
    var acc: array<i32, 16>;
    for (var j = 0u; j < 16u; j++) {
        x[j] = (id.x + j) * 2654435761u;
        acc[j] = 0;
    }
    var y = id.x * 40503u;
    for (var i = 0u; i < ITERS; i++) {
        for (var j = 0u; j < 16u; j++) {
            acc[j] += dot4I8Packed(x[j], y);
        }
        y += 0x01010101u;
    }
    var s = 0;
    for (var j = 0u; j < 16u; j++) {
        s += acc[j];
    }
    out[id.x] = s;
}
