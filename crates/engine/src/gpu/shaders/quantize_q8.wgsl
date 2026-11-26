// X[M,K] (f32) to Q8_0 as `gemm_q8` takes it: per 32 values along a row, a scale
// d = max|x| / 127 and int8 values round(x / d) (half away from zero, as ggml rounds), four
// to a word. Eight lanes per block, four values each; the block's maximum by shuffles.
//
// Needs K % 32 == 0 and a subgroup size of at least 8.

struct Params {
    // Blocks of 32 values in all (M·K/32).
    blocks: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> q: array<u32>;
@group(0) @binding(2) var<storage, read_write> scales: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    // Four values per lane: global lane index = vec4 index.
    let i = id.x + id.y * groups.x * 256u;
    let block = i / 8u;
    // Every lane of a block takes part in the shuffles, so none returns early.
    var v = vec4<f32>(0.0);
    if block < p.blocks {
        v = x[i];
    }
    var amax = max(max(abs(v.x), abs(v.y)), max(abs(v.z), abs(v.w)));
    amax = max(amax, subgroupShuffleXor(amax, 1u));
    amax = max(amax, subgroupShuffleXor(amax, 2u));
    amax = max(amax, subgroupShuffleXor(amax, 4u));
    if block >= p.blocks {
        return;
    }
    let d = amax / 127.0;
    let inv = select(0.0, 1.0 / d, d > 0.0);
    let s = v * inv;
    let r = vec4<i32>(sign(s) * floor(abs(s) + 0.5));
    q[i] = pack4xI8(r);
    if i % 8u == 0u {
        scales[block] = d;
    }
}
