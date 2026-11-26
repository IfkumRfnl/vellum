enable f16;
enable wgpu_cooperative_matrix;

// The encoder's self-attention on tensor cores, flash-style: for every window and head of a
// batch, context = softmax(q·kᵀ / 8) · v over all T positions, without materializing the
// scores. q, k and v are the fused q/k/v product's rows (f32, rounded to f16 as they're
// staged); scores, softmax statistics and the output accumulate in f32.
//
// A workgroup takes 64 queries of one head (4 subgroups of 32, 16 queries each) and walks the
// keys in blocks of 64: S = Q·Kᵀ (16×64 per subgroup, four 16×16 accumulators), the running
// row maximum and sum updated from S (two lanes per row), P = exp(S − max) staged as f16, the
// output rescaled by exp(old max − new max) and P·V added to it. Each subgroup's S and O pass
// through its own patch of workgroup memory for the row-wise steps.
//
// The accumulators are named variables and the products unrolled (the host writes them at the
// `//@` lines): an array of cooperative matrices spills on NVIDIA's compiler.
//
// Prepended by the host: const HEADS, D_MODEL, T (positions per window). D_HEAD is 64.

@group(0) @binding(0) var<storage, read> qkv: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> context: array<vec4<f32>>;

const ROW4: u32 = 3u * D_MODEL / 4u;
const M4: u32 = D_MODEL / 4u;
// Staged rows of 64 halves, padded by 8 against bank conflicts.
const LD: u32 = 72u;
// A subgroup's 16 rows of 64 f32 scores (then outputs), padded to 68.
const LS: u32 = 68u;
const PATCH: u32 = 16u * LS;

var<workgroup> ks: array<f16, 64u * LD>;
var<workgroup> vs: array<f16, 64u * LD>;
// The queries, then each subgroup's probabilities in its own 16 rows.
var<workgroup> qp: array<f16, 64u * LD>;
var<workgroup> so: array<f32, 4u * PATCH>;
// Whether any row's maximum rose in this block (else the output needs no rescaling).
var<workgroup> rose: atomic<u32>;

fn stage(dst: u32, v: vec4<f32>, which: u32) {
    let h = vec4<f16>(v);
    if which == 0u {
        qp[dst] = h.x;
        qp[dst + 1u] = h.y;
        qp[dst + 2u] = h.z;
        qp[dst + 3u] = h.w;
    } else if which == 1u {
        ks[dst] = h.x;
        ks[dst + 1u] = h.y;
        ks[dst + 2u] = h.z;
        ks[dst + 3u] = h.w;
    } else {
        vs[dst] = h.x;
        vs[dst + 1u] = h.y;
        vs[dst + 2u] = h.z;
        vs[dst + 3u] = h.w;
    }
}

@compute @workgroup_size(128)
fn main(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(subgroup_id) sg: u32,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    let q0 = wg.x * 64u;
    let h = wg.y;
    let base = wg.z * T;
    let col = h * 16u;

    // The queries, scaled by 1/√64 (exact in f16 as in f32).
    for (var s = 0u; s < 8u; s++) {
        let e = tid + s * 128u;
        let r = e / 16u;
        let c4 = e % 16u;
        var v = vec4<f32>(0.0);
        if q0 + r < T {
            v = qkv[(base + q0 + r) * ROW4 + col + c4] * 0.125;
        }
        stage(r * LD + c4 * 4u, v, 0u);
    }
    workgroupBarrier();
    let r0 = sg * 16u;
    //@LOAD_Q
    //@ZERO_O

    // This lane's row of the subgroup's 16 and its half of the columns (interleaved).
    let row = lane / 2u;
    let half = lane % 2u;
    let mine = sg * PATCH + row * LS + half;
    var m = -3.0e38;
    var l = 0.0;

    // Each block's keys and values are loaded into registers while the block before is
    // multiplied, then staged.
    var kr: array<vec4<f32>, 8>;
    var vr: array<vec4<f32>, 8>;
    for (var s = 0u; s < 8u; s++) {
        let r = (tid + s * 128u) / 16u;
        let c4 = (tid + s * 128u) % 16u;
        let at = (base + r) * ROW4 + col + c4;
        kr[s] = qkv[at + M4];
        vr[s] = qkv[at + 2u * M4];
    }
    for (var j0 = 0u; j0 < T; j0 += 64u) {
        workgroupBarrier();
        if tid == 0u {
            atomicStore(&rose, 0u);
        }
        for (var s = 0u; s < 8u; s++) {
            let e = tid + s * 128u;
            let r = e / 16u;
            let c4 = e % 16u;
            var k = kr[s];
            var v = vr[s];
            if j0 + r >= T {
                k = vec4<f32>(0.0);
                v = vec4<f32>(0.0);
            }
            stage(r * LD + c4 * 4u, k, 1u);
            stage(r * LD + c4 * 4u, v, 2u);
        }
        workgroupBarrier();
        if j0 + 64u < T {
            for (var s = 0u; s < 8u; s++) {
                let r = (tid + s * 128u) / 16u;
                let c4 = (tid + s * 128u) % 16u;
                let row = min(j0 + 64u + r, T - 1u);
                let at = (base + row) * ROW4 + col + c4;
                kr[s] = qkv[at + M4];
                vr[s] = qkv[at + 2u * M4];
            }
        }

        //@SCORES
        workgroupBarrier();

        // Softmax statistics of this lane's row, then its probabilities.
        var mx = -3.0e38;
        for (var c = 0u; c < 32u; c++) {
            if j0 + 2u * c + half < T {
                mx = max(mx, so[mine + 2u * c]);
            }
        }
        mx = max(mx, subgroupShuffleXor(mx, 1u));
        let m_new = max(m, mx);
        let alpha = exp(m - m_new);
        if m_new != m {
            atomicStore(&rose, 1u);
        }
        var sum = 0.0;
        for (var c = 0u; c < 32u; c++) {
            var pv = 0.0;
            if j0 + 2u * c + half < T {
                pv = exp(so[mine + 2u * c] - m_new);
            }
            sum += pv;
            qp[(r0 + row) * LD + 2u * c + half] = f16(pv);
        }
        sum += subgroupShuffleXor(sum, 1u);
        l = l * alpha + sum;
        m = m_new;
        workgroupBarrier();

        // The output so far, rescaled to the new maximum (unless no row's rose: then every
        // factor is 1).
        if workgroupUniformLoad(&rose) != 0u {
            //@STORE_O
            workgroupBarrier();
            for (var c = 0u; c < 32u; c++) {
                so[mine + 2u * c] *= alpha;
            }
            workgroupBarrier();
            //@LOAD_O
        }

        //@VALUES
    }

    workgroupBarrier();
    //@STORE_O
    workgroupBarrier();
    let inv = 1.0 / l;
    let q = q0 + r0 + row;
    for (var c = 0u; c < 32u; c++) {
        so[mine + 2u * c] *= inv;
    }
    workgroupBarrier();
    // Lane pairs write their row back as vec4s (lane half h: vec4s h, h + 2, …).
    if q < T {
        let src = sg * PATCH + row * LS;
        for (var c = half; c < 16u; c += 2u) {
            let v = vec4<f32>(so[src + 4u * c], so[src + 4u * c + 1u], so[src + 4u * c + 2u], so[src + 4u * c + 3u]);
            context[(base + q) * M4 + col + c] = v;
        }
    }
}
