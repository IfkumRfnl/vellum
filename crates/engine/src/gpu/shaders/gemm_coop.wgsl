enable f16;
enable wgpu_cooperative_matrix;

// Y[M,N] = epilogue(X[M,K] · W[N,K]ᵀ + b) on tensor cores: 16×16×16 f16 products with f32
// accumulators (`coop_mat16x16`). X is f32 and rounded to f16 as it's staged; W is f16.
//
// A workgroup computes a BM×BN tile of Y with (BM/WM)·(BN/WN) subgroups of 32, each owning a
// WM×WN block: (WM/16)·(WN/16) accumulator matrices. BK-wide slices of X and W are staged in
// workgroup memory (rows padded by 8 halves, 16 bytes, against bank conflicts), the next slice
// loaded into registers while the current one is multiplied. The staged W is [n][k], which is
// Wᵀ in column-major order, so it loads as the B operand directly.
//
// The epilogue (bias, activation, scale, accumulate) goes through workgroup memory: each
// subgroup stores one accumulator at a time to its own 16×16 patch and its lanes finish it.
//
// The accumulators are named variables, not an array, and the multiply and the epilogue are
// unrolled: the host writes those parts for the variant's shape where the `//@` lines are (an
// array of cooperative matrices indexed in a loop spills to memory on NVIDIA's compiler, and
// one passed to a function comes back zero).
//
// Prepended by the host: const BM, BN, BK, WM, WN (u32). Needs a subgroup size of 32,
// K % BK == 0, BK % 16 == 0; M and N may be anything.

struct Params {
    m: u32,
    n: u32,
    k: u32,
    // Bit 0: add the bias. Bits 1–2: activation (0 none, 1 ReLU, 2 SiLU, 3 GELU). Bit 3: accumulate.
    flags: u32,
    scale: f32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

const SG_M: u32 = BM / WM;
const SG_N: u32 = BN / WN;
const SUBGROUPS: u32 = SG_M * SG_N;
const THREADS: u32 = SUBGROUPS * 32u;
const FM: u32 = WM / 16u;
const FN: u32 = WN / 16u;
const LD: u32 = BK + 8u;
const A_VECS: u32 = BM * BK / 4u;
const B_VECS: u32 = BN * BK / 8u;
const A_LOADS: u32 = (A_VECS + THREADS - 1u) / THREADS;
const B_LOADS: u32 = (B_VECS + THREADS - 1u) / THREADS;

var<workgroup> xs: array<f16, BM * LD>;
var<workgroup> ws: array<f16, BN * LD>;
var<workgroup> cs: array<f32, SUBGROUPS * 256u>;

fn activate(v: f32, kind: u32) -> f32 {
    if kind == 1u {
        return max(v, 0.0);
    }
    if kind == 2u {
        return v / (1.0 + exp(-v));
    }
    if kind == 3u {
        return 0.5 * v * (1.0 + erf(v * 0.70710678));
    }
    return v;
}

// The error function as the CPU's `cpu::erf` computes it (Cody's rational approximation, to
// about 1e-7), in f32.
fn erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.5 * abs(x));
    let y = 1.0 - t * exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196
        + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398
        + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return select(-y, y, x >= 0.0);
}

@compute @workgroup_size(THREADS)
fn main(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(subgroup_id) sg: u32,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    let m0 = group.y * BM;
    let n0 = group.x * BN;
    let wm0 = (sg / SG_N) * WM;
    let wn0 = (sg % SG_N) * WN;
    let k4 = p.k / 4u;
    let k8 = p.k / 8u;

    //@ACCUMULATORS
    var xr: array<vec4<f32>, A_LOADS>;
    var wr: array<vec4<u32>, B_LOADS>;
    for (var s = 0u; s < A_LOADS; s++) {
        let v = tid + s * THREADS;
        let row = m0 + v / (BK / 4u);
        xr[s] = vec4<f32>(0.0);
        if v < A_VECS && row < p.m {
            xr[s] = x[row * k4 + v % (BK / 4u)];
        }
    }
    for (var s = 0u; s < B_LOADS; s++) {
        let v = tid + s * THREADS;
        let row = n0 + v / (BK / 8u);
        wr[s] = vec4<u32>(0u);
        if v < B_VECS && row < p.n {
            wr[s] = w[row * k8 + v % (BK / 8u)];
        }
    }

    let slices = p.k / BK;
    for (var t = 0u; t < slices; t++) {
        workgroupBarrier();
        for (var s = 0u; s < A_LOADS; s++) {
            let v = tid + s * THREADS;
            if v < A_VECS {
                let at = (v / (BK / 4u)) * LD + (v % (BK / 4u)) * 4u;
                let h = vec4<f16>(xr[s]);
                xs[at] = h.x;
                xs[at + 1u] = h.y;
                xs[at + 2u] = h.z;
                xs[at + 3u] = h.w;
            }
        }
        for (var s = 0u; s < B_LOADS; s++) {
            let v = tid + s * THREADS;
            if v < B_VECS {
                let at = (v / (BK / 8u)) * LD + (v % (BK / 8u)) * 8u;
                let q = wr[s];
                let h0 = vec2<f16>(unpack2x16float(q.x));
                let h1 = vec2<f16>(unpack2x16float(q.y));
                let h2 = vec2<f16>(unpack2x16float(q.z));
                let h3 = vec2<f16>(unpack2x16float(q.w));
                ws[at] = h0.x;
                ws[at + 1u] = h0.y;
                ws[at + 2u] = h1.x;
                ws[at + 3u] = h1.y;
                ws[at + 4u] = h2.x;
                ws[at + 5u] = h2.y;
                ws[at + 6u] = h3.x;
                ws[at + 7u] = h3.y;
            }
        }
        workgroupBarrier();

        if t + 1u < slices {
            let kx = (t + 1u) * (BK / 4u);
            let kw = (t + 1u) * (BK / 8u);
            for (var s = 0u; s < A_LOADS; s++) {
                let v = tid + s * THREADS;
                let row = m0 + v / (BK / 4u);
                if v < A_VECS && row < p.m {
                    xr[s] = x[row * k4 + kx + v % (BK / 4u)];
                }
            }
            for (var s = 0u; s < B_LOADS; s++) {
                let v = tid + s * THREADS;
                let row = n0 + v / (BK / 8u);
                if v < B_VECS && row < p.n {
                    wr[s] = w[row * k8 + kw + v % (BK / 8u)];
                }
            }
        }

        //@MULTIPLY
    }

    let has_bias = (p.flags & 1u) != 0u;
    let kind = (p.flags >> 1u) & 3u;
    let accumulate = (p.flags & 8u) != 0u;
    //@EPILOGUE
}
