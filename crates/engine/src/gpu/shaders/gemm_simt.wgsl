// Y[M,N] = epilogue(X[M,K] · W[N,K]ᵀ + b): X f32, W f16, f32 accumulation and output. The
// portable kernel: no f16 arithmetic, no subgroups, no tensor cores.
//
// Each workgroup computes a BM×BN tile of Y from BK-wide slices of X and W staged in workgroup
// memory (W converted to f32 on the way in); each of its (BM/TM)·(BN/TN) threads holds a TM×TN
// block of accumulators in registers. A thread's rows are strided by BM/TM and its columns by
// BN/TN, so a warp's reads of the staged W rows hit distinct banks and its writes of Y are
// contiguous. The next slice is loaded into registers while the current one is multiplied.
//
// Prepended by the host: const BM, BN, BK, TM, TN (u32). Needs K % BK == 0 and BK % 8 == 0;
// M and N may be anything.

struct Params {
    m: u32,
    n: u32,
    k: u32,
    // Bit 0: add the bias. Bits 1–2: activation (0 none, 1 ReLU, 2 SiLU, 3 GELU). Bit 3: accumulate
    // (Y += result instead of Y = result).
    flags: u32,
    // The result is multiplied by this (after the activation) before it's stored or added.
    scale: f32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
// Eight f16 weights per element, in row order.
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

const WX: u32 = BN / TN;
const WY: u32 = BM / TM;
const THREADS: u32 = WX * WY;
// A staged row: BK values as vec4s, plus one vec4 of padding against bank conflicts.
const LD: u32 = BK / 4u + 1u;
const A_VECS: u32 = BM * BK / 4u;
const B_VECS: u32 = BN * BK / 8u;
const A_LOADS: u32 = (A_VECS + THREADS - 1u) / THREADS;
const B_LOADS: u32 = (B_VECS + THREADS - 1u) / THREADS;

var<workgroup> xs: array<vec4<f32>, BM * LD>;
var<workgroup> ws: array<vec4<f32>, BN * LD>;

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
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) group: vec3<u32>) {
    let m0 = group.y * BM;
    let n0 = group.x * BN;
    let tx = tid % WX;
    let ty = tid / WX;
    let k4 = p.k / 4u;
    let k8 = p.k / 8u;

    var acc: array<f32, TM * TN>;
    for (var i = 0u; i < TM * TN; i++) {
        acc[i] = 0.0;
    }
    var xr: array<vec4<f32>, A_LOADS>;
    var wr: array<vec4<u32>, B_LOADS>;

    // Loads slice `k0` of X and W into registers (zeros past the matrices' edges).
    // (Inlined twice below: WGSL can't pass the register arrays by reference cheaply.)
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
                xs[(v / (BK / 4u)) * LD + v % (BK / 4u)] = xr[s];
            }
        }
        for (var s = 0u; s < B_LOADS; s++) {
            let v = tid + s * THREADS;
            if v < B_VECS {
                let at = (v / (BK / 8u)) * LD + (v % (BK / 8u)) * 2u;
                let q = wr[s];
                ws[at] = vec4<f32>(unpack2x16float(q.x), unpack2x16float(q.y));
                ws[at + 1u] = vec4<f32>(unpack2x16float(q.z), unpack2x16float(q.w));
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

        for (var kk = 0u; kk < BK / 4u; kk++) {
            var a: array<vec4<f32>, TM>;
            var b: array<vec4<f32>, TN>;
            for (var i = 0u; i < TM; i++) {
                a[i] = xs[(ty + i * WY) * LD + kk];
            }
            for (var j = 0u; j < TN; j++) {
                b[j] = ws[(tx + j * WX) * LD + kk];
            }
            for (var i = 0u; i < TM; i++) {
                for (var j = 0u; j < TN; j++) {
                    let c = acc[i * TN + j];
                    acc[i * TN + j] = fma(a[i].w, b[j].w, fma(a[i].z, b[j].z,
                        fma(a[i].y, b[j].y, fma(a[i].x, b[j].x, c))));
                }
            }
        }
    }

    let has_bias = (p.flags & 1u) != 0u;
    let kind = (p.flags >> 1u) & 3u;
    let accumulate = (p.flags & 8u) != 0u;
    for (var i = 0u; i < TM; i++) {
        let row = m0 + ty + i * WY;
        if row >= p.m {
            continue;
        }
        for (var j = 0u; j < TN; j++) {
            let col = n0 + tx + j * WX;
            if col >= p.n {
                continue;
            }
            var v = acc[i * TN + j];
            if has_bias {
                v += bias[col];
            }
            v = activate(v, kind) * p.scale;
            let at = row * p.n + col;
            if accumulate {
                v += y[at];
            }
            y[at] = v;
        }
    }
}
