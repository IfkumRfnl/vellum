// Y[M,N] = epilogue(X[M,K] · W[N,K]ᵀ + b) on packed int8 dot products (`dot4I8Packed`): X
// and W both quantized per 32 values along K (Q8_0: int8 values four to a word, one f32
// scale per block), each block's integer sum exact in i32, scaled by its row's and column's
// scales and summed in f32. The same arithmetic as the reference's CPU path (its 8-bit
// activations); W is the GGUF's own Q8_0, X comes from `quantize_q8.wgsl`.
//
// Each workgroup computes a BM×BN tile of Y from BK-wide slices of X and W (and their
// scales) staged in workgroup memory; each of its (BM/TM)·(BN/TN) threads holds a TM×TN
// block of f32 accumulators in registers, rows strided by BM/TM and columns by BN/TN as in
// `gemm_simt.wgsl`. The next slice is loaded into registers while the current one is
// multiplied.
//
// Prepended by the host: const BM, BN, BK, TM, TN (u32). Needs K % BK == 0 and BK % 32 == 0;
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

// X's int8 values, 16 to a vector, row-major [M][K/16], and its scales [M][K/32].
@group(0) @binding(0) var<storage, read> xq: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> xsc: array<f32>;
// W's, [N][K/16] and [N][K/32].
@group(0) @binding(2) var<storage, read> wq: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> wsc: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;

const WX: u32 = BN / TN;
const WY: u32 = BM / TM;
const THREADS: u32 = WX * WY;
// Blocks of 32 per slice, and vectors of 16 values per staged row.
const KB: u32 = BK / 32u;
const KV: u32 = BK / 16u;
// A staged row: KV vectors plus one of padding against bank conflicts.
const LD: u32 = KV + 1u;
const A_VECS: u32 = BM * KV;
const B_VECS: u32 = BN * KV;
const A_LOADS: u32 = (A_VECS + THREADS - 1u) / THREADS;
const B_LOADS: u32 = (B_VECS + THREADS - 1u) / THREADS;
const SA_LOADS: u32 = (BM * KB + THREADS - 1u) / THREADS;
const SB_LOADS: u32 = (BN * KB + THREADS - 1u) / THREADS;

var<workgroup> xsh: array<vec4<u32>, BM * LD>;
var<workgroup> wsh: array<vec4<u32>, BN * LD>;
// Scales block-major: [block][row] and [block][col].
var<workgroup> xss: array<f32, KB * BM>;
var<workgroup> wss: array<f32, KB * BN>;

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

fn dot16(a: vec4<u32>, b: vec4<u32>) -> i32 {
    return dot4I8Packed(a.x, b.x) + dot4I8Packed(a.y, b.y) + dot4I8Packed(a.z, b.z)
        + dot4I8Packed(a.w, b.w);
}

@compute @workgroup_size(THREADS)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) group: vec3<u32>) {
    let m0 = group.y * BM;
    let n0 = group.x * BN;
    let tx = tid % WX;
    let ty = tid / WX;
    let k16 = p.k / 16u;
    let k32 = p.k / 32u;

    var acc: array<f32, TM * TN>;
    for (var i = 0u; i < TM * TN; i++) {
        acc[i] = 0.0;
    }
    var xr: array<vec4<u32>, A_LOADS>;
    var wr: array<vec4<u32>, B_LOADS>;
    var xsr: array<f32, SA_LOADS>;
    var wsr: array<f32, SB_LOADS>;

    // Slice 0 of X and W and their scales into registers (zeros past the matrices' edges).
    for (var s = 0u; s < A_LOADS; s++) {
        let v = tid + s * THREADS;
        let row = m0 + v / KV;
        xr[s] = vec4<u32>(0u);
        if v < A_VECS && row < p.m {
            xr[s] = xq[row * k16 + v % KV];
        }
    }
    for (var s = 0u; s < B_LOADS; s++) {
        let v = tid + s * THREADS;
        let row = n0 + v / KV;
        wr[s] = vec4<u32>(0u);
        if v < B_VECS && row < p.n {
            wr[s] = wq[row * k16 + v % KV];
        }
    }
    for (var s = 0u; s < SA_LOADS; s++) {
        let v = tid + s * THREADS;
        let row = m0 + v / KB;
        xsr[s] = 0.0;
        if v < BM * KB && row < p.m {
            xsr[s] = xsc[row * k32 + v % KB];
        }
    }
    for (var s = 0u; s < SB_LOADS; s++) {
        let v = tid + s * THREADS;
        let row = n0 + v / KB;
        wsr[s] = 0.0;
        if v < BN * KB && row < p.n {
            wsr[s] = wsc[row * k32 + v % KB];
        }
    }

    let slices = p.k / BK;
    for (var t = 0u; t < slices; t++) {
        workgroupBarrier();
        for (var s = 0u; s < A_LOADS; s++) {
            let v = tid + s * THREADS;
            if v < A_VECS {
                xsh[(v / KV) * LD + v % KV] = xr[s];
            }
        }
        for (var s = 0u; s < B_LOADS; s++) {
            let v = tid + s * THREADS;
            if v < B_VECS {
                wsh[(v / KV) * LD + v % KV] = wr[s];
            }
        }
        for (var s = 0u; s < SA_LOADS; s++) {
            let v = tid + s * THREADS;
            if v < BM * KB {
                xss[(v % KB) * BM + v / KB] = xsr[s];
            }
        }
        for (var s = 0u; s < SB_LOADS; s++) {
            let v = tid + s * THREADS;
            if v < BN * KB {
                wss[(v % KB) * BN + v / KB] = wsr[s];
            }
        }
        workgroupBarrier();

        if t + 1u < slices {
            let kv = (t + 1u) * KV;
            let kb = (t + 1u) * KB;
            for (var s = 0u; s < A_LOADS; s++) {
                let v = tid + s * THREADS;
                let row = m0 + v / KV;
                if v < A_VECS && row < p.m {
                    xr[s] = xq[row * k16 + kv + v % KV];
                }
            }
            for (var s = 0u; s < B_LOADS; s++) {
                let v = tid + s * THREADS;
                let row = n0 + v / KV;
                if v < B_VECS && row < p.n {
                    wr[s] = wq[row * k16 + kv + v % KV];
                }
            }
            for (var s = 0u; s < SA_LOADS; s++) {
                let v = tid + s * THREADS;
                let row = m0 + v / KB;
                if v < BM * KB && row < p.m {
                    xsr[s] = xsc[row * k32 + kb + v % KB];
                }
            }
            for (var s = 0u; s < SB_LOADS; s++) {
                let v = tid + s * THREADS;
                let row = n0 + v / KB;
                if v < BN * KB && row < p.n {
                    wsr[s] = wsc[row * k32 + kb + v % KB];
                }
            }
        }

        for (var kb = 0u; kb < KB; kb++) {
            // One block of 32: two vectors per row and column, an exact i32 sum, then
            // scaled into the f32 accumulator.
            var a0: array<vec4<u32>, TM>;
            var a1: array<vec4<u32>, TM>;
            var b0: array<vec4<u32>, TN>;
            var b1: array<vec4<u32>, TN>;
            var sa: array<f32, TM>;
            var sb: array<f32, TN>;
            for (var i = 0u; i < TM; i++) {
                let r = ty + i * WY;
                a0[i] = xsh[r * LD + 2u * kb];
                a1[i] = xsh[r * LD + 2u * kb + 1u];
                sa[i] = xss[kb * BM + r];
            }
            for (var j = 0u; j < TN; j++) {
                let c = tx + j * WX;
                b0[j] = wsh[c * LD + 2u * kb];
                b1[j] = wsh[c * LD + 2u * kb + 1u];
                sb[j] = wss[kb * BN + c];
            }
            for (var i = 0u; i < TM; i++) {
                for (var j = 0u; j < TN; j++) {
                    let d = dot16(a0[i], b0[j]) + dot16(a1[i], b1[j]);
                    acc[i * TN + j] = fma(f32(d), sa[i] * sb[j], acc[i * TN + j]);
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
