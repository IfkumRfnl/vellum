// The start of a decoder step: each decoding window's input token embedded (its Q8_0 row of
// the token embedding, exactly) plus the learned position of where it goes, into its row of
// x. Also decides word 0 of the state, whether any window is decoding, which the step's other
// kernels check first. One workgroup does every window.
//
// The state: word 0, then SLOT_WORDS words per window (see `gpu/decoder.rs`).
//
// Prepended by the host: const D_MODEL, SLOT_WORDS, MODE, LEN, INPUT (word offsets).

struct Params {
    slots: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read> qs: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> positions: array<f32>;
@group(0) @binding(3) var<storage, read_write> state: array<u32>;
@group(0) @binding(4) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32) {
    if tid == 0u {
        var a = 0u;
        for (var s = 0u; s < p.slots; s++) {
            if state[1u + s * SLOT_WORDS + MODE] != 0u {
                a = 1u;
            }
        }
        state[0] = a;
    }
    workgroupBarrier();
    for (var s = 0u; s < p.slots; s++) {
        let at = 1u + s * SLOT_WORDS;
        if state[at + MODE] == 0u {
            continue;
        }
        let token = state[at + INPUT];
        let len = state[at + LEN];
        for (var c = tid; c < D_MODEL; c += 256u) {
            let i = token * D_MODEL + c;
            let q = i32(qs[i / 4u] << (24u - 8u * (i % 4u))) >> 24u;
            let pair = unpack2x16float(scales[i / 64u]);
            let scale = select(pair.x, pair.y, ((i / 32u) & 1u) == 1u);
            x[s * D_MODEL + c] = scale * f32(q) + positions[len * D_MODEL + c];
        }
    }
}
