// The end of a decoder step, for every decoding window (one workgroup each): where the window
// goes next. During the prompt, its next prompt token. After it, for a greedy window, the
// next token as the CPU oracle chooses it (`decoding.rs`): suppressed tokens and
// `<|notimestamps|>` masked, the timestamp rules for what it has generated, then the best
// token (the first of equals), its log-probability and the gap to the second best, written to
// the window's list of choices; it's done at the end token or the most choices a pass takes. A
// window the host chooses for (sampling at a temperature) only advances; the host writes its
// token. At the first position the probability of `<|nospeech|>` in the raw logits is kept.
//
// The state: word 0, then SLOT_WORDS words per window (see `gpu/decoder.rs`). Masked logits
// are NEG (no infinities in WGSL).
//
// Prepended by the host: const VOCAB, EOS, NO_TIMESTAMPS, TS_BEGIN, MAX_INITIAL, NO_SPEECH_TOKEN,
// MASK_WORDS, MAX_CHOICES, CTX, SLOT_WORDS and the state's word offsets (MODE, LEN, CHOICES,
// PROMPT_LEN, PROMPT, INPUT, LAST, PENULT, LAST_TS, NO_SPEECH, LIMIT).

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> masks: array<u32>;
@group(0) @binding(2) var<storage, read_write> state: array<u32>;
@group(0) @binding(3) var<storage, read_write> choices: array<u32>;

const NEG: f32 = -3.0e38;
const FINITE: f32 = -1.0e38;

var<workgroup> slot: array<u32, 16>;
// Per thread: best and second-best text logit and the best's index; the same for timestamps.
var<workgroup> t1: array<f32, 256>;
var<workgroup> ti: array<u32, 256>;
var<workgroup> t2: array<f32, 256>;
var<workgroup> s1: array<f32, 256>;
var<workgroup> si: array<u32, 256>;
var<workgroup> s2: array<f32, 256>;
var<workgroup> sums: array<vec2<f32>, 256>;

fn bit(word0: u32, i: u32) -> bool {
    return ((masks[word0 + i / 32u] >> (i % 32u)) & 1u) == 1u;
}

fn is_ts(t: u32) -> bool {
    return t >= TS_BEGIN && t < VOCAB;
}

// `a` and `b` merged: the better best (the lower index on ties), and the second best of all.
fn merge(a1: f32, ai: u32, a2: f32, b1: f32, bi: u32, b2: f32) -> vec3<f32> {
    var best = a1;
    var index = ai;
    if b1 > a1 || (b1 == a1 && bi < ai) {
        best = b1;
        index = bi;
    }
    let second = max(max(a2, b2), min(a1, b1));
    return vec3<f32>(best, bitcast<f32>(index), second);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let at = 1u + wg.x * SLOT_WORDS;
    if tid < SLOT_WORDS {
        slot[tid] = state[at + tid];
    }
    // Uniform across the workgroup, so it can branch around barriers.
    let words = workgroupUniformLoad(&slot);
    let mode = words[MODE];
    if mode == 0u {
        return;
    }
    let len = words[LEN];
    let row = wg.x * VOCAB;

    // The raw logits' probability of `<|nospeech|>`, at `<|startoftranscript|>`.
    if len == 0u {
        var m = NEG;
        for (var i = tid; i < VOCAB; i += 256u) {
            m = max(m, logits[row + i]);
        }
        t1[tid] = m;
        workgroupBarrier();
        for (var st = 128u; st > 0u; st >>= 1u) {
            if tid < st {
                t1[tid] = max(t1[tid], t1[tid + st]);
            }
            workgroupBarrier();
        }
        let mx = t1[0];
        var sum = 0.0;
        for (var i = tid; i < VOCAB; i += 256u) {
            sum += exp(logits[row + i] - mx);
        }
        workgroupBarrier();
        s1[tid] = sum;
        workgroupBarrier();
        for (var st = 128u; st > 0u; st >>= 1u) {
            if tid < st {
                s1[tid] += s1[tid + st];
            }
            workgroupBarrier();
        }
        if tid == 0u {
            let p = exp(logits[row + NO_SPEECH_TOKEN] - mx) / s1[0];
            state[at + NO_SPEECH] = bitcast<u32>(p);
        }
        workgroupBarrier();
    }

    let next = len + 1u;
    if next < words[PROMPT_LEN] {
        if tid == 0u {
            state[at + INPUT] = words[PROMPT + next];
            state[at + LEN] = next;
        }
        return;
    }
    if mode == 2u {
        if tid == 0u {
            state[at + LEN] = next;
        }
        return;
    }

    // The timestamp rules' view of what's been generated.
    let n = words[CHOICES];
    let first = n == 0u;
    let last_ts = n >= 1u && is_ts(words[LAST]);
    let penult_ts = n < 2u || is_ts(words[PENULT]);
    let has_ts = words[LAST_TS] != 0u;
    var low = words[LAST_TS] + 1u;
    if last_ts && !penult_ts {
        low = words[LAST_TS];
    }

    var a1 = NEG;
    var ai = 0u;
    var a2 = NEG;
    var b1 = NEG;
    var bi = TS_BEGIN;
    var b2 = NEG;
    for (var i = tid; i < VOCAB; i += 256u) {
        var v = logits[row + i];
        if bit(0u, i) || (first && bit(MASK_WORDS, i)) || i == NO_TIMESTAMPS {
            v = NEG;
        }
        if i >= TS_BEGIN {
            if (last_ts && penult_ts) || (has_ts && i < low) || (first && i > TS_BEGIN + MAX_INITIAL) {
                v = NEG;
            }
            if v > b1 {
                b2 = b1;
                b1 = v;
                bi = i;
            } else if v > b2 {
                b2 = v;
            }
        } else {
            if (last_ts && !penult_ts && i < EOS) || first {
                v = NEG;
            }
            if v > a1 {
                a2 = a1;
                a1 = v;
                ai = i;
            } else if v > a2 {
                a2 = v;
            }
        }
    }
    t1[tid] = a1;
    ti[tid] = ai;
    t2[tid] = a2;
    s1[tid] = b1;
    si[tid] = bi;
    s2[tid] = b2;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st >>= 1u) {
        if tid < st {
            let t = merge(t1[tid], ti[tid], t2[tid], t1[tid + st], ti[tid + st], t2[tid + st]);
            t1[tid] = t.x;
            ti[tid] = bitcast<u32>(t.y);
            t2[tid] = t.z;
            let s = merge(s1[tid], si[tid], s2[tid], s1[tid + st], si[tid + st], s2[tid + st]);
            s1[tid] = s.x;
            si[tid] = bitcast<u32>(s.y);
            s2[tid] = s.z;
        }
        workgroupBarrier();
    }
    let max_text = t1[0];
    let max_ts = s1[0];

    // Each part's sum of exp(v − its maximum), for the log-probabilities.
    var sum = vec2<f32>(0.0);
    for (var i = tid; i < VOCAB; i += 256u) {
        var v = logits[row + i];
        if bit(0u, i) || (first && bit(MASK_WORDS, i)) || i == NO_TIMESTAMPS {
            v = NEG;
        }
        if i >= TS_BEGIN {
            if (last_ts && penult_ts) || (has_ts && i < low) || (first && i > TS_BEGIN + MAX_INITIAL) {
                v = NEG;
            }
            if v > FINITE {
                sum.y += exp(v - max_ts);
            }
        } else {
            if (last_ts && !penult_ts && i < EOS) || first {
                v = NEG;
            }
            if v > FINITE {
                sum.x += exp(v - max_text);
            }
        }
    }
    sums[tid] = sum;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st >>= 1u) {
        if tid < st {
            sums[tid] += sums[tid + st];
        }
        workgroupBarrier();
    }
    if tid != 0u {
        return;
    }
    let total = sums[0];
    // A timestamp is forced when their total probability beats the best text token's.
    var ts_lse = NEG;
    if max_ts > FINITE {
        ts_lse = max_ts + log(total.y);
    }
    var best: f32;
    var index: u32;
    var second: f32;
    var lse: f32;
    if ts_lse > max_text {
        best = max_ts;
        index = si[0];
        second = s2[0];
        lse = ts_lse;
    } else {
        let m = merge(max_text, ti[0], t2[0], max_ts, si[0], s2[0]);
        best = m.x;
        index = bitcast<u32>(m.y);
        second = m.z;
        lse = best + log(total.x * exp(max_text - best) + total.y * exp(max_ts - best));
    }
    let out = (wg.x * MAX_CHOICES + n) * 3u;
    choices[out] = index;
    choices[out + 1u] = bitcast<u32>(best - lse);
    choices[out + 2u] = bitcast<u32>(best - second);
    state[at + CHOICES] = n + 1u;
    state[at + PENULT] = words[LAST];
    state[at + LAST] = index;
    if is_ts(index) {
        state[at + LAST_TS] = index;
    }
    state[at + INPUT] = index;
    state[at + LEN] = next;
    if index == EOS || n + 1u >= words[LIMIT] || next + 1u >= CTX {
        state[at + MODE] = 0u;
    }
}
