//! The little SPIR-V handling the engine's glslc-compiled kernels need: reading the module's
//! words and setting its specialization constants (wgpu passes none to a raw module).

const MAGIC: u32 = 0x0723_0203;
const OP_DECORATE: u32 = 71;
const OP_SPEC_CONSTANT: u32 = 50;
const DECORATION_SPEC_ID: u32 = 1;
/// The header: magic, version, generator, bound, schema.
const HEADER: usize = 5;

/// The module's words with each `(SpecId, value)` in `values` as its constant's default
/// (32-bit integer constants only, which is all the kernels use).
pub fn specialize(bytes: &[u8], values: &[(u32, u32)]) -> Result<Vec<u32>, String> {
    if !bytes.len().is_multiple_of(4) || bytes.len() < HEADER * 4 {
        return Err("not a SPIR-V module".into());
    }
    let mut words: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    if words[0] != MAGIC {
        return Err("not a little-endian SPIR-V module".into());
    }
    // (result id, SpecId) of every decorated constant, then each constant's literal.
    let mut spec_ids = Vec::new();
    let mut at = HEADER;
    while at < words.len() {
        let (count, op) = ((words[at] >> 16) as usize, words[at] & 0xffff);
        if count == 0 || at + count > words.len() {
            return Err(format!("malformed instruction at word {at}"));
        }
        if op == OP_DECORATE && count == 4 && words[at + 2] == DECORATION_SPEC_ID {
            spec_ids.push((words[at + 1], words[at + 3]));
        }
        at += count;
    }
    let mut found = vec![false; values.len()];
    let mut at = HEADER;
    while at < words.len() {
        let (count, op) = ((words[at] >> 16) as usize, words[at] & 0xffff);
        if op == OP_SPEC_CONSTANT && count == 4 {
            let id = words[at + 2];
            if let Some(&(_, spec)) = spec_ids.iter().find(|(i, _)| *i == id) {
                for (i, &(want, value)) in values.iter().enumerate() {
                    if want == spec {
                        words[at + 3] = value;
                        found[i] = true;
                    }
                }
            }
        }
        at += count;
    }
    match values.iter().zip(&found).find(|(_, f)| !**f) {
        Some(((spec, _), _)) => Err(format!("no 32-bit specialization constant {spec}")),
        None => Ok(words),
    }
}
