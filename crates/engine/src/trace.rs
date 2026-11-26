//! What a run shows the lab of its insides: intermediate tensors under the same names as
//! transcribe.cpp's dumps (`enc.mel.in`, `encoder.block.<L>.out`, …), and decoder decisions as
//! JSON events, so a divergence can be pinned to the first checkpoint or step where it appears.

/// Receives a run's checkpoints. Every method defaults to doing nothing, and
/// [`wants`](Trace::wants) lets the engine skip copying tensors nobody asked for.
pub trait Trace {
    /// Whether `tensor(name, …)` would be used. Checked before reading a tensor back.
    fn wants(&self, _name: &str) -> bool {
        false
    }

    /// A tensor, row-major, its `shape` slowest dimension first (as numpy and the reference's
    /// dumps write it, without trailing ones).
    fn tensor(&mut self, _name: &str, _shape: &[usize], _data: &[f32]) {}

    /// A decoder decision or other event, e.g. `("rnnt.step", {"frame":12,"token":5,…})`.
    fn event(&mut self, _kind: &str, _fields: serde_json::Value) {}

    /// A replacement for the tensor the engine is about to use at checkpoint `name` (where it
    /// computed one of `shape`): the reference engine's own, with its shape, so one stage can
    /// be tested on exactly the input the reference gave it (its mel into our encoder, its
    /// block 11 into our block 12…), even before the stage that feeds it exists.
    fn inject(&mut self, _name: &str, _shape: &[usize]) -> Option<(Vec<usize>, Vec<f32>)> {
        None
    }

    /// Whether [`inject`](Trace::inject) will replace checkpoint `name`, so the engine can skip
    /// computing what would be thrown away.
    fn injects(&self, _name: &str) -> bool {
        false
    }
}

/// A checkpoint: the reference's tensor replaces `x` if the trace injects one, then `x` is
/// recorded if the trace wants it. Every stage boundary goes through this, so any stage can be
/// tested alone on the reference's input.
pub fn checkpoint(trace: &mut dyn Trace, name: &str, x: &mut crate::cpu::Mat) {
    if let Some((shape, data)) = trace.inject(name, &x.shape()) {
        let (rows, cols) = match shape[..] {
            [rows, cols] => (rows, cols),
            _ => (1, data.len()),
        };
        *x = crate::cpu::Mat::from_vec(rows, cols, data);
    }
    if trace.wants(name) {
        trace.tensor(name, &x.shape(), &x.data);
    }
}

/// Records nothing.
pub struct NoTrace;

impl Trace for NoTrace {}
