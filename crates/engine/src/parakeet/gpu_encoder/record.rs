//! Recording a batch's GPU work: compute passes on a command encoder, split into submits so no
//! one submit keeps the GPU busy for long, optionally timed with GPU timestamps — per pass (the
//! fast path's own time) or per dispatch (the lab's per-kernel breakdown, measurement only).

use crate::gpu::{Gpu, Kernel, Profiler, Tensor};

/// How the work is timed.
pub enum Timing<'a> {
    None,
    /// One timed pass per submit: the GPU time the batch takes, at the fast path's own speed.
    Passes(&'a mut Profiler),
    /// Every dispatch its own timed pass, labelled `family/name`.
    Dispatches(&'a mut Profiler),
}

pub struct Recorder<'a> {
    gpu: &'a Gpu,
    encoder: Option<wgpu::CommandEncoder>,
    pass: Option<wgpu::ComputePass<'static>>,
    timing: Timing<'a>,
}

impl<'a> Recorder<'a> {
    pub fn new(gpu: &'a Gpu, timing: Timing<'a>) -> Recorder<'a> {
        Recorder {
            gpu,
            encoder: None,
            pass: None,
            timing,
        }
    }

    fn encoder(&mut self) -> &mut wgpu::CommandEncoder {
        self.encoder.get_or_insert_with(|| {
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encoder batch"),
                })
        })
    }

    fn end_pass(&mut self) {
        self.pass = None;
    }

    /// A pass to record `label`'s dispatch into: the open one, or a new timed one.
    fn pass(&mut self, label: &str) -> &mut wgpu::ComputePass<'static> {
        let per_dispatch = matches!(self.timing, Timing::Dispatches(_));
        if per_dispatch {
            self.end_pass();
        }
        if self.pass.is_none() {
            self.encoder();
            let encoder = self.encoder.as_mut().unwrap();
            let pass = match &mut self.timing {
                Timing::None => encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("encoder"),
                    timestamp_writes: None,
                }),
                Timing::Passes(profiler) => profiler.pass(encoder, "batch"),
                Timing::Dispatches(profiler) => profiler.pass(encoder, label),
            };
            self.pass = Some(pass.forget_lifetime());
        }
        self.pass.as_mut().unwrap()
    }

    /// One kernel dispatch: `kernel` over `tensors` (bindings in order), `immediates` as u32
    /// words, `groups` workgroups.
    pub fn dispatch(
        &mut self,
        label: &str,
        kernel: &Kernel,
        tensors: &[&Tensor],
        immediates: &[u32],
        groups: [u32; 3],
    ) {
        let group = self.gpu.bind(kernel, tensors);
        let bytes: Vec<u8> = immediates.iter().flat_map(|v| v.to_le_bytes()).collect();
        let pass = self.pass(label);
        pass.set_pipeline(&kernel.pipeline);
        pass.set_bind_group(0, &group, &[]);
        if !bytes.is_empty() {
            pass.set_immediates(0, &bytes);
        }
        pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
    }

    /// Records into a pass with a caller's own encoding (a `Gemm::encode`).
    pub fn with_pass(&mut self, label: &str, f: impl FnOnce(&mut wgpu::ComputePass<'static>)) {
        f(self.pass(label));
    }

    /// A buffer-to-buffer copy (between passes).
    pub fn copy(
        &mut self,
        from: &Tensor,
        from_byte: u64,
        to: &wgpu::Buffer,
        to_byte: u64,
        bytes: u64,
    ) {
        self.end_pass();
        self.encoder()
            .copy_buffer_to_buffer(&from.buffer, from_byte, to, to_byte, bytes);
    }

    /// Sends everything recorded so far to the GPU (without waiting for it).
    pub fn submit(&mut self) -> Option<wgpu::SubmissionIndex> {
        self.end_pass();
        self.encoder
            .take()
            .map(|encoder| self.gpu.queue.submit([encoder.finish()]))
    }
}
