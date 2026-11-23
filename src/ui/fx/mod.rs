//! Vellum's compositor: Rune draws the window, and Vellum's dot shader (`dots.wgsl`) fills the
//! progress plate's effect slot.

use crate::State;
use rune::compose::{Compositor, DeviceKey, Frame};
use rune::render::{EffectSite, KIND_EFFECT};
use std::collections::HashMap;

/// The progress plate's dots. Data: progress, finished at (−1: running).
pub const PLATE: u32 = 1;

/// Dot pitch, points, and rows.
const PITCH: f32 = 9.0;
const ROWS: f32 = 5.0;

/// What one draw needs, as the shader's `Dots` reads it.
struct Dots {
    rect: [f32; 4],
    pass_size: [f32; 4],
    state: [f32; 4],
    style: [f32; 4],
}

impl Dots {
    fn at(site: &EffectSite<'_>, state: [f32; 4], style: [f32; 4]) -> Dots {
        let [x, y, w, h] = site.instance.rect;
        Dots {
            rect: [x - site.origin[0], y - site.origin[1], w, h],
            pass_size: [
                site.size[0] as f32,
                site.size[1] as f32,
                site.scale,
                if site.srgb_blend { 1.0 } else { 0.0 },
            ],
            state,
            style,
        }
    }

    fn bytes(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        let values = [self.rect, self.pass_size, self.state, self.style];
        for (i, v) in values.iter().flatten().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        out
    }
}

struct Pipelines {
    shader: wgpu::ShaderModule,
    layout: wgpu::PipelineLayout,
    by_format: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
}

impl Pipelines {
    fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vellum dots"),
            source: wgpu::ShaderSource::Wgsl(include_str!("dots.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vellum dots"),
            bind_group_layouts: &[],
            immediate_size: 64,
        });
        Self {
            shader,
            layout,
            by_format: HashMap::new(),
        }
    }

    /// Makes the pipeline for a target format ahead of the passes that need it.
    fn add_format(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) {
        if self.by_format.contains_key(&format) {
            return;
        }
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("vellum dots"),
            layout: Some(&self.layout),
            vertex: wgpu::VertexState {
                module: &self.shader,
                entry_point: Some("dots_vertex"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.shader,
                entry_point: Some("dots_fragment"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        self.by_format.insert(format, pipeline);
    }

    fn draw(&self, pass: &mut wgpu::RenderPass<'_>, format: wgpu::TextureFormat, dots: &Dots) {
        let Some(pipeline) = self.by_format.get(&format) else {
            return;
        };
        pass.set_pipeline(pipeline);
        pass.set_immediates(0, &dots.bytes());
        pass.draw(0..6, 0..1);
    }
}

#[derive(Default)]
pub struct Gfx {
    key: Option<DeviceKey>,
    pipelines: Option<Pipelines>,
    /// The plate was on screen last frame: its dots move on their own.
    live: bool,
}

impl Compositor<State> for Gfx {
    fn features(&self) -> wgpu::Features {
        wgpu::Features::IMMEDIATES
    }

    fn animating(&self) -> bool {
        self.live
    }

    fn render(&mut self, f: &mut Frame<'_, State>) {
        let marks = &f.rt.marks;
        let slots =
            f.rt.instances
                .iter()
                .enumerate()
                .filter(|(_, i)| i.params[1] == KIND_EFFECT)
                .map(|(n, _)| marks.get(n).copied().unwrap_or(0) & 0xff)
                .collect::<Vec<_>>();
        self.live = slots.contains(&PLATE);
        let device = f.device;
        if slots.is_empty() || !device.features().contains(wgpu::Features::IMMEDIATES) {
            f.draw_ui();
            return;
        }
        if self.key != Some(f.device_key()) {
            self.key = Some(f.device_key());
            self.pipelines = None;
        }
        let pipelines = self.pipelines.get_or_insert_with(|| Pipelines::new(device));
        pipelines.add_format(device, f.format);
        pipelines.add_format(device, wgpu::TextureFormat::Rgba16Float);
        let pipelines = &*pipelines;
        let clock = f.rt.clock;
        let since = |at: f32| {
            if at >= 0.0 {
                (clock - f64::from(at)) as f32
            } else {
                -1.0
            }
        };
        let clear = f.rt.clear_color();
        f.ui.prepare(
            device,
            f.queue,
            &mut f.rt.text.atlas,
            &f.rt.instances,
            f.size,
        );
        f.ui.draw_with(
            f.encoder,
            f.target,
            clear,
            &mut |pass, s: &EffectSite<'_>| {
                if s.mark & 0xff != PLATE {
                    return;
                }
                let dots = Dots::at(
                    s,
                    [clock as f32, s.data[0], since(s.data[1]), 0.0],
                    [PITCH, ROWS, 1.0, 0.0],
                );
                pipelines.draw(pass, s.format, &dots);
            },
        );
    }
}
