//! The GPU path's foundation: the engine's own `wgpu` device, persistent buffers, compute
//! kernels compiled from the WGSL under `gpu/shaders/`, GPU timestamps and readback. The models'
//! GPU encoders are built on this; the matrix multiply they all stand on is [`gemm`].
//!
//! The device is the engine's alone (never the UI's), so a long queue or a lost device can't
//! stall the window. Which adapter: the discrete GPU, on Vulkan where it has it (that's where
//! cooperative matrices are), unless `VELLUM_GPU` names another (see [`GpuOptions::from_env`]).
//!
//! Storage is f16 for weights and f32 for everything else; shaders accumulate in f32.

pub mod gemm;

/// The f16 type weights are uploaded as.
pub use half::f16;
/// The wgpu the engine is built on, for callers that set its options.
pub use wgpu;

use crate::{Error, Result};

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};
use wgpu::util::DeviceExt;

/// Every embedded shader, for the pipeline cache's key: a changed shader starts a new cache.
const SHADERS: &[&str] = &[gemm::SIMT_SOURCE, gemm::COOP_SOURCE];

/// How to pick and drive the device.
#[derive(Debug, Clone, Default)]
pub struct GpuOptions {
    /// An adapter by index (in [`adapters`] order) or by a case-insensitive part of its name
    /// ("intel", "4060"). `None`: the best one.
    pub adapter: Option<String>,
    /// Only these backends (default: every one built in).
    pub backends: Option<wgpu::Backends>,
    /// Keep the runtime bounds checks in every shader (slower; for debugging a kernel).
    pub checked: bool,
    /// Where the Vulkan pipeline cache lives across runs (none: compile every run).
    pub cache_dir: Option<PathBuf>,
}

impl GpuOptions {
    /// `VELLUM_GPU` (adapter index or name part), `VELLUM_GPU_BACKEND` (`vulkan`, `dx12`,
    /// `metal`, comma-separated), `VELLUM_GPU_CHECKED=1`. (`VELLUM_GPU_DUMP=<dir>` writes
    /// every kernel's WGSL as compiled there, for kernel work.)
    pub fn from_env() -> GpuOptions {
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        GpuOptions {
            adapter: var("VELLUM_GPU"),
            backends: var("VELLUM_GPU_BACKEND").map(|b| wgpu::Backends::from_comma_list(&b)),
            checked: var("VELLUM_GPU_CHECKED").is_some_and(|v| v == "1"),
            cache_dir: None,
        }
    }
}

/// The features the engine uses when the adapter has them. Only `IMMEDIATES` is required.
fn wanted_features() -> wgpu::Features {
    use wgpu::Features as F;
    F::SHADER_F16
        | F::SUBGROUP
        | F::IMMEDIATES
        | F::TIMESTAMP_QUERY
        | F::TIMESTAMP_QUERY_INSIDE_ENCODERS
        | F::TIMESTAMP_QUERY_INSIDE_PASSES
        | F::PIPELINE_CACHE
        | F::EXPERIMENTAL_COOPERATIVE_MATRIX
}

/// Every adapter the built-in backends see, in the order `VELLUM_GPU=<index>` counts them.
pub fn adapters() -> Vec<wgpu::AdapterInfo> {
    let instance = instance(wgpu::Backends::all());
    block_on(instance.enumerate_adapters(wgpu::Backends::all()))
        .iter()
        .map(|a| a.get_info())
        .collect()
}

fn instance(backends: wgpu::Backends) -> wgpu::Instance {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = backends;
    desc.flags = wgpu::InstanceFlags::empty().with_env();
    wgpu::Instance::new(desc)
}

/// The bytes of f16 values, without copying them.
fn bytemuck_f16(data: &[f16]) -> &[u8] {
    // SAFETY: f16 is a plain 2-byte value with no padding; u8 has alignment 1.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) }
}

/// The engine's device and what it can do.
pub struct Gpu {
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    /// The features the device was created with (the wanted ones the adapter has).
    pub features: wgpu::Features,
    /// The adapter's own limits, which the device was created with.
    pub limits: wgpu::Limits,
    /// The cooperative-matrix configurations the driver reports (empty without the feature).
    pub cooperative_matrices: Vec<wgpu::CooperativeMatrixProperties>,
    checked: bool,
    pipeline_cache: Option<(wgpu::PipelineCache, PathBuf)>,
    kernels: Mutex<HashMap<String, Arc<Kernel>>>,
    bind_groups: Mutex<HashMap<(u64, Vec<u64>), wgpu::BindGroup>>,
    next_id: AtomicU64,
}

/// The usual case — no adapter or backend asked for — without starting every backend: listing
/// adapters on all of them costs a few hundred milliseconds of load time. The first-choice
/// backend (Vulkan, or Metal on Apple) is asked alone for a discrete GPU (any GPU on Apple);
/// `None` sends the choice through [`choose_adapter`].
fn quick_adapter(options: &GpuOptions) -> Option<wgpu::Adapter> {
    if options.adapter.is_some() || options.backends.is_some() {
        return None;
    }
    let apple = cfg!(target_vendor = "apple");
    let first = if apple {
        wgpu::Backends::METAL
    } else {
        wgpu::Backends::VULKAN
    };
    let mut found = block_on(instance(first).enumerate_adapters(first));
    found.retain(|a| apple || a.get_info().device_type == wgpu::DeviceType::DiscreteGpu);
    found.sort_by_key(|a| rank(&a.get_info()));
    found.into_iter().next()
}

/// The adapter `options` asks for, among every backend's, numbered as [`adapters`] lists them.
fn choose_adapter(options: &GpuOptions) -> Result<wgpu::Adapter> {
    // Every adapter, numbered as `adapters` lists them, then only the allowed backends.
    let instance = instance(wgpu::Backends::all());
    let backends = options.backends.unwrap_or(wgpu::Backends::all());
    let adapters: Vec<(usize, wgpu::Adapter)> =
        block_on(instance.enumerate_adapters(wgpu::Backends::all()))
            .into_iter()
            .enumerate()
            .filter(|(_, a)| backends.contains(a.get_info().backend.into()))
            .collect();
    let count = adapters.last().map_or(0, |(i, _)| i + 1);
    let mut candidates: Vec<&wgpu::Adapter> = match &options.adapter {
        // A number is an index when there's an adapter with it ("4060" is a name part).
        Some(want) => match want.parse::<usize>().ok().filter(|i| *i < count) {
            Some(index) => adapters
                .iter()
                .filter(|(i, _)| *i == index)
                .map(|(_, a)| a)
                .collect(),
            None => {
                let want = want.to_lowercase();
                adapters
                    .iter()
                    .map(|(_, a)| a)
                    .filter(|a| a.get_info().name.to_lowercase().contains(&want))
                    .collect()
            }
        },
        None => adapters
            .iter()
            .map(|(_, a)| a)
            .filter(|a| a.get_info().device_type != wgpu::DeviceType::Cpu)
            .collect(),
    };
    // The best of the matching ones (the Vulkan one before the DX12 one).
    candidates.sort_by_key(|a| rank(&a.get_info()));
    Ok((*candidates.first().ok_or_else(|| {
        Error::Other(format!(
            "no GPU adapter matches {:?} on {backends:?}",
            options.adapter.as_deref().unwrap_or("any")
        ))
    })?)
    .clone())
}

impl Gpu {
    pub fn new(options: &GpuOptions) -> Result<Gpu> {
        let adapter = match quick_adapter(options) {
            Some(adapter) => adapter,
            None => choose_adapter(options)?,
        };
        let info = adapter.get_info();
        let features = adapter.features() & wanted_features();
        if !features.contains(wgpu::Features::IMMEDIATES) {
            return Err(Error::Other(format!(
                "{} ({:?}) has no immediates (push constants)",
                info.name, info.backend
            )));
        }
        let limits = adapter.limits();
        let experimental = features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX);
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("vellum engine"),
            required_features: features,
            required_limits: limits.clone(),
            experimental_features: if experimental {
                // SAFETY: cooperative matrices are the one experimental feature used, only by
                // `gemm`'s cooperative kernel, which the lab checks against the CPU.
                unsafe { wgpu::ExperimentalFeatures::enabled() }
            } else {
                wgpu::ExperimentalFeatures::disabled()
            },
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| Error::Other(format!("GPU device on {}: {e}", info.name)))?;
        // wgpu's default panics on an error nobody scoped; the engine reports it and goes on
        // (kernel creation scopes its own errors and returns them).
        device.on_uncaptured_error(std::sync::Arc::new(|e| eprintln!("GPU error: {e}")));
        let cooperative_matrices = if experimental {
            adapter.cooperative_matrix_properties()
        } else {
            Vec::new()
        };
        let pipeline_cache = match (&options.cache_dir, wgpu::util::pipeline_cache_key(&info)) {
            (Some(dir), Some(key)) if features.contains(wgpu::Features::PIPELINE_CACHE) => {
                let path = dir.join(format!("{key}-{:016x}.bin", fnv(SHADERS.iter().copied())));
                let data = std::fs::read(&path).ok();
                // SAFETY: the data is what `get_data` wrote for this adapter and driver (the
                // key names both), and the driver validates its own header besides.
                let cache = unsafe {
                    device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                        label: Some("vellum engine"),
                        data: data.as_deref(),
                        fallback: true,
                    })
                };
                Some((cache, path))
            }
            _ => None,
        };
        Ok(Gpu {
            adapter,
            device,
            queue,
            info,
            features,
            limits,
            cooperative_matrices,
            checked: options.checked,
            pipeline_cache,
            kernels: Mutex::default(),
            bind_groups: Mutex::default(),
            next_id: AtomicU64::new(1),
        })
    }

    /// "NVIDIA GeForce RTX 4060 Laptop GPU (Vulkan, driver 616.56)".
    pub fn name(&self) -> String {
        format!(
            "{} ({:?}, {})",
            self.info.name, self.info.backend, self.info.driver_info
        )
    }

    pub fn has(&self, feature: wgpu::Features) -> bool {
        self.features.contains(feature)
    }

    /// Tensor cores for the multiply: f16 16×16×16 products with f32 accumulation, on
    /// 32-wide subgroups (what `gemm`'s cooperative kernel is written for).
    pub fn has_f16_tensor_cores(&self) -> bool {
        self.has(wgpu::Features::SHADER_F16)
            && self.info.subgroup_min_size <= 32
            && self.info.subgroup_max_size >= 32
            && self.cooperative_matrices.iter().any(|p| {
                (p.m_size, p.n_size, p.k_size) == (16, 16, 16)
                    && p.ab_type == wgpu::CooperativeScalarType::F16
                    && p.cr_type == wgpu::CooperativeScalarType::F32
            })
    }

    /// The device as the lab prints it: adapter, features, cooperative matrices, key limits.
    pub fn describe(&self) -> String {
        let l = &self.limits;
        let coop: Vec<String> = self
            .cooperative_matrices
            .iter()
            .map(|p| {
                format!(
                    "{}×{}×{} {:?}→{:?}",
                    p.m_size, p.n_size, p.k_size, p.ab_type, p.cr_type
                )
            })
            .collect();
        format!(
            "adapter: {} ({:?}, {:?}, driver {} {})\n\
             adapter features: {:?}\n\
             device features: {:?}\n\
             cooperative matrices: {}\n\
             subgroups: {}–{} lanes\n\
             limits: storage binding {} MiB, buffer {} MiB, workgroup memory {} KiB, \
             {} invocations/workgroup, {} storage buffers/stage, {} B immediates",
            self.info.name,
            self.info.backend,
            self.info.device_type,
            self.info.driver,
            self.info.driver_info,
            self.adapter.features(),
            self.features,
            if coop.is_empty() {
                "none".into()
            } else {
                coop.join(", ")
            },
            self.info.subgroup_min_size,
            self.info.subgroup_max_size,
            l.max_storage_buffer_binding_size >> 20,
            l.max_buffer_size >> 20,
            l.max_compute_workgroup_storage_size >> 10,
            l.max_compute_invocations_per_workgroup,
            l.max_storage_buffers_per_shader_stage,
            l.max_immediate_size,
        )
    }

    fn id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    fn tensor(&self, label: &str, dtype: Dtype, len: usize, bytes: Option<&[u8]>) -> Tensor {
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        // Bindings stay under the adapter's per-binding cap; a model's tensors fit easily
        // (the largest Parakeet weight is 8 MiB in f16).
        let size = (len * dtype.bytes()).next_multiple_of(4).max(4) as u64;
        let buffer = match bytes {
            Some(bytes) if bytes.len() as u64 == size => {
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(label),
                        contents: bytes,
                        usage,
                    })
            }
            Some(bytes) => {
                let mut padded = bytes.to_vec();
                padded.resize(size as usize, 0);
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(label),
                        contents: &padded,
                        usage,
                    })
            }
            None => self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            }),
        };
        Tensor {
            buffer,
            len,
            dtype,
            id: self.id(),
        }
    }

    /// A persistent f32 buffer holding `data`.
    pub fn upload_f32(&self, label: &str, data: &[f32]) -> Tensor {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.tensor(label, Dtype::F32, data.len(), Some(&bytes))
    }

    /// A persistent f16 buffer holding `data` (weights).
    pub fn upload_f16(&self, label: &str, data: &[f16]) -> Tensor {
        // f16 is two little-endian bytes on every target wgpu runs on.
        let bytes: &[u8] = bytemuck_f16(data);
        self.tensor(label, Dtype::F16, data.len(), Some(bytes))
    }

    /// An f32 buffer of `len` values, zeroed.
    pub fn zeros_f32(&self, label: &str, len: usize) -> Tensor {
        self.tensor(label, Dtype::F32, len, None)
    }

    /// Overwrites the start of a buffer with `data`.
    pub fn write_f32(&self, tensor: &Tensor, data: &[f32]) {
        assert!(tensor.dtype == Dtype::F32 && data.len() <= tensor.len);
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.queue.write_buffer(&tensor.buffer, 0, &bytes);
    }

    /// Copies a buffer back to the CPU, waiting for the GPU to finish everything before it.
    pub fn read_f32(&self, tensor: &Tensor) -> Result<Vec<f32>> {
        assert_eq!(tensor.dtype, Dtype::F32);
        let bytes = self.read_bytes(&tensor.buffer, (tensor.len * 4) as u64)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }

    fn read_bytes(&self, buffer: &wgpu::Buffer, size: u64) -> Result<Vec<u8>> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| drop(tx.send(r)));
        self.wait()?;
        rx.recv()
            .map_err(|_| Error::Other("readback never mapped".into()))?
            .map_err(|e| Error::Other(format!("readback: {e}")))?;
        let data = staging
            .slice(..)
            .get_mapped_range()
            .map_err(|e| Error::Other(format!("readback: {e}")))?
            .to_vec();
        staging.unmap();
        Ok(data)
    }

    /// Blocks until the GPU has finished all submitted work.
    pub fn wait(&self) -> Result<()> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(drop)
            .map_err(|e| Error::Other(format!("GPU: {e}")))
    }

    /// A compute kernel: `source` (WGSL) with `consts` prepended as `const NAME: u32 = V;`,
    /// entry point `main`, `bindings` storage buffers in order (`true` = read-write) and
    /// `immediates` bytes of immediate data. Compiled once per (name, consts); compile and
    /// validation errors come back as errors, not panics.
    pub fn kernel(
        &self,
        name: &str,
        source: &str,
        consts: &[(&str, u32)],
        bindings: &[bool],
        immediates: u32,
    ) -> Result<Arc<Kernel>> {
        let mut key = name.to_owned();
        for (k, v) in consts {
            key.push_str(&format!(" {k}={v}"));
        }
        if let Some(k) = self.kernels.lock().unwrap().get(&key) {
            return Ok(k.clone());
        }
        let mut wgsl = String::new();
        // `enable` directives must come before any declaration.
        let (enables, body): (Vec<&str>, Vec<&str>) = source
            .lines()
            .partition(|l| l.trim_start().starts_with("enable "));
        for line in enables {
            wgsl.push_str(line);
            wgsl.push('\n');
        }
        for (k, v) in consts {
            wgsl.push_str(&format!("const {k}: u32 = {v}u;\n"));
        }
        for line in body {
            wgsl.push_str(line);
            wgsl.push('\n');
        }
        // `VELLUM_GPU_DUMP=<dir>`: the WGSL as compiled, one file per kernel variant.
        if let Some(dir) = std::env::var_os("VELLUM_GPU_DUMP") {
            let file = key.replace([' ', '='], "_") + ".wgsl";
            std::fs::create_dir_all(&dir)?;
            std::fs::write(std::path::Path::new(&dir).join(file), &wgsl)?;
        }
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let desc = wgpu::ShaderModuleDescriptor {
            label: Some(&key),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        };
        let module = if self.checked {
            self.device.create_shader_module(desc)
        } else {
            // SAFETY: the engine's kernels guard every index themselves (checked against the
            // CPU by `cargo lab kernels`; `VELLUM_GPU_CHECKED=1` brings the checks back).
            unsafe {
                self.device
                    .create_shader_module_trusted(desc, wgpu::ShaderRuntimeChecks::unchecked())
            }
        };
        let entries: Vec<wgpu::BindGroupLayoutEntry> = bindings
            .iter()
            .enumerate()
            .map(|(i, &rw)| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: !rw },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&key),
                entries: &entries,
            });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&key),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: immediates,
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&key),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    // Every kernel writes its workgroup memory before reading it.
                    zero_initialize_workgroup_memory: false,
                },
                cache: self.pipeline_cache.as_ref().map(|(c, _)| c),
            });
        if let Some(error) = block_on(scope.pop()) {
            return Err(Error::Other(format!("kernel {key}: {error}")));
        }
        let kernel = Arc::new(Kernel {
            pipeline,
            layout,
            name: key.clone(),
            id: self.id(),
        });
        self.kernels.lock().unwrap().insert(key, kernel.clone());
        Ok(kernel)
    }

    /// The bind group of `kernel` over `tensors` (bindings 0, 1, …), made once and kept: it
    /// keeps those buffers alive until [`clear_bind_groups`](Self::clear_bind_groups).
    pub fn bind(&self, kernel: &Kernel, tensors: &[&Tensor]) -> wgpu::BindGroup {
        let key = (kernel.id, tensors.iter().map(|t| t.id).collect());
        self.bind_groups
            .lock()
            .unwrap()
            .entry(key)
            .or_insert_with(|| {
                let entries: Vec<wgpu::BindGroupEntry> = tensors
                    .iter()
                    .enumerate()
                    .map(|(i, t)| wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: t.buffer.as_entire_binding(),
                    })
                    .collect();
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&kernel.name),
                    layout: &kernel.layout,
                    entries: &entries,
                })
            })
            .clone()
    }

    pub fn clear_bind_groups(&self) {
        self.bind_groups.lock().unwrap().clear();
    }

    /// Writes the pipeline cache to disk (Vulkan, with a `cache_dir`), so the next run skips
    /// the driver's compiles.
    pub fn save_pipeline_cache(&self) -> Result<()> {
        let Some((cache, path)) = &self.pipeline_cache else {
            return Ok(());
        };
        let Some(data) = cache.get_data() else {
            return Ok(());
        };
        let Some(dir) = path.parent() else {
            return Ok(());
        };
        std::fs::create_dir_all(dir)?;
        let temp = path.with_extension("tmp");
        std::fs::write(&temp, data)?;
        std::fs::rename(&temp, path)?;
        // This adapter's caches for older shaders are dead weight.
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if let Some((key, _)) = name.rsplit_once('-') {
            for entry in std::fs::read_dir(dir)?.flatten() {
                let other = entry.file_name().to_string_lossy().into_owned();
                if other != name && other.starts_with(&format!("{key}-")) {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
        Ok(())
    }

    /// Where the pipeline cache is kept, if it is.
    pub fn pipeline_cache_path(&self) -> Option<&std::path::Path> {
        self.pipeline_cache.as_ref().map(|(_, p)| p.as_path())
    }
}

/// Adapter preference: discrete before integrated before the rest, and on one GPU, Vulkan
/// (cooperative matrices, pipeline cache) before Metal before DX12 before GL.
fn rank(info: &wgpu::AdapterInfo) -> (u8, u8) {
    let device = match info.device_type {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Other => 3,
        wgpu::DeviceType::Cpu => 4,
    };
    let backend = match info.backend {
        wgpu::Backend::Vulkan => 0,
        wgpu::Backend::Metal => 1,
        wgpu::Backend::Dx12 => 2,
        _ => 3,
    };
    (device, backend)
}

/// How a buffer's values are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
}

impl Dtype {
    pub fn bytes(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 => 2,
        }
    }
}

/// A persistent GPU buffer of `len` values.
pub struct Tensor {
    pub buffer: wgpu::Buffer,
    pub len: usize,
    pub dtype: Dtype,
    id: u64,
}

/// A compiled compute kernel and its bind group layout.
pub struct Kernel {
    pub pipeline: wgpu::ComputePipeline,
    pub layout: wgpu::BindGroupLayout,
    pub name: String,
    id: u64,
}

/// GPU timestamps around compute passes, for the profile mode and the lab. Each
/// [`pass`](Profiler::pass) is timed from its start to its end; [`read`](Profiler::read) gives
/// each pass's label and milliseconds in order. Without `TIMESTAMP_QUERY` passes are untimed.
pub struct Profiler {
    queries: Option<(wgpu::QuerySet, wgpu::Buffer)>,
    capacity: u32,
    labels: Vec<String>,
    period_ns: f64,
}

impl Profiler {
    /// Room for `passes` timed passes between reads.
    pub fn new(gpu: &Gpu, passes: u32) -> Profiler {
        let queries = gpu.has(wgpu::Features::TIMESTAMP_QUERY).then(|| {
            let set = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("profiler"),
                ty: wgpu::QueryType::Timestamp,
                count: passes * 2,
            });
            let resolve = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("profiler"),
                size: passes as u64 * 16,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            (set, resolve)
        });
        Profiler {
            queries,
            capacity: passes,
            labels: Vec::new(),
            period_ns: gpu.queue.get_timestamp_period() as f64,
        }
    }

    pub fn enabled(&self) -> bool {
        self.queries.is_some()
    }

    /// A compute pass, timed when there's room and timestamps are available.
    pub fn pass<'e>(
        &mut self,
        encoder: &'e mut wgpu::CommandEncoder,
        label: &str,
    ) -> wgpu::ComputePass<'e> {
        let index = self.labels.len() as u32;
        let timestamp_writes = match &self.queries {
            Some((set, _)) if index < self.capacity => {
                self.labels.push(label.to_owned());
                Some(wgpu::ComputePassTimestampWrites {
                    query_set: set,
                    beginning_of_pass_write_index: Some(index * 2),
                    end_of_pass_write_index: Some(index * 2 + 1),
                })
            }
            _ => None,
        };
        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes,
        })
    }

    /// Every timed pass since the last read: label and GPU milliseconds. Submits a resolve
    /// and waits for it.
    pub fn read(&mut self, gpu: &Gpu) -> Result<Vec<(String, f64)>> {
        let Some((set, resolve)) = &self.queries else {
            return Ok(Vec::new());
        };
        let n = self.labels.len() as u32;
        if n == 0 {
            return Ok(Vec::new());
        }
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.resolve_query_set(set, 0..n * 2, resolve, 0);
        gpu.queue.submit([encoder.finish()]);
        let bytes = gpu.read_bytes(resolve, n as u64 * 16)?;
        let ticks: Vec<u64> = bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| u64::from_le_bytes(*b))
            .collect();
        let labels = std::mem::take(&mut self.labels);
        Ok(labels
            .into_iter()
            .zip(ticks.as_chunks::<2>().0)
            .map(|(l, [t0, t1])| (l, t1.saturating_sub(*t0) as f64 * self.period_ns * 1e-6))
            .collect())
    }
}

/// FNV-1a over some strings: a stable hash for cache keys.
fn fnv<'a>(parts: impl Iterator<Item = &'a str>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for part in parts {
        for b in part.bytes() {
            h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

/// Runs a future to completion on this thread. wgpu's native futures are ready once the
/// device is polled, so a spin with a no-op waker is all they need.
pub fn block_on<F: std::future::Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    loop {
        if let std::task::Poll::Ready(v) = future.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::yield_now();
    }
}
