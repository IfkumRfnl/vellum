//! The NVIDIA GPU through NVML: what it is, and a sampler that records its clocks, temperature,
//! power and throttling while a run happens. NVML ships with the driver (nvml.dll), so it's
//! loaded at run time; without it (no NVIDIA GPU or driver) everything here returns `None`.

use nvml_wrapper::bitmasks::device::ThrottleReasons;
use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};
use nvml_wrapper::{Device, Nvml};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// NVML, initialized once for the whole process (init takes a while) and never shut down.
pub(super) fn nvml() -> Option<&'static Nvml> {
    static NVML: OnceLock<Option<Nvml>> = OnceLock::new();
    NVML.get_or_init(|| Nvml::init().ok()).as_ref()
}

/// The first NVIDIA GPU: a laptop has one, and that's the one the engines run on.
pub(super) fn device() -> Option<Device<'static>> {
    nvml()?.device_by_index(0).ok()
}

/// One reading of the GPU. A field is `None` when NVML doesn't report it on this GPU.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Sample {
    /// Milliseconds since the sampler started.
    pub t_ms: u32,
    pub sm_mhz: Option<u32>,
    pub graphics_mhz: Option<u32>,
    pub memory_mhz: Option<u32>,
    pub temperature_c: Option<u32>,
    /// Board power draw, in milliwatts.
    pub power_mw: Option<u32>,
    /// Percent of the last sample period a kernel was running.
    pub utilization: Option<u32>,
    pub memory_used_mb: Option<u32>,
    /// NVML's clock event (throttle) reasons bitmask, `nvmlClocksEventReason*`.
    pub reasons: Option<u64>,
    /// Percent of the last sample period device memory was being read or written.
    #[serde(default)]
    pub memory_utilization: Option<u32>,
    /// The PCIe link as it runs now (it drops gen and width to save power).
    #[serde(default)]
    pub pcie_gen: Option<u32>,
    #[serde(default)]
    pub pcie_width: Option<u32>,
}

/// Reads every field once. NVML calls only: nothing here allocates.
pub(super) fn read(device: &Device, t0: Instant) -> Sample {
    let utilization = device.utilization_rates().ok();
    Sample {
        t_ms: t0.elapsed().as_millis() as u32,
        sm_mhz: device.clock_info(Clock::SM).ok(),
        graphics_mhz: device.clock_info(Clock::Graphics).ok(),
        memory_mhz: device.clock_info(Clock::Memory).ok(),
        temperature_c: device.temperature(TemperatureSensor::Gpu).ok(),
        power_mw: device.power_usage().ok(),
        utilization: utilization.as_ref().map(|u| u.gpu),
        memory_used_mb: device.memory_info().ok().map(|m| (m.used >> 20) as u32),
        reasons: device.current_throttle_reasons().ok().map(|r| r.bits()),
        memory_utilization: utilization.map(|u| u.memory),
        pcie_gen: device.current_pcie_link_gen().ok(),
        pcie_width: device.current_pcie_link_width().ok(),
    }
}

/// Records the GPU on a background thread until stopped.
pub struct Sampler {
    stop: mpsc::Sender<()>,
    thread: JoinHandle<GpuTrace>,
}

impl Sampler {
    /// Starts sampling every `interval` (~50 ms is right for runs of a second or more), or
    /// returns `None` when there's no NVML or no NVIDIA GPU. The thread sleeps between readings
    /// and only pushes onto a vector reserved up front, so it doesn't disturb the run.
    pub fn start(interval: Duration) -> Option<Sampler> {
        let device = device()?;
        let gpu = device.name().unwrap_or_default();
        let (stop, stopped) = mpsc::channel();
        let thread = thread::Builder::new()
            .name("gpu-sampler".into())
            .spawn(move || {
                let t0 = Instant::now();
                let mut samples = Vec::with_capacity(8192);
                let mut next = t0;
                loop {
                    samples.push(read(&device, t0));
                    // Keep a steady cadence; if a reading ran long, don't try to catch up.
                    next = (next + interval).max(Instant::now());
                    match stopped.recv_timeout(next - Instant::now()) {
                        Err(RecvTimeoutError::Timeout) => {}
                        _ => break,
                    }
                }
                samples.push(read(&device, t0));
                GpuTrace {
                    gpu,
                    interval_ms: interval.as_millis() as u32,
                    samples,
                }
            })
            .ok()?;
        Some(Sampler { stop, thread })
    }

    /// Stops sampling (with one last reading) and returns what was recorded.
    pub fn stop(self) -> GpuTrace {
        let _ = self.stop.send(());
        self.thread.join().unwrap_or_default()
    }
}

/// What the GPU did during a run.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GpuTrace {
    pub gpu: String,
    pub interval_ms: u32,
    pub samples: Vec<Sample>,
}

/// A run's GPU in a few numbers. Clocks and throttling are taken over the samples where the GPU
/// was running kernels (utilization above 0), so a run's model load and its waits on the CPU,
/// when the GPU sits at low clocks by choice, don't look like throttling.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GpuSummary {
    pub gpu: String,
    pub samples: usize,
    /// The samples where the GPU was running kernels.
    pub working: usize,
    pub seconds: f64,
    pub sm_mhz_median: Option<u32>,
    pub sm_mhz_min: Option<u32>,
    pub temperature_max_c: Option<u32>,
    pub power_mean_w: Option<f64>,
    pub power_max_w: Option<f64>,
    pub memory_used_peak_mb: Option<u32>,
    pub utilization_mean: Option<f64>,
    /// The reasons the GPU was held below its clocks, with the share of working samples each
    /// was seen in, e.g. "power cap (62% of the time)". Empty when the run ran unhindered.
    pub throttled: Vec<String>,
}

/// The clock event reasons that mean a run was slowed. Idle, application and user clock
/// settings and the display clock are left out: they say nothing about this run's speed.
const SLOWDOWNS: [(ThrottleReasons, &str); 6] = [
    (ThrottleReasons::SW_POWER_CAP, "power cap"),
    (ThrottleReasons::HW_SLOWDOWN, "hardware slowdown"),
    (
        ThrottleReasons::HW_THERMAL_SLOWDOWN,
        "hardware thermal slowdown",
    ),
    (
        ThrottleReasons::SW_THERMAL_SLOWDOWN,
        "software thermal slowdown",
    ),
    (
        ThrottleReasons::HW_POWER_BRAKE_SLOWDOWN,
        "hardware power brake",
    ),
    (ThrottleReasons::SYNC_BOOST, "sync boost"),
];

impl GpuTrace {
    pub fn summary(&self) -> GpuSummary {
        let samples = &self.samples;
        let working: Vec<&Sample> = samples
            .iter()
            .filter(|s| s.utilization.is_some_and(|u| u > 0))
            .collect();
        let count = working.len();
        let working = if working.is_empty() {
            samples.iter().collect()
        } else {
            working
        };
        let mut sm: Vec<u32> = working.iter().filter_map(|s| s.sm_mhz).collect();
        sm.sort_unstable();
        let power: Vec<f64> = samples
            .iter()
            .filter_map(|s| s.power_mw)
            .map(|mw| mw as f64 / 1000.0)
            .collect();
        let throttled = SLOWDOWNS
            .iter()
            .filter_map(|(reason, name)| {
                let seen = working
                    .iter()
                    .filter(|s| s.reasons.is_some_and(|r| r & reason.bits() != 0))
                    .count();
                (seen > 0).then(|| {
                    let share = 100.0 * seen as f64 / working.len() as f64;
                    format!("{name} ({share:.0}% of the time)")
                })
            })
            .collect();
        GpuSummary {
            gpu: self.gpu.clone(),
            samples: samples.len(),
            working: count,
            seconds: samples.last().map_or(0.0, |s| s.t_ms as f64 / 1000.0),
            sm_mhz_median: sm.get(sm.len() / 2).copied(),
            sm_mhz_min: sm.first().copied(),
            temperature_max_c: samples.iter().filter_map(|s| s.temperature_c).max(),
            power_mean_w: mean(&power),
            power_max_w: power.iter().copied().reduce(f64::max),
            memory_used_peak_mb: samples.iter().filter_map(|s| s.memory_used_mb).max(),
            utilization_mean: mean(
                &samples
                    .iter()
                    .filter_map(|s| s.utilization.map(f64::from))
                    .collect::<Vec<_>>(),
            ),
            throttled,
        }
    }
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}
