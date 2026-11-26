//! The machine a run happens on, and what its GPU does while it runs.
//!
//! Laptop speed numbers move with conditions: the GPU shares power with the CPU (Dynamic
//! Boost), throttles when hot, and clocks differently with the power plan and on battery. So
//! every timing run records the `Machine` it ran on, checks `conditions()` before it starts,
//! and samples the GPU while it runs (`Sampler`), to tell a slower engine from a slower laptop.

// The bench flows that call this are written separately.
#![allow(dead_code, unused_imports)]

mod gpu;
#[cfg(windows)]
mod windows;

pub use gpu::{GpuSummary, GpuTrace, Sample, Sampler};

use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::{Duration, Instant};

/// What a run ran on. `fingerprint` names the parts that decide timings, so runs are compared
/// only against runs from the same machine setup.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Machine {
    pub os: String,
    /// E.g. "24H2".
    pub os_version: String,
    /// E.g. "26100.9974".
    pub os_build: String,
    pub cpu: String,
    pub logical_cores: u32,
    pub physical_cores: Option<u32>,
    pub ram_bytes: Option<u64>,
    pub gpus: Vec<Gpu>,
    /// None when the machine doesn't say (a desktop says it's on AC).
    pub on_ac: Option<bool>,
    /// None without a battery.
    pub battery_percent: Option<u8>,
    /// The active power plan's name, e.g. "Balanced".
    pub power_plan: Option<String>,
    /// The Windows 11 power mode for the current power source, e.g. "Best performance".
    pub power_mode: Option<String>,
    /// A short hash of the CPU model, each GPU's name and driver, and the OS build number.
    pub fingerprint: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Gpu {
    pub name: String,
    pub vendor: String,
    /// NVML's version for NVIDIA (e.g. "616.56"), the user-mode driver's otherwise.
    pub driver: Option<String>,
    pub vram_bytes: Option<u64>,
}

pub fn machine() -> Machine {
    #[cfg(windows)]
    let mut machine = {
        let (os, os_version, os_build) = windows::os();
        let (on_ac, battery_percent) = windows::power();
        Machine {
            os,
            os_version,
            os_build,
            cpu: windows::cpu().unwrap_or_default(),
            logical_cores: logical_cores(),
            physical_cores: windows::physical_cores(),
            ram_bytes: windows::ram_bytes(),
            gpus: windows::gpus(),
            on_ac,
            battery_percent,
            power_plan: windows::power_plan().map(|(_, name)| name),
            power_mode: windows::power_mode(on_ac.unwrap_or(true)),
            fingerprint: String::new(),
        }
    };
    #[cfg(not(windows))]
    let mut machine = Machine {
        os: std::env::consts::OS.into(),
        os_version: String::new(),
        os_build: String::new(),
        cpu: String::new(),
        logical_cores: logical_cores(),
        physical_cores: None,
        ram_bytes: None,
        gpus: Vec::new(),
        on_ac: None,
        battery_percent: None,
        power_plan: None,
        power_mode: None,
        fingerprint: String::new(),
    };
    add_nvidia(&mut machine.gpus);
    machine.fingerprint = fingerprint(&machine);
    machine
}

fn logical_cores() -> u32 {
    std::thread::available_parallelism().map_or(0, |n| n.get() as u32)
}

/// NVML knows the NVIDIA GPU better than DXGI: the driver's own version and the full VRAM.
fn add_nvidia(gpus: &mut Vec<Gpu>) {
    let Some(nvml) = gpu::nvml() else { return };
    let driver = nvml.sys_driver_version().ok();
    for index in 0..nvml.device_count().unwrap_or(0) {
        let Ok(device) = nvml.device_by_index(index) else {
            continue;
        };
        let nvidia = Gpu {
            name: device.name().unwrap_or_default(),
            vendor: "NVIDIA".into(),
            driver: driver.clone(),
            vram_bytes: device.memory_info().ok().map(|m| m.total),
        };
        match gpus.iter_mut().find(|g| g.name == nvidia.name) {
            Some(gpu) => *gpu = nvidia,
            None => gpus.push(nvidia),
        }
    }
}

/// The build number without the monthly update revision: cumulative updates don't move
/// timings, and a fingerprint that changed every month would split comparable runs.
fn fingerprint(machine: &Machine) -> String {
    let mut parts = vec![machine.cpu.clone()];
    for gpu in &machine.gpus {
        parts.push(format!(
            "{} {}",
            gpu.name,
            gpu.driver.as_deref().unwrap_or("")
        ));
    }
    parts.push(format!(
        "{} {}",
        machine.os,
        machine.os_build.split('.').next().unwrap_or("")
    ));
    let hash = Sha256::digest(parts.join("\n"));
    hash[..6].iter().map(|b| format!("{b:02x}")).collect()
}

/// Why a timing run started now might measure the laptop rather than the engine, each as a
/// short sentence. Empty means go. Takes about a second (it watches the GPU).
pub fn conditions() -> Vec<String> {
    let mut warnings = Vec::new();
    #[cfg(windows)]
    {
        let (on_ac, battery) = windows::power();
        if on_ac == Some(false) {
            let charge = battery.map_or(String::new(), |b| format!(" ({b}%)"));
            warnings.push(format!(
                "Running on battery{charge}: plug in, the GPU and CPU slow down unplugged."
            ));
        }
        let on_ac = on_ac.unwrap_or(true);
        if let Some((guid, name)) = windows::power_plan() {
            let lower = name.to_lowercase();
            if guid == windows::POWER_SAVER || lower.contains("saver") {
                warnings.push(format!(
                    "The power plan is \"{name}\", which slows the machine."
                ));
            }
        }
        if let Some(mode) = windows::power_mode(on_ac)
            && mode == "Best power efficiency"
        {
            warnings.push("The power mode is \"Best power efficiency\".".into());
        }
        if let Some(max) = windows::max_processor_state(on_ac)
            && max < 100
        {
            warnings.push(format!(
                "The power plan caps the CPU at {max}% of its speed."
            ));
        }
    }
    let Some(sampler) = Sampler::start(Duration::from_millis(50)) else {
        return warnings;
    };
    std::thread::sleep(Duration::from_millis(1000));
    let trace = sampler.stop();
    let summary = trace.summary();
    // NVML's utilization trails by more than half a second (a run that just ended still reads
    // busy for a while), so only the last 300 ms of the look count.
    let recent: Vec<u32> = trace
        .samples
        .iter()
        .filter(|s| s.t_ms >= 700)
        .filter_map(|s| s.utilization)
        .collect();
    let busy =
        (!recent.is_empty()).then(|| recent.iter().sum::<u32>() as f64 / recent.len() as f64);
    if let Some(busy) = busy
        && busy > 10.0
    {
        warnings.push(format!(
            "The GPU is already {busy:.0}% busy{}.",
            gpu_users()
        ));
    }
    if let Some(temperature) = summary.temperature_max_c
        && temperature > 80
    {
        warnings.push(format!(
            "The GPU is already at {temperature} °C: let it cool first."
        ));
    }
    let total_mb = gpu::device()
        .and_then(|d| d.memory_info().ok())
        .map(|m| (m.total >> 20) as u32);
    if let (Some(used), Some(total)) = (summary.memory_used_peak_mb, total_mb)
        && used * 4 > total
    {
        warnings.push(format!(
            "Other programs hold {:.1} GB of the GPU's {:.0} GB of memory.",
            used as f64 / 1024.0,
            total as f64 / 1024.0
        ));
    }
    warnings
}

/// ", used by a.exe, b.exe": the processes that ran kernels on the NVIDIA GPU in the last few
/// seconds, busiest first. (Every desktop app has a context on it, so having one says little;
/// and on Windows the driver can't say how much memory each process holds.)
fn gpu_users() -> String {
    let Some(samples) = gpu::device().and_then(|d| d.process_utilization_stats(None).ok()) else {
        return String::new();
    };
    let mut busiest: Vec<(u32, u32)> = Vec::new();
    for sample in samples
        .iter()
        .filter(|s| s.sm_util > 0 && s.pid != std::process::id())
    {
        match busiest.iter_mut().find(|(pid, _)| *pid == sample.pid) {
            Some((_, util)) => *util = (*util).max(sample.sm_util),
            None => busiest.push((sample.pid, sample.sm_util)),
        }
    }
    busiest.sort_by_key(|&(_, util)| std::cmp::Reverse(util));
    #[cfg(windows)]
    let names: Vec<String> = busiest
        .iter()
        // Those it can't name are protected system processes (csrss.exe).
        .filter_map(|&(pid, _)| windows::process_name(pid))
        .collect();
    #[cfg(not(windows))]
    let names: Vec<String> = busiest
        .iter()
        .map(|(pid, _)| format!("pid {pid}"))
        .collect();
    if names.is_empty() {
        String::new()
    } else {
        format!(", used by {}", names.join(", "))
    }
}

/// Waits until the GPU is at or below `target_c`, checking each second, for at most
/// `max_wait`. Returns the last temperature read (above the target if time ran out), or None
/// without an NVIDIA GPU.
pub fn wait_for_cool(target_c: u32, max_wait: Duration) -> Option<u32> {
    let device = gpu::device()?;
    let start = Instant::now();
    loop {
        let temperature = device.temperature(TemperatureSensor::Gpu).ok()?;
        if temperature <= target_c || start.elapsed() >= max_wait {
            return Some(temperature);
        }
        std::thread::sleep(Duration::from_secs(1).min(max_wait.saturating_sub(start.elapsed())));
    }
}
