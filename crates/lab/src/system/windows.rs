//! What Windows says about the machine: OS and CPU from the registry, memory and cores from
//! kernel32, the GPUs DXGI enumerates, the power source, and the power plan and mode.

use super::Gpu;
use std::process::Command;
use windows::Win32::Foundation::{CloseHandle, ERROR_SUCCESS};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, IDXGIDevice, IDXGIFactory1,
};
use windows::Win32::System::Power::{GetSystemPowerStatus, SYSTEM_POWER_STATUS};
use windows::Win32::System::Registry::{
    HKEY_LOCAL_MACHINE, RRF_RT_REG_DWORD, RRF_RT_REG_SZ, RegGetValueW,
};
use windows::Win32::System::SystemInformation::{
    GetLogicalProcessorInformationEx, GlobalMemoryStatusEx, MEMORYSTATUSEX, RelationProcessorCore,
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
};
use windows::Win32::System::Threading::{
    OpenProcess, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION, QueryFullProcessImageNameW,
};
use windows::core::{HSTRING, Interface, PWSTR};

const CURRENT_VERSION: &str = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion";
const POWER_SCHEMES: &str = r"SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes";

fn registry_string(key: &str, value: &str) -> Option<String> {
    let mut buffer = [0u16; 512];
    let mut size = (buffer.len() * 2) as u32;
    let status = unsafe {
        RegGetValueW(
            HKEY_LOCAL_MACHINE,
            &HSTRING::from(key),
            &HSTRING::from(value),
            RRF_RT_REG_SZ,
            None,
            Some(buffer.as_mut_ptr().cast()),
            Some(&mut size),
        )
    };
    (status == ERROR_SUCCESS).then(|| wide(&buffer).trim().to_string())
}

fn registry_dword(key: &str, value: &str) -> Option<u32> {
    let mut data = 0u32;
    let mut size = 4u32;
    let status = unsafe {
        RegGetValueW(
            HKEY_LOCAL_MACHINE,
            &HSTRING::from(key),
            &HSTRING::from(value),
            RRF_RT_REG_DWORD,
            None,
            Some((&mut data as *mut u32).cast()),
            Some(&mut size),
        )
    };
    (status == ERROR_SUCCESS).then_some(data)
}

/// A NUL-terminated UTF-16 buffer as a string.
fn wide(buffer: &[u16]) -> String {
    let end = buffer.iter().position(|&c| c == 0).unwrap_or(buffer.len());
    String::from_utf16_lossy(&buffer[..end])
}

/// The OS name, version (e.g. "24H2") and build (e.g. "26100.9974").
pub fn os() -> (String, String, String) {
    let build = registry_string(CURRENT_VERSION, "CurrentBuildNumber").unwrap_or_default();
    let mut name = registry_string(CURRENT_VERSION, "ProductName").unwrap_or("Windows".into());
    // Windows 11 still calls itself Windows 10 here; the build tells them apart.
    if build.parse::<u32>().is_ok_and(|b| b >= 22000) {
        name = name.replacen("Windows 10", "Windows 11", 1);
    }
    let version = registry_string(CURRENT_VERSION, "DisplayVersion").unwrap_or_default();
    let full_build = match registry_dword(CURRENT_VERSION, "UBR") {
        Some(ubr) => format!("{build}.{ubr}"),
        None => build,
    };
    (name, version, full_build)
}

pub fn cpu() -> Option<String> {
    registry_string(
        r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        "ProcessorNameString",
    )
}

/// Physical cores: one `RelationProcessorCore` record each.
pub fn physical_cores() -> Option<u32> {
    let mut length = 0u32;
    let _ = unsafe { GetLogicalProcessorInformationEx(RelationProcessorCore, None, &mut length) };
    let mut buffer = vec![0u8; length as usize];
    unsafe {
        GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            Some(buffer.as_mut_ptr().cast()),
            &mut length,
        )
    }
    .ok()?;
    let (mut offset, mut cores) = (0usize, 0u32);
    while offset < length as usize {
        let record = unsafe {
            &*buffer
                .as_ptr()
                .add(offset)
                .cast::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>()
        };
        cores += 1;
        offset += record.Size as usize;
    }
    Some(cores)
}

pub fn ram_bytes() -> Option<u64> {
    let mut status = MEMORYSTATUSEX {
        dwLength: size_of::<MEMORYSTATUSEX>() as u32,
        ..Default::default()
    };
    unsafe { GlobalMemoryStatusEx(&mut status) }.ok()?;
    Some(status.ullTotalPhys)
}

/// The hardware adapters DXGI sees (not the software renderer). The driver version is the
/// user-mode driver's, e.g. 32.0.15.6156 for NVIDIA 616.56.
pub fn gpus() -> Vec<Gpu> {
    let Ok(factory) = (unsafe { CreateDXGIFactory1::<IDXGIFactory1>() }) else {
        return Vec::new();
    };
    let mut gpus = Vec::new();
    for index in 0.. {
        let Ok(adapter) = (unsafe { factory.EnumAdapters1(index) }) else {
            break;
        };
        let Ok(desc) = (unsafe { adapter.GetDesc1() }) else {
            continue;
        };
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
            continue;
        }
        let driver = unsafe { adapter.CheckInterfaceSupport(&IDXGIDevice::IID) }
            .ok()
            .map(|v| {
                let v = v as u64;
                format!(
                    "{}.{}.{}.{}",
                    v >> 48,
                    (v >> 32) & 0xffff,
                    (v >> 16) & 0xffff,
                    v & 0xffff
                )
            });
        let gpu = Gpu {
            name: wide(&desc.Description).trim().to_string(),
            vendor: vendor(desc.VendorId),
            driver,
            vram_bytes: Some(desc.DedicatedVideoMemory as u64),
        };
        // A hybrid laptop's iGPU can be listed twice, once per display path.
        if !gpus.contains(&gpu) {
            gpus.push(gpu);
        }
    }
    gpus
}

fn vendor(id: u32) -> String {
    match id {
        0x10de => "NVIDIA".into(),
        0x8086 => "Intel".into(),
        0x1002 | 0x1022 => "AMD".into(),
        0x5143 => "Qualcomm".into(),
        0x1414 => "Microsoft".into(),
        other => format!("{other:#06x}"),
    }
}

/// Whether the machine is on AC power, and the battery charge (None without a battery).
pub fn power() -> (Option<bool>, Option<u8>) {
    let mut status = SYSTEM_POWER_STATUS::default();
    if unsafe { GetSystemPowerStatus(&mut status) }.is_err() {
        return (None, None);
    }
    let on_ac = match status.ACLineStatus {
        0 => Some(false),
        1 => Some(true),
        _ => None,
    };
    // Flag 128: no system battery. Percent 255: unknown.
    let battery = (status.BatteryFlag & 128 == 0 && status.BatteryLifePercent <= 100)
        .then_some(status.BatteryLifePercent);
    (on_ac, battery)
}

/// The well-known power plans; a custom plan has its own GUID and only its name to go by.
pub const POWER_SAVER: &str = "a1841308-3541-4fab-bc81-f71556f20b4a";

/// The active power plan as `powercfg /getactivescheme` prints it: its GUID and name.
pub fn power_plan() -> Option<(String, String)> {
    let output = Command::new("powercfg")
        .arg("/getactivescheme")
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    // "Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced)"; the label is
    // localized, so find the GUID by shape.
    let line = text.lines().find(|l| l.contains('('))?;
    let guid = line
        .split_whitespace()
        .find(|w| w.len() == 36 && w.matches('-').count() == 4)?
        .to_lowercase();
    let name = line[line.find('(')? + 1..line.rfind(')')?].to_string();
    Some((guid, name))
}

/// The Windows 11 power mode slider (an overlay on the plan) for the current power source.
pub fn power_mode(on_ac: bool) -> Option<String> {
    let value = if on_ac {
        "ActiveOverlayAcPowerScheme"
    } else {
        "ActiveOverlayDcPowerScheme"
    };
    let guid = registry_string(POWER_SCHEMES, value)?.to_lowercase();
    Some(
        match guid.as_str() {
            "961cc777-2547-4f9d-8174-7d86181b8a7a" => "Best power efficiency",
            "00000000-0000-0000-0000-000000000000" => "Balanced",
            "3af9b8d9-7c97-431d-ad78-34a8bfea439f" => "Better performance",
            "ded574b5-45a0-4f42-8737-46345c09c238" => "Best performance",
            other => return Some(other.to_string()),
        }
        .to_string(),
    )
}

/// The plan's maximum processor state for the current power source, in percent: a plan that
/// caps the CPU below 100% slows the CPU side of every engine.
pub fn max_processor_state(on_ac: bool) -> Option<u32> {
    let output = Command::new("powercfg")
        .args([
            "/query",
            "SCHEME_CURRENT",
            "SUB_PROCESSOR",
            "PROCTHROTTLEMAX",
        ])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    // The last two "…: 0x00000064" lines are the AC and then the DC value.
    let values: Vec<u32> = text
        .lines()
        .filter_map(|l| l.trim().rsplit_once(": 0x"))
        .filter_map(|(_, hex)| u32::from_str_radix(hex.trim(), 16).ok())
        .collect();
    let [.., ac, dc] = values[..] else {
        return None;
    };
    Some(if on_ac { ac } else { dc })
}

/// A process's executable name, e.g. "vellum.exe".
pub fn process_name(pid: u32) -> Option<String> {
    let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) }.ok()?;
    let mut buffer = [0u16; 1024];
    let mut size = buffer.len() as u32;
    let result = unsafe {
        QueryFullProcessImageNameW(
            handle,
            PROCESS_NAME_WIN32,
            PWSTR(buffer.as_mut_ptr()),
            &mut size,
        )
    };
    let _ = unsafe { CloseHandle(handle) };
    result.ok()?;
    let path = String::from_utf16_lossy(&buffer[..size as usize]);
    path.rsplit('\\').next().map(str::to_string)
}
