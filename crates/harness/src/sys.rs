//! What the operating system says about this process.

/// Peak resident memory so far, in bytes (0 when unknown).
pub fn peak_rss() -> u64 {
    imp::peak_rss()
}

#[cfg(windows)]
mod imp {
    use windows_sys::Win32::System::{
        ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS},
        Threading::GetCurrentProcess,
    };

    pub fn peak_rss() -> u64 {
        // SAFETY: a zeroed counters struct with its size set is what the call expects, and the
        // pseudo-handle of the current process needs no closing.
        unsafe {
            let mut counters: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
            counters.cb = size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
            if GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) != 0 {
                counters.PeakWorkingSetSize as u64
            } else {
                0
            }
        }
    }
}

#[cfg(unix)]
mod imp {
    pub fn peak_rss() -> u64 {
        // SAFETY: getrusage fills the struct it's given.
        let usage = unsafe {
            let mut usage: libc::rusage = std::mem::zeroed();
            if libc::getrusage(libc::RUSAGE_SELF, &mut usage) != 0 {
                return 0;
            }
            usage
        };
        let max = usage.ru_maxrss.max(0) as u64;
        // Kilobytes on Linux, bytes on macOS.
        if cfg!(target_os = "macos") {
            max
        } else {
            max * 1024
        }
    }
}

#[cfg(not(any(windows, unix)))]
mod imp {
    pub fn peak_rss() -> u64 {
        0
    }
}
