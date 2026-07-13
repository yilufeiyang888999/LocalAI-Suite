use std::{path::PathBuf, process::Command};

use serde::{Deserialize, Serialize};
use sysinfo::{Disks, System};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub gpu_list: Vec<GpuInfo>,
    pub gpu_name: String,
    pub gpu_vram_gb: f64,
    pub gpu_count: usize,
    pub has_gpu: bool,
    pub cpu: CpuInfo,
    pub memory: MemoryInfo,
    pub disk: DiskInfo,
    pub backend: BackendType,
    pub needs_manual_tier: bool,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub index: usize,
    pub vendor: GpuVendor,
    pub name: String,
    pub vram_gb: f64,
    pub driver_version: String,
    /// Memory bandwidth (GB/s) looked up from the model name, used for speed
    /// estimation. `None` when the card isn't in the table.
    pub bandwidth_gbs: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuVendor {
    NVIDIA,
    AMD,
    Intel,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackendType {
    CUDA,
    CPUx86,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    pub name: String,
    pub cores: usize,
    pub threads: usize,
    pub arch: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub total_gb: f64,
    pub available_gb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskInfo {
    pub install_path: String,
    pub available_gb: f64,
}

#[tauri::command]
pub async fn get_hardware_info() -> HardwareInfo {
    tokio::task::spawn_blocking(detect_hardware)
        .await
        .unwrap_or_else(|_| fallback_hardware())
}

/// Primary outbound LAN IPv4 of this machine, shown so users can hand the
/// service address to other computers when LAN access is enabled. Uses the
/// UDP-connect trick: a connected UDP socket picks the outbound interface
/// without sending any packet. Returns `None` if it can't be determined.
#[tauri::command]
pub fn local_ip() -> Option<String> {
    use std::net::UdpSocket;
    let socket = UdpSocket::bind("0.0.0.0:0").ok()?;
    // 10.255.255.255 is a non-routable target used only to select an interface.
    socket.connect("10.255.255.255:1").ok()?;
    let ip = socket.local_addr().ok()?.ip();
    if ip.is_loopback() || ip.is_unspecified() {
        None
    } else {
        Some(ip.to_string())
    }
}

pub(crate) fn detect_hardware() -> HardwareInfo {
    let mut system = System::new_all();
    system.refresh_all();

    let cpu_name = system
        .cpus()
        .first()
        .map(|cpu| cpu.brand().to_string())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "暂未识别到CPU型号".to_string());
    let cpu = CpuInfo {
        name: cpu_name,
        cores: system
            .physical_core_count()
            .unwrap_or_else(|| system.cpus().len()),
        threads: system.cpus().len(),
        arch: std::env::consts::ARCH.to_string(),
    };

    let total_gb = bytes_to_gb(system.total_memory());
    let available_gb = bytes_to_gb(system.available_memory());
    let memory = MemoryInfo {
        total_gb,
        available_gb,
    };

    let install_path = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("LocalAI-Installer");
    let disk = detect_disk(&install_path);
    let nvidia_gpus = detect_nvidia_gpus();
    // Acceleration availability + sizing come only from nvidia-smi, which
    // reports accurate VRAM. WMI's AdapterRAM is capped at ~4GB, so it is used
    // only to surface non-NVIDIA (AMD/Intel) cards for display.
    let has_nvidia = nvidia_gpus.iter().any(|gpu| gpu.vram_gb > 0.0);
    let gpu_vram_gb = nvidia_gpus
        .iter()
        .map(|gpu| gpu.vram_gb)
        .fold(0.0_f64, f64::max);
    let gpu_list = merge_gpus(nvidia_gpus, detect_gpus_via_wmi());
    let has_other_gpu = !gpu_list.is_empty();
    let gpu_name = gpu_list
        .iter()
        .find(|gpu| matches!(gpu.vendor, GpuVendor::NVIDIA))
        .or_else(|| gpu_list.first())
        .map(|gpu| gpu.name.clone())
        .unwrap_or_else(|| "未检测到独立显卡".to_string());
    // `has_gpu` means "GPU acceleration is usable"; MVP only auto-uses NVIDIA,
    // so AMD/Intel-only machines report CPU mode (but their card is still shown).
    let has_gpu = has_nvidia;
    let backend = if has_nvidia {
        BackendType::CUDA
    } else {
        BackendType::CPUx86
    };
    let needs_manual_tier = !has_nvidia || gpu_vram_gb <= 0.0;
    let summary = if has_nvidia {
        format!(
            "你的电脑是 {}，约 {:.1}GB 显卡内存，可以优先尝试显卡加速运行本地AI。",
            gpu_name, gpu_vram_gb
        )
    } else if has_other_gpu {
        format!(
            "检测到 {}，但当前版本只自动支持 NVIDIA 显卡加速，将以 CPU 模式运行，速度会较慢。",
            gpu_name
        )
    } else {
        "未检测到独立显卡，运行速度会较慢，适合简单问答。".to_string()
    };

    tracing::info!(
        backend = ?backend,
        gpu = %gpu_name,
        gpu_vram_gb,
        gpu_count = gpu_list.len(),
        has_gpu,
        cpu_cores = cpu.cores,
        memory_gb = memory.total_gb,
        "hardware detected"
    );

    HardwareInfo {
        gpu_count: gpu_list.len(),
        gpu_list,
        gpu_name,
        gpu_vram_gb,
        has_gpu,
        cpu,
        memory,
        disk,
        backend,
        needs_manual_tier,
        summary,
    }
}

fn detect_disk(install_path: &PathBuf) -> DiskInfo {
    let disks = Disks::new_with_refreshed_list();
    let available = disks
        .iter()
        .find(|disk| install_path.starts_with(disk.mount_point()))
        .or_else(|| disks.iter().next())
        .map(|disk| bytes_to_gb(disk.available_space()))
        .unwrap_or(0.0);

    DiskInfo {
        install_path: install_path.to_string_lossy().to_string(),
        available_gb: available,
    }
}

fn detect_nvidia_gpus() -> Vec<GpuInfo> {
    let mut command = Command::new("nvidia-smi");
    command.args([
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]);
    crate::proc::hide_console(&mut command);
    let output = command.output();

    let Ok(output) = output else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            let parts: Vec<_> = line.split(',').map(|part| part.trim()).collect();
            if parts.len() < 4 {
                return None;
            }
            let index = parts[0].parse::<usize>().ok()?;
            let memory_mb = parts[2].parse::<f64>().ok()?;
            let name = parts[1].to_string();
            Some(GpuInfo {
                index,
                bandwidth_gbs: lookup_bandwidth(&GpuVendor::NVIDIA, &name),
                vendor: GpuVendor::NVIDIA,
                name,
                vram_gb: memory_mb / 1024.0,
                driver_version: parts[3].to_string(),
            })
        })
        .collect()
}

// Memory bandwidth (GB/s) by GPU model, per tech doc §3.1.4. Keys are matched
// case-insensitively as substrings of the adapter name; the longest matching key
// wins so "RTX 4080 Super" isn't shadowed by "RTX 4080".
const NVIDIA_BANDWIDTH: &[(&str, u32)] = &[
    ("RTX 5090", 1792), ("RTX 5080", 960), ("RTX 5070 Ti", 896), ("RTX 5070", 672),
    ("RTX 4090", 1008), ("RTX 4080 Super", 736), ("RTX 4080", 717), ("RTX 4070 Ti Super", 672),
    ("RTX 4070 Ti", 504), ("RTX 4070 Super", 504), ("RTX 4070", 504), ("RTX 4060 Ti", 288),
    ("RTX 4060", 272),
    ("RTX 3090 Ti", 1008), ("RTX 3090", 936), ("RTX 3080 Ti", 912), ("RTX 3080", 760),
    ("RTX 3070 Ti", 608), ("RTX 3070", 448), ("RTX 3060 Ti", 448), ("RTX 3060", 360),
    ("RTX 3050", 224),
    ("RTX 2080 Ti", 616), ("RTX 2080 Super", 496), ("RTX 2080", 448), ("RTX 2070 Super", 448),
    ("RTX 2070", 448), ("RTX 2060 Super", 448), ("RTX 2060", 336),
    ("GTX 1660 Ti", 288), ("GTX 1660 Super", 336), ("GTX 1660", 192), ("GTX 1650", 192),
];

// AMD GPU 显存带宽表（GB/s）。
const AMD_BANDWIDTH: &[(&str, u32)] = &[
    ("RX 7900 XTX", 960), ("RX 7900 XT", 800), ("RX 7800 XT", 624),
    ("RX 6950 XT", 576), ("RX 6900 XT", 512), ("RX 6800 XT", 512),
];

// Apple Silicon 统一内存带宽（GB/s）。
const APPLE_BANDWIDTH: &[(&str, u32)] = &[
    ("M1 Ultra", 800), ("M1 Max", 400), ("M1 Pro", 200),
    ("M2 Ultra", 800), ("M2 Max", 400), ("M2 Pro", 200),
    ("M3 Ultra", 800), ("M3 Max", 400), ("M3 Pro", 150),
];

/// Look up memory bandwidth (GB/s) for a GPU by matching its adapter name against
/// the vendor's bandwidth table. Returns the value for the longest key that is a
/// case-insensitive substring of `name`, or `None` when nothing matches.
fn lookup_bandwidth(vendor: &GpuVendor, name: &str) -> Option<u32> {
    let table: &[(&str, u32)] = match vendor {
        GpuVendor::NVIDIA => NVIDIA_BANDWIDTH,
        GpuVendor::AMD => AMD_BANDWIDTH,
        // Apple Silicon shows up as "Unknown" via generic detection; still try it.
        GpuVendor::Unknown => APPLE_BANDWIDTH,
        GpuVendor::Intel => return None,
    };
    let lower = name.to_ascii_lowercase();
    table
        .iter()
        .filter(|(key, _)| lower.contains(&key.to_ascii_lowercase()))
        .max_by_key(|(key, _)| key.len())
        .map(|(_, bandwidth)| *bandwidth)
}

/// Rough decode-speed estimate (tokens/sec) for a memory-bandwidth-bound LLM.
/// Each token reads the resident weights once, so `tok/s ≈ bandwidth / weight_bytes`
/// scaled by a real-world efficiency factor. `weight_gb` should be the resident
/// weight size (GGUF file size for dense models, active fraction for MoE).
pub fn estimate_tokens_per_second(bandwidth_gbs: u32, weight_gb: f64) -> f64 {
    if bandwidth_gbs == 0 || weight_gb <= 0.0 {
        return 0.0;
    }
    // Real llama.cpp decode reaches ~70% of the theoretical roofline in practice.
    const EFFICIENCY: f64 = 0.7;
    f64::from(bandwidth_gbs) / weight_gb * EFFICIENCY
}

/// True for virtual / remote-desktop display adapters that aren't real GPUs
/// (e.g. Sunlogin/Oray IDD, Parsec, spacedesk, the Microsoft Basic fallback).
/// These are enumerated by WMI but must not be shown to the user as their
/// graphics card or counted toward acceleration.
fn is_virtual_adapter(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    const VIRTUAL_KEYWORDS: &[&str] = &[
        "iddriver",
        "idd device",
        "indirect display",
        "oray",
        "sunlogin",
        "todesk",
        "rustdesk",
        "anydesk",
        "parsec",
        "spacedesk",
        "splashtop",
        "displaylink",
        "usb display",
        "remote display",
        "mirror driver",
        "virtual display",
        "virtual monitor",
        "vmware",
        "virtualbox",
        "citrix",
        "vnc",
        "microsoft basic display",
        "microsoft remote display",
        "basic render",
    ];
    VIRTUAL_KEYWORDS.iter().any(|kw| lower.contains(kw))
}

/// Classify a GPU by the vendor keywords in its adapter name.
fn classify_vendor(name: &str) -> GpuVendor {
    let lower = name.to_ascii_lowercase();
    if ["nvidia", "geforce", "rtx", "gtx", "quadro", "tesla"]
        .iter()
        .any(|kw| lower.contains(kw))
    {
        GpuVendor::NVIDIA
    } else if ["amd", "radeon", "firepro"].iter().any(|kw| lower.contains(kw)) {
        GpuVendor::AMD
    } else if ["intel", "arc", "iris", "uhd graphics", "hd graphics"]
        .iter()
        .any(|kw| lower.contains(kw))
    {
        GpuVendor::Intel
    } else {
        GpuVendor::Unknown
    }
}

/// Merge the authoritative nvidia-smi list with WMI-discovered controllers,
/// dropping WMI duplicates (by name) and re-indexing the result. nvidia-smi
/// entries win because their VRAM is accurate.
fn merge_gpus(mut primary: Vec<GpuInfo>, secondary: Vec<GpuInfo>) -> Vec<GpuInfo> {
    for gpu in secondary {
        let duplicate = primary
            .iter()
            .any(|existing| existing.name.eq_ignore_ascii_case(&gpu.name));
        if !duplicate {
            primary.push(gpu);
        }
    }
    for (index, gpu) in primary.iter_mut().enumerate() {
        gpu.index = index;
    }
    primary
}

/// Enumerate every display adapter via WMI (`Win32_VideoController`). Best-effort:
/// any COM/query failure yields an empty list so hardware detection never fails.
#[cfg(windows)]
fn detect_gpus_via_wmi() -> Vec<GpuInfo> {
    use serde::Deserialize;
    use wmi::{COMLibrary, WMIConnection};

    #[derive(Deserialize)]
    #[serde(rename_all = "PascalCase")]
    struct VideoController {
        name: Option<String>,
        #[serde(rename = "AdapterRAM")]
        adapter_ram: Option<u32>,
        driver_version: Option<String>,
    }

    let Ok(com) = COMLibrary::new() else {
        return Vec::new();
    };
    let Ok(connection) = WMIConnection::new(com) else {
        return Vec::new();
    };
    let query = "SELECT Name, AdapterRAM, DriverVersion FROM Win32_VideoController";
    let Ok(rows) = connection.raw_query::<VideoController>(query) else {
        return Vec::new();
    };

    rows.into_iter()
        .enumerate()
        .filter_map(|(index, row)| {
            let name = row.name?.trim().to_string();
            if name.is_empty() {
                return None;
            }
            // Skip virtual / remote-desktop display drivers so they aren't shown
            // as the user's graphics card.
            if is_virtual_adapter(&name) {
                return None;
            }
            let vram_gb = row
                .adapter_ram
                .map(|bytes| bytes as f64 / 1024.0 / 1024.0 / 1024.0)
                .unwrap_or(0.0);
            let vendor = classify_vendor(&name);
            Some(GpuInfo {
                index,
                bandwidth_gbs: lookup_bandwidth(&vendor, &name),
                vendor,
                name,
                vram_gb,
                driver_version: row.driver_version.unwrap_or_default(),
            })
        })
        .collect()
}

#[cfg(not(windows))]
fn detect_gpus_via_wmi() -> Vec<GpuInfo> {
    Vec::new()
}

pub(crate) fn fallback_hardware() -> HardwareInfo {
    HardwareInfo {
        gpu_list: Vec::new(),
        gpu_name: "暂未识别到显卡".to_string(),
        gpu_vram_gb: 0.0,
        gpu_count: 0,
        has_gpu: false,
        cpu: CpuInfo {
            name: "暂未识别到CPU型号".to_string(),
            cores: 0,
            threads: 0,
            arch: std::env::consts::ARCH.to_string(),
        },
        memory: MemoryInfo {
            total_gb: 0.0,
            available_gb: 0.0,
        },
        disk: DiskInfo {
            install_path: ".".to_string(),
            available_gb: 0.0,
        },
        backend: BackendType::CPUx86,
        needs_manual_tier: true,
        summary: "暂时没能识别你的电脑配置，可以手动选择硬件档位继续。".to_string(),
    }
}

fn bytes_to_gb(bytes: u64) -> f64 {
    bytes as f64 / 1024.0 / 1024.0 / 1024.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(name: &str, vram_gb: f64) -> GpuInfo {
        let vendor = classify_vendor(name);
        GpuInfo {
            index: 0,
            bandwidth_gbs: lookup_bandwidth(&vendor, name),
            vendor,
            name: name.to_string(),
            vram_gb,
            driver_version: String::new(),
        }
    }

    #[test]
    fn vendor_classification_from_name() {
        assert!(matches!(classify_vendor("NVIDIA GeForce RTX 3060"), GpuVendor::NVIDIA));
        assert!(matches!(classify_vendor("AMD Radeon RX 6800"), GpuVendor::AMD));
        assert!(matches!(classify_vendor("Intel(R) Arc(TM) A770"), GpuVendor::Intel));
        assert!(matches!(classify_vendor("Intel(R) UHD Graphics 630"), GpuVendor::Intel));
        assert!(matches!(classify_vendor("Some Virtual Display"), GpuVendor::Unknown));
    }

    #[test]
    fn virtual_adapters_are_detected_but_real_gpus_are_not() {
        // Remote-desktop / virtual display drivers must be filtered out.
        assert!(is_virtual_adapter("OrayIddDriver Device"));
        assert!(is_virtual_adapter("Parsec Virtual Display Adapter"));
        assert!(is_virtual_adapter("spacedesk Graphics Adapter"));
        assert!(is_virtual_adapter("Microsoft Basic Display Adapter"));
        assert!(is_virtual_adapter("ToDesk Virtual Display"));
        // Real GPUs must never be treated as virtual.
        assert!(!is_virtual_adapter("NVIDIA GeForce RTX 4060 Laptop GPU"));
        assert!(!is_virtual_adapter("AMD Radeon RX 7900 XTX"));
        assert!(!is_virtual_adapter("Intel(R) UHD Graphics 630"));
    }

    #[test]
    fn merge_prefers_nvidia_smi_and_dedups_by_name() {
        // nvidia-smi entry (accurate VRAM) plus WMI listing the same NVIDIA card
        // and a separate integrated Intel GPU.
        let nvidia = vec![gpu("NVIDIA GeForce RTX 3060", 12.0)];
        let wmi = vec![
            gpu("NVIDIA GeForce RTX 3060", 4.0), // WMI-capped duplicate, must drop
            gpu("Intel(R) UHD Graphics 630", 0.0),
        ];

        let merged = merge_gpus(nvidia, wmi);

        assert_eq!(merged.len(), 2);
        // The NVIDIA entry keeps its accurate 12GB (nvidia-smi wins).
        assert_eq!(merged[0].name, "NVIDIA GeForce RTX 3060");
        assert_eq!(merged[0].vram_gb, 12.0);
        // The Intel card is added and indices are contiguous.
        assert!(matches!(merged[1].vendor, GpuVendor::Intel));
        assert_eq!(merged[0].index, 0);
        assert_eq!(merged[1].index, 1);
    }

    #[test]
    fn merge_surfaces_non_nvidia_when_no_nvidia_present() {
        let merged = merge_gpus(Vec::new(), vec![gpu("AMD Radeon RX 6800", 0.0)]);
        assert_eq!(merged.len(), 1);
        assert!(matches!(merged[0].vendor, GpuVendor::AMD));
    }

    #[test]
    fn bandwidth_lookup_matches_longest_key() {
        // Full adapter names as reported by nvidia-smi / WMI.
        assert_eq!(
            lookup_bandwidth(&GpuVendor::NVIDIA, "NVIDIA GeForce RTX 4060 Laptop GPU"),
            Some(272)
        );
        // "RTX 4080 Super" must not be shadowed by the shorter "RTX 4080" key.
        assert_eq!(
            lookup_bandwidth(&GpuVendor::NVIDIA, "NVIDIA GeForce RTX 4080 SUPER"),
            Some(736)
        );
        assert_eq!(
            lookup_bandwidth(&GpuVendor::NVIDIA, "NVIDIA GeForce RTX 4080"),
            Some(717)
        );
        assert_eq!(lookup_bandwidth(&GpuVendor::AMD, "AMD Radeon RX 7900 XTX"), Some(960));
        // Unknown card / Intel iGPU have no table entry.
        assert_eq!(lookup_bandwidth(&GpuVendor::NVIDIA, "NVIDIA H100"), None);
        assert_eq!(lookup_bandwidth(&GpuVendor::Intel, "Intel Arc A770"), None);
    }

    #[test]
    fn tokens_per_second_scales_with_bandwidth_and_size() {
        // ~4GB weights on a 448 GB/s card: 448/4*0.7 ≈ 78 t/s.
        let fast = estimate_tokens_per_second(448, 4.0);
        assert!((70.0..90.0).contains(&fast), "unexpected estimate {fast}");
        // Same card, a heavier 8GB model must be slower.
        let slow = estimate_tokens_per_second(448, 8.0);
        assert!(slow < fast);
        // Missing inputs degrade to zero rather than dividing by zero.
        assert_eq!(estimate_tokens_per_second(0, 4.0), 0.0);
        assert_eq!(estimate_tokens_per_second(448, 0.0), 0.0);
    }
}
