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
    /// 硬件探测推荐的**首选**后端，用来决定「装哪个引擎」（`App.vue::prepareEngineAndStart`）。
    ///
    /// 🔴 **不是**服务当前实际加载的后端 —— 那个只认 `starting llama-server … backend=`
    /// 或 `显存对账 … backend=`。名字保留 `backend` 是为了不打断已发布的跨语言边界
    /// （改字段名前端会静默读不到，且不报错）。
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
    Vulkan,
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
    // `has_gpu` means "GPU acceleration is usable". With Vulkan as the default
    // GPU backend we can drive any real adapter (NVIDIA/AMD/Intel; virtual/remote
    // displays already filtered out), so any detected GPU enables acceleration.
    let has_gpu = has_other_gpu;
    let backend = select_backend(&gpu_list);
    // Reliable VRAM only comes from nvidia-smi. When it's unknown (no NVIDIA card
    // or read failed) we ask the user to pick a hardware tier so recommendations
    // don't over-reach on an unknown AMD/Intel card.
    let needs_manual_tier = gpu_vram_gb <= 0.0;
    let summary = if has_nvidia {
        format!(
            "你的电脑是 {}，约 {:.1}GB 显卡内存，可以优先尝试显卡加速运行本地AI。",
            gpu_name, gpu_vram_gb
        )
    } else if has_other_gpu {
        format!(
            "检测到 {}，将尝试通用显卡加速运行本地AI；如果加速用不上会自动改用更稳妥的 CPU 模式。",
            gpu_name
        )
    } else {
        "未检测到独立显卡，运行速度会较慢，适合简单问答。".to_string()
    };

    // 🔴 `preferred_backend` 的语义：**硬件探测给出的首选后端**（决定装哪个
    // 引擎），**不是本轮服务实际加载的后端**。后者由 service.rs 打印：
    //   `starting llama-server … backend=` / `显存对账 … backend=`
    //
    // 早先这个字段叫 `backend`，但它在本机跑 Vulkan 的那一轮**仍然报 CUDA**
    // （2026-10-18 B-4 实跑：两轮的后端切换是真的，但这一行没跟着变）。
    // 两个 `backend` 语义不同、名字相同 ⇒ 拿这一行判「本轮实际后端」会得到
    // 「两轮都跑 CUDA」的**假阴性**，进而把一次有效校验判成无效。
    tracing::info!(
        preferred_backend = ?backend,
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

/// Query the free VRAM (in GB) of the primary NVIDIA GPU (index 0).
///
/// Returns `None` when nvidia-smi is unavailable or the output can't be parsed.
/// This is used by `cuda_launch_params` to compute offload parameters based on
/// *actual* free VRAM rather than the total VRAM stored in `GpuInfo.vram_gb`,
/// preventing the "pretend success via shared-memory spill" scenario where
/// llama.cpp overcommits VRAM and silently degrades.
pub fn nvidia_free_vram_gb() -> Option<f64> {
    let mut command = Command::new("nvidia-smi");
    command.args([
        "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
    ]);
    crate::proc::hide_console(&mut command);
    let output = command.output().ok()?;
    if !output.status.success() {
        return None;
    }
    parse_free_vram_gb(&String::from_utf8_lossy(&output.stdout))
}

/// Parse the free-VRAM (GB) of GPU 0 from `nvidia-smi --query-gpu=memory.free`
/// output. Handles both the `nounits` form ("1234") and a stray " MiB" suffix.
/// Pure function so the parsing is unit-testable without invoking nvidia-smi.
fn parse_free_vram_gb(stdout: &str) -> Option<f64> {
    let line = stdout.lines().next()?;
    let num_str = line.trim().trim_end_matches("MiB").trim();
    let free_mib = num_str.parse::<f64>().ok()?;
    Some(free_mib / 1024.0)
}

/// 跑一条 `nvidia-smi --query-gpu=<查询>`，返回去掉表头与首尾空白后的**首行**。
///
/// 拿不到（没有 nvidia-smi / 无 N 卡 / 字段不被该驱动版本支持）返回 `None`。
/// 注意 `driver_model.current`、`compute_cap` 这两个字段在旧驱动上未必存在，
/// 所以调用方必须容忍 `None`，不能把它当成「一定拿得到」。
fn query_nvidia_smi(查询: &str) -> Option<String> {
    let 参数 = format!("--query-gpu={}", 查询);
    let mut command = Command::new("nvidia-smi");
    command.args([参数.as_str(), "--format=csv,noheader,nounits"]);
    crate::proc::hide_console(&mut command);
    let output = command.output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// 主 N 卡的驱动模型，取值 `WDDM` 或 `TCC`。
///
/// **为什么需要它**（2026-09-30 P100 实测教训）：`TCC`（Tesla Compute Cluster）
/// 模式把显卡从显示子系统里摘出去了 —— 它是**纯计算模式、没有显示输出能力**。
/// 而 **Vulkan 是图形 API，设备枚举要求适配器具备显示/呈现能力**，于是 TCC 卡上
/// Vulkan **枚举不到任何设备**：`-ngl` 无处可卸，全部层静默落 CPU。
///
/// 这个失败**不报任何错、只表现为慢一个数量级**（实测 8.62 t/s），比报错更难发现。
/// 所以选后端、以及做「后端 DLL 隔离」之前都要先看这一列：
/// **TCC 的卡只能走 CUDA 或 CPU，不要再考虑 Vulkan。**
pub fn nvidia_driver_model() -> Option<String> {
    query_nvidia_smi("driver_model.current")
}

/// 该值是否表示「计算专用模式」（TCC → Vulkan 不可用）。
///
/// 抽成纯函数便于单测；比较不区分大小写，容忍 nvidia-smi 的大小写差异。
pub fn is_tcc_mode(model: &str) -> bool {
    model.trim().eq_ignore_ascii_case("TCC")
}

/// 主 N 卡的算力（compute capability），如 `6.0`（P100）/ `12.0`（RTX 50 系）。
///
/// 用于把「引擎未包含你这张卡的架构」这类报错**按实际显卡代际**说明清楚，
/// 而不是笼统地猜一个型号。旧驱动上该字段可能不存在 → `None`，调用方需容忍。
pub fn nvidia_compute_cap() -> Option<String> {
    query_nvidia_smi("compute_cap")
}

/// 主 N 卡的型号名（如 `Tesla P100-PCIE-16GB`）。
///
/// 只用于把报错说清楚 —— 「你的显卡是 X（算力 Y）」比笼统的「当前显卡」
/// 可行动得多，尤其当引擎编入的架构档位和用户的卡对不上时。
pub fn primary_nvidia_name() -> Option<String> {
    query_nvidia_smi("name")
}

/// 主 N 卡**当前已用**显存（GiB）。
///
/// 用途：服务启动后校验「显卡加速是否真的生效」。权重已进显存时该值会跳到
/// 权重体积量级；静默退回 CPU 时它仍停在空载的个位数 MiB（实测 8 MiB）。
/// 这是唯一能抓住「不报错但也没上卡」的判据。
pub fn nvidia_used_vram_gb() -> Option<f64> {
    let 值 = query_nvidia_smi("memory.used")?;
    let mib = 值.trim().trim_end_matches("MiB").trim().parse::<f64>().ok()?;
    Some(mib / 1024.0)
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

/// Real-world efficiency factor for the roofline decode estimate, bucketed by
/// GPU generation **and** by how big the weights are.
///
/// `tok/s ≈ bandwidth / weight_bytes` is the *theoretical* ceiling; how close a
/// real llama.cpp build gets depends mostly on whether the silicon has tensor
/// (fp16/bf16) acceleration and how well the backend kernels are tuned for it.
/// A single constant cannot cover both ends:
///
/// | Class | Measured efficiency | Example |
/// |---|---|---|
/// | Modern, tensor-accelerated | 49–67% | RTX 4060 Laptop 8B-Q4: 35 t/s of 52 theoretical |
/// | Pascal, small weights (≤ 4 GiB) | ~26% | Tesla P100 1.7B-Q8: 111 t/s of 428 theoretical |
/// | Pascal, large weights (> 4 GiB) | ~18% | Tesla P100 27B-Q3: 11 t/s of 61 theoretical |
/// | AMD / Intel (Vulkan/Vulkan-only) | ~35% | conservative middle ground |
///
/// **Why Pascal splits in two** (2026-10-20, P100 / 732 GB/s, both measured):
/// 1.7B-Q8_0 (1.708 GiB) ran 110–113 t/s ⇒ 25.9%; 27B-UD-Q3_K_XL (11.92 GiB)
/// ran 10.5–11.4 t/s ⇒ 17.9%. Same card, same backend, same KV precision class —
/// the only change is the weight size. The small model spends proportionally
/// more of each token on fixed per-step work (sampling, logits, kernel launch)
/// that the roofline model doesn't account for, so it lands closer to the
/// ceiling. Using 0.20 for both understated the 1.7B by 1.3× — which is exactly
/// the model the VRAM planner cares most about on an 8–16 GiB card.
///
/// ⚠️ **单位口径**：这里的 `weight_gb` 是 **GiB**（与 `nvidia-smi` 一致），
/// 而 `bandwidth_gbs` 是十进制 GB/s。理论值 = `732 / 1.708 = 428`（不是
/// `732 / 1.834 = 399`）—— 系数必须按代码里的单位标定，否则会凭空多算 7%。
///
/// ⚠️ **Two data points only.** The 4 GiB threshold is a gap between them, not
/// a measured knee; anything in 2–10 GiB is interpolation. If a 4B/8B lands on
/// this card later, measure it before trusting the estimate.
///
/// The previous flat `0.7` matched modern consumer cards but overstated a
/// bandwidth-rich Pascal card by 3.8× — a P100 would be advertised at ~42 t/s
/// while actually delivering ~11. Every figure below is anchored to a measured
/// system, and `estimate_tokens_per_second` has regression tests pinning them.
fn efficiency_for(name: &str, vendor: &GpuVendor, weight_gb: f64) -> f64 {
    let lower = name.to_ascii_lowercase();

    // NVIDIA: tensor cores arrived with Volta (V100) and became broadly useful
    // for LLM inference from Turing onward. Pascal (P100/P40/GTX 10xx) has no
    // usable fp16 path at all, so decode stays close to the fp32 roofline.
    if matches!(vendor, GpuVendor::NVIDIA) {
        const PASCAL_OR_OLDER: &[&str] = &[
            "p100", "p40", "p4", "gtx 10", "gtx 9", "gtx 7", "tesla k", "tesla m",
            "quadro p", "mx1", "mx2", "mx3", "mx250", "mx330", "mx350",
        ];
        if PASCAL_OR_OLDER.iter().any(|kw| lower.contains(kw)) {
            // 4 GiB 是实测两点（1.708 / 11.92）之间的空档，不是测出来的拐点。
            // 系数按「weight 用 GiB」的口径标定：1.7B 111/428=0.26、27B 11/61=0.18。
            return if weight_gb <= 4.0 { 0.26 } else { 0.18 };
        }
        return 0.60;
    }

    // Apple Silicon shares unified memory and has a well-tuned Metal backend.
    if matches!(vendor, GpuVendor::Unknown) {
        return 0.55;
    }

    // AMD / Intel run the Vulkan backend, which is usable but less tuned than
    // the NVIDIA path.
    0.35
}

/// Rough decode-speed estimate (tokens/sec) for a memory-bandwidth-bound LLM.
/// Each token reads the resident weights once, so `tok/s ≈ bandwidth / weight_bytes`
/// scaled by a generation-dependent efficiency factor. `weight_gb` should be the
/// resident weight size (GGUF file size for dense models, active fraction for MoE).
///
/// `gpu_name` / `vendor` select the efficiency bucket; pass the adapter name and
/// vendor of the GPU that will actually serve the model. `weight_gb` is also
/// used to pick the Pascal sub-bucket (small vs large weights — see
/// `efficiency_for`).
pub fn estimate_tokens_per_second(
    bandwidth_gbs: u32,
    weight_gb: f64,
    gpu_name: &str,
    vendor: &GpuVendor,
) -> f64 {
    if bandwidth_gbs == 0 || weight_gb <= 0.0 {
        return 0.0;
    }
    f64::from(bandwidth_gbs) / weight_gb * efficiency_for(gpu_name, vendor, weight_gb)
}

/// Windows NVIDIA driver floor for the bundled CUDA 12.4 `llama.cpp` build. An
/// NVIDIA card whose driver meets this can take the faster native CUDA engine;
/// older NVIDIA drivers and all AMD/Intel cards fall back to the universal Vulkan
/// backend. CPU stays the safety net if the chosen GPU backend fails to launch
/// (see `service.rs`).
const CUDA_MIN_NVIDIA_DRIVER: f64 = 551.78;

/// Pick the engine backend from the detected GPUs: NVIDIA with a CUDA-capable
/// driver takes the CUDA fast lane, any other real GPU takes Vulkan, and no GPU
/// runs on CPU. Only nvidia-smi-sourced entries (`vram_gb > 0`) carry a
/// trustworthy NVIDIA branch driver version, so a WMI-only NVIDIA card (when
/// nvidia-smi is unavailable) conservatively falls back to Vulkan.
fn select_backend(gpus: &[GpuInfo]) -> BackendType {
    if gpus.is_empty() {
        return BackendType::CPUx86;
    }
    let cuda_capable = gpus.iter().any(|gpu| {
        matches!(gpu.vendor, GpuVendor::NVIDIA)
            && gpu.vram_gb > 0.0
            && parse_driver_version(&gpu.driver_version)
                .is_some_and(|version| version >= CUDA_MIN_NVIDIA_DRIVER)
    });
    if cuda_capable {
        BackendType::CUDA
    } else {
        BackendType::Vulkan
    }
}

/// Parse an NVIDIA driver version like `551.86` (or `551.86.01`) into a
/// `major.minor` float for threshold comparison. Returns `None` when the string
/// isn't a recognizable version (e.g. `n/a`).
pub(crate) fn parse_driver_version(raw: &str) -> Option<f64> {
    let raw = raw.trim();
    let mut parts = raw.split('.');
    let major = parts.next()?;
    let minor = parts.next().unwrap_or("0");
    format!("{}.{}", major, minor).parse::<f64>().ok()
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

/// 归一化适配器名字，用于**跨来源**去重。
///
/// `nvidia-smi` 报 `Tesla P100-PCIE-16GB`，WMI 报 `NVIDIA Tesla P100-PCIE-16GB` ——
/// 只差一个厂商前缀，按完整名字相等就判不出重复，同一张卡被数两次
/// （P100 虚机实测 `gpu_count=3`，实际只有 1 张；这是 P2-6「3 → 1 → 3 抖动」的
/// 真正来源，此前一直当成 WMI 不稳）。
///
/// 归一步骤：小写 → 去掉 `(R)` / `(TM)` 这类商标噪声 → 去掉厂商前缀 →
/// 只留字母数字。`NVIDIA Tesla P100-PCIE-16GB` 与 `Tesla P100-PCIE-16GB`
/// 因此都变成 `teslap100pcie16gb`。
///
/// ⚠️ 归一化后为空（名字全是符号）时**不能**据此判重 —— 那会把两个无名条目
/// 合成一个。判重那边保留了原始名相等的兜底。
fn 归一化适配器名(name: &str) -> String {
    let mut s = name.trim().to_ascii_lowercase();
    for 噪声 in ["(r)", "(tm)", "®", "™"] {
        s = s.replace(噪声, "");
    }
    for 前缀 in ["nvidia ", "amd ", "intel ", "advanced micro devices inc "] {
        if let Some(rest) = s.strip_prefix(前缀) {
            s = rest.to_string();
            break;
        }
    }
    s.retain(|c| c.is_ascii_alphanumeric());
    s
}

/// Merge the authoritative nvidia-smi list with WMI-discovered controllers,
/// dropping WMI duplicates (by name) and re-indexing the result. nvidia-smi
/// entries win because their VRAM is accurate.
fn merge_gpus(mut primary: Vec<GpuInfo>, secondary: Vec<GpuInfo>) -> Vec<GpuInfo> {
    for gpu in secondary {
        let 候选 = 归一化适配器名(&gpu.name);
        let duplicate = primary.iter().any(|existing| {
            // 归一化后空名字不参与判重（见 `归一化适配器名` 的备注）。
            let 现有 = 归一化适配器名(&existing.name);
            (!现有.is_empty() && 现有 == 候选) || existing.name.eq_ignore_ascii_case(&gpu.name)
        });
        if !duplicate {
            primary.push(gpu);
        }
    }
    for (index, gpu) in primary.iter_mut().enumerate() {
        gpu.index = index;
    }
    primary
}

/// **P2-6 纯函数**：本次 WMI 结果与「上次成功结果」之间取哪个。
///
/// 规则：**只增不减** —— 本次读到的条目比上次更少 ⇒ 视作 WMI 抖动，沿用上次。
/// 为什么可以这么做：这个列表只用于**展示非 NVIDIA 显卡**（加速可用性、显存大小
/// 都只认 nvidia-smi），少一条不该让界面上的显卡数量跳来跳去。
/// 代价：真换卡/拔卡要重启 APP 才反映 —— 对展示字段可接受。
#[cfg(windows)]
fn 取较长的适配器列表(上次: Option<&Vec<GpuInfo>>, 本次: Vec<GpuInfo>) -> Vec<GpuInfo> {
    match 上次 {
        Some(旧) if 旧.len() > 本次.len() => 旧.clone(),
        _ => 本次,
    }
}

/// Enumerate every display adapter via WMI (`Win32_VideoController`). Best-effort:
/// any COM/query failure yields an empty list so hardware detection never fails.
#[cfg(windows)]
fn detect_gpus_via_wmi() -> Vec<GpuInfo> {
    use serde::Deserialize;
    use wmi::{COMLibrary, WMIConnection};

    /// **P2-6**：同机实测 `gpu_count` 曾出现 **3 → 1 → 3**（WMI 不稳）。
    /// 记住上一次成功读到的列表，本次更少时沿用（见 `取较长的适配器列表`）。
    static 上次成功: std::sync::Mutex<Option<Vec<GpuInfo>>> = std::sync::Mutex::new(None);

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

    let 本次: Vec<GpuInfo> = rows
        .into_iter()
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
        .collect();

    // 毒化锁不 panic（硬件检测不该因为一次线程 panic 就整体失败）。
    let mut 缓存 = 上次成功
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let 输出 = 取较长的适配器列表(缓存.as_ref(), 本次);
    if 缓存.as_ref().map(|旧| 旧.len()) != Some(输出.len()) {
        tracing::debug!(
            previous = 缓存.as_ref().map(|旧| 旧.len()),
            current = 输出.len(),
            "WMI 适配器列表长度变化（P2-6：更少时沿用上次）"
        );
    }
    *缓存 = Some(输出.clone());
    输出
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

    #[test]
    fn tcc_mode_is_detected_case_insensitively() {
        // TCC = 计算专用模式：Vulkan 在该模式下枚举不到设备，
        // 所以这个判定直接决定「要不要放弃后端 DLL 隔离」。
        assert!(is_tcc_mode("TCC"));
        assert!(is_tcc_mode("tcc"));
        assert!(is_tcc_mode("  Tcc  "));
        // WDDM 是正常显示模式 —— 绝不能误判成 TCC，否则隔离会被无谓放弃，
        // 「切后端无效」的老问题就回来了。
        assert!(!is_tcc_mode("WDDM"));
        assert!(!is_tcc_mode("wddm"));
        // 空串 / 异常输出同样不能判成 TCC（探测失败要保守）。
        assert!(!is_tcc_mode(""));
        assert!(!is_tcc_mode("N/A"));
    }

    /// **P2-6 护栏**：WMI 读到的适配器列表「只增不减」。
    ///
    /// 正反两面都要钉：
    /// ① 本次更少（WMI 抖动）→ 沿用上次（否则界面显卡数 3 → 1 → 3 乱跳）；
    /// ② 本次更多（真发现了新适配器）→ **必须采纳** —— 不能因为「只增不减」
    ///    把新出现的卡吞掉，那样就从「去噪」变成了「永久错误」。
    #[cfg(windows)]
    #[test]
    fn wmi_adapter_list_never_shrinks() {
        let 三块 = vec![
            gpu("NVIDIA GeForce RTX 5060 Laptop GPU", 8.0),
            gpu("Intel(R) UHD Graphics", 0.0),
            gpu("Microsoft Basic Display Adapter", 0.0),
        ];
        let 一块 = vec![gpu("NVIDIA GeForce RTX 5060 Laptop GPU", 8.0)];
        // ① 抖动：本次 1 块、上次 3 块 → 沿用 3 块
        assert_eq!(取较长的适配器列表(Some(&三块), 一块.clone()).len(), 3);
        // ② 增长：本次 3 块、上次 1 块 → 采纳本次（3 块）
        assert_eq!(取较长的适配器列表(Some(&一块), 三块.clone()).len(), 3);
        // ③ 首次（还没有上次）→ 采纳本次
        assert_eq!(取较长的适配器列表(None, 一块.clone()).len(), 1);
        // ④ 一样多 → 采纳本次（等价替换，不把旧值锁死）
        assert_eq!(取较长的适配器列表(Some(&一块), 一块.clone()).len(), 1);
    }

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
    fn merge_dedups_the_same_card_reported_with_different_prefixes() {
        // P100 虚机实测的现场：同一张卡两个来源两个名字，前缀只差 `NVIDIA `。
        // 旧实现按完整名字相等去重 ⇒ 判不出重复 ⇒ `gpu_count=3`（实际 1 张）。
        let nvidia = vec![gpu("Tesla P100-PCIE-16GB", 16.0)];
        let wmi = vec![
            gpu("NVIDIA Tesla P100-PCIE-16GB", 4.0),
            gpu("Microsoft Basic Display Adapter", 0.0),
        ];
        // WMI 那条 Basic Display 在真实链路里已被 `is_virtual_adapter` 滤掉，
        // 这里直接从源头去掉，只看前缀差异这一件事。
        let wmi = wmi
            .into_iter()
            .filter(|g| !is_virtual_adapter(&g.name))
            .collect();

        let merged = merge_gpus(nvidia, wmi);

        assert_eq!(merged.len(), 1, "同一张卡不该被数成两张");
        assert_eq!(merged[0].name, "Tesla P100-PCIE-16GB");
        assert_eq!(merged[0].vram_gb, 16.0, "保留 nvidia-smi 的准确显存");
    }

    #[test]
    fn merge_still_separates_two_genuinely_different_cards() {
        // 去重不能过火：两张真卡名字不同就别合并。
        let nvidia = vec![gpu("Tesla P100-PCIE-16GB", 16.0)];
        let wmi = vec![gpu("NVIDIA GeForce RTX 3060", 12.0)];

        let merged = merge_gpus(nvidia, wmi);

        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn normalizes_adapter_names_across_sources() {
        // 前缀 + 商标噪声 + 大小写，三种差异都要吃掉。
        assert_eq!(
            归一化适配器名("NVIDIA Tesla P100-PCIE-16GB"),
            归一化适配器名("Tesla P100-PCIE-16GB")
        );
        assert_eq!(
            归一化适配器名("AMD Radeon RX 7900 XTX"),
            归一化适配器名("Radeon RX 7900 XTX")
        );
        assert_eq!(
            归一化适配器名("Intel(R) UHD Graphics 630"),
            归一化适配器名("Intel UHD Graphics 630")
        );
        // 不同卡不能被归一成同一个串。
        assert_ne!(
            归一化适配器名("Tesla P100-PCIE-16GB"),
            归一化适配器名("Tesla P40")
        );
        // 纯符号名字归一后为空 —— 判重那边靠这个空值拒绝比较。
        assert!(归一化适配器名("---").is_empty());
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
        // ~4.9GB weights on a 1008 GB/s Ada card: 1008/4.9*0.60 ≈ 123 t/s.
        let nv = GpuVendor::NVIDIA;
        let fast = estimate_tokens_per_second(1008, 4.9, "NVIDIA GeForce RTX 4090", &nv);
        assert!((100.0..150.0).contains(&fast), "unexpected estimate {fast}");
        // Same card, a heavier 39GB model must be slower.
        let slow = estimate_tokens_per_second(1008, 39.0, "NVIDIA GeForce RTX 4090", &nv);
        assert!(slow < fast);
        // Missing inputs degrade to zero rather than dividing by zero.
        assert_eq!(estimate_tokens_per_second(0, 4.0, "NVIDIA GeForce RTX 4090", &nv), 0.0);
        assert_eq!(estimate_tokens_per_second(1008, 0.0, "NVIDIA GeForce RTX 4090", &nv), 0.0);
    }

    /// Regression guard anchored to the 2026-09-28 P100 measurement.
    ///
    /// Measured on a Tesla P100 (Pascal, HBM2 732 GB/s): Qwen3.8-27B-Q3_K_XL,
    /// 11.92 GiB resident weights → **10.47 tok/s (Vulkan) / 11.36 tok/s (CUDA)**,
    /// i.e. ~18% of the 61.4 tok/s roofline. The old flat `0.7` factor predicted
    /// ~42 tok/s and overstated reality by 3.8×.
    ///
    /// This test exists so nobody re-flattens the efficiency table and silently
    /// brings back a 4×-inflated number on old hardware.
    #[test]
    fn p100_estimate_matches_measured_tok_per_second() {
        let nv = GpuVendor::NVIDIA;
        let est = estimate_tokens_per_second(732, 11.92, "Tesla P100-PCIE-16GB", &nv);
        // Measured 10.47–11.36 t/s. Allow the estimate to land in a band that
        // covers both backends without accepting a wildly optimistic value.
        assert!(
            (9.5..15.0).contains(&est),
            "P100 estimate {est} drifted away from the measured 10.5–11.4 t/s"
        );
        // The flat 0.7 factor would have produced ~43 t/s — reject anything close.
        assert!(est < 20.0, "P100 estimate {est} is suspiciously close to the old 0.7 factor");
    }

    #[test]
    fn pascal_is_bucketed_below_tensor_accelerated_cards() {
        let nv = GpuVendor::NVIDIA;
        // Same bandwidth, different generations → different efficiency.
        assert!(efficiency_for("Tesla P100-PCIE-16GB", &nv, 1.708) < 0.3);
        assert!(efficiency_for("NVIDIA GeForce RTX 4090", &nv, 20.0) > 0.5);
        assert!(efficiency_for("NVIDIA GeForce GTX 1080 Ti", &nv, 5.0) < 0.3);
        assert!(efficiency_for("NVIDIA GeForce RTX 4060 Laptop GPU", &nv, 8.0) > 0.5);
        // Non-NVIDIA vendors fall into their own buckets.
        assert!(
            (0.3..0.5).contains(&efficiency_for("AMD Radeon RX 7900 XTX", &GpuVendor::AMD, 20.0))
        );
    }

    #[test]
    fn pascal_splits_by_weight_size_on_the_two_measured_points() {
        // 2026-10-20，同一张 P100（732 GB/s），两个实测点：
        //   1.7B-Q8_0  1.708 GiB → 110~113 t/s
        //   27B-Q3_K_XL 11.92 GiB → 10.5~11.4 t/s
        // 一个 0.20 覆盖不了这两档（1.7B 会被低估 1.4×），所以按权重分档。
        let nv = GpuVendor::NVIDIA;
        let p100 = "Tesla P100-PCIE-16GB";
        assert!((efficiency_for(p100, &nv, 1.708) - 0.26).abs() < 1e-9);
        assert!((efficiency_for(p100, &nv, 11.92) - 0.18).abs() < 1e-9);

        // 估算值必须落在实测区间里（不是"接近"，是落进去）。
        let 小 = estimate_tokens_per_second(732, 1.708, p100, &nv);
        assert!((110.0..=113.0).contains(&小), "1.7B 估出 {小} t/s，实测 110~113");
        let 大 = estimate_tokens_per_second(732, 11.92, p100, &nv);
        assert!(
            (10.5..=11.4).contains(&大),
            "27B 估出 {大} t/s，实测 10.5~11.4"
        );

        // 阈值本身不是测出来的拐点：4 GiB 只是两点之间的空档。
        assert!(efficiency_for(p100, &nv, 4.0) > efficiency_for(p100, &nv, 4.01));
        // 现代卡不受权重分档影响（样本只有一端，不改）。
        assert!((efficiency_for("NVIDIA GeForce RTX 4090", &nv, 1.0) - 0.60).abs() < 1e-9);
        assert!((efficiency_for("NVIDIA GeForce RTX 4090", &nv, 20.0) - 0.60).abs() < 1e-9);
    }

    fn nvidia(name: &str, vram_gb: f64, driver: &str) -> GpuInfo {
        GpuInfo {
            index: 0,
            bandwidth_gbs: lookup_bandwidth(&GpuVendor::NVIDIA, name),
            vendor: GpuVendor::NVIDIA,
            name: name.to_string(),
            vram_gb,
            driver_version: driver.to_string(),
        }
    }

    #[test]
    fn driver_version_parsing() {
        assert_eq!(parse_driver_version("551.86"), Some(551.86));
        assert_eq!(parse_driver_version(" 552.44 "), Some(552.44));
        assert_eq!(parse_driver_version("551.86.01"), Some(551.86));
        assert_eq!(parse_driver_version("n/a"), None);
    }

    #[test]
    fn free_vram_parsing() {
        // nounits form (what --format=csv,noheader,nounits produces).
        assert_eq!(parse_free_vram_gb("1024\n"), Some(1.0));
        assert_eq!(parse_free_vram_gb(" 5218 "), Some(5218.0 / 1024.0));
        // Defensive: a stray "MiB" suffix from a differently-formatted build.
        assert_eq!(parse_free_vram_gb("2048 MiB"), Some(2.0));
        // Multi-GPU: only GPU 0 (first line) is taken.
        assert_eq!(parse_free_vram_gb("8192\n4096\n"), Some(8.0));
        // Garbage / empty → None so the caller falls back to total VRAM.
        assert_eq!(parse_free_vram_gb(""), None);
        assert_eq!(parse_free_vram_gb("[N/A]"), None);
    }

    #[test]
    fn no_gpu_selects_cpu() {
        assert!(matches!(select_backend(&[]), BackendType::CPUx86));
    }

    #[test]
    fn nvidia_with_capable_driver_takes_cuda_fast_lane() {
        // NVIDIA card from nvidia-smi (accurate VRAM) with a driver at/above the
        // CUDA 12.4 floor uses the native CUDA engine.
        let gpus = vec![nvidia("NVIDIA GeForce RTX 4070", 12.0, "552.44")];
        assert!(matches!(select_backend(&gpus), BackendType::CUDA));
    }

    #[test]
    fn nvidia_with_old_driver_falls_back_to_vulkan() {
        // Driver below the CUDA 12.4 floor (e.g. the R535-era 539.72) can't run the
        // bundled CUDA build, so use the universal Vulkan backend instead.
        let gpus = vec![nvidia("NVIDIA Tesla P100-PCIE-16GB", 16.0, "539.72")];
        assert!(matches!(select_backend(&gpus), BackendType::Vulkan));
    }

    #[test]
    fn nvidia_without_smi_vram_is_untrusted_and_uses_vulkan() {
        // A WMI-only NVIDIA entry (nvidia-smi unavailable) has no reliable VRAM and
        // a Windows display-driver version string, not the NVIDIA branch version —
        // don't trust it for the CUDA gate.
        let gpus = vec![nvidia("NVIDIA GeForce RTX 4070", 0.0, "31.0.15.5222")];
        assert!(matches!(select_backend(&gpus), BackendType::Vulkan));
    }

    #[test]
    fn amd_and_intel_use_vulkan() {
        assert!(matches!(select_backend(&[gpu("AMD Radeon RX 7900 XTX", 0.0)]), BackendType::Vulkan));
        assert!(matches!(select_backend(&[gpu("Intel(R) Arc(TM) A770", 0.0)]), BackendType::Vulkan));
    }
}
