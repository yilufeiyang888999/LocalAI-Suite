use std::path::Path;

use serde::{Deserialize, Serialize};
use sysinfo::Disks;

use crate::{
    app_state::AppState,
    error::{AppError, AppResult},
    hardware::{detect_hardware, parse_driver_version},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnoseResult {
    pub issue_id: String,
    pub title: String,
    pub description: String,
    pub auto_level: AutoLevel,
    pub fixable: bool,
    pub ok: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AutoLevel {
    FullAuto,
    SemiAuto,
    ManualGuide,
}

/// Largest builtin model (~4.7GB) plus engine and working room.
const REQUIRED_FREE_GB: f64 = 8.0;
/// Smallest builtin model's VRAM need; below this GPU acceleration is skipped.
const SMALLEST_VRAM_NEED_GB: f64 = 1.8;
/// Windows driver floor for the bundled CUDA 12.4 `llama.cpp` build.
const MIN_NVIDIA_DRIVER: f64 = 551.61;

#[tauri::command]
pub fn run_diagnose(state: tauri::State<'_, AppState>) -> AppResult<Vec<DiagnoseResult>> {
    // Gather raw facts under the lock, then release it before subprocess/fs work.
    let (install_dir, model_path, engine_path, safe_mode) = {
        let inner = state.inner.lock().expect("state lock poisoned");
        (
            inner.settings.install_dir.clone(),
            inner.settings.local_model_path.clone(),
            inner.settings.llama_server_path.clone(),
            inner.service.safe_mode,
        )
    };

    let dir_usable = install_dir.exists() || std::fs::create_dir_all(&install_dir).is_ok();
    let available_gb = available_disk_gb(&install_dir);

    let (model_exists, model_size) = match model_path.as_ref() {
        Some(path) => match std::fs::metadata(path) {
            Ok(meta) => (meta.is_file(), meta.len()),
            Err(_) => (false, 0),
        },
        None => (false, 0),
    };

    let engine_present = engine_path.as_ref().map(|p| p.exists()).unwrap_or(false);
    let dll_count = count_backend_dlls(&install_dir);

    let hardware = detect_hardware();
    let driver_version = hardware
        .gpu_list
        .first()
        .and_then(|gpu| parse_driver_version(&gpu.driver_version));

    let results = vec![
        eval_disk(dir_usable, available_gb, REQUIRED_FREE_GB),
        eval_model(model_exists, model_size),
        eval_port(portpicker::is_free(8000)),
        eval_driver(hardware.has_gpu, driver_version, MIN_NVIDIA_DRIVER),
        eval_runtime_libs(engine_present, dll_count),
        eval_vram(
            hardware.has_gpu,
            hardware.gpu_vram_gb,
            SMALLEST_VRAM_NEED_GB,
            safe_mode,
        ),
    ];

    if results.is_empty() {
        return Err(AppError::Message("暂时没有可用的检查项。".to_string()));
    }
    let failing: Vec<&str> = results.iter().filter(|item| !item.ok).map(|item| item.issue_id.as_str()).collect();
    tracing::info!(total = results.len(), failing = ?failing, "diagnostics completed");
    Ok(results)
}

// --- Pure evaluators (unit-tested) -----------------------------------------

fn eval_disk(dir_usable: bool, available_gb: f64, required_gb: f64) -> DiagnoseResult {
    let ok = dir_usable && available_gb >= required_gb;
    let description = if !dir_usable {
        "模型保存位置暂时不可用，请在设置里换一个文件夹。".to_string()
    } else if available_gb >= required_gb {
        format!("模型保存位置正常，约还有 {:.1} GB 可用空间。", available_gb)
    } else {
        format!(
            "保存位置可用空间约 {:.1} GB，可能不够下载模型，建议清理空间或换到更大的磁盘。",
            available_gb
        )
    };
    DiagnoseResult {
        issue_id: "disk_space".to_string(),
        title: "磁盘空间检查".to_string(),
        description,
        auto_level: AutoLevel::SemiAuto,
        fixable: !ok,
        ok,
    }
}

fn eval_model(exists: bool, size_bytes: u64) -> DiagnoseResult {
    let ok = exists && size_bytes > 0;
    let description = if !exists {
        "还没有可用的本地模型文件，完成一次模型安装后就会自动就绪。".to_string()
    } else if size_bytes == 0 {
        "模型文件好像不完整，稍后会帮你重新下载。".to_string()
    } else {
        "本地模型文件检查通过。".to_string()
    };
    DiagnoseResult {
        issue_id: "model_file".to_string(),
        title: "模型文件完整性检查".to_string(),
        description,
        // Corrupt/incomplete model -> auto re-download (§3.7.2 A级).
        auto_level: AutoLevel::FullAuto,
        fixable: !ok,
        ok,
    }
}

fn eval_port(default_free: bool) -> DiagnoseResult {
    let description = if default_free {
        "服务入口可用；实际启动时会自动选择空闲入口。".to_string()
    } else {
        "默认服务入口被占用了，启动时会自动换一个空闲入口，无需手动处理。".to_string()
    };
    DiagnoseResult {
        issue_id: "port".to_string(),
        title: "端口占用检查".to_string(),
        description,
        // Port conflicts are auto-resolved by picking a free port (§3.7.2 A级).
        auto_level: AutoLevel::FullAuto,
        fixable: false,
        ok: true,
    }
}

fn eval_driver(has_gpu: bool, driver_version: Option<f64>, min: f64) -> DiagnoseResult {
    // With Vulkan as the default GPU backend, acceleration only needs the
    // driver's Vulkan ICD (bundled with modern NVIDIA/AMD/Intel drivers), not the
    // CUDA 12.4 runtime's driver floor. So this check never fails the launch — a
    // low driver is just an informational nudge; if acceleration can't start we
    // still fall back to CPU automatically.
    let (ok, description) = if !has_gpu {
        (
            true,
            "未检测到独立显卡，将使用 CPU 模式运行，无需显卡驱动。".to_string(),
        )
    } else {
        match driver_version {
            Some(version) if version < min => (
                true,
                format!(
                    "显卡驱动版本较早（{:.2}）。当前使用通用显卡加速（Vulkan），一般仍可正常加速；如果加速用不上，建议到显卡官网升级驱动，在此之前会自动使用更稳妥的 CPU 模式。",
                    version
                ),
            ),
            Some(_) => (
                true,
                "显卡驱动版本满足显卡加速要求。".to_string(),
            ),
            None => (
                true,
                "检测到独立显卡，将尝试通用显卡加速；若启动失败会自动切换 CPU 模式。".to_string(),
            ),
        }
    };
    DiagnoseResult {
        issue_id: "gpu_driver".to_string(),
        title: "显卡驱动版本检查".to_string(),
        description,
        // Driver upgrade needs the user (§3.7.2 C级), but Vulkan means it's never
        // a hard blocker — this stays informational.
        auto_level: AutoLevel::ManualGuide,
        fixable: !ok,
        ok,
    }
}

fn eval_runtime_libs(engine_present: bool, dll_count: usize) -> DiagnoseResult {
    let ok = engine_present && dll_count > 0;
    let description = if !engine_present {
        "本地AI运行引擎会在安装模型时自动准备；自动下载不可用时可用高级设置手动选择。".to_string()
    } else if dll_count == 0 {
        "运行引擎的部分依赖文件好像缺失了，重新准备一次运行引擎即可修复。".to_string()
    } else {
        "本地AI运行引擎和依赖文件检查通过。".to_string()
    };
    DiagnoseResult {
        issue_id: "runtime_libs".to_string(),
        title: "运行依赖库检查".to_string(),
        description,
        // Missing runtime libs -> re-prepare engine with user confirm (§3.7.2 B级).
        auto_level: AutoLevel::SemiAuto,
        fixable: !ok,
        ok,
    }
}

fn eval_vram(has_gpu: bool, vram_gb: f64, smallest_need_gb: f64, safe_mode: bool) -> DiagnoseResult {
    let (ok, description) = if !has_gpu {
        (
            true,
            "当前使用 CPU 模式，速度可能较慢，适合简单问答。".to_string(),
        )
    } else if safe_mode {
        (
            true,
            "显卡加速暂时没用上，正在以更稳妥的 CPU 模式运行；显存充足时可尝试恢复显卡加速。".to_string(),
        )
    } else if vram_gb <= 0.0 {
        // Non-NVIDIA (AMD/Intel) cards: VRAM isn't reliably readable, so this is
        // informational rather than a failure — the launch tries acceleration and
        // falls back to CPU automatically if VRAM turns out to be too small.
        (
            true,
            "检测到显卡但暂时读不到显存大小，将尝试显卡加速；如果显存不足会自动改用更省资源的方式或 CPU 模式。".to_string(),
        )
    } else if vram_gb >= smallest_need_gb {
        (
            true,
            format!("显卡内存约 {:.1} GB，可以尝试显卡加速。", vram_gb),
        )
    } else {
        (
            false,
            format!(
                "显卡内存约 {:.1} GB 偏小，启动时会自动选择更省显存的版本或改用 CPU 模式。",
                vram_gb
            ),
        )
    };
    DiagnoseResult {
        issue_id: "vram".to_string(),
        title: "显存可用性检查".to_string(),
        description,
        // Low VRAM -> auto pick smaller/CPU (§3.7.2 A级).
        auto_level: AutoLevel::FullAuto,
        fixable: !ok,
        ok,
    }
}

// --- Impure helpers ---------------------------------------------------------

/// 统计引擎目录里可用的 DLL 数量，**同时算上被后端隔离挪进 `.pool/` 的那些**。
///
/// 为什么要把池子算进来：`service::isolate_backend_dlls` 会让
/// `backends/` 根目录下只剩当前后端的 DLL，另一个 GPU 后端的 DLL
/// （CUDA 那个有 547MB）会被挪到 `backends/.pool/<后端>/`。
/// 若这里只数根目录，隔离生效后计数会凭空少一大截，
/// 诊断就会误报「运行库缺失」并提示用户重新下载引擎 —— 白白让用户
/// 重下几百 MB，而实际上文件好端端地在池子里。
fn count_backend_dlls(install_dir: &Path) -> usize {
    let backend_dir = install_dir.join("engine").join("backends");
    统计目录内dll数(&backend_dir) + 统计目录内dll数(&backend_dir.join(".pool"))
}

/// 递归统计一个目录下的 `.dll` 文件数。目录不存在时返回 0。
fn 统计目录内dll数(目录: &Path) -> usize {
    let Ok(entries) = std::fs::read_dir(目录) else {
        return 0;
    };
    let mut 数 = 0;
    for entry in entries.flatten() {
        let 路径 = entry.path();
        if 路径.is_dir() {
            // 池子结构是 .pool/<后端名>/*.dll，递归一层足矣；
            // 用递归实现是为了不依赖「恰好两层」这个假设。
            数 += 统计目录内dll数(&路径);
            continue;
        }
        if 路径
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("dll"))
            .unwrap_or(false)
        {
            数 += 1;
        }
    }
    数
}

fn available_disk_gb(install_dir: &Path) -> f64 {
    let disks = Disks::new_with_refreshed_list();
    disks
        .iter()
        .find(|disk| install_dir.starts_with(disk.mount_point()))
        .or_else(|| disks.iter().next())
        .map(|disk| disk.available_space() as f64 / 1024.0 / 1024.0 / 1024.0)
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disk_check_flags_low_space() {
        let ok = eval_disk(true, 20.0, REQUIRED_FREE_GB);
        assert!(ok.ok && !ok.fixable);

        let low = eval_disk(true, 2.0, REQUIRED_FREE_GB);
        assert!(!low.ok && low.fixable);
        assert_eq!(low.auto_level, AutoLevel::SemiAuto);

        let unusable = eval_disk(false, 100.0, REQUIRED_FREE_GB);
        assert!(!unusable.ok);
    }

    #[test]
    fn model_check_distinguishes_missing_and_empty() {
        assert!(eval_model(true, 1024).ok);
        assert!(!eval_model(false, 0).ok);
        let empty = eval_model(true, 0);
        assert!(!empty.ok && empty.fixable);
        // Corrupt model is auto-fixable per §3.7.2.
        assert_eq!(empty.auto_level, AutoLevel::FullAuto);
    }

    #[test]
    fn port_check_is_always_ok_and_full_auto() {
        let occupied = eval_port(false);
        assert!(occupied.ok && !occupied.fixable);
        assert_eq!(occupied.auto_level, AutoLevel::FullAuto);
    }

    #[test]
    fn driver_check_is_informational_only() {
        // No GPU: ok, not fixable.
        assert!(eval_driver(false, None, MIN_NVIDIA_DRIVER).ok);
        // Recent driver: ok.
        assert!(eval_driver(true, Some(552.44), MIN_NVIDIA_DRIVER).ok);

        // A driver below the old CUDA floor no longer fails — Vulkan doesn't need
        // it, so this is informational (ok, never fixable).
        let low = eval_driver(true, Some(537.58), MIN_NVIDIA_DRIVER);
        assert!(low.ok && !low.fixable);

        // Unknown version must not hard-fail either.
        assert!(eval_driver(true, None, MIN_NVIDIA_DRIVER).ok);
    }

    #[test]
    fn runtime_libs_check_needs_engine_and_dlls() {
        assert!(eval_runtime_libs(true, 5).ok);
        assert!(!eval_runtime_libs(false, 0).ok);
        let no_dll = eval_runtime_libs(true, 0);
        assert!(!no_dll.ok && no_dll.fixable);
        assert_eq!(no_dll.auto_level, AutoLevel::SemiAuto);
    }

    #[test]
    fn vram_check_reflects_mode_and_size() {
        assert!(eval_vram(false, 0.0, SMALLEST_VRAM_NEED_GB, false).ok);
        assert!(eval_vram(true, 8.0, SMALLEST_VRAM_NEED_GB, false).ok);
        assert!(eval_vram(true, 8.0, SMALLEST_VRAM_NEED_GB, true).ok);
        // GPU present but VRAM unknown (AMD/Intel): informational, not a failure.
        let unknown = eval_vram(true, 0.0, SMALLEST_VRAM_NEED_GB, false);
        assert!(unknown.ok && !unknown.fixable);
        // Known-small NVIDIA VRAM still flags for the auto smaller-model fix.
        let small = eval_vram(true, 1.0, SMALLEST_VRAM_NEED_GB, false);
        assert!(!small.ok);
        assert_eq!(small.auto_level, AutoLevel::FullAuto);
    }
}
