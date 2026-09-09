use std::{
    fs::File,
    io::copy,
    path::{Path, PathBuf},
};

use futures_util::StreamExt;
use reqwest::{header, Client, StatusCode};
use serde::{Deserialize, Serialize};
use tauri::AppHandle;
use tokio::{fs, io::AsyncWriteExt};
use zip::ZipArchive;

use crate::{
    app_state::AppState,
    error::{AppError, AppResult},
    settings::save_settings,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineInfo {
    pub backend: String,
    pub version: String,
    pub executable_path: String,
    pub message: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EnsureEngineRequest {
    pub backend: String,
}

struct EnginePackage {
    backend: &'static str,
    version: &'static str,
    file_name: &'static str,
    urls: &'static [&'static str],
    extra_archives: &'static [EngineArchive],
    executable_name: &'static str,
    sha256: &'static str,
}

struct EngineArchive {
    file_name: &'static str,
    urls: &'static [&'static str],
    sha256: &'static str,
}

#[tauri::command]
pub async fn ensure_engine_ready(
    request: EnsureEngineRequest,
    _app: AppHandle,
    state: tauri::State<'_, AppState>,
) -> AppResult<EngineInfo> {
    let package = engine_package(&request.backend);
    tracing::info!(backend = %package.backend, version = %package.version, "engine preparation requested");
    let (data_dir, install_dir) = {
        let inner = state.inner.lock().expect("state lock poisoned");
        (inner.data_dir.clone(), inner.settings.install_dir.clone())
    };

    let engine_dir = install_dir.join("engine");
    let backend_dir = engine_dir.join("backends");
    let executable_path = backend_dir.join(package.executable_name);

    prepare_package(package, &engine_dir, &backend_dir).await?;

    if !executable_path.exists() {
        return Err(AppError::Message(
            "AI运行引擎文件暂时找不到，请重新安装运行引擎。".to_string(),
        ));
    }

    // A GPU backend can still fail to launch on a given machine (missing Vulkan
    // ICD, driver too old for the CUDA build, etc.), in which case the service
    // falls back to CPU safe mode — which needs `llama-server-cpu.exe` present.
    //
    // Run the CPU download in the *background* rather than awaiting it on the
    // startup path. A slow network can stretch a 16 MB package over several
    // minutes, and the user just asked for the GPU backend — making them wait
    // for the CPU fallback they may never need is a bad tradeoff. If the GPU
    // path does fail later, the service flow re-runs `ensure_engine_ready`
    // for the CPU backend and will wait for the in-flight download to finish
    // before giving up.
    if package.backend != CPU.backend {
        let engine_dir_bg = engine_dir.to_path_buf();
        let backend_dir_bg = backend_dir.to_path_buf();
        tokio::spawn(async move {
            match prepare_package(&CPU, &engine_dir_bg, &backend_dir_bg).await {
                Ok(()) => tracing::info!("CPU fallback engine ready in background"),
                Err(error) => tracing::warn!(error = %error, "background CPU fallback engine download failed"),
            }
        });
    }

    fs::write(
        engine_dir.join("current_backend.txt"),
        package.executable_name,
    )
    .await?;
    fs::write(
        engine_dir.join("versions.json"),
        serde_json::json!({
            "current_backend": package.backend,
            "version": package.version,
            "executable": package.executable_name,
            "source": package.urls[0],
        })
        .to_string(),
    )
    .await?;

    {
        let mut inner = state.inner.lock().expect("state lock poisoned");
        inner.settings.llama_server_path = Some(executable_path.clone());
        save_settings(&data_dir, &inner.settings)?;
    }

    tracing::info!(backend = %package.backend, version = %package.version, "engine ready");
    Ok(EngineInfo {
        backend: package.backend.to_string(),
        version: package.version.to_string(),
        executable_path: executable_path.to_string_lossy().to_string(),
        message: "AI运行引擎已准备好。".to_string(),
    })
}

// ModelScope mirror (CN) is listed first: from within CN, direct github.com
// release downloads frequently connect but then stall to a crawl, so we
// prefer the ModelScope CDN, then a generic GitHub proxy, then the direct
// URL. The per-chunk idle timeout in try_download_archive catches any stall
// on any of them.
static CPU_URLS: [&str; 3] = [
    "https://www.modelscope.cn/models/yilufeiyang888999/llama-cpp-engine-zh-2026/resolve/master/llama-b10853-bin-win-cpu-x64.zip",
    "https://gh-proxy.com/https://github.com/ggml-org/llama.cpp/releases/download/b10853/llama-b10853-bin-win-cpu-x64.zip",
    "https://github.com/ggml-org/llama.cpp/releases/download/b10853/llama-b10853-bin-win-cpu-x64.zip",
];
static CUDA_URLS: [&str; 3] = [
    "https://www.modelscope.cn/models/yilufeiyang888999/llama-cpp-engine-zh-2026/resolve/master/llama-b10853-bin-win-cuda-12.4-x64.zip",
    "https://gh-proxy.com/https://github.com/ggml-org/llama.cpp/releases/download/b10853/llama-b10853-bin-win-cuda-12.4-x64.zip",
    "https://github.com/ggml-org/llama.cpp/releases/download/b10853/llama-b10853-bin-win-cuda-12.4-x64.zip",
];
static CUDA_RUNTIME_URLS: [&str; 3] = [
    "https://www.modelscope.cn/models/yilufeiyang888999/llama-cpp-engine-zh-2026/resolve/master/cudart-llama-bin-win-cuda-12.4-x64.zip",
    "https://gh-proxy.com/https://github.com/ggml-org/llama.cpp/releases/download/b10853/cudart-llama-bin-win-cuda-12.4-x64.zip",
    "https://github.com/ggml-org/llama.cpp/releases/download/b10853/cudart-llama-bin-win-cuda-12.4-x64.zip",
];
static VULKAN_URLS: [&str; 3] = [
    "https://www.modelscope.cn/models/yilufeiyang888999/llama-cpp-engine-zh-2026/resolve/master/llama-b10853-bin-win-vulkan-x64.zip",
    "https://gh-proxy.com/https://github.com/ggml-org/llama.cpp/releases/download/b10853/llama-b10853-bin-win-vulkan-x64.zip",
    "https://github.com/ggml-org/llama.cpp/releases/download/b10853/llama-b10853-bin-win-vulkan-x64.zip",
];
static CPU: EnginePackage = EnginePackage {
    backend: "CPUx86",
    version: "b10853",
    file_name: "llama-b10853-bin-win-cpu-x64.zip",
    urls: &CPU_URLS,
    extra_archives: &[],
    executable_name: "llama-server-cpu.exe",
    sha256: "10b520be35483161a0746de3d05d66a36ce3e2bb6b243d806564ff798e28f01c",
};
static CUDA_EXTRA: [EngineArchive; 1] = [EngineArchive {
    file_name: "cudart-llama-bin-win-cuda-12.4-x64.zip",
    urls: &CUDA_RUNTIME_URLS,
    sha256: "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
}];
static CUDA: EnginePackage = EnginePackage {
    backend: "CUDA",
    version: "b10853",
    file_name: "llama-b10853-bin-win-cuda-12.4-x64.zip",
    urls: &CUDA_URLS,
    extra_archives: &CUDA_EXTRA,
    executable_name: "llama-server-cuda.exe",
    sha256: "26898c5f8519e59b61f78b4fc285c9e330b2a0afa670329d303ccc871f68ef5b",
};
// Vulkan is the default GPU backend: it relies only on the driver's Vulkan
// ICD (bundled with modern NVIDIA/AMD/Intel drivers), so it dodges the CUDA
// 12.4 runtime's driver floor and covers all three vendors from one package.
static VULKAN: EnginePackage = EnginePackage {
    backend: "Vulkan",
    version: "b10853",
    file_name: "llama-b10853-bin-win-vulkan-x64.zip",
    urls: &VULKAN_URLS,
    extra_archives: &[],
    executable_name: "llama-server-vulkan.exe",
    sha256: "e4c1a9d47a79e9352ec0d9085840bed1ac5ebea0ef134f263be2cc90434a60ed",
};

fn engine_package(backend: &str) -> &'static EnginePackage {
    if backend.eq_ignore_ascii_case("vulkan") {
        &VULKAN
    } else if backend.eq_ignore_ascii_case("cuda") {
        &CUDA
    } else {
        &CPU
    }
}

/// Every engine archive that ships with the app, as `(file_id, sha256)` pairs
/// (`file_id` is `engine:<zip name>`). This is the single source of truth the
/// offline manifest signer reads, so the signed manifest can never drift from
/// the hashes the downloader actually enforces.
pub fn engine_manifest_entries() -> Vec<(String, String)> {
    let mut entries = vec![
        (format!("engine:{}", CPU.file_name), CPU.sha256.to_string()),
        (format!("engine:{}", VULKAN.file_name), VULKAN.sha256.to_string()),
        (format!("engine:{}", CUDA.file_name), CUDA.sha256.to_string()),
    ];
    for extra in CUDA.extra_archives {
        entries.push((format!("engine:{}", extra.file_name), extra.sha256.to_string()));
    }
    entries
}

// Download + extract one engine package (and any dependency archives such as
// the CUDA runtime). No-op if the target executable is already present
// **and** its version marker is current — stale engines from an older
// release get replaced when the app updates.
async fn prepare_package(
    package: &EnginePackage,
    engine_dir: &Path,
    backend_dir: &Path,
) -> AppResult<()> {
    let executable_path = backend_dir.join(package.executable_name);
    let version_marker = backend_dir.join(format!("{}.version.json", package.executable_name));
    if executable_path.exists() {
        if let Ok(raw) = fs::read_to_string(&version_marker).await {
            use serde_json::Value;
            if let Ok(json) = serde_json::from_str::<Value>(&raw) {
                if json.get("version").and_then(|v| v.as_str()) == Some(package.version) {
                    return Ok(());
                }
            }
        }
        // Marker missing, unreadable, or version mismatch (b9837 → b10853).
        tracing::info!(old = %executable_path.display(), version = %package.version, "stale engine detected; re-downloading");
        let _ = fs::remove_file(&executable_path).await;
    }

    tracing::info!(file = %package.file_name, "engine not present; downloading package");
    fs::create_dir_all(backend_dir).await?;
    let package_path = engine_dir.join(package.file_name);
    download_archive(package.urls, package.sha256, package.file_name, &package_path).await?;
    extract_engine_package(package, &package_path, backend_dir).await?;
    for archive in package.extra_archives {
        let archive_path = engine_dir.join(archive.file_name);
        download_archive(archive.urls, archive.sha256, archive.file_name, &archive_path).await?;
        extract_dependency_archive(&archive_path, backend_dir).await?;
    }
    // Record which build this executable came from so a later app update can
    // tell it is stale and pull the new one (the shared `versions.json` only
    // tracks the last-prepared backend, which is not enough here: the GPU
    // engine and the CPU fallback are prepared independently).
    let _ = fs::write(
        &version_marker,
        serde_json::json!({
            "backend": package.backend,
            "version": package.version,
            "file_name": package.file_name,
        })
        .to_string(),
    )
    .await;
    Ok(())
}

/// Public wrapper around `prepare_package` for the CPU engine, so the GPU→CPU
/// fallback path in `service::start_service_core` can pull the CPU package
/// without going through the full `ensure_engine_ready` Tauri command (which
/// would also write `current_backend.txt` and overwrite settings — not what
/// we want when GPU is the user's chosen backend and we're just shoring up
/// the fallback exe).
pub async fn prepare_cpu_package(engine_dir: &Path, backend_dir: &Path) -> AppResult<()> {
    prepare_package(&CPU, engine_dir, backend_dir).await
}

async fn download_archive(urls: &[&str], sha256: &str, file_name: &str, package_path: &Path) -> AppResult<()> {
    let mut last_error = "当前网络无法连接运行引擎下载源，请稍后重试。".to_string();
    for url in urls {
        match try_download_archive_with_retry(url, sha256, file_name, package_path).await {
            Ok(()) => return Ok(()),
            Err(error) => last_error = error.to_string(),
        }
    }
    Err(AppError::Message(last_error))
}

async fn try_download_archive_with_retry(url: &str, sha256: &str, file_name: &str, package_path: &Path) -> AppResult<()> {
    let mut last_error = "当前网络无法连接运行引擎下载源，请稍后重试。".to_string();
    for attempt in 0..3 {
        match try_download_archive(url, sha256, file_name, package_path).await {
            Ok(()) => return Ok(()),
            Err(error) => {
                last_error = error.to_string();
                if attempt < 2 {
                    tokio::time::sleep(std::time::Duration::from_secs(2_u64.pow(attempt + 1))).await;
                }
            }
        }
    }
    Err(AppError::Message(last_error))
}

async fn try_download_archive(url: &str, sha256: &str, file_name: &str, package_path: &Path) -> AppResult<()> {
    if !url.starts_with("https://") || url.contains("placeholder") {
        return Err(AppError::Message("运行引擎下载源尚未配置。".to_string()));
    }

    let tmp_path = package_path.with_extension("zip.tmp");
    let client = Client::builder()
        .connect_timeout(std::time::Duration::from_secs(20))
        .timeout(std::time::Duration::from_secs(60 * 30))
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|_| AppError::Message("运行引擎下载组件暂时没能启动，请稍后再试。".to_string()))?;

    // Resume from a partial `.tmp` left by an interrupted attempt or mirror.
    // All URLs point at the same release asset, so bytes are interchangeable.
    let existing = fs::metadata(&tmp_path)
        .await
        .map(|meta| meta.len())
        .unwrap_or(0);
    let mut request = client.get(url);
    if existing > 0 {
        request = request.header(header::RANGE, format!("bytes={}-", existing));
    }
    // Bound the wait for response headers independently. connect_timeout only
    // covers TCP/TLS setup, and the 30-min total timeout is far too long: if a
    // mirror accepts the connection but then stalls before sending headers, we
    // want to fail fast and try the next URL rather than hang.
    let response = match tokio::time::timeout(
        std::time::Duration::from_secs(30),
        request.send(),
    )
    .await
    {
        Ok(Ok(response)) => response,
        Ok(Err(_)) => {
            return Err(AppError::Message(
                "当前网络无法连接运行引擎下载源，请稍后重试。".to_string(),
            ))
        }
        Err(_) => {
            return Err(AppError::Message(
                "运行引擎下载源响应超时，正在尝试其他线路。".to_string(),
            ))
        }
    };

    if !(response.status().is_success() || response.status() == StatusCode::PARTIAL_CONTENT) {
        return Err(AppError::Message(
            "运行引擎下载源暂时不可用，请稍后重试。".to_string(),
        ));
    }

    // 206 means the server honored the range: append. Anything else (including a
    // 200 that ignored our range) starts the file over.
    let mut file = if response.status() == StatusCode::PARTIAL_CONTENT {
        fs::OpenOptions::new().append(true).open(&tmp_path).await?
    } else {
        fs::File::create(&tmp_path).await?
    };
    let mut stream = response.bytes_stream();
    // Idle timeout: GitHub release downloads from within CN often connect but
    // then stall (no bytes for a long time). Without this the download would
    // hang until the 30-minute total timeout before trying the next mirror.
    // Bounding each chunk wait lets us fail fast and fall through to gh-proxy.
    const IDLE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);
    loop {
        let next = tokio::time::timeout(IDLE_TIMEOUT, stream.next()).await;
        let chunk = match next {
            Ok(Some(chunk)) => chunk.map_err(|_| {
                AppError::Message("运行引擎下载中断了，请稍后重试。".to_string())
            })?,
            Ok(None) => break,
            Err(_) => {
                return Err(AppError::Message(
                    "运行引擎下载停滞了（长时间没有数据），正在尝试其他线路。".to_string(),
                ))
            }
        };
        file.write_all(&chunk).await?;
    }
    file.flush().await?;

    // A verification failure means the bytes on disk are unusable (corrupt or a
    // botched resume), so drop the `.tmp` to force a clean re-download next time.
    if let Err(error) = verify_sha256(&tmp_path, sha256).await {
        let _ = fs::remove_file(&tmp_path).await;
        return Err(error);
    }
    if let Err(error) = crate::manifest::verify_file(&format!("engine:{}", file_name), &tmp_path).await {
        let _ = fs::remove_file(&tmp_path).await;
        return Err(error);
    }

    if package_path.exists() {
        fs::remove_file(package_path).await?;
    }
    fs::rename(&tmp_path, package_path).await?;
    Ok(())
}

async fn verify_sha256(path: &Path, expected: &str) -> AppResult<()> {
    if !crate::hash::is_configured(expected) {
        return Err(AppError::Message(
            "运行引擎校验信息尚未配置，请先补充真实校验值。".to_string(),
        ));
    }

    let actual = crate::hash::file_sha256_hex(path).await?;
    if !crate::hash::matches(expected, &actual) {
        return Err(AppError::Message(
            "运行引擎文件检查不通过，请重新下载。".to_string(),
        ));
    }
    Ok(())
}

async fn extract_engine_package(
    package: &EnginePackage,
    package_path: &Path,
    backend_dir: &Path,
) -> AppResult<()> {
    let package_path = package_path.to_path_buf();
    let backend_dir = backend_dir.to_path_buf();
    let executable_name = package.executable_name.to_string();
    tokio::task::spawn_blocking(move || {
        extract_zip_blocking(&package_path, &backend_dir, &executable_name)
    })
    .await
    .map_err(|_| AppError::Message("运行引擎解压失败，请稍后重试。".to_string()))?
}

async fn extract_dependency_archive(package_path: &Path, backend_dir: &Path) -> AppResult<()> {
    let package_path = package_path.to_path_buf();
    let backend_dir = backend_dir.to_path_buf();
    tokio::task::spawn_blocking(move || {
        extract_dependency_zip_blocking(&package_path, &backend_dir)
    })
    .await
    .map_err(|_| AppError::Message("运行引擎依赖解压失败，请稍后重试。".to_string()))?
}

fn extract_dependency_zip_blocking(package_path: &Path, backend_dir: &Path) -> AppResult<()> {
    std::fs::create_dir_all(backend_dir)?;
    let file = File::open(package_path)?;
    let mut archive = ZipArchive::new(file)
        .map_err(|_| AppError::Message("运行引擎依赖包暂时无法打开，请重新下载。".to_string()))?;

    for index in 0..archive.len() {
        let mut entry = archive.by_index(index).map_err(|_| {
            AppError::Message("运行引擎依赖包暂时无法读取，请重新下载。".to_string())
        })?;
        if entry.is_dir() {
            continue;
        }

        let Some(name) = PathBuf::from(entry.name())
            .file_name()
            .and_then(|name| name.to_str())
            .map(str::to_string)
        else {
            continue;
        };
        if !name.to_ascii_lowercase().ends_with(".dll") {
            continue;
        }

        let output_path = backend_dir.join(name);
        let mut output = File::create(output_path)?;
        copy(&mut entry, &mut output)?;
    }

    Ok(())
}

fn extract_zip_blocking(
    package_path: &Path,
    backend_dir: &Path,
    executable_name: &str,
) -> AppResult<()> {
    std::fs::create_dir_all(backend_dir)?;
    let file = File::open(package_path)?;
    let mut archive = ZipArchive::new(file)
        .map_err(|_| AppError::Message("运行引擎压缩包暂时无法打开，请重新下载。".to_string()))?;
    let mut found_server = false;

    for index in 0..archive.len() {
        let mut entry = archive.by_index(index).map_err(|_| {
            AppError::Message("运行引擎压缩包暂时无法读取，请重新下载。".to_string())
        })?;
        if entry.is_dir() {
            continue;
        }

        let Some(name) = PathBuf::from(entry.name())
            .file_name()
            .and_then(|name| name.to_str())
            .map(str::to_string)
        else {
            continue;
        };
        let lower_name = name.to_ascii_lowercase();
        let should_extract = lower_name.ends_with(".dll")
            || lower_name == "ggml-metal.metal"
            || lower_name == "ggml-cuda.dll"
            || lower_name == "llama-server.exe";
        if !should_extract {
            continue;
        }

        let output_name = if lower_name == "llama-server.exe" {
            found_server = true;
            executable_name.to_string()
        } else {
            name
        };
        let output_path = backend_dir.join(output_name);
        let mut output = File::create(output_path)?;
        copy(&mut entry, &mut output)?;
    }

    if !found_server {
        return Err(AppError::Message(
            "运行引擎包里没有找到可启动文件，请稍后重试。".to_string(),
        ));
    }
    Ok(())
}
