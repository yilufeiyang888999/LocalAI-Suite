use std::{
    net::TcpListener,
    path::PathBuf,
    process::{Child, Command},
    time::{Duration, Instant},
};

use portpicker::pick_unused_port;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter};

use crate::{
    app_state::{AppState, ServiceStatus},
    database::{clear_service_state, load_service_state, save_service_state, PersistedServiceState},
    error::{AppError, AppResult},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceInfo {
    pub model_id: Option<String>,
    pub port: Option<u16>,
    pub host: Option<String>,
    pub status: ServiceStatus,
    pub safe_mode: bool,
    pub message: String,
}

/// Concrete `llama-server` launch tuning. `n_gpu_layers` is `None` for CPU-only
/// launches; `Some(n)` offloads `n` transformer layers (999 = offload all).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LaunchParams {
    n_gpu_layers: Option<u32>,
    ctx_size: u32,
}

/// Default CUDA tuning when the model or VRAM is unknown — matches the old
/// fixed behavior so imported models keep working.
const DEFAULT_CUDA_PARAMS: LaunchParams = LaunchParams {
    n_gpu_layers: Some(24),
    ctx_size: 4096,
};

/// Pick GPU offload + context size from available VRAM relative to the model's
/// requirement. More headroom → offload everything with a larger context;
/// tighter VRAM → offload fewer layers and shrink the context to avoid a CUDA
/// out-of-memory launch failure.
fn cuda_launch_params(available_vram_gb: f64, model_required_gb: f64) -> LaunchParams {
    if available_vram_gb >= model_required_gb + 1.5 {
        LaunchParams { n_gpu_layers: Some(999), ctx_size: 4096 }
    } else if available_vram_gb >= model_required_gb {
        LaunchParams { n_gpu_layers: Some(999), ctx_size: 3072 }
    } else if available_vram_gb >= model_required_gb * 0.7 {
        LaunchParams { n_gpu_layers: Some(20), ctx_size: 3072 }
    } else if available_vram_gb >= model_required_gb * 0.5 {
        LaunchParams { n_gpu_layers: Some(12), ctx_size: 2048 }
    } else {
        LaunchParams { n_gpu_layers: Some(8), ctx_size: 2048 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EngineBackend {
    Cpu,
    Cuda,
}

#[tauri::command]
pub async fn start_model_service(
    model_id: String,
    app: AppHandle,
    state: tauri::State<'_, AppState>,
) -> AppResult<ServiceInfo> {
    start_service_core(&model_id, &app, &state).await
}

/// Start the local `llama-server` for `model_id`, falling back to CPU safe mode
/// when CUDA fails. Shared by the Tauri command and the model-update flow.
pub async fn start_service_core(
    model_id: &str,
    app: &AppHandle,
    state: &AppState,
) -> AppResult<ServiceInfo> {
    let (server_path, model_path, install_dir) = resolve_paths(state)?;
    let (host, extra_args) = {
        let inner = state.inner.lock().expect("state lock poisoned");
        let host = inner.settings.service_host.clone();
        let extra = parse_extra_args(inner.settings.extra_launch_args.as_deref().unwrap_or(""));
        (host, extra)
    };
    let configured_port = {
        let inner = state.inner.lock().expect("state lock poisoned");
        inner.settings.service_port
    };
    let backend = detect_backend(&server_path, &install_dir);
    // A fixed port lets external clients keep a stable base_url. 0 means "auto".
    let port = if configured_port == 0 {
        pick_unused_port().unwrap_or(8000)
    } else {
        configured_port
    };

    // Are we just restarting our own service on the same address (e.g. to apply
    // a settings change)? If so, skip the availability check — the port is
    // "taken" only by the process we're about to stop and rebind.
    let restarting_same_address = {
        let inner = state.inner.lock().expect("state lock poisoned");
        inner.service.port == Some(port) && inner.service.host.as_deref() == Some(host.as_str())
    };

    // Fail early with a friendly message if the fixed port is already taken,
    // rather than letting llama-server fail to bind and time out the health check.
    if configured_port != 0 && !restarting_same_address && !is_port_available(&host, port) {
        let info = set_failed(
            app,
            state,
            format!("接口端口 {} 已被占用，请在设置里换一个端口。", port),
        );
        return Err(AppError::Message(info.message));
    }

    let params = if backend == EngineBackend::Cuda {
        let hw = tokio::task::spawn_blocking(crate::hardware::detect_hardware)
            .await
            .unwrap_or_else(|_| crate::hardware::fallback_hardware());
        let vram = hw.gpu_list.iter().map(|g| g.vram_gb).fold(0.0_f64, f64::max);
        let data_dir = {
            let inner = state.inner.lock().expect("state lock poisoned");
            inner.data_dir.clone()
        };
        let model_required = crate::catalog::effective_models(&data_dir)
            .iter()
            .find(|m| m.id == model_id)
            .map(|m| m.vram_required_gb)
            .unwrap_or(0.0);
        if vram > 0.0 {
            cuda_launch_params(vram, model_required)
        } else {
            DEFAULT_CUDA_PARAMS
        }
    } else {
        LaunchParams { n_gpu_layers: None, ctx_size: 4096 }
    };

    set_starting(app, state, model_id, &host, port, false);
    tracing::info!(
        %model_id,
        backend = ?backend,
        host = %host,
        port,
        gpu_layers = ?params.n_gpu_layers,
        ctx_size = params.ctx_size,
        "starting llama-server"
    );
    match launch_service(&server_path, &model_path, &host, port, &params, &extra_args).await {
        Ok(child) => set_running(app, state, child, false, "AI服务已启动，可以开始聊天。".to_string()),
        Err(error) => {
            if backend == EngineBackend::Cpu {
                let info = set_failed(app, state, error.to_string());
                return Err(AppError::Message(info.message));
            }

            tracing::warn!(%model_id, error = %error, "CUDA launch failed; falling back to CPU safe mode");
            let Some(cpu_path) = find_cpu_engine_executable(&install_dir) else {
                let info = set_failed(
                    app,
                    state,
                    "显卡加速暂时没启动成功，也没有找到 CPU 备用运行引擎。".to_string(),
                );
                return Err(AppError::Message(info.message));
            };

            let safe_params = LaunchParams { n_gpu_layers: None, ctx_size: 2048 };
            set_starting(app, state, model_id, &host, port, true);
            match launch_service(&cpu_path, &model_path, &host, port, &safe_params, &extra_args).await {
                Ok(child) => set_running(
                    app,
                    state,
                    child,
                    true,
                    "显卡加速暂时没启动成功，已切换到更稳妥的 CPU 模式。".to_string(),
                ),
                Err(safe_error) => {
                    let info = set_failed(app, state, safe_error.to_string());
                    Err(AppError::Message(info.message))
                }
            }
        }
    }
}

#[tauri::command]
pub fn stop_model_service(
    app: AppHandle,
    state: tauri::State<'_, AppState>,
) -> AppResult<ServiceInfo> {
    Ok(stop_service_core(&app, &state))
}

/// Stop the running local service (killing our spawned child or a reattached
/// pid) and clear its persisted snapshot. Shared by the command and update flow.
pub fn stop_service_core(app: &AppHandle, state: &AppState) -> ServiceInfo {
    let info = {
        let mut inner = state.inner.lock().expect("state lock poisoned");
        if let Some(mut child) = inner.service.process.take() {
            let _ = child.kill();
        } else if let Some(pid) = inner.service.pid {
            // Reattached process from a previous run: we have no Child handle,
            // so stop it by pid instead.
            kill_pid(pid);
        }
        let _ = clear_service_state(&inner.db_path);
        inner.service.status = ServiceStatus::Stopped;
        inner.service.port = None;
        inner.service.host = None;
        inner.service.model_id = None;
        inner.service.pid = None;
        inner.service.safe_mode = false;
        ServiceInfo {
            model_id: None,
            port: None,
            host: None,
            status: ServiceStatus::Stopped,
            safe_mode: false,
            message: "AI服务已停止。".to_string(),
        }
    };
    let _ = app.emit("service_status_changed", &info);
    tracing::info!("llama-server stopped");
    info
}

/// Reattach to a `llama-server` left running by a previous app session, or clean
/// it up if it is no longer healthy. Should be called once on startup before the
/// frontend decides whether to auto-start a fresh service.
#[tauri::command]
pub async fn recover_service(
    app: AppHandle,
    state: tauri::State<'_, AppState>,
) -> AppResult<ServiceInfo> {
    let (db_path, persisted) = {
        let inner = state.inner.lock().expect("state lock poisoned");
        (inner.db_path.clone(), load_service_state(&inner.db_path)?)
    };

    let Some(persisted) = persisted else {
        return Ok(service_status(&state));
    };

    // A healthy port means the old process is still serving — reattach to it.
    if health_check_once(persisted.port).await {
        tracing::info!(port = persisted.port, model_id = %persisted.model_id, safe_mode = persisted.safe_mode, "reattached to previous llama-server");
        let info = {
            let mut inner = state.inner.lock().expect("state lock poisoned");
            inner.service.model_id = Some(persisted.model_id.clone());
            inner.service.port = Some(persisted.port);
            inner.service.host = Some(persisted.host.clone());
            inner.service.pid = Some(persisted.pid);
            inner.service.status = ServiceStatus::Running;
            inner.service.safe_mode = persisted.safe_mode;
            inner.service.process = None;
            ServiceInfo {
                model_id: Some(persisted.model_id),
                port: Some(persisted.port),
                host: Some(persisted.host),
                status: ServiceStatus::Running,
                safe_mode: persisted.safe_mode,
                message: if persisted.safe_mode {
                    "已恢复上次的 CPU 安全模式 AI服务。".to_string()
                } else {
                    "已恢复上次运行的本地AI服务。".to_string()
                },
            }
        };
        let _ = app.emit("service_status_changed", &info);
        return Ok(info);
    }

    // The old process is gone or unresponsive. Best-effort kill of any lingering
    // pid (in case it is a zombie holding the port), then clear the record.
    tracing::warn!(port = persisted.port, pid = persisted.pid, "persisted service not healthy; cleaning up stale record");
    kill_pid(persisted.pid);
    let _ = clear_service_state(&db_path);
    Ok(service_status(&state))
}

#[tauri::command]
pub fn get_service_status(state: tauri::State<'_, AppState>) -> ServiceInfo {
    service_status(&state)
}

fn service_status(state: &AppState) -> ServiceInfo {
    let inner = state.inner.lock().expect("state lock poisoned");
    ServiceInfo {
        model_id: inner.service.model_id.clone(),
        port: inner.service.port,
        host: inner.service.host.clone(),
        status: inner.service.status.clone(),
        safe_mode: inner.service.safe_mode,
        message: match inner.service.status {
            ServiceStatus::Running if inner.service.safe_mode => "AI服务正在 CPU 安全模式运行。".to_string(),
            ServiceStatus::Running => "AI服务正在运行。".to_string(),
            ServiceStatus::Starting => "AI服务正在启动。".to_string(),
            ServiceStatus::Failed => "AI服务遇到了问题。".to_string(),
            ServiceStatus::Stopped => "AI服务尚未启动。".to_string(),
        },
    }
}

fn resolve_paths(state: &AppState) -> AppResult<(PathBuf, PathBuf, PathBuf)> {
    let (server_path, model_path, install_dir) = {
        let inner = state.inner.lock().expect("state lock poisoned");
        (
            inner.settings.llama_server_path.clone(),
            inner.settings.local_model_path.clone(),
            inner.settings.install_dir.clone(),
        )
    };

    let server_path = if let Some(path) = server_path {
        if path.exists() {
            path
        } else {
            find_engine_executable(&install_dir)
                .ok_or_else(|| AppError::Message("AI运行引擎文件暂时找不到，请重新安装运行引擎。".to_string()))?
        }
    } else {
        find_engine_executable(&install_dir)
            .ok_or_else(|| AppError::Message("本地AI运行引擎还没准备好，请稍后重试。".to_string()))?
    };

    let model_path = model_path
        .ok_or_else(|| AppError::Message("模型文件还没准备好，请先完成安装。".to_string()))?;

    if !model_path.exists() {
        return Err(AppError::Message(
            "模型文件暂时找不到，请重新安装模型。".to_string(),
        ));
    }

    Ok((server_path, model_path, install_dir))
}

fn set_starting(
    app: &AppHandle,
    state: &AppState,
    model_id: &str,
    host: &str,
    port: u16,
    safe_mode: bool,
) -> ServiceInfo {
    let info = {
        let mut inner = state.inner.lock().expect("state lock poisoned");
        if let Some(mut old) = inner.service.process.take() {
            let _ = old.kill();
        } else if let Some(pid) = inner.service.pid {
            kill_pid(pid);
        }
        inner.service.pid = None;
        inner.service.model_id = Some(model_id.to_string());
        inner.service.host = Some(host.to_string());
        inner.service.port = Some(port);
        inner.service.status = ServiceStatus::Starting;
        inner.service.safe_mode = safe_mode;
        ServiceInfo {
            model_id: inner.service.model_id.clone(),
            port: inner.service.port,
            host: inner.service.host.clone(),
            status: inner.service.status.clone(),
            safe_mode: inner.service.safe_mode,
            message: if safe_mode {
                "正在切换到更稳妥的 CPU 模式。".to_string()
            } else {
                "本地AI服务正在启动。".to_string()
            },
        }
    };
    let _ = app.emit("service_status_changed", &info);
    info
}

async fn launch_service(
    server_path: &PathBuf,
    model_path: &PathBuf,
    host: &str,
    port: u16,
    params: &LaunchParams,
    extra_args: &[String],
) -> AppResult<Child> {
    let mut command = Command::new(server_path);
    crate::proc::hide_console(&mut command);
    command
        .arg("--model")
        .arg(model_path)
        .arg("--host")
        .arg(host)
        .arg("--port")
        .arg(port.to_string())
        .arg("--ctx-size")
        .arg(params.ctx_size.to_string());

    if let Some(n_gpu_layers) = params.n_gpu_layers {
        command.arg("--n-gpu-layers").arg(n_gpu_layers.to_string());
    }

    // Advanced users can append raw llama.cpp flags (risk-at-own) via settings.
    if !extra_args.is_empty() {
        command.args(extra_args);
    }

    let mut child = command.spawn().map_err(|_| {
        AppError::Message("AI服务暂时没能启动，请检查模型文件是否完整。".to_string())
    })?;

    // Require two consecutive healthy probes so a single transient 200 during
    // llama.cpp warmup doesn't get mistaken for a stable service.
    if let Err(error) = wait_for_health(port, 2).await {
        let _ = child.kill();
        return Err(error);
    }

    Ok(child)
}

fn set_running(
    app: &AppHandle,
    state: &AppState,
    child: Child,
    safe_mode: bool,
    message: String,
) -> AppResult<ServiceInfo> {
    let info = {
        let mut inner = state.inner.lock().expect("state lock poisoned");
        let pid = child.id();
        inner.service.process = Some(child);
        inner.service.pid = Some(pid);
        inner.service.status = ServiceStatus::Running;
        inner.service.safe_mode = safe_mode;
        if let (Some(model_id), Some(port), Some(host)) = (
            inner.service.model_id.clone(),
            inner.service.port,
            inner.service.host.clone(),
        ) {
            let persisted = PersistedServiceState {
                model_id,
                port,
                pid,
                safe_mode,
                host,
            };
            let _ = save_service_state(&inner.db_path, &persisted);
        }
        ServiceInfo {
            model_id: inner.service.model_id.clone(),
            port: inner.service.port,
            host: inner.service.host.clone(),
            status: inner.service.status.clone(),
            safe_mode: inner.service.safe_mode,
            message,
        }
    };

    let _ = app.emit("service_status_changed", &info);
    tracing::info!(model_id = ?info.model_id, port = ?info.port, safe_mode, "llama-server running");
    Ok(info)
}

async fn wait_for_health(port: u16, required_consecutive: u32) -> AppResult<()> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .map_err(|_| AppError::Message("AI服务检查组件暂时不可用。".to_string()))?;
    // Always probe via 127.0.0.1: both supported bind modes (127.0.0.1 and
    // 0.0.0.0) are reachable on the loopback locally, so the local health check
    // works regardless of whether LAN access is enabled.
    let url = format!("http://127.0.0.1:{}/v1/models", port);
    let deadline = Instant::now() + Duration::from_secs(90);
    let mut consecutive = 0;

    while Instant::now() < deadline {
        let healthy = client
            .get(&url)
            .send()
            .await
            .map(|response| response.status().is_success())
            .unwrap_or(false);
        if healthy {
            consecutive += 1;
            if consecutive >= required_consecutive.max(1) {
                return Ok(());
            }
        } else {
            consecutive = 0;
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }

    Err(AppError::Message(
        "AI服务启动超时，请稍后重试或换一个更小的模型。".to_string(),
    ))
}

/// Single health probe used by startup recovery to decide whether a persisted
/// service is still alive.
async fn health_check_once(port: u16) -> bool {
    let Ok(client) = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
    else {
        return false;
    };
    let url = format!("http://127.0.0.1:{}/v1/models", port);
    client
        .get(&url)
        .send()
        .await
        .map(|response| response.status().is_success())
        .unwrap_or(false)
}

/// Best-effort termination of a process (and its children) by pid. Used to stop
/// a reattached server or clean up a stale/zombie one on Windows.
fn kill_pid(pid: u32) {
    let mut command = Command::new("taskkill");
    command.args(["/PID", &pid.to_string(), "/T", "/F"]);
    crate::proc::hide_console(&mut command);
    let _ = command.output();
}

fn set_failed(app: &AppHandle, state: &AppState, message: String) -> ServiceInfo {
    tracing::error!(reason = %message, "llama-server failed to start");
    let info = {
        let mut inner = state.inner.lock().expect("state lock poisoned");
        if let Some(mut child) = inner.service.process.take() {
            let _ = child.kill();
        } else if let Some(pid) = inner.service.pid {
            kill_pid(pid);
        }
        inner.service.pid = None;
        let _ = clear_service_state(&inner.db_path);
        inner.service.status = ServiceStatus::Failed;
        inner.service.port = None;
        ServiceInfo {
            model_id: inner.service.model_id.clone(),
            port: inner.service.port,
            host: inner.service.host.clone(),
            status: inner.service.status.clone(),
            safe_mode: inner.service.safe_mode,
            message,
        }
    };
    let _ = app.emit("service_status_changed", &info);
    info
}

fn detect_backend(server_path: &PathBuf, install_dir: &PathBuf) -> EngineBackend {
    if server_path
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.to_ascii_lowercase().contains("cuda"))
        .unwrap_or(false)
    {
        return EngineBackend::Cuda;
    }

    let marker = install_dir.join("engine").join("current_backend.txt");
    std::fs::read_to_string(marker)
        .map(|content| {
            if content.to_ascii_lowercase().contains("cuda") {
                EngineBackend::Cuda
            } else {
                EngineBackend::Cpu
            }
        })
        .unwrap_or(EngineBackend::Cpu)
}

fn find_engine_executable(install_dir: &PathBuf) -> Option<PathBuf> {
    let engine_dir = install_dir.join("engine");
    let backend_dir = engine_dir.join("backends");

    let candidates = vec![
        backend_dir.join("llama-server-cuda.exe"),
        backend_dir.join("llama-server-cpu.exe"),
        engine_dir.join("llama-server.exe"),
    ];

    for candidate in candidates {
        if candidate.exists() {
            return Some(candidate);
        }
    }

    None
}

/// Whether `host:port` can be bound right now. Used as a pre-flight check for a
/// user-fixed port so we can surface a clear "port in use" message instead of
/// letting llama-server fail to bind and time out the health check. The listener
/// is dropped immediately, so there is a tiny bind race before the real spawn —
/// acceptable for a single-user desktop tool.
fn is_port_available(host: &str, port: u16) -> bool {
    TcpListener::bind((host, port)).is_ok()
}

/// Split a user-provided extra-args string into individual arguments on
/// whitespace, dropping empty tokens. Simple by design: quoted arguments with
/// embedded spaces are out of MVP scope.
fn parse_extra_args(raw: &str) -> Vec<String> {
    raw.split_whitespace().map(|token| token.to_string()).collect()
}

fn find_cpu_engine_executable(install_dir: &PathBuf) -> Option<PathBuf> {
    let engine_dir = install_dir.join("engine");
    let backend_dir = engine_dir.join("backends");
    let candidates = vec![
        backend_dir.join("llama-server-cpu.exe"),
        engine_dir.join("llama-server.exe"),
    ];

    for candidate in candidates {
        if candidate.exists() {
            return Some(candidate);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_params_scale_with_available_vram() {
        let required = 5.0;

        // Comfortable headroom: offload everything with the full context.
        assert_eq!(
            cuda_launch_params(7.0, required),
            LaunchParams { n_gpu_layers: Some(999), ctx_size: 4096 }
        );
        // Just fits: still full offload but a smaller context for KV headroom.
        assert_eq!(
            cuda_launch_params(5.2, required),
            LaunchParams { n_gpu_layers: Some(999), ctx_size: 3072 }
        );
        // Tight (>= 70% of requirement): partial offload.
        assert_eq!(
            cuda_launch_params(4.0, required),
            LaunchParams { n_gpu_layers: Some(20), ctx_size: 3072 }
        );
        // Low (>= 50%): fewer layers, minimal context.
        assert_eq!(
            cuda_launch_params(2.6, required),
            LaunchParams { n_gpu_layers: Some(12), ctx_size: 2048 }
        );
        // Very low: minimal offload, minimal context.
        assert_eq!(
            cuda_launch_params(1.0, required),
            LaunchParams { n_gpu_layers: Some(8), ctx_size: 2048 }
        );
    }

    #[test]
    fn cuda_params_are_monotonic_in_vram() {
        // More VRAM never offloads fewer layers than less VRAM.
        let required = 5.0;
        let layers = |vram: f64| cuda_launch_params(vram, required).n_gpu_layers.unwrap();
        assert!(layers(1.0) <= layers(2.6));
        assert!(layers(2.6) <= layers(4.0));
        assert!(layers(4.0) <= layers(5.2));
    }

    #[test]
    fn extra_args_split_on_whitespace() {
        assert_eq!(
            parse_extra_args("  --threads 8   --no-mmap "),
            vec![
                "--threads".to_string(),
                "8".to_string(),
                "--no-mmap".to_string()
            ]
        );
        assert!(parse_extra_args("").is_empty());
        assert!(parse_extra_args("   ").is_empty());
    }

    #[test]
    fn port_availability_reflects_binding() {
        // Bind an ephemeral port, then confirm it reads as unavailable while held
        // and available again once released.
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind ephemeral");
        let port = listener.local_addr().unwrap().port();
        assert!(!is_port_available("127.0.0.1", port));
        drop(listener);
        assert!(is_port_available("127.0.0.1", port));
    }
}
