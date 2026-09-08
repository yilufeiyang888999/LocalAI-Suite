use std::{
    net::TcpListener,
    path::PathBuf,
    process::{Child, Command},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
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
/// `parallel` is the number of llama.cpp slots — we pin it to 1 for the desktop
/// single-user case so the KV cache isn't multiplied across idle slots (the
/// default of 4 blows past small VRAM budgets). `kv_quant` requests an 8-bit KV
/// cache (`--cache-type-k/v q8_0` + `--flash-attn`), which roughly halves the
/// per-token KV footprint so a larger context fits — only enabled when fp16
/// can't hold the target context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LaunchParams {
    n_gpu_layers: Option<u32>,
    ctx_size: u32,
    parallel: u32,
    kv_quant: bool,
}

/// Default CUDA tuning when the model or VRAM is unknown — matches the old
/// fixed behavior so imported models keep working.
const DEFAULT_CUDA_PARAMS: LaunchParams = LaunchParams {
    n_gpu_layers: Some(24),
    ctx_size: 4096,
    parallel: 1,
    kv_quant: false,
};

/// Health-check budget for a GPU launch.
///
/// The first GPU cold start can be slow: Windows Defender scans the large CUDA
/// DLLs (`ggml-cuda.dll` is ~565MB), and CUDA may JIT-compile kernels for a
/// newer GPU arch not precompiled in the build (both results get cached, so
/// later starts are fast). 90s was empirically too tight and misfired as a
/// failure on a perfectly usable GPU, falling back to CPU.
///
/// 180s was too generous: with the two-attempt strategy (aggressive params
/// then conservative params) it made the user wait ~6 minutes before falling
/// back to CPU safe mode on a launch that was never going to succeed — most
/// commonly because Windows SmartScreen blocked the unsigned
/// `llama-server-vulkan.exe` at the OS layer (no process is ever spawned,
/// so the in-process fast-fail on `child.try_wait()` cannot see it).
///
/// 120s is the middle ground: still leaves headroom for cold-start DLL scan
/// + JIT, but trims the two-attempt total from ~6 min to ~4 min when the GPU
/// path is going to fail anyway. Real (working) GPU starts are observed at
/// 26–60s in practice.
const GPU_STARTUP_TIMEOUT_SECS: u64 = 120;

/// Health-check budget for a CPU launch (and CPU safe-mode fallback). CPU has no
/// JIT step, so the original tighter budget is fine.
const CPU_STARTUP_TIMEOUT_SECS: u64 = 90;

/// Runtime reserve on top of model weights + KV cache, in GB. Matches the
/// `0.5GB` run reserve in the VRAM formula, plus an extra `0.8GB` safety pad for
/// GPU compute buffers / driver overhead so we don't launch right at the edge
/// and spill into shared system memory (which tanks throughput).
const GPU_OVERHEAD_GB: f64 = 0.5 + 0.8;

/// Round a context length down to a multiple of 2048, with a 2048 floor.
fn round_ctx(ctx: f64) -> u32 {
    let stepped = (ctx / 2048.0).floor() as u32 * 2048;
    stepped.max(2048)
}

/// Choose GPU offload, context size, and KV-cache precision by inverting the
/// project's VRAM formula against the *actual* available VRAM, rather than a
/// fixed small context. The formula (see CLAUDE.md) is:
///
/// ```text
/// VRAM = weight + KV + reserve
/// KV   = 0.000008 × active_params(B) × ctx × kv_factor
/// ```
///
/// so given a VRAM budget we solve for the largest `ctx` that fits. fp16 KV is
/// preferred for stability; we only drop to an 8-bit KV cache (`kv_factor` 0.5)
/// when fp16 can't hold the target context. When even the weights barely fit we
/// fall back to partial offload with a small context (the old tiered behavior).
///
/// `weight_gb` is model weights (params × bpp), `active_params_billions` drives
/// the KV size, and `max_context` caps the context at the model's trained limit.
fn cuda_launch_params(
    available_vram_gb: f64,
    weight_gb: f64,
    active_params_billions: f64,
    max_context: u32,
) -> LaunchParams {
    let target = max_context.max(2048) as f64;
    // Per-token KV footprint in GB at fp16 (kv_factor = 1.0).
    let kv_per_token_fp16 = 0.000008 * active_params_billions;
    let budget = available_vram_gb - weight_gb - GPU_OVERHEAD_GB;

    // Weights don't fit (or barely) with any usable KV headroom: keep the old
    // partial-offload tiers so a low-VRAM GPU still starts, just with a small
    // context and an 8-bit KV cache to squeeze what we can.
    if budget <= 0.0 || kv_per_token_fp16 <= 0.0 {
        let (layers, ctx) = if available_vram_gb >= weight_gb * 0.7 {
            (20, 3072)
        } else if available_vram_gb >= weight_gb * 0.5 {
            (12, 2048)
        } else {
            (8, 2048)
        };
        return LaunchParams {
            n_gpu_layers: Some(layers),
            ctx_size: ctx,
            parallel: 1,
            kv_quant: true,
        };
    }

    // Prefer an fp16 KV cache when it holds the target context: for the app's
    // core case — short chats at a moderate context — fp16 is fastest, because
    // q8 KV adds a per-token dequant cost and its flash-attention win only pays
    // off at long contexts. Only drop to q8 (which this build pairs with flash
    // attention) when fp16 can't fit the target — e.g. an advanced user raised
    // `--ctx-size` well past the default — trading a little short-prompt speed
    // for the long-context throughput q8 + flash attention provides.
    let max_ctx_fp16 = budget / kv_per_token_fp16;
    if max_ctx_fp16 >= target {
        return LaunchParams {
            n_gpu_layers: Some(999),
            ctx_size: target as u32,
            parallel: 1,
            kv_quant: false,
        };
    }

    // fp16 falls short: use q8 KV (half the footprint + flash attention) to
    // reach the target, or the largest context q8 allows if even that's tight.
    let max_ctx_q8 = budget / (kv_per_token_fp16 * 0.5);
    let ctx_size = if max_ctx_q8 >= target {
        target as u32
    } else {
        round_ctx(max_ctx_q8)
    };
    LaunchParams {
        n_gpu_layers: Some(999),
        ctx_size,
        parallel: 1,
        kv_quant: true,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EngineBackend {
    Cpu,
    Cuda,
    Vulkan,
}

impl EngineBackend {
    /// Whether this backend offloads layers to the GPU (Vulkan or CUDA), so it
    /// should be launched with VRAM-tiered `-ngl` params and can fall back to CPU
    /// safe mode on failure.
    fn is_gpu(self) -> bool {
        matches!(self, EngineBackend::Cuda | EngineBackend::Vulkan)
    }
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
    // Resolve engine/model paths first: this can fail (missing engine or model
    // file) and must return *before* the guard below flips status to Starting,
    // otherwise a resolve failure would leave the service stuck in Starting.
    let (server_path, model_path, install_dir) = resolve_paths(state)?;

    // ---- Guard: kill any existing process and prevent concurrent starts ----
    //
    // Without this, a second call to start_service_core (e.g. from a frontend
    // double-click or a race between switchModel and an auto-recovery) would
    // spawn a duplicate llama-server whose Child handle overwrites the first
    // one's in set_running — but on Windows, dropping the Child does NOT kill
    // the process, so it becomes an orphan holding the port and VRAM.
    //
    // We do this *synchronously* before any `.await` point so the Mutex scope
    // is atomic: the first task to reach this sets status=Starting, and any
    // concurrent task sees it and bails.
    {
        let mut inner = state.inner.lock().expect("state lock poisoned");
        if inner.service.status == ServiceStatus::Starting {
            return Err(AppError::Message(
                "AI服务正在启动中，请稍候。".to_string(),
            ));
        }
        // Kill any existing process (whether spawned this session or reattached).
        if let Some(mut child) = inner.service.process.take() {
            let _ = child.kill();
        } else if let Some(pid) = inner.service.pid {
            kill_pid(pid);
        }
        inner.service.pid = None;
        inner.service.status = ServiceStatus::Starting;
        inner.service.model_id = Some(model_id.to_string());
        // Keep host/port — they are needed for the restarting_same_address
        // check below and will be overwritten by set_starting later.
    }
    // Emit *now* so the frontend's ThirdPartyConnect/ChatView cards see the
    // new model_id immediately, rather than waiting for set_starting /
    // set_running to fire after the await points below. Without this, if any
    // later step (port-availability check, GPU param computation) takes a
    // moment, the UI briefly (or in some Tauri 2 cases persistently) shows the
    // previous model_id even though the service is about to start with the
    // new one.
    let _ = app.emit(
        "service_status_changed",
        &service_status(state),
    );
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
    // We poll briefly rather than checking once: when we (or the update flow) just
    // killed the previous server, the OS may not have released the port yet — a
    // single immediate check would misfire and wrongly report "port taken" (which
    // made model updates always roll back). A genuinely external holder stays
    // taken past the wait window.
    if configured_port != 0 && !restarting_same_address && !wait_for_port_available(&host, port).await {
        let info = set_failed(
            app,
            state,
            format!("接口端口 {} 已被占用，请在设置里换一个端口。", port),
        );
        return Err(AppError::Message(info.message));
    }

    let params = if backend.is_gpu() {
        let hw = tokio::task::spawn_blocking(crate::hardware::detect_hardware)
            .await
            .unwrap_or_else(|_| crate::hardware::fallback_hardware());
        let total_vram = hw.gpu_list.iter().map(|g| g.vram_gb).fold(0.0_f64, f64::max);
        // Compute offload params from *free* VRAM, not total: other processes
        // (or a model we just stopped whose VRAM hasn't been reclaimed yet) may
        // hold much of the card. Using total VRAM here makes llama.cpp overcommit
        // and spill to shared system memory — a slow "pretend success" instead of
        // a proper partial-offload or CPU fallback. Fall back to total when
        // nvidia-smi can't report free VRAM (no worse than the old behavior).
        let vram = tokio::task::spawn_blocking(crate::hardware::nvidia_free_vram_gb)
            .await
            .ok()
            .flatten()
            .filter(|free| *free > 0.0)
            .unwrap_or(total_vram);
        tracing::info!(total_vram_gb = total_vram, usable_vram_gb = vram, "computing GPU launch params from usable VRAM");
        let data_dir = {
            let inner = state.inner.lock().expect("state lock poisoned");
            inner.data_dir.clone()
        };
        let model = crate::catalog::effective_models(&data_dir)
            .into_iter()
            .find(|m| m.id == model_id);
        match (vram > 0.0, model) {
            (true, Some(m)) => {
                // Weights in GB and active params (billions) drive the VRAM
                // formula; active_parameters covers MoE, else the full count.
                let weight_gb = (m.parameters_raw as f64) * m.quant_bpp / 1e9;
                let active = m.active_parameters.unwrap_or(m.parameters_raw);
                let active_billions = active as f64 / 1e9;
                let max_ctx = (m.context_length.min(16384)) as u32;
                cuda_launch_params(vram, weight_gb, active_billions, max_ctx)
            }
            // Unknown VRAM or an imported model without metadata: keep the old
            // conservative fixed params.
            _ => DEFAULT_CUDA_PARAMS,
        }
    } else {
        LaunchParams { n_gpu_layers: None, ctx_size: 4096, parallel: 1, kv_quant: false }
    };

    set_starting(app, state, model_id, &host, port, false);
    tracing::info!(
        %model_id,
        backend = ?backend,
        host = %host,
        port,
        gpu_layers = ?params.n_gpu_layers,
        ctx_size = params.ctx_size,
        parallel = params.parallel,
        kv_quant = params.kv_quant,
        "starting llama-server"
    );
    let startup_timeout = if backend.is_gpu() {
        GPU_STARTUP_TIMEOUT_SECS
    } else {
        CPU_STARTUP_TIMEOUT_SECS
    };
    match launch_service(&server_path, &model_path, &host, port, &params, &extra_args, startup_timeout).await {
        Ok(child) => set_running(app, state, child, false, "AI服务已启动，可以开始聊天。".to_string()),
        Err(error) => {
            if backend == EngineBackend::Cpu {
                let info = set_failed(app, state, error.to_string());
                return Err(AppError::Message(info.message));
            }

            // Before giving up on the GPU, retry once with conservative GPU
            // params (small context, fp16 KV, single slot). The aggressive
            // params above may have asked for a context the VRAM couldn't hold;
            // a smaller one often still runs fully on the GPU — far better than
            // crawling on the CPU when the card is actually fine.
            let conservative = LaunchParams {
                n_gpu_layers: Some(999),
                ctx_size: 4096,
                parallel: 1,
                kv_quant: false,
            };
            if conservative != params {
                tracing::warn!(%model_id, backend = ?backend, error = %error, "GPU launch failed; retrying with conservative GPU params");
                set_starting(app, state, model_id, &host, port, false);
                if let Ok(child) = launch_service(&server_path, &model_path, &host, port, &conservative, &extra_args, startup_timeout).await {
                    return set_running(app, state, child, false, "AI服务已启动，可以开始聊天。".to_string());
                }
            }

            tracing::warn!(%model_id, backend = ?backend, error = %error, "GPU launch failed; falling back to CPU safe mode");
            // The CPU fallback may not be on disk yet because `ensure_engine_ready`
            // for a GPU backend now downloads it in the background (so a slow
            // network doesn't block the user who asked for GPU). Try once to
            // fetch it synchronously here — if a download is already in flight,
            // the call is fast; if not, it's a 16 MB package and worth a short
            // wait before we tell the user "no CPU fallback either."
            let cpu_path = match find_cpu_engine_executable(&install_dir) {
                Some(path) => path,
                None => {
                    let engine_dir = install_dir.join("engine");
                    let backend_dir = engine_dir.join("backends");
                    if let Err(fallback_error) = crate::engine::prepare_cpu_package(&engine_dir, &backend_dir).await {
                        tracing::warn!(error = %fallback_error, "CPU fallback engine fetch failed during GPU fallback");
                        let info = set_failed(
                            app,
                            state,
                            "显卡加速暂时没启动成功，也没有找到 CPU 备用运行引擎。".to_string(),
                        );
                        return Err(AppError::Message(info.message));
                    }
                    match find_cpu_engine_executable(&install_dir) {
                        Some(path) => path,
                        None => {
                            let info = set_failed(
                                app,
                                state,
                                "显卡加速暂时没启动成功，也没有找到 CPU 备用运行引擎。".to_string(),
                            );
                            return Err(AppError::Message(info.message));
                        }
                    }
                }
            };

            let safe_params = LaunchParams { n_gpu_layers: None, ctx_size: 2048, parallel: 1, kv_quant: false };
            set_starting(app, state, model_id, &host, port, true);
            match launch_service(&cpu_path, &model_path, &host, port, &safe_params, &extra_args, CPU_STARTUP_TIMEOUT_SECS).await {
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
    // Stop the auto-restart watchdog *first* so it doesn't try to revive the
    // process we are about to kill.
    stop_service_watchdog(state);
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
        // Reattach path: the old process is alive and answering, so we want
        // the same auto-restart safety net as a freshly spawned service.
        spawn_service_watchdog(app.clone(), (*state).clone());
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
    startup_timeout_secs: u64,
) -> AppResult<Child> {
    // True iff `extra_args` already contains a flag the user explicitly
    // passed in `extra_launch_args`. We skip our auto-derived copy in that
    // case so llama.cpp doesn't see the same flag twice (it now logs a
    // DEPRECATED warning for repeated `--ctx-size` etc. and only honors
    // the last value, which can silently overrule the computed one).
    let extra_has = |flag: &str| -> bool {
        let prefix_eq = format!("{}=", flag);
        extra_args
            .iter()
            .any(|token| token == flag || token.starts_with(&prefix_eq))
    };

    let mut command = Command::new(server_path);
    crate::proc::hide_console(&mut command);
    command
        .arg("--model")
        .arg(model_path)
        .arg("--host")
        .arg(host)
        .arg("--port")
        .arg(port.to_string());
    if !extra_has("--ctx-size") {
        command.arg("--ctx-size").arg(params.ctx_size.to_string());
    }
    // Pin to a single slot: this is a single-user desktop app, and the
    // llama.cpp default of 4 slots multiplies the KV cache 4× — the root
    // cause of blowing past small VRAM budgets and spilling to system RAM.
    if !extra_has("--parallel") {
        command.arg("--parallel").arg(params.parallel.to_string());
    }

    if let Some(n_gpu_layers) = params.n_gpu_layers {
        if !extra_has("--n-gpu-layers") {
            command.arg("--n-gpu-layers").arg(n_gpu_layers.to_string());
        }
    }

    // An 8-bit KV cache roughly halves the per-token KV footprint so a larger
    // context fits in tight VRAM. This matches the verified-working manual
    // config (`--cache-type-k/v q8_0`); this llama.cpp build auto-enables flash
    // attention when the KV cache is quantized, so we don't pass `--flash-attn`
    // explicitly (an untested flag on this build).
    if params.kv_quant
        && !extra_has("--cache-type-k")
        && !extra_has("--cache-type-v")
    {
        command
            .arg("--cache-type-k")
            .arg("q8_0")
            .arg("--cache-type-v")
            .arg("q8_0");
    }

    // Advanced users can append raw llama.cpp flags (risk-at-own) via settings.
    if !extra_args.is_empty() {
        command.args(extra_args);
    }

    let mut child = command.spawn().map_err(|_| {
        AppError::Message("AI服务暂时没能启动，请检查模型文件是否完整。".to_string())
    })?;

    // Require two consecutive healthy probes so a single transient 200 during
    // llama.cpp warmup doesn't get mistaken for a stable service. The health
    // wait also watches the child: if the process exits early (corrupted exe,
    // missing DLL, incompatible GPU backend), we fail fast instead of waiting
    // out the full 180s timeout.
    if let Err(error) = wait_for_health(&mut child, port, 2, startup_timeout_secs).await {
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
    spawn_service_watchdog(app.clone(), state.clone());
    tracing::info!(model_id = ?info.model_id, port = ?info.port, safe_mode, "llama-server running");
    Ok(info)
}

async fn wait_for_health(
    child: &mut Child,
    port: u16,
    required_consecutive: u32,
    timeout_secs: u64,
) -> AppResult<()> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .map_err(|_| AppError::Message("AI服务检查组件暂时不可用。".to_string()))?;
    // Always probe via 127.0.0.1: both supported bind modes (127.0.0.1 and
    // 0.0.0.0) are reachable on the loopback locally, so the local health check
    // works regardless of whether LAN access is enabled.
    let url = format!("http://127.0.0.1:{}/v1/models", port);
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    let mut consecutive = 0;

    while Instant::now() < deadline {
        // Fast-fail: if the server process has already exited (corrupted exe,
        // missing DLL, incompatible GPU backend, out-of-memory abort), stop
        // waiting immediately rather than burning the whole timeout. This is
        // what keeps a failed GPU launch from making the user wait ~180s per
        // attempt before falling back to CPU safe mode.
        if let Ok(Some(exit_status)) = child.try_wait() {
            let code = exit_status.code().unwrap_or(-1);
            return Err(AppError::Message(format!(
                "AI服务进程启动后异常退出（退出码 {}），可能是运行引擎与硬件不兼容或模型文件损坏。",
                code
            )));
        }

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

/// Spawn a background task that watches the running llama-server process and
/// automatically restarts it if it dies unexpectedly (e.g. OOM abort, GPU
/// driver crash, user killed it from Task Manager). User-initiated stops go
/// through `stop_service_core` which sets the stop flag first, so the loop
/// exits cleanly without trying to revive a service the user just killed.
///
/// This is intentionally minimal: process-alive polling + bounded restart
/// with exponential backoff. It does NOT do periodic health-probe of the
/// HTTP endpoint — the in-process `try_wait` is cheaper and catches all
/// crash classes the user would care about (process gone = unhealthy). A
/// hung process that still answers `/v1/models` but never produces tokens
/// is a separate, much harder problem and is out of scope here.
pub fn spawn_service_watchdog(app: AppHandle, state: AppState) {
    // Stop any previous watchdog (e.g. when switching models restarts the
    // service, or recover_service re-spawns after a crash). The old loop
    // sees the flag and exits on its next iteration; new loop takes over.
    stop_service_watchdog(&state);

    let stop = Arc::new(AtomicBool::new(false));
    {
        let mut inner = state.inner.lock().expect("state lock poisoned");
        inner.watchdog_stop = Some(stop.clone());
    }

    let app_for_task = app.clone();
    let state_for_task = state.clone();
    let stop_for_task = stop.clone();

    tokio::spawn(async move {
        service_watchdog_loop(app_for_task, state_for_task, stop_for_task).await;
    });

    tracing::info!("service watchdog spawned");
}

/// Signal the watchdog loop to exit on its next iteration. Safe to call when
/// no watchdog is running.
pub fn stop_service_watchdog(state: &AppState) {
    let prev = {
        let mut inner = state.inner.lock().expect("state lock poisoned");
        inner.watchdog_stop.take()
    };
    if let Some(stop_flag) = prev {
        stop_flag.store(true, Ordering::Relaxed);
    }
}

async fn service_watchdog_loop(app: AppHandle, state: AppState, stop: Arc<AtomicBool>) {
    /// How often the watchdog polls `child.try_wait()`.
    const POLL: Duration = Duration::from_secs(5);
    /// Backoff between successive restart attempts. The first attempt is
    /// immediate; subsequent ones wait progressively longer to give the
    /// system (GPU driver, free VRAM, etc.) time to settle, and to avoid
    /// tight crash-restart loops when the underlying cause persists.
    const BACKOFF: [Duration; 6] = [
        Duration::from_secs(0),  // 1st: immediate
        Duration::from_secs(5),  // 2nd
        Duration::from_secs(15), // 3rd
        Duration::from_secs(30), // 4th
        Duration::from_secs(60), // 5th
        Duration::from_secs(60), // 6th+: cap at 60s
    ];
    /// After this many consecutive restart failures the watchdog gives up and
    /// the service is left in `Failed` state for the user to handle (likely
    /// the same root cause that killed the original process will keep
    /// killing every restart: bad driver, OOM, missing file, etc.).
    const MAX_FAILURES: u32 = 5;

    let mut consecutive_failures: u32 = 0;
    tracing::info!("service watchdog loop entered");

    while !stop.load(Ordering::Relaxed) {
        tokio::time::sleep(POLL).await;
        if stop.load(Ordering::Relaxed) {
            break;
        }

        // Step 1: bail out if the service is no longer expected to be running.
        // (User pressed Stop, or a previous restart already failed and
        // set_failed flipped the status — either way, no point watching.)
        let (should_monitor, model_id) = {
            let inner = state.inner.lock().expect("state lock poisoned");
            (
                matches!(inner.service.status, ServiceStatus::Running),
                inner.service.model_id.clone(),
            )
        };
        if !should_monitor {
            break;
        }

        // Step 2: is the process still alive? `try_wait` is non-blocking and
        // returns Some(exit) iff the OS has reaped the child.
        let exit_status = {
            let mut inner = state.inner.lock().expect("state lock poisoned");
            match inner.service.process.as_mut() {
                Some(child) => child.try_wait().ok().flatten(),
                // Reattached process: we have no Child handle, so we can't
                // poll the kernel for it. Skip — the next start_service_core
                // path will respawn us with a fresh handle.
                None => None,
            }
        };
        let Some(exit) = exit_status else {
            continue;
        };

        let Some(model_id) = model_id else {
            tracing::warn!("watchdog: process died but no model_id recorded; exiting");
            break;
        };

        tracing::warn!(
            model_id = %model_id,
            exit_code = ?exit.code(),
            "watchdog: llama-server crashed, scheduling restart"
        );

        // Clear the dead child handle and reset status so start_service_core
        // can run its "kill any existing + set Starting" guard cleanly. The
        // previous `process` is already dead so dropping it is a no-op.
        {
            let mut inner = state.inner.lock().expect("state lock poisoned");
            inner.service.process = None;
            inner.service.pid = None;
            inner.service.status = ServiceStatus::Stopped;
        }

        // Step 3: apply the backoff schedule. Skip the sleep entirely on the
        // first attempt so a one-off hiccup (model load race, transient VRAM
        // pressure) recovers without the user noticing.
        let backoff_idx = (consecutive_failures as usize).min(BACKOFF.len() - 1);
        let backoff = BACKOFF[backoff_idx];
        if !backoff.is_zero() {
            tracing::info!(
                model_id = %model_id,
                backoff_secs = backoff.as_secs(),
                "watchdog: backoff before restart"
            );
            tokio::time::sleep(backoff).await;
            if stop.load(Ordering::Relaxed) {
                break;
            }
        }

        // Step 4: try to bring the service back up. On success, the normal
        // start_service_core → set_running path will respawn *this* watchdog
        // (with a fresh flag) and the old loop here is already past its
        // useful life; that's fine, the new loop will take over.
        match start_service_core(&model_id, &app, &state).await {
            Ok(info) => {
                tracing::info!(
                    model_id = ?info.model_id,
                    "watchdog: restart succeeded"
                );
                consecutive_failures = 0;
            }
            Err(error) => {
                consecutive_failures += 1;
                tracing::error!(
                    model_id = %model_id,
                    attempt = consecutive_failures,
                    max = MAX_FAILURES,
                    error = %error,
                    "watchdog: restart failed"
                );
                if consecutive_failures >= MAX_FAILURES {
                    tracing::error!(
                        model_id = %model_id,
                        "watchdog: giving up after {} consecutive failures; user intervention required",
                        MAX_FAILURES
                    );
                    break;
                }
            }
        }
    }

    tracing::info!("service watchdog loop exiting");
}

fn detect_backend(server_path: &PathBuf, install_dir: &PathBuf) -> EngineBackend {
    if let Some(name) = server_path
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.to_ascii_lowercase())
    {
        if name.contains("vulkan") {
            return EngineBackend::Vulkan;
        }
        if name.contains("cuda") {
            return EngineBackend::Cuda;
        }
    }

    let marker = install_dir.join("engine").join("current_backend.txt");
    std::fs::read_to_string(marker)
        .map(|content| {
            let content = content.to_ascii_lowercase();
            if content.contains("vulkan") {
                EngineBackend::Vulkan
            } else if content.contains("cuda") {
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
        backend_dir.join("llama-server-vulkan.exe"),
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

/// Poll `is_port_available` for a short window. Returns true as soon as the port
/// is free. This absorbs the brief delay between killing our previous server and
/// the OS releasing its socket, so a same-port restart (settings change, model
/// update) doesn't misread the not-yet-released port as an external conflict.
/// A port genuinely held by another program stays taken for the whole window and
/// returns false.
async fn wait_for_port_available(host: &str, port: u16) -> bool {
    const ATTEMPTS: u32 = 12;
    const INTERVAL: Duration = Duration::from_millis(250);
    for attempt in 0..ATTEMPTS {
        if is_port_available(host, port) {
            return true;
        }
        if attempt + 1 < ATTEMPTS {
            tokio::time::sleep(INTERVAL).await;
        }
    }
    false
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

    // Qwen3-8B Q4_K_M: 8.2B params × 0.4 bpp = 3.28GB weights, 8.2B active.
    // Used across the tuning tests below.
    const W8B: f64 = 3.28;
    const A8B: f64 = 8.2;

    #[test]
    fn ample_vram_uses_fp16_at_default_context() {
        // 8GB card holds 8B weights + 16k fp16 KV with comfort. fp16 is fastest
        // for short prompts — the core use case — so we prefer it over q8.
        let p = cuda_launch_params(8.0, W8B, A8B, 16384);
        assert_eq!(p.n_gpu_layers, Some(999));
        assert_eq!(p.ctx_size, 16384);
        assert_eq!(p.parallel, 1);
        assert!(!p.kv_quant, "fp16 should hold the default context easily");
    }

    #[test]
    fn medium_vram_still_reaches_jumbo_context_on_q8() {
        // 6GB: fp16 falls short of 32k but q8 (half footprint + flash attention)
        // reaches it. This is the path an advanced user hits by raising
        // --ctx-size past what fp16 can hold on a smaller card.
        let p = cuda_launch_params(6.0, W8B, A8B, 32768);
        assert_eq!(p.n_gpu_layers, Some(999));
        assert_eq!(p.ctx_size, 32768);
        assert!(p.kv_quant, "large context beyond fp16 capacity should use q8 KV");
        assert_eq!(p.parallel, 1);
    }

    #[test]
    fn tight_vram_takes_largest_q8_context() {
        // 5.5GB: even q8 can't reach 32k, so take the largest q8 context,
        // rounded down to a 2048 multiple (and never below 2048).
        let p = cuda_launch_params(5.5, W8B, A8B, 32768);
        assert_eq!(p.n_gpu_layers, Some(999));
        assert!(p.kv_quant);
        assert!(p.ctx_size < 32768 && p.ctx_size >= 2048);
        assert_eq!(p.ctx_size % 2048, 0);
    }

    #[test]
    fn very_low_vram_falls_back_to_partial_offload() {
        // Weights barely fit: keep the old partial-offload tiers so the GPU
        // still starts with a small context.
        let tight = cuda_launch_params(4.0, W8B, A8B, 32768);
        assert_eq!(tight.n_gpu_layers, Some(20));
        assert_eq!(tight.parallel, 1);

        let lower = cuda_launch_params(2.0, W8B, A8B, 32768);
        assert_eq!(lower.n_gpu_layers, Some(12));

        let lowest = cuda_launch_params(1.0, W8B, A8B, 32768);
        assert_eq!(lowest.n_gpu_layers, Some(8));
    }

    #[test]
    fn context_never_exceeds_model_trained_limit() {
        // A tiny model with a small trained context must not be over-extended
        // just because there's plenty of VRAM.
        let p = cuda_launch_params(16.0, 1.5, 1.7, 8192);
        assert!(p.ctx_size <= 8192);
        assert!(!p.kv_quant, "small context at high VRAM should use fp16");
    }

    #[test]
    fn context_is_monotonic_in_vram() {
        // More VRAM never yields a smaller context (at equal or better KV
        // precision), across the full range.
        let ctx = |vram: f64| cuda_launch_params(vram, W8B, A8B, 32768).ctx_size;
        assert!(ctx(4.0) <= ctx(5.5));
        assert!(ctx(5.5) <= ctx(6.0));
        assert!(ctx(6.0) <= ctx(8.0));
    }

    #[test]
    fn all_gpu_launches_pin_single_slot() {
        // The single-slot pin is the core fix for the 4× KV blowup; it must
        // hold across every VRAM tier.
        for vram in [1.0, 2.0, 4.0, 5.5, 6.0, 8.0, 16.0] {
            let p = cuda_launch_params(vram, W8B, A8B, 32768);
            assert_eq!(p.parallel, 1, "vram={vram}");
        }
    }

    #[test]
    fn round_ctx_floors_to_2048_multiple_with_floor() {
        assert_eq!(round_ctx(28048.0), 26624);
        assert_eq!(round_ctx(4095.0), 2048);
        assert_eq!(round_ctx(100.0), 2048);
        assert_eq!(round_ctx(4096.0), 4096);
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

    #[test]
    fn detect_backend_reads_executable_name() {
        let install = PathBuf::from("/does/not/matter");
        assert_eq!(
            detect_backend(&PathBuf::from("/x/llama-server-vulkan.exe"), &install),
            EngineBackend::Vulkan
        );
        assert_eq!(
            detect_backend(&PathBuf::from("/x/llama-server-cuda.exe"), &install),
            EngineBackend::Cuda
        );
        // A generic name with no missing marker file falls back to CPU.
        assert_eq!(
            detect_backend(&PathBuf::from("/x/llama-server.exe"), &install),
            EngineBackend::Cpu
        );
    }

    #[test]
    fn gpu_backends_report_is_gpu() {
        assert!(EngineBackend::Vulkan.is_gpu());
        assert!(EngineBackend::Cuda.is_gpu());
        assert!(!EngineBackend::Cpu.is_gpu());
    }
}
