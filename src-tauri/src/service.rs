use std::{
    net::TcpListener,
    path::{Path, PathBuf},
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

/// fp16 与目标上下文的差距在这个比例以内时，不再为那点长度降精度。
///
/// 这是一个真实存在的边界情形：Qwen3-1.7B 在 8G 卡上 fp16 能开到约 38.9K，
/// 离原生 40960 只差 5% —— 为这点长度把整段 KV 换成 q8 并不划算。
const FP16_SHORTFALL_TOLERANCE: f64 = 1.05;

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
/// KV   = kv_gb_per_token × ctx × kv_factor + kv_fixed_gb
/// ```
///
/// so given a VRAM budget we solve for the largest `ctx` that fits. fp16 KV is
/// preferred for stability; we only drop to an 8-bit KV cache (`kv_factor` 0.5)
/// when fp16 can't hold the target context. When even the weights barely fit we
/// fall back to partial offload with a small context (the old tiered behavior).
///
/// `weight_gb`, `kv_gb_per_token_fp16` and `kv_fixed_gb` are all in **GiB**,
/// matching `nvidia-smi`. Both KV figures come from the model's real GGUF layer
/// structure (`resolve_kv_profile`), *not* from parameter count: hybrid-attention
/// models only grow KV in their full-attention layers, so a parameter-count rule
/// of thumb errs in both directions — measured on local files, Qwen3-1.7B comes
/// out 7.8× too low and Ornith-1.5-9B 2.1× too high.
///
/// `kv_fixed_gb` is the slice that stops growing once the sliding window fills
/// (54 MiB on Spark-X2.5-4B). It is subtracted from the budget *before* the
/// division, because only the per-token term is proportional to ctx — folding it
/// into the per-token figure would over-estimate the reachable context.
/// `max_context` caps the context at the model's trained limit.
fn cuda_launch_params(
    available_vram_gb: f64,
    weight_gb: f64,
    kv_gb_per_token_fp16: f64,
    kv_fixed_gb: f64,
    max_context: u32,
) -> LaunchParams {
    let target = max_context.max(2048) as f64;
    let budget = available_vram_gb - weight_gb - GPU_OVERHEAD_GB - kv_fixed_gb;
    // q8 KV 是半量（kv_factor 0.5），也是这里能给出的最省档；先算出来用于
    // 判断「是否还剩任何可用余量」。
    let max_ctx_q8 = budget / (kv_gb_per_token_fp16 * 0.5);

    // 三种情况下放弃全量卸载，退回按层数分档的部分卸载：
    //   - 权重加保留量已经吃满显存（budget ≤ 0）；
    //   - KV 参数不可用（层结构读不到、兜底系数也为 0）；
    //   - 连最小可用上下文（2048）都放不下。
    // 最后一条是必需的：`round_ctx` 有 2048 下限，若在这里硬给 2048，
    // 实际占用会超出预算，llama.cpp 把 KV 溢出到共享内存 —— 结果就是
    // 「启动成功但龟速」，比明确走部分卸载更糟。
    if budget <= 0.0 || kv_gb_per_token_fp16 <= 0.0 || max_ctx_q8 < 2048.0 {
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
    // q8 KV adds a per-token dequant cost, and its flash-attention win only pays
    // off at long contexts. Only drop to q8 (which this build pairs with flash
    // attention) when fp16 genuinely can't fit the target — e.g. an advanced
    // user raised `--ctx-size` well past what the card holds — trading a little
    // short-prompt speed for the long-context throughput q8 + flash attention
    // provides.
    let max_ctx_fp16 = budget / kv_gb_per_token_fp16;
    if max_ctx_fp16 * FP16_SHORTFALL_TOLERANCE >= target {
        // 够到目标就开满；只差一点点（容忍带内）则开 fp16 的物理上限。
        // 后者不是绣花：Qwen3-1.7B 在 8G 卡上 fp16 能开到约 38.9K，离原生
        // 40960 只差 5%，为这点长度把整段 KV 降成 q8 并不划算。
        let ctx_size = if max_ctx_fp16 >= target {
            target as u32
        } else {
            round_ctx(max_ctx_fp16)
        };
        return LaunchParams {
            n_gpu_layers: Some(999),
            ctx_size,
            parallel: 1,
            kv_quant: false,
        };
    }

    // fp16 falls short: use q8 KV (half the footprint + flash attention) to
    // reach the target, or the largest context q8 allows if even that's tight.
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

/// 旧的 KV 经验系数：每十亿参数每 token 的 fp16 KV 占用（GiB）。
///
/// 只在拿不到 GGUF 层结构时兜底。它对常规全注意力模型尚可，对混合注意力
/// 模型会错得离谱（实测低估 7.8× 到高估 2.1×），所以**不该**是首选路径。
const LEGACY_KV_GIB_PER_TOKEN_PER_BILLION: f64 = 0.000008;

/// fp16（f16）元素字节数。
const KV_ELEMENT_BYTES_FP16: f64 = 2.0;

/// 权重显存占用（GiB）。
///
/// 优先用真实文件大小 —— 它与 `recommend.rs::model_weight_gb` 同一口径，
/// 也是唯一能反映「这个文件到底是什么量化档」的量。内置清单里的
/// `quant_bpp` 是理论值（Q4_K_M 记 0.4），实测为 0.614，按它算会把
/// Qwen3-8B 的权重低估 1.53×。文件大小缺失时才回落到参数量 × bpp。
fn resolve_weight_gb(model: &crate::models::ModelMetadata) -> f64 {
    if model.file_size_mb > 0 {
        // `file_size_mb` 是十进制的 10⁶ 字节，换算成 GiB 要再除 1024³ —— 直接
        // 除 1024 得到的是「按 1024 进制读十进制 MB」，比真值高约 4.9%。
        return (model.file_size_mb as f64) * 1_000_000.0 / crate::gguf::每GiB字节;
    }
    (model.parameters_raw as f64) * model.quant_bpp / crate::gguf::每GiB字节
}

/// 一个模型在 fp16 精度下的 KV 占用画像（GiB）。
///
/// 拆成两项是因为它们在预算里的行为完全不同：`per_token_fp16` 随上下文
/// 线性增长，`fixed_fp16` 是滑窗层封顶后的常量。预算反解 ctx 时只能对前者
/// 做除法，后者必须先扣掉 —— 否则算出的上下文会虚高。
#[derive(Debug, Clone, Copy, Default)]
struct KvProfile {
    /// 每 token 的增量（只含全注意力层）。
    per_token_fp16: f64,
    /// 不随上下文增长的常量（滑动窗口层封顶后）。
    fixed_fp16: f64,
}

/// 从模型文件读 KV 占用画像（fp16，GiB）。
///
/// 现读模型文件的 GGUF 头，按真实层结构正算：`per_token = 全注意力层数 ×
/// KV头数 × (key维 + value维) × 元素字节`。只有全注意力层的 KV 随上下文
/// 线性增长，混合注意力模型（状态空间 / 滑动窗口）按总层数或参数量算都会
/// 失真。
///
/// 现读而非落库，是因为**事实来源就是文件本身** —— 用户换掉同名文件、
/// 或从别的量化档重下，估算会自动跟上，不会留下失同步的缓存。
/// 解析失败时回落旧经验公式（文件缺失 / safetensors 原版 / 字段不全），
/// 保持原有行为不倒退；回落路径的 `fixed_fp16` 取 0，因为旧公式本来就
/// 只描述线性增长的那一部分。
fn resolve_kv_profile(
    model: &crate::models::ModelMetadata,
    fallback_params_billions: f64,
) -> KvProfile {
    let from_layers = model
        .local_path
        .as_deref()
        .filter(|path| !path.is_empty())
        .and_then(|path| crate::gguf::parse_file(std::path::Path::new(path)).ok())
        .map(|meta| KvProfile {
            per_token_fp16: meta.kv_bytes_per_token(KV_ELEMENT_BYTES_FP16)
                / crate::gguf::每GiB字节,
            fixed_fp16: meta.kv_fixed_gb(KV_ELEMENT_BYTES_FP16),
        })
        .filter(|profile| profile.per_token_fp16 > 0.0);

    if let Some(profile) = from_layers {
        tracing::info!(
            model_id = %model.id,
            kv_gib_per_token = profile.per_token_fp16,
            kv_fixed_gib = profile.fixed_fp16,
            "resolved KV footprint from real GGUF layer structure"
        );
        return profile;
    }

    tracing::debug!(
        model_id = %model.id,
        "no GGUF layer structure available; falling back to parameter-count KV estimate"
    );
    KvProfile {
        per_token_fp16: LEGACY_KV_GIB_PER_TOKEN_PER_BILLION * fallback_params_billions,
        fixed_fp16: 0.0,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineBackend {
    Cpu,
    Cuda,
    Vulkan,
}

impl EngineBackend {
    /// Whether this backend offloads layers to the GPU (Vulkan or CUDA), so it
    /// should be launched with VRAM-tiered `-ngl` params and can fall back to CPU
    /// safe mode on failure.
    pub fn is_gpu(self) -> bool {
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
                // 权重取真实文件大小，KV 现读 GGUF 层结构，两者都是 GiB。
                // active_parameters 覆盖 MoE，否则用总参数量做 KV 兜底。
                let active = m.active_parameters.unwrap_or(m.parameters_raw);
                let active_billions = active as f64 / 1e9;
                // 目标上下文取模型原生上限，不再硬编码 16384。
                //
                // 放开是安全的：实际开多大由 `cuda_launch_params` 按可用显存
                // 反解，物理上封顶、不会超预算，算不出来还有保守重试档与
                // CPU 安全模式兜底；而 KV 现在也是按真实层结构正算，不再是
                // 那个按参数量反推的经验式。
                //
                // 不放开则长上下文模型永远吃不到自己的长处 —— Spark-X2.5-4B
                // 原生 1M，过去被死死卡在 16K，且**换任何量化档都突破不了**。
                // 想要更短的上下文，仍可用 extra_launch_args 的 `--ctx-size`
                // 显式覆盖（launch_service 会跳过自动值，只保留用户那一份）。
                // context_length 在结构体里统一是 u64，但真实模型最大也就
                // 1M 量级；夹一次 u32 上限只为防异常元数据，不是常规路径。
                let max_ctx = m.context_length.min(u32::MAX as u64) as u32;
                let weight_gb = resolve_weight_gb(&m);
                // 读 GGUF 头是文件 IO：即使只需毫秒，也不该占着 async runtime
                // 的线程（与上面的硬件探测同理）。
                let kv_profile = tokio::task::spawn_blocking(move || {
                    resolve_kv_profile(&m, active_billions)
                })
                .await
                .unwrap_or(KvProfile {
                    per_token_fp16: LEGACY_KV_GIB_PER_TOKEN_PER_BILLION * active_billions,
                    fixed_fp16: 0.0,
                });
                cuda_launch_params(
                    vram,
                    weight_gb,
                    kv_profile.per_token_fp16,
                    kv_profile.fixed_fp16,
                    max_ctx,
                )
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

/// Path of a specific backend's `llama-server` executable inside `install_dir`,
/// when that backend's binary is present. Used by the backend-consistency check
/// to launch a second backend without going through the engine installer.
///
/// The CPU binary is deliberately not offered: comparing GPU backends is the
/// point, and a CPU run would take minutes per generation.
pub fn backend_executable(install_dir: &Path, backend: EngineBackend) -> Option<PathBuf> {
    let 名 = match backend {
        EngineBackend::Cuda => "llama-server-cuda.exe",
        EngineBackend::Vulkan => "llama-server-vulkan.exe",
        EngineBackend::Cpu => return None,
    };
    let 路径 = install_dir.join("engine").join("backends").join(名);
    路径.exists().then_some(路径)
}

/// List the GPU backends whose executables are actually installed, so the UI can
/// offer the consistency check only when there is something to compare.
pub fn available_gpu_backends(install_dir: &Path) -> Vec<EngineBackend> {
    [EngineBackend::Cuda, EngineBackend::Vulkan]
        .into_iter()
        .filter(|b| backend_executable(install_dir, *b).is_some())
        .collect()
}

/// Which GPU backend the currently selected engine path resolves to. Exposed so
/// the check can decide which backend to run first and which one to compare
/// against.
pub fn current_gpu_backend(state: &AppState) -> EngineBackend {
    // 一次取锁拿全所需字段。原先分两次 lock 虽不会死锁（顺序执行、各自及时释放），
    // 但两次 `lock().expect()` 是两处 panic 点，且中间有空窗；合并更稳。
    let (install_dir, server_path) = {
        let inner = state.inner.lock().expect("state lock poisoned");
        (
            inner.settings.install_dir.clone(),
            inner.settings.llama_server_path.clone(),
        )
    };
    let server_path = server_path.or_else(|| find_engine_executable(&install_dir));
    match server_path {
        Some(path) => detect_backend(&path, &install_dir),
        None => EngineBackend::Cpu,
    }
}

/// The port the service is currently listening on, if running.
pub fn current_port(state: &AppState) -> Option<u16> {
    let inner = state.inner.lock().expect("state lock poisoned");
    inner.service.port
}

/// Temporarily point the engine path at a different backend and start the
/// service. Returns the previous path so the caller can restore it.
///
/// This is the only mutation the consistency check performs on user settings,
/// and it is always paired with `restore_engine_path` in a guard so an error
/// mid-check cannot leave the app pinned to the "wrong" backend.
pub fn swap_engine_path_for(
    state: &AppState,
    install_dir: &Path,
    target: EngineBackend,
) -> AppResult<Option<PathBuf>> {
    let 目标 = backend_executable(install_dir, target).ok_or_else(|| {
        AppError::Message(format!(
            "没有找到 {} 后端的运行引擎，无法进行一致性校验。",
            backend_label(target)
        ))
    })?;
    let mut inner = state.inner.lock().expect("state lock poisoned");
    let 原值 = inner.settings.llama_server_path.take();
    inner.settings.llama_server_path = Some(目标);
    Ok(原值)
}

/// Undo `swap_engine_path_for`.
pub fn restore_engine_path(state: &AppState, previous: Option<PathBuf>) {
    let mut inner = state.inner.lock().expect("state lock poisoned");
    inner.settings.llama_server_path = previous;
}

/// Human-readable backend name, used in UI messages.
pub fn backend_label(backend: EngineBackend) -> &'static str {
    match backend {
        EngineBackend::Cuda => "CUDA",
        EngineBackend::Vulkan => "Vulkan",
        EngineBackend::Cpu => "CPU",
    }
}

/// Wait for a just-stopped service to actually release its port and VRAM.
///
/// `stop_service_core` kills the process, but on Windows the OS may take a
/// moment to tear down the socket and reclaim GPU memory. The backend check
/// restarts the service immediately afterwards with a *different* backend, and
/// without this pause the new process can fail to bind the port or hit a
/// transient out-of-VRAM condition — which would look like "the other backend
/// doesn't work" instead of "we didn't wait long enough".
///
/// On the desktop single-user path `--no-mmap` means the next launch re-reads the
/// whole model from disk anyway, so a short settle delay is not on the critical
/// path.
pub async fn wait_for_service_stop(state: &AppState) {
    let 端口 = {
        let mut inner = state.inner.lock().expect("state lock poisoned");
        // Only the first call sees a port; clear it so a second wait is a no-op
        // rather than sleeping again for a service that is already down.
        inner.service.port.take()
    };
    if let Some(端口) = 端口 {
        let _ = wait_for_port_available("127.0.0.1", 端口).await;
    }
    // GPU memory release is not observable through the port check; give the
    // driver a fixed grace period before the next process claims the card.
    tokio::time::sleep(Duration::from_millis(700)).await;
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

    // 权重与 KV 一律用本机真实模型文件的实测值（详见 CLAUDE.md「GGUF 元数据
    // 解析与 KV 缓存估算（实机校准）」）。用真值而不是拍脑袋的数字：公式一旦
    // 偏离真实硬件表现，下面的上下文推断会立刻给出离谱结果。
    //
    // Qwen3-8B Q4_K_M：文件 5,027,783,488 字节（file_size_mb 5028）→ 权重 4.68 GiB；
    //   36 层全注意力 / 8 KV 头 / 128 维 → 147,456 字节/token。
    const W8B: f64 = 4.68;
    const KV8B: f64 = 147_456.0 / crate::gguf::每GiB字节;

    // Qwen3-1.7B Q8_0：1834 MB → 权重 1.71 GiB；28 层全注意力 / 8 KV 头 /
    //   128 维 → 114,688 字节/token。小模型在 8G 卡上全量卸载的正面样本。
    const W1_7B: f64 = 1.71;
    const KV1_7B: f64 = 114_688.0 / crate::gguf::每GiB字节;

    /// Spark-X2.5-4B Q6_K：3.147 GB → 权重 2.93 GiB；36 层里只有 9 层全注意力
    /// （3:1 滑窗:全注意力）/ 4 KV 头 / 256 维 → 36,864 字节/token；另 27 层
    /// 滑窗窗口 512，封顶后是一块不随上下文增长的固定占用。
    const W_SPARK_Q6: f64 = 2.93;
    const KV_SPARK: f64 = 36_864.0 / crate::gguf::每GiB字节;
    const KV_FIXED_SPARK: f64 = 27.0 * 4096.0 * 512.0 / crate::gguf::每GiB字节;
    /// 原生 1M —— 正是「放开 16384 上限」要服务的对象。
    const CTX_SPARK_NATIVE: u32 = 1_048_576;

    /// 纯全注意力模型没有滑窗层，KV 没有固定项。
    const NO_KV_FIXED: f64 = 0.0;

    #[test]
    fn small_model_fits_fp16_at_default_context() {
        // 8G 卡跑 Qwen3-1.7B：权重 1.71 + 保留 1.3 = 3.01，余 4.99 GiB。
        // fp16 KV（112 KB/token）够约 46K，16K 目标档轻松放下 → 走 fp16 最快。
        let p = cuda_launch_params(8.0, W1_7B, KV1_7B, NO_KV_FIXED, 16384);
        assert_eq!(p.n_gpu_layers, Some(999));
        assert_eq!(p.ctx_size, 16384);
        assert_eq!(p.parallel, 1);
        assert!(!p.kv_quant, "fp16 should hold the default context easily");
    }

    #[test]
    fn qwen3_8b_needs_q8_to_reach_default_context() {
        // 8G 卡跑 Qwen3-8B Q4_K_M 的真实情形：权重 4.68 + 保留 1.3 = 5.98，
        // 只剩约 2.0 GiB。fp16 KV（144 KB/token）只够约 15K，差一点点不到
        // 16K 目标档，于是退一档用 q8 半量 + flash attention 补上。
        // 这正是「按参数量估算 KV」看不到的差别。
        let p = cuda_launch_params(8.0, W8B, KV8B, NO_KV_FIXED, 16384);
        assert_eq!(p.n_gpu_layers, Some(999));
        assert_eq!(p.ctx_size, 16384);
        assert_eq!(p.parallel, 1);
        assert!(p.kv_quant, "fp16 can't hold 16k for Qwen3-8B on an 8G card");
    }

    #[test]
    fn medium_vram_still_reaches_jumbo_context_on_q8() {
        // 6GB: fp16 falls short of 32k but q8 (half footprint + flash attention)
        // reaches it. This is the path an advanced user hits by raising
        // --ctx-size past what fp16 can hold on a smaller card.
        let p = cuda_launch_params(6.0, W1_7B, KV1_7B, NO_KV_FIXED, 32768);
        assert_eq!(p.n_gpu_layers, Some(999));
        assert_eq!(p.ctx_size, 32768);
        assert!(p.kv_quant, "large context beyond fp16 capacity should use q8 KV");
        assert_eq!(p.parallel, 1);
    }

    #[test]
    fn tight_vram_takes_largest_q8_context() {
        // 4.5GB: even q8 can't reach 32k, so take the largest q8 context,
        // rounded down to a 2048 multiple (and never below 2048).
        let p = cuda_launch_params(4.5, W1_7B, KV1_7B, NO_KV_FIXED, 32768);
        assert_eq!(p.n_gpu_layers, Some(999));
        assert!(p.kv_quant);
        assert!(p.ctx_size < 32768 && p.ctx_size >= 2048);
        assert_eq!(p.ctx_size % 2048, 0);
    }

    #[test]
    fn very_low_vram_falls_back_to_partial_offload() {
        // Qwen3-8B 权重 4.68 + 保留 1.3 = 5.98：8G 以下的卡全量装不进，
        // 按层数分档部分卸载，至少还能起来。
        let tight = cuda_launch_params(4.0, W8B, KV8B, NO_KV_FIXED, 32768);
        assert_eq!(tight.n_gpu_layers, Some(20));
        assert_eq!(tight.parallel, 1);

        let lower = cuda_launch_params(2.8, W8B, KV8B, NO_KV_FIXED, 32768);
        assert_eq!(lower.n_gpu_layers, Some(12));

        let lowest = cuda_launch_params(1.5, W8B, KV8B, NO_KV_FIXED, 32768);
        assert_eq!(lowest.n_gpu_layers, Some(8));
    }

    #[test]
    fn context_never_exceeds_model_trained_limit() {
        // A tiny model with a small trained context must not be over-extended
        // just because there's plenty of VRAM.
        let p = cuda_launch_params(16.0, W1_7B, KV1_7B, NO_KV_FIXED, 8192);
        assert!(p.ctx_size <= 8192);
        assert!(!p.kv_quant, "small context at high VRAM should use fp16");
    }

    #[test]
    fn budget_too_small_for_minimum_context_goes_partial() {
        // 余量只够几百个 token 时不能硬给 2048：`round_ctx` 的 2048 下限会让
        // 实际占用超出预算，llama.cpp 把 KV 溢出到共享内存，表现为
        // 「启动成功但龟速」。这种情况必须明确退回部分卸载。
        let p = cuda_launch_params(6.0, W8B, KV8B, NO_KV_FIXED, 32768);
        assert_eq!(
            p.n_gpu_layers,
            Some(20),
            "应走部分卸载，而不是超预算的全量卸载"
        );
        assert!(p.ctx_size <= 3072);
    }

    #[test]
    fn context_is_monotonic_in_vram() {
        // More VRAM never yields a smaller context (at equal or better KV
        // precision), across the full range.
        let ctx =
            |vram: f64| cuda_launch_params(vram, W8B, KV8B, NO_KV_FIXED, 32768).ctx_size;
        assert!(ctx(4.0) <= ctx(5.5));
        assert!(ctx(5.5) <= ctx(6.0));
        assert!(ctx(6.0) <= ctx(8.0));
    }

    #[test]
    fn all_gpu_launches_pin_single_slot() {
        // The single-slot pin is the core fix for the 4× KV blowup; it must
        // hold across every VRAM tier.
        for vram in [1.0, 2.0, 4.0, 5.5, 6.0, 8.0, 16.0] {
            let p = cuda_launch_params(vram, W8B, KV8B, NO_KV_FIXED, 32768);
            assert_eq!(p.parallel, 1, "vram={vram}");
        }
    }

    #[test]
    fn model_native_context_replaces_the_16k_hard_cap() {
        // 盯防「16384 上限被重新写死」。Qwen3-1.7B 原生 40960，8G 卡上
        //   budget = 8.0 − 1.71 − 1.3 = 4.99 GiB，fp16 每 token 112 KB
        //   → 够约 46K，足以放满原生上下文。
        // 放在旧代码里这里只会得到 16384（`min(context_length, 16384)`）。
        let p = cuda_launch_params(8.0, W1_7B, KV1_7B, NO_KV_FIXED, 40_960);
        assert!(
            p.ctx_size > 16_384,
            "原生 40960 的模型不该被卡在 16384，实际 {}",
            p.ctx_size
        );
        assert!(p.ctx_size <= 40_960, "不得超过模型原生上下文");
    }

    #[test]
    fn tiny_shortfall_keeps_fp16_instead_of_dropping_to_q8() {
        // Qwen3-1.7B 在 8G 卡实测空闲量 7.2 GiB 下：
        //   budget = 7.2 − 1.71 − 1.3 = 4.19 GiB → fp16 约 39.2K，
        //   离原生 40960 只差 5% 出头，正落在容忍带内。
        // 为这点长度把整段 KV 降成 q8 并不划算，应留在 fp16。
        let p = cuda_launch_params(7.2, W1_7B, KV1_7B, NO_KV_FIXED, 40_960);
        assert!(!p.kv_quant, "差距在容忍带内就该保留 fp16");
        assert!(
            p.ctx_size < 40_960 && p.ctx_size >= 36_864,
            "应开 fp16 的物理上限（而非降到 2048 档），实际 {}",
            p.ctx_size
        );
    }

    #[test]
    fn spark_reaches_far_past_16k_on_an_8g_card() {
        // Spark-X2.5-4B Q6_K 在 8G 卡上的真实情形 —— 放开上限的主要受益者。
        //   budget = 7.2（实测空闲）− 2.93（权重）− 1.3（保留）− 0.05（滑窗固定）
        //          = 2.92 GiB
        //   fp16 每 token 只有 36 KB（9 层全注意力），够约 85K，仍远不到原生 1M，
        //   于是走 q8 取物理最大。关键是这个结果远大于过去硬编码的 16384。
        let p = cuda_launch_params(7.2, W_SPARK_Q6, KV_SPARK, KV_FIXED_SPARK, CTX_SPARK_NATIVE);
        assert_eq!(p.n_gpu_layers, Some(999), "全量卸载应能装下");
        assert_eq!(p.parallel, 1);
        assert!(p.kv_quant, "fp16 放不下 1M 原生上下文，退 q8 换长度");
        assert_eq!(p.ctx_size % 2048, 0, "上下文按 2048 对齐");
        assert!(
            p.ctx_size >= 131_072,
            "放开上限后应能开到 128K 以上，实际 {}",
            p.ctx_size
        );
    }

    #[test]
    fn sliding_window_fixed_kv_is_deducted_from_the_budget() {
        // 同一模型，同一显存：把滑窗固定占用从 0 改成正数后，可开上下文必须
        // 变小 —— 这就是「固定项要先扣掉」的意义所在。若哪天有人把
        // kv_fixed_gb 误加进每 token 项并相乘，这条会立刻炸掉。
        let 无固定 =
            cuda_launch_params(7.2, W_SPARK_Q6, KV_SPARK, NO_KV_FIXED, CTX_SPARK_NATIVE);
        let 有固定 = cuda_launch_params(
            7.2,
            W_SPARK_Q6,
            KV_SPARK,
            KV_FIXED_SPARK,
            CTX_SPARK_NATIVE,
        );
        assert!(
            有固定.ctx_size < 无固定.ctx_size,
            "扣掉 54 MiB 固定开销后上下文应变小：有固定 {} vs 无固定 {}",
            有固定.ctx_size,
            无固定.ctx_size
        );
        // 54 MiB 只占预算的 1.8%，所以差距应该是几千 token 量级，而不是翻倍。
        let 差 = 无固定.ctx_size - 有固定.ctx_size;
        assert!(
            (2_048..=8_192).contains(&差),
            "差距应在几千 token 量级，实际 {差}"
        );
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
