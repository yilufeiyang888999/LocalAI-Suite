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
    /// 启动成功但**性能打折**的原因（当前只有一种：显卡上还有别的程序占着显存，
    /// 只好少卸几层）。`None` = 没有可说的，界面不该凭空报喜或报忧。
    ///
    /// 为什么必须单独一个字段，而不是塞进 `message`（2026-10-20，0.1.7-1）：
    /// `message` 挂在状态条上，而 `running` 的状态条是**绿色**的 —— 把
    /// 「只有一半层上了卡」写进绿条，等于用配色宣布一切正常。这是「让用户
    /// 误以为成功」的另一种形态，跟不提示一样糟。
    ///
    /// ⚠️ 不持久化：重开 APP 走接管时无从知道当初的计划，`None` 是**未知**，
    /// 不是「一切正常」。
    pub warning: Option<String>,
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

/// 模型权重与 KV 之外的运行期 GPU 占用（GiB）—— **全项目唯一来源**。
///
/// 含 CUDA context、compute buffer、驱动与运行库，外加「别贴着边启动」的余量。
///
/// 旧值 `0.5 + 0.8 = 1.3` 偏乐观。2026-10-18 在本机 8G 卡（RTX 5060 Laptop）
/// 对 Qwen3-1.7B Q8_0、ctx=40960、KV q8 做了三次实测，用
/// `nvidia-smi used − 权重 1.7084 − KV 2.1876` 反推开销：
/// **1.8394 / 1.8384 / 1.8237 GiB**（三次独立吻合，取最坏值 1.84）。
///
/// 低估的代价不是「少开点上下文」这么温和：规划器会以为还够放 fp16 KV，
/// 实际开出来溢出到共享内存 —— llama.cpp 不报错，只是慢得莫名其妙，
/// 排错时极难定位到「预留少算了 0.5 GiB」。
///
/// ⚠️ 改这个数必须同步改 `cuda_launch_params` 的单测期望值 —— 它们全是按
/// 这个常量算出来的预算。`models::estimate_vram_gb` 也直接引用本常量
/// （那里曾另有一份硬编码 1.3，两处各自漂移）。
pub(crate) const GPU_OVERHEAD_GB: f64 = 1.84;

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
    /// 主干层数（已扣掉 MTP 层）。只用于把「上了 20 层」说成人话 ——
    /// 光说 20 层，用户不知道这是一半还是九成。读不到层结构时为 `None`。
    主干层数: Option<u32>,
}

/// 定位一个模型真实的 GGUF 文件，按可靠性排序返回候选路径。
///
/// **为什么需要多个候选**（P1-7，2026-09-30 发现）：原先只读
/// `model.local_path`，而它**只在「导入本地模型」时才会被填写**
/// （`model_repo.rs::upsert_imported_model`）—— 清单里的内置模型
/// （`models.rs::builtin_models`）全部硬编码 `local_path: None`。
/// 于是**所有内置模型**都拿不到层结构、一律走回落经验公式：
/// 实测 Qwen3-1.7B 的 KV 被低估 **7.8×**、Ornith-1.5-9B 高估 2.1×。
/// 低估的直接后果是 8G 卡上按错误预算开出过大上下文 → **OOM**，
/// 而长上下文正是知识库问答（作品一）的必需项。
///
/// 内置模型下载后落在 `install_dir/models/<id>/<清单文件名>`
/// （见 `models::model_install_path`）。候选顺序：
/// 1. `local_path` —— 本地导入模型的唯一来源；
/// 2. `install_dir/models/<id>/<清单文件名>` —— 内置模型的标准落点；
/// 3. 该模型目录下任意 `*.gguf` —— 用户换量化档重下、文件名与清单不同时兜底。
///
/// 逐个 `parse_file`，第一个解析成功的即为事实来源；全失败才回落公式。
fn kv_profile_candidate_paths(
    model: &crate::models::ModelMetadata,
    install_dir: &Path,
) -> Vec<PathBuf> {
    let mut 候选: Vec<PathBuf> = Vec::new();
    if let Some(path) = model.local_path.as_deref().filter(|p| !p.is_empty()) {
        候选.push(PathBuf::from(path));
    }
    let 标准落点 = crate::models::model_install_path(install_dir, model);
    候选.push(标准落点.clone());
    if let Some(目录) = 标准落点.parent() {
        if let Ok(条目) = std::fs::read_dir(目录) {
            let mut 同目录gguf: Vec<PathBuf> = 条目
                .flatten()
                .map(|项| 项.path())
                .filter(|p| {
                    p.extension()
                        .and_then(|e| e.to_str())
                        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
                })
                .collect();
            // 排序只为让结果确定（目录枚举顺序不保证稳定）；
            // 标准落点已排在前面，这里只是兜底，不会抢在它之前。
            同目录gguf.sort();
            候选.extend(同目录gguf);
        }
    }
    候选
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
///
/// 候选路径（含内置模型的标准落点）见 `kv_profile_candidate_paths`。
fn resolve_kv_profile(
    model: &crate::models::ModelMetadata,
    install_dir: &Path,
    fallback_params_billions: f64,
) -> KvProfile {
    // 逐个候选试解析，取第一个成功的；同时把命中的路径带出来写进日志 ——
    // 这条日志本身就是「层结构到底生效没有」的判据（有它 = 真的读到了）。
    let 命中 = kv_profile_candidate_paths(model, install_dir)
        .into_iter()
        .find_map(|path| {
            crate::gguf::parse_file(&path)
                .ok()
                .map(|meta| (path, meta))
        });

    if let Some((path, meta)) = 命中 {
        let profile = KvProfile {
            per_token_fp16: meta.kv_bytes_per_token(KV_ELEMENT_BYTES_FP16)
                / crate::gguf::每GiB字节,
            fixed_fp16: meta.kv_fixed_gb(KV_ELEMENT_BYTES_FP16),
            // 与 `full_attention_layers` 同一口径：`block_count` 含 MTP 层，
            // 而 llama.cpp 不加载那些层，所以「总共几层」也要先扣掉 ——
            // 否则「20/33 层」这种说法会把 MTP 算进分母。
            主干层数: meta
                .block_count
                .map(|总| 总.saturating_sub(meta.mtp_layers.unwrap_or(0)))
                .filter(|主干| *主干 > 0),
        };
        if profile.per_token_fp16 > 0.0 {
            tracing::info!(
                model_id = %model.id,
                gguf = %path.display(),
                kv_gib_per_token = profile.per_token_fp16,
                kv_fixed_gib = profile.fixed_fp16,
                "resolved KV footprint from real GGUF layer structure"
            );
            return profile;
        }
    }

    tracing::debug!(
        model_id = %model.id,
        "no GGUF layer structure available; falling back to parameter-count KV estimate"
    );
    KvProfile {
        per_token_fp16: LEGACY_KV_GIB_PER_TOKEN_PER_BILLION * fallback_params_billions,
        fixed_fp16: 0.0,
        // 读不到层结构就没法说「20/32 层」。留 `None`，文案会自动省略这一段，
        // 而不是编一个分母出来。
        主干层数: None,
    }
}

/// 取「本次启动要用的模型元数据」：**清单 → 数据库 → 磁盘**三级逐级回退。
///
/// 🔴 **P2-11（2026-10-16 第一次修复）**：`catalog::effective_models` 只由「内置
/// 清单 + 在线清单缓存」拼成，**从不含 `is_imported = 1` 的导入模型** —— 而导入
/// 模型的 id 形如 `local-<文件主名>`，在那里必然查不到。修复前这里只查清单，于是
/// 导入模型的元数据恒为 `None`，启动参数（`-ngl` / 上下文）直接落到
/// `DEFAULT_CUDA_PARAMS`：本机 8G 卡与虚机 16G 卡**算出同一组 24 层 / 4096**
/// （两台机器、两个不同模型，同日复现），P100 的 16 GB 只用到 6424 MiB。
///
/// 🔴 **P2-11 续二（同日第二次修复）**：补上第二级之后，P100 上**照旧** 24 层 /
/// 4096 —— 因为「库里查得到这一行」本身还是个**假设**。真实用法是把 `.gguf` 丢进
/// `models/<目录>/` 再由界面扫描发现它：`installed_model_info` 算得出 id、UI 也
/// 拿得到，**但没有任何一处把它写进库**（`switch_installed_model` 只设置路径与
/// 当前模型指针，不落库）。于是两级全落空，而且**全程静默** —— `find_model`
/// 查不到是「正常结果」不是错误，`Ok(None)` 分支不打日志，日志里与「一切正常」
/// 同形。所以第三级改为直接读**磁盘上的文件本身**
/// （见 `model_repo::从文件读取模型元数据`），并在第二级落空时补一条 DEBUG，
/// 把试过的候选 id 一并打出来 —— 否则「库里有行但 id 是第三种写法」这种最难查的
/// 失配，在日志里永远不露头。
///
/// **回落失败 ≠ 启动失败**：库读不出来只当「未知模型」，退回 `DEFAULT_CUDA_PARAMS`
/// —— 那是本函数存在之前的既有行为，不会更坏；但**必须留一条 WARN**，否则
/// 「回落生效」与「回落没生效」在日志里同形。
///
/// 抽成独立函数是为了能被单测直接钉住。判据有四条，缺一条都会漏掉一类真问题：
/// ① 导入 id 能从库里取回完整元数据；② **库里没有行**、磁盘上有文件时也能取回；
/// ③ 内置 id 的结果一字不变；④ 库里磁盘都没有时老老实实返回 `None`（不许乱猜）。
fn 查找启动模型(
    data_dir: &Path,
    db_path: &Path,
    install_dir: &Path,
    current_model_path: Option<&Path>,
    model_id: &str,
) -> Option<crate::models::ModelMetadata> {
    // ① 内置 / 在线清单。
    if let Some(model) = crate::catalog::effective_models(data_dir)
        .into_iter()
        .find(|m| m.id == model_id)
    {
        return Some(model);
    }
    // ② 数据库（含老库 `imported-…` 写法的兼容，见 `model_repo::候选标识`）。
    match crate::model_repo::find_model(db_path, model_id) {
        Ok(Some(model)) => {
            // 这条日志是「回落真的发生了」的判据 —— 它出现，就说明模型不在
            // 清单里（内置模型永远走不到这里）。参数也一并记下，因为下一次
            // 排查「-ngl 为什么是这个值」时，先要看的就是元数据对不对。
            tracing::info!(
                model_id = %model.id,
                file_size_mb = model.file_size_mb,
                context_length = model.context_length,
                parameters_raw = model.parameters_raw,
                "模型不在内置/在线清单里，已从本地库取回完整元数据用于计算启动参数"
            );
            return Some(model);
        }
        // 查不到是**正常结果**（模型可能压根没进过库），但必须留痕。
        Ok(None) => {
            tracing::debug!(
                model_id = %model_id,
                试过的候选 = ?crate::model_repo::候选标识(model_id),
                "库里没有这个模型的元数据行，转从磁盘上的 GGUF 兜底"
            );
        }
        Err(err) => {
            tracing::warn!(
                model_id = %model_id,
                error = %err,
                "从本地库读取模型元数据失败，按未知模型处理（启动参数将用兜底默认值）"
            );
        }
    }
    // ③ 磁盘：文件在，元数据就一定有 —— 这是最后的事实来源。
    for 路径 in 磁盘候选路径(install_dir, current_model_path, model_id) {
        if let Some(model) = crate::model_repo::从文件读取模型元数据(model_id, &路径) {
            tracing::info!(
                model_id = %model.id,
                路径 = %路径.display(),
                file_size_mb = model.file_size_mb,
                context_length = model.context_length,
                parameters_raw = model.parameters_raw,
                "清单与本地库里都没有这个模型，已改从磁盘上的 GGUF 现场读取元数据"
            );
            return Some(model);
        }
    }
    None
}

/// 按 `model_id` 反查磁盘上可能的 `.gguf` 落点（P2-11 续二的第 ③ 级）。
///
/// 命名口径与 `models::installed_model_info` **逐字一致**（同一个 `导入模型标识`）：
/// 两边只要差一个字符，就会出现「UI 拿到的 id 与本函数算的 id 谁也不认谁」——
/// 这类失配在日志里只表现为「没查到」，最难定位，所以宁可共用同一个函数。
fn 磁盘候选路径(
    install_dir: &Path,
    current_model_path: Option<&Path>,
    model_id: &str,
) -> Vec<PathBuf> {
    let mut 候选: Vec<PathBuf> = Vec::new();

    // ① 当前正在用的那个文件 —— 用户刚在界面上选过它，命中率最高，也最可信。
    //    **必须排在目录扫描之前，且不能被排序打乱**：一个目录里可能同时躺着
    //    同名的不同量化档，一旦选错那份，算出的启动参数就与用户选的模型不符。
    if let Some(路径) = current_model_path.filter(|路径| 主名对得上(路径, model_id)) {
        候选.push(路径.to_path_buf());
    }

    // ② 托管目录 `models/<目录>/<文件>.gguf`：手工拷进来、靠扫描发现的那种模型
    //    就落在这里，而库里没有它们的行 —— 本函数要救的正是这一种。
    let mut 目录候选: Vec<PathBuf> = Vec::new();
    if let Ok(目录们) = std::fs::read_dir(install_dir.join("models")) {
        for 目录 in 目录们.flatten() {
            let Ok(文件们) = std::fs::read_dir(目录.path()) else {
                continue;
            };
            for 文件 in 文件们.flatten() {
                let 路径 = 文件.path();
                let 是gguf = 路径
                    .extension()
                    .and_then(|后缀| 后缀.to_str())
                    .is_some_and(|后缀| 后缀.eq_ignore_ascii_case("gguf"));
                if 是gguf && 主名对得上(&路径, model_id) {
                    目录候选.push(路径);
                }
            }
        }
    }
    // 排序 + 去重**只作用于目录扫描这一段**：目录枚举顺序不保证稳定，不排的话
    // 同一台机器两次启动可能选到不同文件，日志就不可比了。① 的位置不受影响。
    目录候选.sort();
    目录候选.dedup();
    for 路径 in 目录候选 {
        if !候选.contains(&路径) {
            候选.push(路径);
        }
    }

    候选
}

/// 文件主名经 `导入模型标识` 之后是否就是这个 id（与扫描侧同一口径）。
///
/// 只比主名、不比后缀：`<主名>.gguf` 与 `<主名>.GGUF` 是同一个模型。
fn 主名对得上(path: &Path, model_id: &str) -> bool {
    path.file_stem()
        .and_then(|主名| 主名.to_str())
        .is_some_and(|主名| crate::models::导入模型标识(主名) == model_id)
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
    let 结果 = start_service_core(&model_id, &app, &state).await;
    // 启动失败此前**一条日志都不留**（错误只回给前端弹窗），日志里于是出现
    // 「有 `starting llama-server`、却没有对应的 `llama-server running`」这种
    // 断头现场 —— 2026-10-16 P100 的日志里就有一例（`qwen3_1_7b_light`：
    // 起了、没起来、也不知道为什么）。补一条 WARN，让原因在日志里可查。
    if let Err(错误) = &结果 {
        tracing::warn!(
            model_id = %model_id,
            错误 = %错误,
            "启动服务失败（前端同时会收到这条错误）"
        );
    }
    结果
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
    // 权限级别只打一次：它是「引擎为什么杀不掉」的前置判据（见 提示提权状态）。
    提示提权状态();

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
    // 记下「进这次启动之前手上那个进程」的 pid：端口随后若仍被占，可能就是它
    // 没退干净（2026-10-19 P100：`taskkill` 拒绝访问，父进程没了、子进程还占着
    // 8000）。判据不能停在我们发过 kill。
    //
    // 同时把**进本次启动前的完整现场**存下来：停旧服务失败时要原样放回去。
    // 只恢复一半（例如把状态改回 Running 却留着新 model_id）比不恢复更糟 ——
    // 界面会显示「正在运行 Ornith」而端口上跑的还是 1.7B，正是这一轮要根治的谎报。
    struct 启动前现场 {
        进程: Option<Child>,
        pid: Option<u32>,
        端口: Option<u16>,
        模型: Option<String>,
        host: Option<String>,
        状态: ServiceStatus,
        安全模式: bool,
        /// 上一轮启动留下的降档提示。切模型失败要把现场**原样**恢复 ——
        /// 只恢复状态不恢复提示，界面上那条「卡被占」会凭空消失，
        /// 而服务其实还是那个降档跑着的服务。
        警告: Option<String>,
    }
    let 现场 = {
        let mut inner = state.inner.lock().expect("state lock poisoned");
        if inner.service.status == ServiceStatus::Starting {
            return Err(AppError::Message(
                "AI服务正在启动中，请稍候。".to_string(),
            ));
        }
        let 现场 = 启动前现场 {
            // Kill any existing process (whether spawned this session or reattached).
            进程: inner.service.process.take(),
            pid: inner.service.pid,
            端口: inner.service.port,
            模型: inner.service.model_id.clone(),
            host: inner.service.host.clone(),
            状态: inner.service.status.clone(),
            安全模式: inner.service.safe_mode,
            警告: inner.service.warning.clone(),
        };
        inner.service.pid = None;
        inner.service.status = ServiceStatus::Starting;
        inner.service.model_id = Some(model_id.to_string());
        // Keep host/port — they are needed for the restarting_same_address
        // check below and will be overwritten by set_starting later.
        现场
    };
    let 刚停掉的pid: Option<u32> = 现场.pid;
    // 🔴 停旧服务这一步**必须看结果**。
    //
    // 此前这里写的是 `结束服务进程(inner.service.process.take(), …);` —— 返回值被
    // 直接丢掉。于是「拒绝访问」只留一条 WARN，启动流程照走，接着就是双重谎报：
    //
    //   1. 旧引擎（1.7B，管理员权限起的）还占着 8000，新引擎绑不上端口；
    //   2. 健康探测命中的是**旧引擎**，连续两次 200 ⇒ 「启动成功」；
    //   3. 日志写下 `llama-server running model_id=Some("local-Ornith-1.5-9B-Q8_0")`，
    //      界面显示已切到 9B，实际跑的还是 1.7B（2026-10-19 P100 `07:42:10`）。
    //
    // 所以在**拉起任何进程之前**就要判「旧的到底停没停」，没停就不许往下走。
    let 终止 = 结束服务进程(现场.进程, 现场.pid, 现场.端口);
    // 「杀完了」不等于「端口已经放开」：见 `等端口上不再有旧进程` 的注释。
    // 判据仍然是现场，只是给现场一点时间 —— 这不是放宽，是**等它变成真话**。
    let (仍在监听, 等端口_ms) = match 现场.端口 {
        Some(端口) => 等端口上不再有旧进程(端口, 现场.pid),
        None => (None, 0),
    };
    if 等端口_ms >= 50 {
        tracing::info!(
            port = ?现场.端口,
            等端口_ms,
            仍在监听 = ?仍在监听,
            "端口没有立刻让出来，已等它（杀掉进程与 socket 拆除之间有时间差）"
        );
    }
    if 停止没能生效(终止, 仍在监听, 现场.pid) {
        let 端口 = 现场.端口.unwrap_or(0);
        let 占用 = 仍在监听.or(现场.pid).unwrap_or(0);
        let 原因 = if 终止 == 终止结果::拒绝访问 {
            "本程序没有权限结束它 —— 它是以管理员身份运行的那一次起的引擎"
        } else {
            "它仍在监听这个端口"
        };
        let 文案 = format!(
            "上一个 AI 服务没能停下来（接口端口 {端口} 上的进程 {占用}）。{原因}，\
             所以这次**没有**启动「{model_id}」—— 否则新引擎绑不上端口，\
             界面上会显示已经换了模型、实际跑的还是旧的那个。{}",
            停止失败建议(终止, 占用)
        );
        tracing::error!(
            port = 端口,
            pid = 占用,
            终止结果 = ?终止,
            拒绝启动 = model_id,
            "停旧服务失败 ⇒ 拒绝启动新模型：状态保持原样，绝不谎报已切换"
        );
        let info = {
            let mut inner = state.inner.lock().expect("state lock poisoned");
            // 句柄已经交给 `结束服务进程` 消费掉了；进程若还活着，我们手里没有句柄，
            // 这正是「接管态」（`process = None`）—— 停止走按端口反查的 kill_pid。
            inner.service.process = None;
            inner.service.pid = 现场.pid;
            inner.service.port = 现场.端口;
            inner.service.host = 现场.host.clone();
            inner.service.model_id = 现场.模型.clone();
            inner.service.status = 现场.状态.clone();
            inner.service.safe_mode = 现场.安全模式;
            inner.service.warning = 现场.警告.clone();
            ServiceInfo {
                model_id: inner.service.model_id.clone(),
                port: inner.service.port,
                host: inner.service.host.clone(),
                status: inner.service.status.clone(),
                safe_mode: inner.service.safe_mode,
                message: 文案.clone(),
                warning: 现场.警告,
            }
        };
        if info.status == ServiceStatus::Running {
            // 状态是 Running 就必须有看门狗：前面进 `Starting` 时看门狗已经停了，
            // 「Running 却没人在看」是另一种不一致（进程真挂了没人发现）。
            spawn_service_watchdog(app.clone(), state.clone());
        }
        let _ = app.emit("service_status_changed", &info);
        return Err(AppError::Message(info.message));
    }
    if 现场.pid.is_some() {
        // 切换模型此前**没有**任何「停了没有」的痕迹：日志里只有 `switched current
        // model` 紧接着 `starting llama-server`，中间那段是空的。补一条，让
        // 「正常切换」与「切换时没停下来」在日志里可区分。
        tracing::info!(
            pid = 现场.pid.unwrap_or(0),
            port = ?现场.端口,
            终止结果 = ?终止,
            "已结束上一个服务，准备启动新模型"
        );
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
    let host = {
        let inner = state.inner.lock().expect("state lock poisoned");
        inner.settings.service_host.clone()
    };
    // 附加参数只在**这里**算一次：三个启动分支（激进参数 / 保守参数 / CPU 兜底）
    // 共用同一份，且配置文件有问题时**在任何一次拉起之前**就失败，不会先起一次再报错。
    //
    // 🔴 这里**绝不能**写成裸 `?`。
    //
    // 状态在上面（guard 里）已经被翻成 `Starting`，而 `Starting` 下**启动与停止
    // 两个按钮都是灰的**。裸 `?` 一返回，没人把状态改回去，界面上就永久停在
    // 「AI服务正在启动」，用户改好配置也点不动，只能关掉 APP 重开。
    //
    // 2026-10-20 P100 实测：四条旁车配置反例（C-13~C-16）**每条都逼着重开一次
    // APP** —— 报错弹窗说得清清楚楚「配置不合法」，改好再点启动却被告知
    // 「AI服务正在启动中，请稍候」，而那个「启动中」永远不会结束。用户会以为
    // 改配置没生效，反复去改配置，实际卡住的是状态机。
    //
    // 复位成 **Stopped**、不是 `现场.状态`：能走到这一行说明上面的「停旧服务」
    // 已经成功，旧进程被 `结束服务进程` 收走了。恢复成 Running 会谎报「服务还在
    // 跑」—— 那是本项目已经栽过一次的坑（见上方「停旧服务失败」分支的注释）。
    let (extra_args, 旁车生效项) = match 推导附加参数(state, &model_path) {
        Ok(值) => 值,
        Err(错误) => {
            tracing::error!(
                模型 = %model_path.display(),
                错误 = %错误,
                "旁车配置不合法 ⇒ 拒绝启动：状态复位为未启动（否则界面永久卡在「正在启动」）"
            );
            let info = 复位为已停止(state, &format!("启动失败，服务已复位为未启动：{错误}"));
            let _ = app.emit("service_status_changed", &info);
            return Err(错误);
        }
    };
    if !旁车生效项.is_empty() {
        // 🔴 「已并入本次启动命令行」这句**只在这条路径上说** —— 这里是真的要起进程。
        // 接管路径（`recover_service`）也调同一个推导函数，但那里没有起进程、没有
        // 往任何命令行里加参数。2026-10-16 实机踩到：那行原先写在被共用的
        // `model_sidecar::读取旁车参数` 里，于是接管日志里出现了「已…附加」，
        // 而那个进程根本不是本次起的 —— 日志宣称了没发生的动作。
        tracing::info!(
            模型 = %model_path.display(),
            附加参数 = ?旁车生效项,
            "已把模型旁车配置并入本次启动命令行（自定义对话模板 / LoRA 适配器）"
        );
    }
    let configured_port = {
        let inner = state.inner.lock().expect("state lock poisoned");
        inner.settings.service_port
    };
    let backend = detect_backend(&server_path, &install_dir);

    // 隔离后端 DLL：让 backends/ 根目录下只剩当前后端需要的 GPU 后端 DLL。
    //
    // 必须做这一步，否则「切后端」形同虚设 —— llama.cpp 的
    // ggml_backend_load_all() 不看可执行文件名，只要目录里有
    // ggml-cuda.dll 就会加载并初始化 CUDA。2026-09-29 虚拟机实测：
    // 选 Vulkan 却仍报 CUDA error: invalid device function，正是此因。
    //
    // CPU 是兜底路径，它的变体不参与隔离；共用底座（ggml.dll / llama.dll
    // 等）也不动。失败不阻断启动（best-effort）—— 见 isolate_backend_dlls 注释。
    //
    // ⚠️ **TCC 保底**（2026-09-30 实测教训）：TCC 模式的计算卡上 Vulkan
    // 枚举不到设备。此时若仍按「选了 Vulkan 就把 CUDA DLL 收走」执行隔离，
    // 等于把唯一**可能**可用的 GPU 路径也一并砍掉 —— 引擎静默退回 CPU
    // （实测 8.62 t/s、零提示），比隔离之前的「明确报 CUDA 错」更难诊断
    // （报错至少指向问题，静默变慢只会让人以为一切正常）。
    // 因此 TCC 卡上把隔离目标**改成 CUDA**（详见下方「实际隔离后端」）。
    let 计算专用模式 = tokio::task::spawn_blocking(nvidia_is_tcc)
        .await
        .unwrap_or(false);
    // 🔴 **P1-1b（2026-09-30 修正，前身是「跳过隔离」）**
    //
    // 保底动作曾经写成「跳过隔离」，但它只保证「不动目录」，不保证
    // 「CUDA 路径可用」：上一轮 `isolate(Cuda)` 早已把 `ggml-vulkan.dll`
    // 收进池子，而**若此刻 `ggml-cuda.dll` 恰在池子里**（被更早一次 Vulkan
    // 隔离收走），跳过不会把它搬回来。表面看根目录仍是「两个 GPU 后端都没
    // 有/只剩一个」，引擎照常起来，实际是**用 CUDA 冒充 Vulkan 完成了比对**，
    // 界面还会报「两个后端完全一致」—— 比失败更危险，因为失败会报警。
    //
    // 正确动作是**显式隔离到 CUDA**：`isolate_backend_dlls` 是双向的，
    // 第二步会把 CUDA DLL 从池子搬回根目录，让「TCC 卡上确保 CUDA 可用」
    // 这个语义真正成立。
    //
    // 🔴 决策本身抽在 `选择实际隔离后端` 里（纯函数）—— 原因见该函数注释：
    // 这条分支在升过驱动的 P100 上**真机不可达**，只能靠单测兜底。
    let 实际隔离后端 = 选择实际隔离后端(backend, 计算专用模式);
    if 实际隔离后端 != backend {
        // WARN 必须保留 —— 它是这条分支的唯一出口，删掉就再也没有
        // 「确实走了保底」的判据了。
        tracing::warn!(
            "显卡处于 TCC 模式，该模式下 Vulkan 枚举不到设备；\
             改用 CUDA 隔离并确保 CUDA 运行库回到根目录，避免静默退回 CPU"
        );
    }
    let 移动数 = isolate_backend_dlls(&install_dir, 实际隔离后端);
    if 移动数 > 0 {
        tracing::info!(
            backend = backend_label(实际隔离后端),
            请求后端 = backend_label(backend),
            移动文件数 = 移动数,
            "已隔离后端 DLL，确保只加载当前后端的 GPU 模块"
        );
    }

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
        // 🔴 端口被占有两种完全不同的情形，此前一律报「请换一个端口」，
        // 把**最常见的那一种**变成了死路。
        //
        // 2026-10-19 P100 实测：关掉 APP（引擎按设计继续跑）再重开，接管没接上
        // （`recover_service` 那条记录没读到，或健康探测那一次没答），前端随即
        // 自动启动，于是撞上**自己上一次起的那个引擎** —— 界面弹出「请换一个端口」。
        // 用户此刻既停不掉它（新会话手里没有它的句柄），也不该换端口（换了就再也
        // 接不回原来那个，还得手工去杀进程）。
        //
        // 所以先问一句「占用者是谁」：能按 llama-server 的协议答出模型列表，
        // 就是我方引擎 ⇒ 直接接管（幂等）。判据**不能**停在「端口在听」——
        // 那是本项目「『活着』≠『能干这件事』」那条的直接应用：任何占端口的程序
        // 都会让端口「在听」，只有引擎能答出 `/v1/models` 的结构化正文。
        match 识别端口上的引擎(&host, port).await {
            Some(引擎自报) => {
                let 期望 = model_path.display().to_string();
                // 🔴 第一步：**先假设它是「刚被我们停掉、还没退干净」的那个进程**。
                //
                // 2026-10-19 P100（`06:47:18`）的原文：
                // `kill_pid：taskkill 未能结束该进程 … 无法终止 PID 11748
                //  （属于 PID 8868 子进程）的原因: 拒绝访问`。
                // 父进程（记录里的 pid）没了、子进程还占着 8000 与 6.4 GiB 显存 ——
                // 于是切换模型时界面报「正在运行另一个模型」，而那个「另一个」正是
                // 我们自己 7 秒前刚停掉的 qwen3-1.7B。
                //
                // 判据：只有**本次进来之前手上确实有进程**（刚停掉的pid 有值）才走这条。
                // 首次启动（新会话、手上没进程）时端口上那份可能是用户自己的服务，
                // 那种情况**不许**动它，仍然只报「另一个模型」。
                let mut 已让出 = false;
                let mut 上次终止 = 终止结果::未尝试;
                if 刚停掉的pid.is_some() {
                    if let Some(端口pid) = 端口上正在监听的进程(port) {
                        tracing::warn!(
                            port,
                            端口pid,
                            刚停掉的pid = 刚停掉的pid.unwrap_or(0),
                            模型 = %引擎自报,
                            "端口仍被本机引擎占着 —— 很可能是刚停掉的那个进程没退干净\
                             （taskkill 只结束了父进程）；按端口找到真正的监听者再终止一次"
                        );
                        上次终止 = kill_pid(端口pid);
                    }
                    已让出 = 等端口释放(&host, port, Duration::from_secs(10)).await;
                    if 已让出 {
                        tracing::info!(port, "端口已让出，继续按本次请求启动（不再判模型）");
                    }
                }
                if !已让出 {
                    let 问题 = crate::model_sidecar::核对模型(&期望, &引擎自报);
                    if 问题.is_empty() {
                        return Ok(接管遗留引擎(app, state, model_id, &host, port, &引擎自报));
                    }
                    // 是我方引擎、但跑的不是要启动的那个模型：**不许**接管（否则界面
                    // 显示 A、实际跑 B）。报出两边名字与占用 pid，让用户能处置。
                    let 占用pid = 端口上正在监听的进程(port);
                    let 占位 = 占用pid
                        .map(|pid| pid.to_string())
                        .unwrap_or_else(|| "未知".to_string());
                    // 🔴 「杀不掉」和「别人的服务」是两回事，文案必须分开。
                    // 2026-10-19 P100（`07:16:31`）：报「端口上正在运行另一个模型」时，
                    // 那个「另一个」其实就是 APP 自己上一次起的 qwen3-1.7B ——
                    // 只是它被以管理员身份启动，普通权限的本次程序 `taskkill` 不掉。
                    // 用户看到「另一个模型」只会去找「那边的服务」，永远找不到。
                    let 文案 = if 上次终止 == 终止结果::拒绝访问 {
                        format!(
                            "接口端口 {port} 上的引擎（进程 {占位}）还在运行，但本程序结束不掉它 —— \
                             Windows 拒绝访问，说明它是用**管理员权限**启动的。\
                             请右键「以管理员身份运行」本程序后再切换，\
                             或打开任务管理器结束这个进程。"
                        )
                    } else {
                        format!(
                            "接口端口 {port} 上正在运行另一个模型（{引擎自报}，进程 {占位}），\
                             不是这次要启动的「{期望}」。请先在那边停止服务，或到设置里换一个端口。"
                        )
                    };
                    let info = set_failed(app, state, 文案);
                    return Err(AppError::Message(info.message));
                }
            }
            None => {
                let info = set_failed(
                    app,
                    state,
                    format!("接口端口 {} 已被占用，请在设置里换一个端口。", port),
                );
                return Err(AppError::Message(info.message));
            }
        }
    }

    // 本次启动「为什么没能全量卸到显卡上」的归因。默认 `无从判断` ——
    // CPU 启动、拿不到显存、模型元数据查不到都会走不到判定，
    // 那时**不许**编一个原因出来（缺字段不能当成某个已知旧值）。
    let mut 降档 = 降档::无从判断;
    // 主干层数，用于把「上了 20 层」说成人话。读不到层结构就是 None。
    let mut 主干层数: Option<u32> = None;

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
        // 驱动模式决定「空载该占多少」：TCC 没有桌面合成，WDDM 有。见 `空载基线_gib`。
        let 驱动模式 = tokio::task::spawn_blocking(crate::hardware::nvidia_driver_model)
            .await
            .ok()
            .flatten();
        let 空载基线 = 空载基线_gib(驱动模式.as_deref());
        let (data_dir, db_path, current_model_path) = {
            let inner = state.inner.lock().expect("state lock poisoned");
            (
                inner.data_dir.clone(),
                inner.db_path.clone(),
                inner.settings.local_model_path.clone(),
            )
        };
        // 🔴 P2-11：这里**必须**走「清单 → 数据库 → 磁盘」三级查找。
        // 只查清单的话，导入模型（清单里天生没有它们）恒拿到 None；
        // 只加数据库还不够 —— 扫目录发现的模型（手工把 gguf 放进 models/）
        // **从不进库**，两级仍然全落空、且全程静默。三层缺一不可。
        let model = 查找启动模型(
            &data_dir,
            &db_path,
            &install_dir,
            current_model_path.as_deref(),
            model_id,
        );
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
                // 的线程（与上面的硬件探测同理）。install_dir 一并传进去 ——
                // 内置模型的 local_path 为空，要靠它拼出 `models/<id>/` 落点。
                let kv_install_dir = install_dir.clone();
                let kv_profile = tokio::task::spawn_blocking(move || {
                    resolve_kv_profile(&m, &kv_install_dir, active_billions)
                })
                .await
                .unwrap_or(KvProfile {
                    per_token_fp16: LEGACY_KV_GIB_PER_TOKEN_PER_BILLION * active_billions,
                    fixed_fp16: 0.0,
                    主干层数: None,
                });
                主干层数 = kv_profile.主干层数;
                // 同一套参数算**两遍**：一遍按「现在真正空着的显存」，一遍按
                // 「整张卡都归我用」。两者的差就是归因的唯一依据 ——
                // 空卡能全量、现在不能，差额只可能被人占走了；
                // 空卡也不够，那就是这个模型本身装不下这张卡，
                // 「卡被占」是冤枉别人（27B 放 8G 卡就是这种）。
                let 计划 = cuda_launch_params(
                    vram,
                    weight_gb,
                    kv_profile.per_token_fp16,
                    kv_profile.fixed_fp16,
                    max_ctx,
                );
                let 空卡计划 = cuda_launch_params(
                    total_vram,
                    weight_gb,
                    kv_profile.per_token_fp16,
                    kv_profile.fixed_fp16,
                    max_ctx,
                );
                降档 = 判定降档(total_vram, vram, 空载基线, 计划, 空卡计划);
                tracing::info!(
                    驱动模式 = ?驱动模式,
                    空载基线_gib = 空载基线,
                    他人占用_gib = total_vram - vram - 空载基线,
                    归因 = ?降档,
                    计划的层数 = ?计划.n_gpu_layers,
                    空卡能给的层数 = ?空卡计划.n_gpu_layers,
                    "显存归因：这次降档算谁的"
                );
                if let 降档::卡被占 { 他人_gib, .. } = 降档 {
                    // 2026-10-19 P100 `07:42`：旧引擎没杀掉（拒绝访问），可用显存
                    // 从 15.898 掉到 9.448，规划器「合理地」降成 20 层 / 3072 ctx，
                    // 而日志里只有一行看起来毫无问题的 `computing GPU launch params`
                    // —— 降档本身看不出异常，真因就此被埋掉。
                    // 2026-10-20：这条 WARN 过去用「可用 < 总量×0.80」判，在
                    // WDDM 笔记本上**每次启动都亮**（实测 5.784/7.960 = 0.726，
                    // 占着卡的只是桌面合成）⇒ 报警恒亮 = 报警失效。现在只在
                    // 「卡被占确实让计划变差了」时才报，见 `判定降档`。
                    tracing::warn!(
                        total_vram_gb = total_vram,
                        usable_vram_gb = vram,
                        他人占用_gib = 他人_gib,
                        计划的层数 = ?计划.n_gpu_layers,
                        空卡能给的层数 = ?空卡计划.n_gpu_layers,
                        "卡上有别的进程占着显存 —— 本次启动只好按剩下的显存降档。\
                         界面上会同时给出提示（这条 WARN 不再是日志里的孤证）"
                    );
                }
                计划
            }
            // Unknown VRAM or an imported model without metadata: keep the old
            // conservative fixed params.
            _ => DEFAULT_CUDA_PARAMS,
        }
    } else {
        LaunchParams { n_gpu_layers: None, ctx_size: 4096, parallel: 1, kv_quant: false }
    };

    set_starting(app, state, model_id, &host, port, false);
    // 提示文案在**开进程之前**就算好：它是计划的一部分，不是事后的猜测。
    // `None` = 没话可说（全量、模型本来就装不下、读不到显存都走这里）。
    let 降档提醒 = 降档提示(降档, params, 主干层数);
    tracing::info!(
        %model_id,
        backend = ?backend,
        // 🔴 报**实际生效**的后端，不是请求的那个。P100（TCC）上请求 Vulkan
        // 会被保底改成 CUDA 隔离，而这一行原先只打 `backend=Vulkan` ——
        // 日志于是宣称了一条没发生的事，后端一致性校验「两轮都跑在 CUDA 上」
        // 就变得无法从日志解释（2026-10-19 实测）。
        实际后端 = ?实际隔离后端,
        host = %host,
        port,
        gpu_layers = ?params.n_gpu_layers,
        ctx_size = params.ctx_size,
        parallel = params.parallel,
        kv_quant = params.kv_quant,
        会提示降档 = 降档提醒.is_some(),
        "starting llama-server"
    );
    let startup_timeout = if backend.is_gpu() {
        GPU_STARTUP_TIMEOUT_SECS
    } else {
        CPU_STARTUP_TIMEOUT_SECS
    };
    match launch_service(&server_path, &model_path, &host, port, &params, &extra_args, startup_timeout).await {
        Ok(child) => {
            // 【2026-09-30 新增】启动后校验「显卡加速是否真的生效」。
            //
            // 背景：健康检查只问 HTTP `/v1/models` 是否 200，而引擎在
            // 「后端 DLL 被隔离走、被迫用 CPU」时**照样**能正常起服务。
            // 于是隔离误伤会以「启动成功」的名义蒙混过关，用户只感到慢
            // （P100 上的真实事故：8.62 t/s，无任何提示）。
            //
            // 判据用 nvidia-smi 的已用显存：真上卡会到权重体积量级，
            // 静默退 CPU 则停在空载的个位数 MiB。读不到显存时不判定
            // （见 `gpu_offload_verified`），以免在 AMD/Intel 上误报。
            //
            // 判定「没上卡」时不报失败，而是按 safe_mode 处理：前端已有的
            // 「恢复显卡加速」按钮会自动出现，用户可一键重试。
            if backend.is_gpu() && !gpu_offload_verified().await {
                tracing::warn!(
                    backend = backend_label(backend),
                    "健康检查通过，但显卡显存占用仍在空载水平 —— 判定为未真正使用显卡加速"
                );
                return set_running(
                    app,
                    state,
                    child,
                    true,
                    "显卡加速没能生效，当前用的是普通（CPU）速度，生成会明显偏慢。\
                     可以点「恢复显卡加速」重试，或在设置里换一种加速方式。"
                        .to_string(),
                    // 已经退到 CPU 了，「卡被占多少」不再是主要矛盾 —— 这里再叠一条
                    // 只会把上面那句「点恢复加速」冲淡。
                    None,
                    &extra_args,
                );
            }
            // 【B-1 新增】档 2：`--n-gpu-layers 999` 只是**请求**。
            //
            // 档 1 的阈值固定 256 MiB，只抓得到「完全没上卡」；显存不足时引擎会
            // **静默**少卸几层，启动成功、健康检查 200、速度掉一半，外面看不出来。
            // 这里按「计划全量却装不下权重」补这一段灰色地带。
            //
            // 只 WARN 不阻断：部分卸载是「能跑但慢」，判成失败会把可用变不可用。
            // 位置在健康检查之后（`:812` 已过）⇒ 权重早已加载完，读显存不会读到半载值。
            if backend.is_gpu() {
                核对计划全量卸载(&model_path, params.n_gpu_layers, 实际隔离后端).await;
            }
            // 🔴 0.1.7-1：这条路径此前**没有任何**降档痕迹 —— 界面写「启动成功」，
            // 用户不知道只有一半层上了卡。现在把归因写进 `warning`，界面用琥珀色条
            // 单独显示。仍然**不阻断**：部分卸载本身是合法工作状态。
            set_running(
                app,
                state,
                child,
                false,
                "AI服务已启动，可以开始聊天。".to_string(),
                降档提醒.clone(),
                &extra_args,
            )
        }
        Err(error) => {
            // 端口被**别的引擎**占着 ⇒ 重试没有意义：保守参数与 CPU 兜底同样绑不上
            // 端口，那两次只会多打两条 `GPU launch failed; retrying…`，把真正的错因
            // 埋掉（2026-10-19 P100 `07:42` 之后就是这样：连报三次「显卡没起来」，
            // 而显卡一点问题都没有）。
            if 是端口占用类错误(&error.to_string()) {
                let info = set_failed(app, state, error.to_string());
                return Err(AppError::Message(info.message));
            }
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
                // 🔴 参数必须写进日志（P2-14）。`llama-server running` 那一行**不带**
                // 启动参数，所以降级成功之后「用的是保守参数」在外面看不出来 ——
                // 而保守参数把上下文压到 4096，肉眼聊天时完全无感，只有遇到长文才暴露。
                tracing::warn!(
                    %model_id,
                    backend = ?backend,
                    error = %error,
                    gpu_layers = ?conservative.n_gpu_layers,
                    ctx_size = conservative.ctx_size,
                    "GPU launch failed; retrying with conservative GPU params"
                );
                set_starting(app, state, model_id, &host, port, false);
                if let Ok(child) = launch_service(&server_path, &model_path, &host, port, &conservative, &extra_args, startup_timeout).await {
                    // 成功也要**明确记成「这是一次降级」**：否则「自动调参正常」与
                    // 「降级到 4096 才起来」在日志里同形（本项目铁律：降级/跳过
                    // 必须与「通过」可区分）。前端文案同样区分开。
                    tracing::info!(
                        %model_id,
                        gpu_layers = ?conservative.n_gpu_layers,
                        ctx_size = conservative.ctx_size,
                        "已用**保守参数**启动（自动调参那一组没跑起来）—— 上下文被压到 4096，长文场景会明显变短"
                    );
                    // 【B-1】保守参数**仍是 999 层**（只压上下文），所以这条判据在这里同样适用 ——
                    // 而且更需要：能走到这里说明激进那组已经起不来了，显存本来就紧。
                    // ⚠️ 不走上面那条 812 的「没上卡」分支（它 return 了），两条不是重复：
                    //    那条报的是「显存停在空载」，这条报的是「上了卡但没全上去」。
                    核对计划全量卸载(&model_path, conservative.n_gpu_layers, 实际隔离后端).await;
                    // 这里**不**沿用 `降档提醒`：那句话念的是激进计划（如「20 层 / 3072」），
                    // 而实际跑的是保守计划（999 层 / 4096）。照抄等于把一份错的计划
                    // 念给用户听 —— 比不提示更糟。保守档本身已有专门文案。
                    return set_running(
                        app,
                        state,
                        child,
                        false,
                        "AI服务已启动，但用的是更保守的参数（上下文更短）。".to_string(),
                        None,
                        &extra_args,
                    );
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
            // CPU 兜底必须把**两个** GPU 后端都隔离掉。
            // 否则 ggml_backend_load_all 仍会去加载 ggml-cuda.dll 并调用它的
            // 初始化函数 —— 在 CUDA 不适配的机器上，那一步本身就会崩，
            // 连 CPU 这条路都走不通（这正是本次虚拟机踩的坑）。
            let 隔离数 = isolate_backend_dlls(&install_dir, EngineBackend::Cpu);
            if 隔离数 > 0 {
                tracing::info!(移动文件数 = 隔离数, "CPU 安全模式：已隔离全部 GPU 后端 DLL");
            }
            set_starting(app, state, model_id, &host, port, true);
            match launch_service(&cpu_path, &model_path, &host, port, &safe_params, &extra_args, CPU_STARTUP_TIMEOUT_SECS).await {
                Ok(child) => set_running(
                    app,
                    state,
                    child,
                    true,
                    "显卡加速暂时没启动成功，已切换到更稳妥的 CPU 模式。".to_string(),
                    // 走到这里已经是 CPU 模式了，`safe_mode` 会触发界面上那条
                    // 「恢复显卡加速」的黄条，不必再叠一条降档提示。
                    None,
                    &extra_args,
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
    let 结果 = stop_service_core(&app, &state);
    // 🔴 没停下就**必须**报错，不能返回 Ok 让界面显示「已停止」。
    // 前端 `stopService()` 的 catch 分支会把这句话原样显示出来，并重新拉一次状态 ——
    // 那正是我们要的：用户看到「没能停下来 + 怎么办」，而不是被一句「已停止」骗着
    // 去切模型，然后撞上「端口上正在运行另一个模型」。
    if 结果.status != ServiceStatus::Stopped {
        return Err(AppError::Message(结果.message));
    }
    Ok(结果)
}

/// 停止之后，「那个进程是不是还在」——**只用现场可见的事实**判断。
///
/// 🔴 这条判据是这一轮的核心。2026-10-19 P100 的日志（`07:16:04`、`07:16:26`）：
///
/// ```text
/// 接管态：按端口找到真正的监听进程并终止 端口pid=2468 记录pid=Some(2468)
/// kill_pid：taskkill 未能结束该进程 pid=2468 状态=Some(128)
///   输出= 无法终止 PID 2468 (属于 PID 11688 子进程) 的原因: 拒绝访问
/// llama-server stopped            ← 状态写成停止、记录清掉，进程却还在
/// ```
///
/// 于是界面说「已停止」，用户接着切模型，撞上那个仍占着 8000 的进程，
/// 报「端口上正在运行另一个模型」——**一个谎引出第二个谎**。
///
/// 参数取 `(端口上现在监听的 pid, 停止前记录的 pid)`：
/// * 同号 ⇒ 没停下；
/// * 端口有人在听、但我们没有 pid 可比（接管时查不到）⇒ 同样不许宣称停掉了；
/// * 端口空、或换成了别的 pid（旧的那个确实退了，端口被别人抢走）⇒ 认作已停。
fn 停止后仍是同一个进程(仍在监听: Option<u32>, 停止前的pid: Option<u32>) -> bool {
    match (仍在监听, 停止前的pid) {
        (Some(现在), Some(旧的)) => 现在 == 旧的,
        (Some(_), None) => true,
        (None, _) => false,
    }
}

/// 启动前那次「停旧服务」到底有没有生效 —— 没生效就**不许拉起新进程**。
///
/// 🔴 与 `停止后仍是同一个进程` 的关键差别：**不能**把「端口有人在听、但我们没有
/// pid 可比」判成「没停下」。启动路径上那种情形多半是**别人的服务**占着端口，
/// 正确处置是交给后面的端口分支去识别（能识出我方引擎就接管，否则报清楚），
/// 在这里拦下来只会把「可以接管」变成「莫名无法启动」。
///
/// 所以只看两条**明确的反证**：
/// * 端口上现在那个 pid，还是我们刚才要停的那个；
/// * `taskkill` 明确回了「拒绝访问」—— 命令都说权限不够了，进程一定还在。
fn 停止没能生效(终止: 终止结果, 仍在监听: Option<u32>, 停止前的pid: Option<u32>) -> bool {
    if 终止 == 终止结果::拒绝访问 {
        return true;
    }
    match (仍在监听, 停止前的pid) {
        (Some(现在), Some(旧的)) => 现在 == 旧的,
        _ => false,
    }
}

/// 杀掉之后，端口上还会**短暂地**挂着旧 pid —— 等它让出来再下结论。
///
/// 🔴 2026-10-20 本机（管理员态切 Spark）：`07:08:01.428 kill_pid：已结束该进程`，
/// `01.472` 就查端口 —— **43 ms** 之后 netstat 仍报 `29328 LISTENING` ⇒ 判
/// 「没停下来」⇒ **拒绝切换模型**（界面上模型已切、服务没起）。几十秒后
/// `Get-Process 29328` 已是 GONE、端口 FREE：**停是停掉了，只是 socket 拆得慢**。
///
/// 而那条提示还写着「请右键以管理员身份运行」—— 用户本来就是管理员，
/// 建议与事实相反，白关一次 APP。
///
/// 预算故意很短（<1 s）：这里等的是**socket 拆除**，不是引擎优雅退出；
/// 等太久会把「真的没停下」也拖成一次界面卡顿。
fn 等端口上不再有旧进程(端口: u16, 旧pid: Option<u32>) -> (Option<u32>, u64) {
    等端口让出_直到(|| 端口上正在监听的进程(端口), 旧pid, 端口等待预算())
}

/// 等待预算：只等 socket 拆除，不等引擎优雅退出。
fn 端口等待预算() -> Duration {
    Duration::from_millis(800)
}

/// 等待循环本体，**把「查端口」抽成闭包**是为了能测 —— 真实实现要跑 `netstat`，
/// 判据（重试有没有发生、超时会不会照样下结论）就一个都测不了。
///
/// 两条必须同时成立：①旧 pid 还在时**确实会重试**；②预算用完仍不让出 ⇒
/// **照实报没让出**（不能靠等久一点把「真没停下」粉饰成「停掉了」）。
fn 等端口让出_直到<F>(mut 查端口: F, 旧pid: Option<u32>, 预算: Duration) -> (Option<u32>, u64)
where
    F: FnMut() -> Option<u32>,
{
    const 间隔: Duration = Duration::from_millis(50);
    let 起点 = Instant::now();
    loop {
        let 现在 = 查端口();
        if 端口已让出(现在, 旧pid) || 起点.elapsed() >= 预算 {
            return (现在, 起点.elapsed().as_millis() as u64);
        }
        std::thread::sleep(间隔);
    }
}

/// 「端口上那个还是旧 pid 吗」—— 从等待循环里抽出来，**因为它才是判据**。
///
/// 等待循环本身跑的是真 `netstat`，测不了；能测的、必须测的，是被等的这个条件。
/// 口径与 `停止没能生效` 一致：没有 pid 可比时，「有人在听」不能当成「没停下」。
fn 端口已让出(现在: Option<u32>, 旧pid: Option<u32>) -> bool {
    match (现在, 旧pid) {
        (Some(甲), Some(乙)) => 甲 != 乙,
        _ => true,
    }
}

/// 「没能停下来」时给用户的下一步。
///
/// 🔴 **只在真是权限问题时才建议提权**。2026-10-20 的那次提示让一个**本来就是
/// 管理员**的用户去「右键以管理员身份运行」，问题照旧、还白关一次 APP ——
/// 建议与事实相反，比没有建议更糟。
fn 停止失败建议(终止: 终止结果, 占用: u32) -> String {
    if 终止 == 终止结果::拒绝访问 {
        format!("请右键「以管理员身份运行」本程序后再试，或打开任务管理器结束进程 {占用}。")
    } else {
        format!("端口一般一两秒内就让开，再点一次即可；若反复出现，打开任务管理器结束进程 {占用}。")
    }
}

/// 把服务状态复位成「已停止」，返回新的状态对象（调用方负责 `emit`）。
///
/// 两个调用点，同一个理由：**状态已经不在真实状态上了，必须收尾**。
///
/// * `stop_service_core` —— 正常停止；
/// * `start_service_core` —— 状态翻成 `Starting` 之后的失败路径。
///
/// 🔴 后者的教训（2026-10-20 P100 实测）：`Starting` 下启动与停止两个按钮都是
/// 灰的，只要有一条失败路径忘了复位，用户就得关掉 APP 重开 —— 而界面上那句
/// 「AI服务正在启动中，请稍候」看起来像是在正常工作，排查时会把人引向配置。
///
/// 抽成一个函数而不是两处各写一遍：本项目栽过「同一逻辑两处定义，改一处跑另
/// 一处，不报错只给错结果」的坑。
fn 复位为已停止(state: &AppState, 说明: &str) -> ServiceInfo {
    let mut inner = state.inner.lock().expect("state lock poisoned");
    // 持久化快照一并清掉：重开 APP 时 `recover_service` 会读它决定要不要接管，
    // 留着一份指向「刚才没起来的那个模型」的快照，只会让下一次重开多绕一圈。
    let _ = clear_service_state(&inner.db_path);
    inner.service.process = None;
    inner.service.status = ServiceStatus::Stopped;
    inner.service.port = None;
    inner.service.host = None;
    inner.service.model_id = None;
    inner.service.pid = None;
    inner.service.safe_mode = false;
    // 降档提示是「这一次启动」的属性：服务不在了，它就过期了。
    // 留着会让界面在已停止状态下还挂着一条「卡被占」的黄条。
    inner.service.warning = None;
    ServiceInfo {
        model_id: None,
        port: None,
        host: None,
        status: ServiceStatus::Stopped,
        safe_mode: false,
        message: 说明.to_string(),
        warning: None,
    }
}

/// Stop the running local service (killing our spawned child or a reattached
/// pid) and clear its persisted snapshot. Shared by the command and update flow.
pub fn stop_service_core(app: &AppHandle, state: &AppState) -> ServiceInfo {
    // Stop the auto-restart watchdog *first* so it doesn't try to revive the
    // process we are about to kill.
    stop_service_watchdog(state);
    // 先把现场取出来：`结束服务进程` 要跑 `netstat` / `taskkill`（外部命令，
    // 几百毫秒），**绝不能握着状态锁做** —— 同线程重复加锁会自死锁。
    let (进程, 旧pid, 旧端口, 旧模型, 旧host, 安全模式) = {
        let mut inner = state.inner.lock().expect("state lock poisoned");
        (
            inner.service.process.take(),
            inner.service.pid,
            inner.service.port,
            inner.service.model_id.clone(),
            inner.service.host.clone(),
            inner.service.safe_mode,
        )
    };
    let 结果 = 结束服务进程(进程, 旧pid, 旧端口);
    // 判据 = 现场，不是「我们发过 kill」；但现场要**等它变成真话**再说 ——
    // 2026-10-20 `06:18:25.709` 停完 `25.775` 就报「仍在监听」，66 ms 的差；
    // 34 秒后端口确实空了。见 `等端口上不再有旧进程`。
    let (仍在监听, 等端口_ms) = match 旧端口 {
        Some(端口) => 等端口上不再有旧进程(端口, 旧pid),
        None => (None, 0),
    };
    if 等端口_ms >= 50 {
        tracing::info!(
            port = ?旧端口,
            等端口_ms,
            仍在监听 = ?仍在监听,
            "停止后端口没有立刻让出来，已等它（杀掉进程与 socket 拆除之间有时间差）"
        );
    }
    if 停止后仍是同一个进程(仍在监听, 旧pid) {
        let 端口 = 旧端口.unwrap_or(0);
        let 占用 = 仍在监听.unwrap_or(0);
        let 原因 = if 结果 == 终止结果::拒绝访问 {
            "（Windows 拒绝访问：它是以管理员权限启动的，当前程序权限不够）"
        } else {
            ""
        };
        let 文案 = format!(
            "AI服务没能停下来：接口端口 {端口} 上的引擎（进程 {占用}）仍在运行{原因}。{}",
            停止失败建议(结果, 占用)
        );
        tracing::error!(
            port = 端口,
            pid = 占用,
            终止结果 = ?结果,
            "停止服务失败：那个进程仍在监听端口 —— 状态**不**改成已停止"
        );
        let info = {
            let mut inner = state.inner.lock().expect("state lock poisoned");
            // 状态照实写回 Running：它**确实**还在服务，用户还能继续聊天。
            // 写 Stopped 只会让人以为停掉了，然后被下一次启动的报错误导。
            inner.service.status = ServiceStatus::Running;
            inner.service.port = 旧端口;
            inner.service.host = 旧host;
            inner.service.model_id = 旧模型;
            inner.service.pid = if 占用 == 0 { None } else { Some(占用) };
            inner.service.safe_mode = 安全模式;
            ServiceInfo {
                model_id: inner.service.model_id.clone(),
                port: inner.service.port,
                host: inner.service.host.clone(),
                status: ServiceStatus::Running,
                safe_mode: 安全模式,
                message: 文案.clone(),
                // 进程还是原来那个 ⇒ 当初的降档提示依然成立，不能因为
                // 这次「停止失败」就把它丢了。
                warning: inner.service.warning.clone(),
            }
        };
        // 记录**不清**：它还在跑，下次重开应当照常接管（而不是靠端口兜底去认领）。
        // 看门狗**要重新拉起**：刚才第一步就把它停了，而状态现在写的是 Running ——
        // 「Running 却没有看门狗」是另一种不一致（进程真挂了没人发现）。
        spawn_service_watchdog(app.clone(), state.clone());
        let _ = app.emit("service_status_changed", &info);
        return info;
    }
    let info = 复位为已停止(state, "AI服务已停止。");
    let _ = app.emit("service_status_changed", &info);
    tracing::info!(端口 = ?旧端口, 终止结果 = ?结果, "llama-server stopped");
    info
}

/// 现在该不该去读持久化记录、尝试接管一个「上次留下的」进程？
///
/// 🔴 **本会话已经在管服务时不许接管**（P2-13，2026-10-16 P100 实机抓到）。
/// 反例就是那天 14:50 的日志：前端重新加载页面时会**再调一次** `recover_service`，
/// 而它只看**持久化记录** —— 那条记录在本会话写出新值之前，还是**上一个会话**的旧值。
/// 偏巧本会话正在起一个新服务（`Starting`、端口已 listening），健康探测于是通过：
/// 一个**正在加载的新进程**被认成「上次留下的旧进程」。同一次竞态的四个指纹：
///
/// 1. `reattached … model_id=local-Ornith-1.5-9B-Q8_0` 与同一时刻回执核对的
///    `期望=…\qwen3_1_7b_light\…` **互相矛盾**（状态里最终留下的是错的那个）；
/// 2. 那个刚起的服务**从没打过自己的** `llama-server running model_id=Some("qwen3_1_7b_light")`；
/// 3. 4 秒内出现**两个** `service watchdog spawned`，其中一个紧接着退出；
/// 4. 随后 `llama-server running model_id=Some("local-Ornith…")` 报的是**旧模型名**，
///    而进程其实是新的那个 ⇒ 持久化记录被写成「旧 model_id + 新 pid」。
///
/// `Failed` **不在**排除之列：既然起步就失败，本会话手里没有服务，此时去接管一个
/// 真正的残留进程仍是原语义 —— 这次只排除「本会话自己正在管」的两态。
fn 应当尝试接管(当前状态: &ServiceStatus) -> bool {
    !matches!(当前状态, ServiceStatus::Starting | ServiceStatus::Running)
}

/// Reattach to a `llama-server` left running by a previous app session, or clean
/// it up if it is no longer healthy. Should be called once on startup before the
/// frontend decides whether to auto-start a fresh service.
#[tauri::command]
pub async fn recover_service(
    app: AppHandle,
    state: tauri::State<'_, AppState>,
) -> AppResult<ServiceInfo> {
    // 权限级别只打一次：起点在接管这条路上，「上次那个引擎杀不掉」是它的主要形态。
    提示提权状态();
    // 先看本会话的状态：已经在管（或正在起）服务时直接返回当前状态，
    // **连持久化记录都不读** —— 那一份此刻代表的是上个会话，不是现场（见 `应当尝试接管`）。
    // 锁必须在调 `service_status` 之前释放：那是同一个 Mutex，同线程重入 = 自死锁。
    let 跳过接管 = {
        let inner = state.inner.lock().expect("state lock poisoned");
        !应当尝试接管(&inner.service.status)
    };
    if 跳过接管 {
        tracing::debug!("recover_service：本会话已有服务，跳过接管（避免把新起的进程认成旧进程）");
        return Ok(service_status(&state));
    }

    let (db_path, persisted) = {
        let inner = state.inner.lock().expect("state lock poisoned");
        (inner.db_path.clone(), load_service_state(&inner.db_path)?)
    };

    let Some(persisted) = persisted else {
        // 🔴 这一步**必须留下痕迹**：2026-10-19 P100 实测「关 APP 再重开没接管」，
        // 日志里一句话都没有 —— 「没有记录」与「记录在但探测没过」从此**同形**，
        // 只能靠猜。两种情形的处置完全不同（前者要按端口重新认领，后者要重探），
        // 判据落在这行与下面那条 WARN 上。
        tracing::info!(
            "recover_service：数据库里没有上一会话的服务记录 ⇒ 本次无记录可接管；\
             端口上是否还有引擎在跑，由启动路径按端口重新认领"
        );
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
            // 接管态对「当初按什么参数起的」一无所知 ⇒ 提示留 `None`。
            // 这是**未知**，不是「一切正常」：判据缺失不许当成已知值。
            inner.service.warning = None;
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
                warning: None,
            }
        };
        let _ = app.emit("service_status_changed", &info);
        // Reattach path: the old process is alive and answering, so we want
        // the same auto-restart safety net as a freshly spawned service.
        spawn_service_watchdog(app.clone(), (*state).clone());

        // 【2026-10-16 新增】接管路径**也要问一次引擎**。
        //
        // 2026-10-16 老李重启 APP，日志里那一行 `reattached to previous llama-server`
        // 走的就是这里 —— 它**不进 `set_running`**，于是上一版加的回执核对在这条路上
        // **一次都没跑**（设计时预告过这条边界，当天就被实测确认）。
        //
        // 而 reattach 恰恰是**日常最常见**的路径（每次重启 APP 都走它），也是**最该查**
        // 的一条：接管的进程不是本次起的，APP 对它一无所知 —— 连「端口上跑的是哪个模型」
        // 都拿不出证据（「端口在听 + /health 200」回答不了这个问题）。
        //
        // 期望值用**当前**配置重新推导 ⇒ 用户改过配置时这里会报「对不上」，那是**正确**的
        // （进程是旧的），措辞由 `回执场景::接管旧进程` 负责区分，不会引到「微调没生效」。
        //
        // 推导失败**不能**让接管失败（否则 APP 起不来），但也不许静默 —— 打 WARN 说明
        // 本次没核对，并点明「下次启动服务会因此失败」（旁车配置损坏时的真实后果）。
        let 模型路径 = {
            let inner = state.inner.lock().expect("state lock poisoned");
            inner.settings.local_model_path.clone()
        };
        match &模型路径 {
            None => tracing::debug!("接管路径：设置里没有模型路径，本次不做回执核对"),
            Some(模型路径) => match 推导附加参数(&state, 模型路径) {
                Err(错误) => tracing::warn!(
                    错误 = %错误,
                    "接管路径：附加参数推导失败，本次不核对 —— **这不代表微调没生效**，\
                     但这个配置问题会让下次启动服务直接失败"
                ),
                Ok((附加参数, _)) => {
                    if let Some(host) = info.host.clone() {
                        let 模型路径 = 模型路径.clone();
                        let 端口 = persisted.port;
                        tokio::spawn(async move {
                            核对引擎回执(
                                &host,
                                端口,
                                Some(&模型路径),
                                &附加参数,
                                回执场景::接管旧进程,
                            )
                            .await;
                        });
                    }
                }
            },
        }

        return Ok(info);
    }

    // The old process is gone or unresponsive. Best-effort kill of any lingering
    // pid (in case it is a zombie holding the port), then clear the record.
    //
    // 🔴 杀之前**先核对 pid 是不是现在占着这个端口的那个**。不核对就等于
    // 「拿一条可能过期的记录去杀一个可能已被系统复用的 pid」—— 杀错进程，
    // 而且现场看起来只是「服务没起来」。记录里的 pid 与端口上的 pid 不一致时，
    // 真正的处置是清掉这条过期记录，不是动手。
    let 端口上的pid = 端口上正在监听的进程(persisted.port);
    match 端口上的pid {
        Some(pid) if pid == persisted.pid => {
            kill_pid(pid);
        }
        Some(pid) => tracing::warn!(
            port = persisted.port,
            记录pid = persisted.pid,
            端口pid = pid,
            "记录里的 pid 不是当前监听该端口的进程（记录已过期，pid 可能被系统复用）\
             ⇒ 不杀任何进程，只清记录"
        ),
        None => tracing::info!(
            port = persisted.port,
            "端口上已没有监听 ⇒ 记录确认过期（进程确实退出了），只清记录"
        ),
    }
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
/// 每台机器最多同时存在的 GPU 后端种类；后端隔离目录用固定名字。
///
/// 之所以要「隔离」，是因为 llama.cpp 的后端加载**根本不看可执行文件名**：
/// `ggml_backend_load_all()` 内部是按硬编码名字列表
/// （blas / cuda / vulkan / ... / cpu）逐个 `ggml_backend_load_best(name)`，
/// 每个 `load_best` 会在**可执行文件所在目录**里找 `ggml-<name>-*.dll` 或
/// `ggml-<name>.dll`，找到就 `dlopen` 并调用它的 `ggml_backend_init`。
///
/// 实测（2026-09-29 虚拟机）后果：`backends/` 里同时躺着 `ggml-cuda.dll` 和
/// `ggml-vulkan.dll` 时，**即使启动的是 `llama-server-vulkan.exe`，CUDA 后端
/// 也会被加载并初始化** —— 于是一张 CUDA 不支持的卡（如 RTX 50 系 +
/// 官方 CUDA 12.4 包）会让 Vulkan 路径也一起崩掉，报
/// `CUDA error: invalid device function`，且报错内容与后端选择无关，极具误导性。
///
/// 上游**没有**提供「只加载某几个后端」的开关（只有 `GGML_DISABLE_VULKAN`
/// 一个特例，没有 CUDA 版本）。所以唯一可靠的隔离手段就是让不被选中的后端
/// DLL **不在搜索目录里**。本模块把非当前后端的 DLL 挪进
/// `backends/.pool/<后端名>/`，启动前再按需换位。
const 后端池目录名: &str = ".pool";

/// N 卡是否处于 **TCC（计算专用）模式** —— 该模式下 Vulkan 用不了。
///
/// 判据来自 `nvidia-smi --query-gpu=driver_model.current`（实测本机返回 `WDDM`，
/// P100 那台返回 `TCC`）。
///
/// **探测失败一律返回 `false`** —— 宁可照旧隔离，也不要因为「拿不到驱动模型」
/// 就放弃隔离，那等于把刚修好的「切后端无效」问题又放回来。隔离保底只在
/// **确知**是 TCC 时才生效。
///
/// `pub(crate)`：后端校验可用性判定（P3-2）也要用 —— TCC 卡上 Vulkan
/// 根本跑不起来，不该提供这项比对。
pub(crate) fn nvidia_is_tcc() -> bool {
    crate::hardware::nvidia_driver_model()
        .as_deref()
        .is_some_and(crate::hardware::is_tcc_mode)
}

/// 这一轮**实际**要隔离到哪个后端 —— TCC 卡上请求 Vulkan 时保底成 CUDA（P1-1b）。
///
/// 抽成纯函数不是为了好看，而是**因为这条分支在真机上不可达**：
///
/// `backend` 来自 `detect_backend(settings.llama_server_path)`，而该路径由
/// `ensure_engine_ready` 按 `hardware.backend` 覆写；`select_backend` 的判据是
/// 「NVIDIA 且驱动 ≥ 551.78」，**它不看 TCC**。所以升到 582.78 的 P100 上
/// 引擎路径恒为 `llama-server-cuda.exe`、`backend` 恒为 `Cuda`，
/// `backend == Vulkan && 计算专用模式` **永远不成立** —— 2026-10-08 想按清单
/// 在虚机上复现这条 WARN，就是这么卡住的。
///
/// 这条分支真正保护的是**驱动低于 551.78 的 TCC 卡**（那时 `select_backend`
/// 会判 Vulkan，而 TCC 下 Vulkan 枚举不到设备 → 静默落 CPU 8.62 t/s）。
/// 本机恰好是「升驱动前属于该场景、升驱动后脱离该场景」，于是它成了纯防御代码。
///
/// **不可达的分支必须由单测兜底，否则等于没验证。** 见
/// `tcc_turns_vulkan_isolation_into_cuda`。
fn 选择实际隔离后端(请求后端: EngineBackend, 计算专用模式: bool) -> EngineBackend {
    if 请求后端 == EngineBackend::Vulkan && 计算专用模式 {
        EngineBackend::Cuda
    } else {
        请求后端
    }
}

/// 校验「显卡加速是否真的生效」：看 N 卡已用显存是否明显高于空载。
///
/// - 空载实测约 8 MiB；只要真把层放上去，占用会跳到几百 MiB 以上。
/// - 阈值取 **256 MiB**：低于它几乎不可能是「已卸载了若干层」。
/// - **只在能读到显存时才下结论**（N 卡 + nvidia-smi 可用）。读不到时返回
///   `true`（视为通过），避免在 AMD/Intel 或探测失败时误报。
///
/// **P2-9**：判定点必须把实测值打进日志 —— 否则「真读到显存且通过」与
/// 「读不到显存直接放过」在日志上**完全一样**，看日志永远无法确认这道护栏
/// 有没有真的起作用。2026-09-30 就是靠人工跑一次 `nvidia-smi` 才把两者分开。
async fn gpu_offload_verified() -> bool {
    // 阈值 256 MiB：低于它几乎不可能是「已卸载了若干层」。
    // 用局部 `let` 而非 `const`：常量名会被 `non_upper_case_globals` 判为
    // 非全大写（中文后跟 `GiB` 这种混排必踩），局部变量无此约束。
    let 阈值_gib = 0.25_f64;
    match tokio::task::spawn_blocking(crate::hardware::nvidia_used_vram_gb).await {
        Ok(Some(used_gib)) => {
            let 通过 = used_gib >= 阈值_gib;
            tracing::info!(
                memory_used_mib = (used_gib * 1024.0).round(),
                threshold_mib = (阈值_gib * 1024.0).round(),
                verified = 通过,
                "checked GPU offload against measured used VRAM"
            );
            通过
        }
        _ => {
            tracing::info!(
                threshold_mib = (阈值_gib * 1024.0).round(),
                verified = true,
                "used VRAM unavailable (non-NVIDIA or nvidia-smi failed); offload check skipped, treated as passed"
            );
            true
        }
    }
}

/// `ggml-*.dll` 里哪些属于哪个后端。用于决定「当前后端需要哪些文件留在
/// `backends/` 根下、哪些要挪进池子」。
///
/// 判定用**文件名前缀匹配**，因为同一后端可能有多个变体
/// （CPU 就有一整套 `ggml-cpu-{alderlake,haswell,...,x64,sse42}.dll`）。
/// 顺序有意义：先匹配到的先返回，所以更具体的前缀要排在前面。
fn 文件名属于哪个后端(文件名: &str) -> Option<BackendFileKind> {
    let 小写 = 文件名.to_ascii_lowercase();
    if !小写.ends_with(".dll") {
        return None;
    }
    if 小写.starts_with("ggml-cuda") {
        Some(BackendFileKind::Cuda)
    } else if 小写.starts_with("ggml-vulkan") {
        Some(BackendFileKind::Vulkan)
    } else if 小写.starts_with("ggml-cpu") {
        Some(BackendFileKind::Cpu)
    } else {
        // ggml.dll / ggml-base.dll / llama.dll / llama-server-impl.dll 等
        // 是所有后端共用的底座，必须始终留在根目录。
        None
    }
}

/// 一个后端专属的 DLL 归属。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendFileKind {
    Cuda,
    Vulkan,
    /// CPU 是兜底路径，任何 GPU 后端启动失败都会退到它，所以 CPU 变体
    /// **永远留在根目录**，不参与隔离（它体积极小，且不会有「架构不适配
    /// 导致崩溃」之外的风险）。
    Cpu,
}

impl BackendFileKind {
    fn 对应后端(self) -> EngineBackend {
        match self {
            BackendFileKind::Cuda => EngineBackend::Cuda,
            BackendFileKind::Vulkan => EngineBackend::Vulkan,
            BackendFileKind::Cpu => EngineBackend::Cpu,
        }
    }
}

/// 让 `backends/` 根目录下**只剩** `目标后端` 需要的 GPU 后端 DLL。
///
/// 幂等：可以反复调用（每次启动前都调）。返回值是本次被移动的文件数，仅供日志。
///
/// 只动 GPU 后端的 DLL：
/// - `目标后端` 自己的 DLL → 确保在根目录（若还在池子里就搬回来）
/// - 另一个 GPU 后端的 DLL → 搬进池子
/// - 共用底座（`ggml.dll` / `llama.dll` / `llama-server-impl.dll` 等）→ 不动
/// - `目标后端 == Cpu` 时，**两个 GPU 后端都搬进池子**：CPU 安全模式的意义就是
///   绕开 GPU 路径，而 CUDA/Vulkan 的 `ggml_backend_init` 在不适配的机器上
///   本身就会崩，留着它们等于 CPU 兜底也走不通。
///
/// 失败不阻断启动（best-effort）：万一某个文件被占用搬不动，只是回到
/// 「两个后端都在」的老行为，不该因此让用户连服务都起不来。
pub fn isolate_backend_dlls(install_dir: &Path, 目标后端: EngineBackend) -> usize {
    let backend_dir = install_dir.join("engine").join("backends");
    if !backend_dir.is_dir() {
        return 0;
    }
    let 池根 = backend_dir.join(后端池目录名);
    let mut 移动数 = 0;

    // 第一步：把不该留在根目录的 GPU 后端 DLL 收进池子。
    if let Ok(条目) = std::fs::read_dir(&backend_dir) {
        for 项 in 条目.flatten() {
            let 路径 = 项.path();
            let Some(名) = 路径.file_name().and_then(|n| n.to_str()).map(str::to_string) else {
                continue;
            };
            let Some(归属) = 文件名属于哪个后端(&名) else {
                continue;
            };
            // CPU 变体常驻根目录（体积极小，且是最终兜底）。
            if 归属 == BackendFileKind::Cpu {
                continue;
            }
            // 当前后端的 DLL 留在根目录 —— 但 CPU 模式例外：它要清掉所有 GPU 后端。
            if 目标后端 != EngineBackend::Cpu && 归属.对应后端() == 目标后端 {
                continue;
            }
            let 目标目录 = 池根.join(backend_label(归属.对应后端()));
            if std::fs::create_dir_all(&目标目录).is_err() {
                continue;
            }
            if std::fs::rename(&路径, 目标目录.join(&名)).is_ok() {
                移动数 += 1;
            }
        }
    }

    // 第二步：把当前后端自己的 DLL 从池子里搬回根目录。
    //
    // 这一步不能省。想象用户先用 Vulkan（此时 CUDA DLL 在池子里），
    // 之后切回 CUDA —— 若只做第一步，CUDA 的 DLL 会永远留在池子里出不来。
    // CPU 模式不需要搬回任何东西（没有任何 GPU 后端属于它）。
    if 目标后端 == EngineBackend::Cpu {
        return 移动数;
    }
    let 池目录 = 池根.join(backend_label(目标后端));
    if let Ok(条目) = std::fs::read_dir(&池目录) {
        for 项 in 条目.flatten() {
            let 路径 = 项.path();
            let Some(名) = 路径.file_name().and_then(|n| n.to_str()).map(str::to_string) else {
                continue;
            };
            // 双保险：只搬确实属于当前后端的文件，避免池子被污染后误搬。
            if 文件名属于哪个后端(&名).map(|k| k.对应后端()) != Some(目标后端) {
                continue;
            }
            if std::fs::rename(&路径, backend_dir.join(&名)).is_ok() {
                移动数 += 1;
            }
        }
    }

    移动数
}

/// 把曾被隔离走的 DLL 全部还原到 `backends/` 根目录。
///
/// 用途：① 安装新引擎包前还原（避免新旧 DLL 混在一个目录，见 `engine.rs`）；
/// ② 卸载 / 校验引擎完整性 / 用户手工排查时需要一个「原样」目录。
/// 正常启动流程**不需要**调用它 —— 隔离状态本身就是期望状态。
pub fn restore_backend_dlls(backend_dir: &Path) -> usize {
    let 池根 = backend_dir.join(后端池目录名);
    if !池根.is_dir() {
        return 0;
    }
    let mut 还原数 = 0;
    if let Ok(子目录) = std::fs::read_dir(&池根) {
        for 子项 in 子目录.flatten() {
            let 子路径 = 子项.path();
            if !子路径.is_dir() {
                continue;
            }
            if let Ok(文件) = std::fs::read_dir(&子路径) {
                for 文件项 in 文件.flatten() {
                    let 路径 = 文件项.path();
                    let Some(名) = 路径.file_name().and_then(|n| n.to_str()).map(str::to_string)
                    else {
                        continue;
                    };
                    if std::fs::rename(&路径, backend_dir.join(&名)).is_ok() {
                        还原数 += 1;
                    }
                }
            }
            let _ = std::fs::remove_dir(&子路径);
        }
    }
    let _ = std::fs::remove_dir(&池根);
    还原数
}

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

/// 扫描 `engine/backends/` **根目录实际存在**的 GPU 后端 DLL，推断引擎这一轮
/// 真正会加载哪个后端 —— 即两次校验到底跑在不在同一个后端上。
///
/// **为什么必须实测**（P1-8，2026-09-30「假一致」事故）：`ggml_backend_load_all()`
/// 只看 DLL 在不在搜索目录，不看可执行文件名（见 `后端池目录名` 的注释）。
/// 所以「我们想把后端切成什么」与「引擎实际会用什么」是两件事。后端一致性
/// 校验原先只记录调用方传进去的标签（`backend_label(目标后端)`），于是
/// **「两轮真的换了后端」与「两轮根本没切过去」在界面和日志上完全无法区分**
/// —— TCC 卡上就这么输出了「两个后端完全一致」（实测 110.51 / 111.48 字/秒、
/// 384 字逐字相同，实际两轮都是 CUDA）。
///
/// 目录里实际有哪家的 DLL，才是引擎真正会用的后端。与 `文件名属于哪个后端`
/// 共用同一套前缀判定，保证「隔离谁」与「读到谁」不会各说各话。
pub fn effective_gpu_backend(install_dir: &Path) -> EngineBackend {
    let backend_dir = install_dir.join("engine").join("backends");
    let (mut 有cuda, mut 有vulkan) = (false, false);
    if let Ok(条目) = std::fs::read_dir(&backend_dir) {
        for 项 in 条目.flatten() {
            let Some(名) = 项.path().file_name().and_then(|n| n.to_str()).map(str::to_string)
            else {
                continue;
            };
            match 文件名属于哪个后端(&名) {
                Some(BackendFileKind::Cuda) => 有cuda = true,
                Some(BackendFileKind::Vulkan) => 有vulkan = true,
                _ => {}
            }
        }
    }
    // 两个都在时按 CUDA 计：`ggml_backend_load_all()` 的硬编码名字列表里
    // cuda 排在 vulkan 之前，先注册的先用（2026-09-29 实测：根目录躺着
    // `ggml-cuda.dll` 时，即使用 `llama-server-vulkan.exe` 启动，
    // `device_info` 也只有 CUDA0）。
    match (有cuda, 有vulkan) {
        (true, _) => EngineBackend::Cuda,
        (false, true) => EngineBackend::Vulkan,
        (false, false) => EngineBackend::Cpu,
    }
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
        warning: inner.service.warning.clone(),
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

/// 推导**最终**传给引擎的附加参数 = 设置里的额外启动参数 + 该模型的旁车配置。
///
/// 做成一个函数是**为了防止两处定义各自漂移**：`start_service_core`（拉起新进程）
/// 与 `recover_service`（接管旧进程后核对回执）都要用同一份口径 ——
/// 本项目已经栽过一次「同一份数据两处定义，改一处跑另一处不报错、只给错报告」。
///
/// 模型旁车配置（`<模型主名>.launch.json`）解决的是这个：微调出来的 adapter 有两个
/// 必须与训练时一致的部署条件 —— 挂载 LoRA、以及与训练契约**相同**的对话模板 ——
/// 而两者都只是 llama.cpp 的启动参数。APP 原先一个都不传，于是「实验室里修好了模板，
/// 生产上照旧」：模板不一致时引擎会把模型输出整段判进 `reasoning_content`、
/// `content` 变空串，同一份 adapter 的增益从 `+39.1 pp` 塌到 `+4.3 pp`
/// （D23 报告「四·补十」缺口 1/3）。
///
/// 🔴 配置损坏 / 引用的文件缺失一律当场失败 —— 静默跳过会让服务「看起来正常」。
/// 注意先在锁内把设置读完、**锁外**再做文件 I/O（锁内做 I/O 是本项目踩过的自死锁形态）。
///
/// 返回值第二项是**真正并入最终参数**的旁车项（被用户手写参数盖掉的不算）。
/// 之所以把它交出来：光看第一项分不清「这些参数是谁带来的」，而「已并入启动命令行」
/// 这句日志**只有真正拉起进程的那条路径才有资格说** —— 接管路径也调本函数，
/// 但那里没有起进程、没有附加任何参数（2026-10-16 实机踩过，见 `model_sidecar.rs`）。
fn 推导附加参数(
    state: &AppState,
    模型路径: &Path,
) -> AppResult<(Vec<String>, Vec<(String, String)>)> {
    let mut 参数 = {
        let inner = state.inner.lock().expect("state lock poisoned");
        parse_extra_args(inner.settings.extra_launch_args.as_deref().unwrap_or(""))
    };
    let 旁车生效项 = 并入旁车项(&mut 参数, crate::model_sidecar::读取旁车参数(模型路径)?);
    Ok((参数, 旁车生效项))
}

/// 把旁车项并入最终参数，返回**真正并入**的那些（被用户手写参数盖掉的不算）。
///
/// 抽成纯函数的原因：它决定「已并入本次启动命令行」这句日志的**真值**。
/// 本项目刚在「日志宣称了没发生的动作」上栽过一次（见 `model_sidecar.rs`
/// 里 `读取旁车参数` 的那段注释），所以这条链上最容易错的一环必须有判据兜着。
///
/// 已知边界：`--lora` 与 `--lora-scaled` 在这里算**两个标志** —— 用户手写
/// `--lora-scaled x:0` 时，旁车给的 `--lora` 仍会并入，引擎会挂两次适配器。
/// 属于既有行为，本轮不改；真要改得先有「适配器参数族」的口径。
fn 并入旁车项(
    参数: &mut Vec<String>,
    旁车项: Vec<(String, String)>,
) -> Vec<(String, String)> {
    let mut 生效: Vec<(String, String)> = Vec::new();
    for (标志, 值) in 旁车项 {
        // 用户在设置里手写的同名标志优先，避免 llama.cpp 收到两份（重复的
        // `--lora` 会挂载两次适配器；重复的 `--chat-template-file` 只认最后一个）。
        let 用户已指定 = 参数
            .iter()
            .any(|词| 词 == &标志 || 词.starts_with(&format!("{}=", 标志)));
        if 用户已指定 {
            tracing::info!(标志 = %标志, "设置里的同名启动参数优先，不使用模型旁车配置里的这一项");
            continue;
        }
        参数.push(标志.clone());
        参数.push(值.clone());
        生效.push((标志, 值));
    }
    生效
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
        // 新一次启动开始 ⇒ 上一轮的降档提示作废（否则旧提示会挂在新启动上，
        // 「正在启动」状态却带着「只上了一半层」的结论，自相矛盾）。
        inner.service.warning = None;
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
            warning: None,
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

    // 捕获 stderr 以便启动失败时能读到 llama.cpp 自己打印的原因。
    // 这是实测教训：只报退出码等于没有诊断信息，用户只能看到
    // 「可能是运行引擎与硬件不兼容或模型文件损坏」这种无法行动的话。
    command.stderr(std::process::Stdio::piped());

    let mut child = command.spawn().map_err(|_| {
        AppError::Message("AI服务暂时没能启动，请检查模型文件是否完整。".to_string())
    })?;

    // Require two consecutive healthy probes so a single transient 200 during
    // llama.cpp warmup doesn't get mistaken for a stable service. The health
    // wait also watches the child: if the process exits early (corrupted exe,
    // missing DLL, incompatible GPU backend), we fail fast instead of waiting
    // out the full 180s timeout.
    if let Err(error) = wait_for_health(&mut child, port, 2, startup_timeout_secs).await {
        结束服务进程(Some(child), None, Some(port));
        return Err(error);
    }

    // 🔴 健康检查之后**再问一句「端口上服务的是不是我刚起的这个进程」**。
    // 旧引擎还占着端口时，`wait_for_health` 会被它的 200 直接骗过 —— 见
    // `确认本次进程在服务` 里 2026-10-19 P100（`07:42:10`）的原文。
    if let Err(error) = 确认本次进程在服务(child.id(), port, model_path).await {
        结束服务进程(Some(child), None, Some(port));
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
    // 启动成功但**性能打折**的原因（见 `ServiceInfo::warning`）。`None` = 无话可说。
    warning: Option<String>,
    附加参数: &[String],
) -> AppResult<ServiceInfo> {
    let info = {
        let mut inner = state.inner.lock().expect("state lock poisoned");
        let pid = child.id();
        inner.service.process = Some(child);
        inner.service.pid = Some(pid);
        inner.service.status = ServiceStatus::Running;
        inner.service.safe_mode = safe_mode;
        // 写进状态而不只是返回：`get_service_status` 会重建 `ServiceInfo`，
        // 不落状态的话前端下一次轮询就把提示抹掉了。
        inner.service.warning = warning.clone();
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
            warning,
        }
    };

    let _ = app.emit("service_status_changed", &info);
    spawn_service_watchdog(app.clone(), state.clone());
    tracing::info!(model_id = ?info.model_id, port = ?info.port, safe_mode, "llama-server running");

    // 【2026-10-16 新增】服务真的起来了 ⇒ 再**问一次引擎**，把「参数传过去了」
    // 升级为「引擎确实生效了」。
    //
    // 为什么必须放在这里、而且必须是「问」而不是「信」：`--lora` 与
    // `--chat-template-file` 都是**我们自己填进命令行的标签**，写进去了不等于
    // 引擎接受了（把「调用方填的标签」当「引擎实测值」，是本项目翻过五次车的
    // 那类错误）。2026-10-16 实机验证证明这两件事都问得出来，见
    // `model_sidecar.rs` 的「启动之后：问引擎要回执」一节与报告「四·补十一-4」。
    //
    // 放在 `set_running` 而不是 `start_service_core` 的末尾：这里是**四个成功
    // 路径共同的汇合点**（激进参数 / 保守参数 / 显卡未生效 / CPU 兜底），
    // 改一处就全覆盖，将来新增分支也不会漏。异步 spawn，不拖慢返回。
    //
    // 先判「有没有值得核对的东西」再 spawn：用户只是在设置里写了 `--ctx-size`
    // 时不该白起一个任务、更不该白打两行日志。条件与 `核对引擎回执` 的内部
    // 早退条件一致（那边是兜底，这边是省一次 spawn）。
    let 有微调产物 = !crate::model_sidecar::期望的适配器(附加参数).is_empty()
        || crate::model_sidecar::期望的对话模板(附加参数).is_some();
    if 有微调产物 {
        if let (Some(port), Some(host)) = (info.port, info.host.clone()) {
            let 附加参数 = 附加参数.to_vec();
            tokio::spawn(async move {
                // 模型路径传 `None`：这条路径下进程是我们**亲手**按 `model_path` 起的，
                // 「跑的是哪个模型」不需要再问（接管路径才需要，见 `回执场景`）。
                核对引擎回执(&host, port, None, &附加参数, 回执场景::本次启动).await;
            });
        }
    }

    Ok(info)
}

/// 这次回执核对是从**哪条路径**来的 —— 决定「对不上」怎么归因、给什么建议。
///
/// 两条路径的**起点不同**，所以判据也不同：
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum 回执场景 {
    /// 刚刚亲手拉起的进程：命令行就是本次传的 ⇒ 对不上就是真对不上。
    本次启动,
    /// 接管（reattach）上一个会话留下的进程。这里有三点不同：
    ///   ① 进程**不是本次起的**，APP 手里只有一个 `model_id` 字符串，
    ///      所以**必须**额外问一次 `/v1/models`：端口在听 ≠ 跑的是我要的模型；
    ///   ② 期望值是用**当前**配置重新推导的，而进程用的是**上次启动时**的配置
    ///      ⇒ 对不上更可能是「配置改过、进程还是旧的」，处置是重启服务；
    ///   ③ 这是**日常最常见**的路径（每次重启 APP 都走它），却最容易漏。
    接管旧进程,
}

/// 要不要额外问一次模型身份（`GET /v1/models`）？
///
/// * **接管场景**：要 —— 那个进程不是本次起的，APP 手里只有一个 `model_id`
///   字符串，「端口在听」证明不了它跑的是我要的模型。
/// * **本次启动**：不要 —— 进程是调用方按 `model_path` 亲手起的，启动日志里
///   已经有 `starting llama-server model_id=…`；日常每次都多打一行是白噪声。
///
/// 抽成纯函数是为了让「接管必须问身份」这条属性**可被单测钉住** ——
/// reattach 分支在单测里跑不到（本项目铁律：不可达分支必须由单测兜底）。
fn 应当核对模型身份(场景: 回执场景, 有模型路径: bool) -> bool {
    matches!(场景, 回执场景::接管旧进程) && 有模型路径
}

/// 接管态要不要再问一次引擎「你按什么上下文在跑」？
///
/// 🔴 **必须问**（P2-14，2026-10-16 P100 实机抓出）。接管**不重算启动参数**，
/// 沿用的是上一个会话留下的那个进程；而在此之前，这条路径上**没有任何一行日志**
/// 记录那个进程的参数。于是：
///
/// * 接管到一个 **262144 上下文 / 全量卸载**的服务，日志说「跑的确实是这个模型」；
/// * 接管到一个 **4096 上下文 / 兜底参数**的服务，日志**也是**这么说的。
///
/// 当天那台 P100 正是后者：14:35 接管到的其实是 07:18 用 24 层 / 4096 起的进程，
/// 用户因此误判成「上一轮的修复没生效」。判据要落到**引擎自报**的值上，
/// 见 `model_sidecar::解析上下文回执`。
///
/// **本次启动不问**：那条路径的参数已经有三重记录（`starting llama-server … ctx_size=`、
/// 启动成功本身、`checked GPU offload … verified=true`），每次启动再多问一次是白噪声。
/// （代价是「引擎把请求的 ctx 悄悄夹小」这一层目前只能靠命令行与 `/props` 手工对照。）
fn 应当报告引擎上下文(场景: 回执场景) -> bool {
    matches!(场景, 回执场景::接管旧进程)
}

/// 接管态的**交叉验证**：引擎自报的那个模型，权重真的进显存了吗（P2-15）。
///
/// 🔴 **P2-15（2026-10-16，老李拍板要做）**。P2-14 让接管态报出了 `n_ctx`，但引擎的
/// `/props` **不暴露 `n_gpu_layers`**（API 里没这个字段）—— 于是「**ctx 对、层数错**」
/// 这个组合仍然**同形**：报出 `262144` 却只卸载了几层，与真·全量卸载在日志上
/// 长得一模一样。
///
/// 引擎不给，就从**它留下的痕迹**反推：`-ngl 999` 的意图是「权重全进显存」，
/// 而权重**必须**占显存 —— 于是「实测整卡 `used` 连模型文件都装不下」就**确凿**
/// 说明层数没（全）上去。
///
/// 为什么**不**用「权重 + KV@n_ctx」当理论值：KV 的量化档（`kv_quant`）在接管态
/// **也拿不到**（`/props` 不给），按 fp16 算会**偏大** ⇒ 必然误报。只认**权重下界**，
/// 是这里唯一**不会误报**的口径。
///
/// **余量取 0.9 而不取 1.0**：GGUF 文件体积含**元数据**（键值表 / 词表 / 对话模板），
/// 比纯权重字节略大；而真出事时缺口远大于 10% —— P100 现场实测「24 层 / 4096」那一次
/// `used = 6424 MiB` vs 文件 `9787 MB`（≈ 9333 MiB），缺口 **31%**。
///
/// **判据单边**（只查偏小）：接管态整卡 `used` 里含**别的进程**（P100 上就常驻一个
/// ComfyUI），偏大是常态 ⇒ 拿双边阈值必然误报。反过来，这也意味着本判据**几乎不可能
/// 误报** —— 任何干扰都只会把 `used` 推高，不会压低。
///
/// 抽成纯函数是为了让**边界由单测钉住**（`不可达分支必须由单测兜底`）。
fn 实测显存够装权重(权重_gib: f64, 实测_gib: f64) -> bool {
    // 用局部 `let` 而非 `const`：常量名会被 `non_upper_case_globals` 判为非全大写
    // （中文后跟 `GiB` 这种混排必踩），局部变量无此约束。见 `gpu_offload_verified`。
    let 余量 = 0.9_f64;
    实测_gib >= 权重_gib * 余量
}

/// 采样失败的原因。**必须带原因**：两个调用点要打不同的措辞（接管态说「引擎回执」、
/// 启动态说「显存对账」），而「读不到」与「查了没事」在日志上必须能区分。
enum 采样缺失 {
    文件(String),
    显存 { 权重_gib: f64 },
}

/// 采样「模型文件体积 + 整卡已用显存」。
///
/// 🔴 **为什么抽出来**：接管态（P2-15）与启动态（B-1）的判据用的是**同一把尺子** ——
/// 同一个 `nvidia_used_vram_gb`、同一种 GiB 换算（GiB 按 1024³，不用十进制 GB）。
/// 两处各写一遍采样，将来改一处漏一处，就会变成「同一个词两个意思」而**不报错**，
/// 正是本项目反复踩的那类失效。日志措辞留给调用方，这里只把**采样与缺失原因**做一份。
async fn 采样权重与显存(模型文件: &Path) -> Result<(f64, f64), 采样缺失> {
    let 权重_gib = match std::fs::metadata(模型文件) {
        Ok(元数据) => 元数据.len() as f64 / (1024.0 * 1024.0 * 1024.0),
        Err(错误) => return Err(采样缺失::文件(错误.to_string())),
    };
    match tokio::task::spawn_blocking(crate::hardware::nvidia_used_vram_gb).await {
        Ok(Some(值)) => Ok((权重_gib, 值)),
        _ => Err(采样缺失::显存 { 权重_gib }),
    }
}

/// 规划器是否**声称**要把权重全部放上卡。
///
/// `999` 是 `cuda_launch_params` 里的全量哨兵（`:181`/`:196`）。按层数算出来的档位
/// （20 / 12 / 8）与保守固定档（24）都是**已经认了**部分卸载 —— 那是合法工作状态，
/// 不该拿「全量」的尺子去量。拿不到层数（`None`，CPU 模式）同样不算全量。
fn 计划全量卸载(n_gpu_layers: Option<u32>) -> bool {
    matches!(n_gpu_layers, Some(层) if 层 >= 999)
}

/// 判据本体：**声称全量、实际没装下** → 才是故障。
///
/// 单独抽出来是为了让单测能钉住「什么时候报、什么时候不许报」——
/// 尤其是 `部分卸载 + 装不下` 这个反例：16 GB 卡装 27B-Q3 时它**本来就装不下**
/// （只卸一部分层），报它就是把正常状态说成故障，属于「判据把错钉成期望行为」的反面。
fn 全量卸载却没上卡(计划层数: Option<u32>, 权重_gib: f64, 实测_gib: f64) -> bool {
    计划全量卸载(计划层数) && !实测显存够装权重(权重_gib, 实测_gib)
}

/// 启动路径的档 2：**计划全量 ⇏ 真上去了**。
///
/// `--n-gpu-layers 999` 只是**请求**。显存不足时 llama.cpp 会**静默**少卸几层，
/// 启动照样成功、健康检查照样 200、`/props` 照样报得出 `n_ctx` —— 只有速度掉一半。
/// 档 1（`gpu_offload_verified`，阈值固定 256 MiB）只能抓「完全没上卡」，
/// 抓不到「上了几层」。这里补的就是这一段灰色地带。
///
/// **只 WARN，不阻断启动**：部分卸载是「能跑但慢」，把它判成失败会让可用变不可用 ——
/// 而这条判据的价值是**让失败可见**，不是替用户做取舍。
///
/// 明说的漏报：缺口 < 10% 不报（与 P2-15 同一条单边口径）；小模型（8G 卡装 4.68 GiB）
/// 的部分卸载基本抓不到。**报不出的护栏好过乱报的护栏。**
async fn 核对计划全量卸载(模型文件: &Path, 计划层数: Option<u32>, 后端: EngineBackend) {
    // 不是全量计划 → 本判据**不适用**，但要说出来，否则「没查」与「查了没事」同形。
    if !计划全量卸载(计划层数) {
        tracing::debug!(
            backend = backend_label(后端),
            gpu_layers = ?计划层数,
            "显存对账：本次计划**不是**全量卸载（规划器已按余量分档），全量判据**不适用**"
        );
        return;
    }
    let (权重_gib, 实测_gib) = match 采样权重与显存(模型文件).await {
        Ok(对) => 对,
        Err(采样缺失::文件(错误)) => {
            tracing::info!(
                路径 = %模型文件.display(),
                错误 = %错误,
                "显存对账：读不到模型文件的体积，本次**未做**全量卸载判据"
            );
            return;
        }
        Err(采样缺失::显存 { 权重_gib }) => {
            tracing::info!(
                权重_gib = 权重_gib,
                "显存对账：读不到显存占用（非 N 卡或 nvidia-smi 不可用），本次**未做**全量卸载判据"
            );
            return;
        }
    };
    if !全量卸载却没上卡(计划层数, 权重_gib, 实测_gib) {
        tracing::info!(
            backend = backend_label(后端),
            权重_gib = 权重_gib,
            实测_gib = 实测_gib,
            "显存对账：计划全量卸载，实测显存也装得下权重 —— 权重确实都在卡上"
        );
        return;
    }
    tracing::warn!(
        backend = backend_label(后端),
        计划层数 = 计划层数.unwrap_or(0),
        权重_gib = 权重_gib,
        实测_gib = 实测_gib,
        "显存对账：**计划全量卸载，但实测显存装不下这个模型的权重** —— \
         引擎静默少卸了几层（`/props` 报得出 n_ctx、报不出层数，外面看不出来）。\
         启动**未受影响**，但生成会明显偏慢；点「停止」再「启动」会按当前显存重算，\
         若仍如此说明这张卡装不下这个量化档"
    );
}

/// 读一次模型文件体积与整卡已用显存，把「装得下吗」的**三态**分别打出来。
///
/// 三态必须可区分（项目铁律「降级 / 跳过 / 未跑完必须与『通过』可区分」）：
///
/// * 装得下   → `INFO`（常态：接管到参数正确的服务）
/// * 装不下   → **`WARN`** + 可行动的建议
/// * **读不到**（文件 stat 失败 / 非 N 卡 / `nvidia-smi` 不可用）→ `INFO`，措辞
///   必须写明「**本次未做**」—— 静默吞掉的话，「没查」与「查了没事」在日志上同形，
///   正是本项目翻过五次车的那类错误。
///
/// ⚠️ 本判据**只覆盖「连权重都没装下」**这一种。**部分卸载**（16 GB 卡装 27B-Q3）
/// 是合法工作状态，缺口不到 10% 时判据**有意**放过 —— 那属于性能取舍，不是故障。
async fn 核对接管态显存(模型文件: &Path) {
    // 采样与启动态（B-1）共用一份 —— 两处判据必须同一把尺子，见 `采样权重与显存`。
    let (权重_gib, 实测_gib) = match 采样权重与显存(模型文件).await {
        Ok(对) => 对,
        Err(采样缺失::文件(错误)) => {
            tracing::info!(
                路径 = %模型文件.display(),
                错误 = %错误,
                "引擎回执：读不到模型文件的体积，本次**未做**显存交叉验证"
            );
            return;
        }
        Err(采样缺失::显存 { 权重_gib }) => {
            tracing::info!(
                权重_gib = 权重_gib,
                "引擎回执：读不到显存占用（非 N 卡或 nvidia-smi 不可用），本次**未做**显存交叉验证"
            );
            return;
        }
    };
    if 实测显存够装权重(权重_gib, 实测_gib) {
        tracing::info!(
            权重_gib = 权重_gib,
            实测_gib = 实测_gib,
            "引擎回执：实测显存装得下这个模型的权重 —— 接管到的服务确实把权重放上了卡"
        );
    } else {
        tracing::warn!(
            权重_gib = 权重_gib,
            实测_gib = 实测_gib,
            "引擎回执：**实测显存明显装不下这个模型的权重** —— 层数没（全）上去。\
             显存不足时引擎会**静默**少卸几层，而 `/props` 报得出 `n_ctx`、报不出层数；\
             点「停止」再「启动」会按当前显存重算；若仍如此，说明这张卡装不下这个量化档"
        );
    }
}

/// 把 llama-server 的**监听地址**换算成**可连接的探测地址**。
///
/// 🔴 **为什么必须换算**（P2-12，2026-10-16 实机抓出）：服务可以监听 `0.0.0.0`
/// （设置里开了「局域网可访问」），而 `0.0.0.0` 是「本机所有网卡」的**通配**
/// 写法，**不是可连接地址** —— 直接连它当场失败：
/// `error sending request for url (http://0.0.0.0:8000/v1/models)`。
/// 本机一直是默认的 `127.0.0.1`，所以这个坑直到 P100 开了局域网才暴露：
/// 那台机器上「接管到的到底是不是这个模型」**永远核对不成**。
///
/// ⚠️ 现象很容易读反：**健康探测走的是 `127.0.0.1`**（日志里那串
/// `connecting to 127.0.0.1:8000` 就是），所以现场看起来是
/// 「端口明明健康，回执却说取不到」—— 别把它读成「引擎没起来」。
///
/// 非通配地址（`127.0.0.1`、具体网卡 IP、主机名）**原样返回**：
/// 一律改成 `127.0.0.1` 会在服务只监听某块网卡时反而连不上。
fn 探测主机(监听地址: &str) -> String {
    match 监听地址.trim() {
        "" | "0.0.0.0" | "::" | "[::]" | "*" => "127.0.0.1".to_string(),
        其他 => 其他.to_string(),
    }
}

/// 服务就绪后**问引擎**要回执：命令行里的参数到底「传过去了」还是「生效了」。
///
/// 三态必须可区分（本项目铁律「降级 / 跳过 / 未跑完必须与『通过』可区分」）：
///
/// * 一致   → `INFO`
/// * 不一致 → **`WARN`**，附问题清单
/// * 取不到 → **`WARN`**，措辞必须写明「**取不到 ≠ 没生效**」——
///   否则「端点还没就绪」会被读成「适配器白挂」，把结论带反。
///
/// 期望值**只从最终附加参数里解析**，不从旁车配置读 ——
/// 那是唯一的事实来源（用户可以在设置里手写同名标志盖过旁车配置）。
async fn 核对引擎回执(
    host: &str,
    port: u16,
    模型路径: Option<&Path>,
    附加参数: &[String],
    场景: 回执场景,
) {
    let 期望适配器 = crate::model_sidecar::期望的适配器(附加参数);
    let 期望模板文件 = crate::model_sidecar::期望的对话模板(附加参数);

    // 🔴 接管场景下**总是**问一次模型身份，即使没有任何微调产物：
    //    那是最基本、也最容易漏掉的一层 —— 2026-10-16 老李重启 APP 时就走的这条路
    //    （`reattached to previous llama-server`），而当时这一层还没接上。
    let 查身份 = 应当核对模型身份(场景, 模型路径.is_some());
    // 🔴 接管态**无条件**报一次引擎自报的上下文（P2-14）—— 即使没配旁车、
    //    即使模型路径没能解析出来。理由见 `应当报告引擎上下文`。
    let 报上下文 = 应当报告引擎上下文(场景);
    if 期望适配器.is_empty() && 期望模板文件.is_none() && !查身份 && !报上下文 {
        return;
    }

    // 探测地址 ≠ 监听地址：`0.0.0.0` 是通配写法、不可连接（见 `探测主机`）。
    let 基址 = format!("http://{}:{port}", 探测主机(host));
    let 客户端 = match reqwest::Client::builder()
        .timeout(Duration::from_secs(8))
        .build()
    {
        Ok(客户端) => 客户端,
        Err(错误) => {
            tracing::warn!(错误 = %错误, "引擎回执：HTTP 客户端建不起来，本次不核对（**这不代表没生效**）");
            return;
        }
    };

    // ── 第 ① 层：端口上跑的是不是我要的那个模型 ─────────────────────────────
    //
    // 引擎自报的那个路径**还要留给第 ②b 层**当「权重体积」的来源（P2-15），
    // 所以在这里带出来 —— 见 `核对接管态显存`。
    let mut 引擎自报模型: Option<String> = None;
    if 查身份 {
        if let Some(期望模型) = 模型路径 {
            let 期望模型 = 期望模型.display().to_string();
            match 取回执(&客户端, &format!("{基址}/v1/models")).await {
                None => tracing::warn!(
                    期望 = %期望模型,
                    "引擎回执：/v1/models 取不到 —— **这不代表跑错了模型**，只代表这次没核对成"
                ),
                Some(正文) => match crate::model_sidecar::解析模型回执(&正文) {
                    Err(错误) => tracing::warn!(
                        错误 = %错误,
                        正文 = %正文.chars().take(200).collect::<String>(),
                        "引擎回执：模型列表回执看不懂，本次不核对"
                    ),
                    Ok(自报) => {
                        let 问题 = crate::model_sidecar::核对模型(&期望模型, &自报);
                        if 问题.is_empty() {
                            tracing::info!(
                                模型 = %自报,
                                "引擎回执：接管到的服务跑的确实是这个模型"
                            );
                        } else {
                            tracing::warn!(
                                期望 = %期望模型,
                                引擎自报 = %自报,
                                问题 = ?问题,
                                "引擎回执：**接管到的服务可能不是这个模型** —— 建议停掉后重新启动服务"
                            );
                        }
                        // 对不上**也照样留着**：第 ②b 层要验的是「显存里装的是不是
                        // 这个文件」，与「是不是我们期望的那个」是两回事。
                        引擎自报模型 = Some(自报);
                    }
                },
            }
        }
    }

    // ── 第 ② 层：接管到的服务**按什么规格在跑** ───────────────────────────────
    //
    // 模板核对（第 ③ 层）也要 `/props`，所以**只取一次**、两个消费者共用。
    let 需要属性 = 期望模板文件.is_some() || 报上下文;
    let 属性正文 = if 需要属性 {
        取回执(&客户端, &format!("{基址}/props")).await
    } else {
        None
    };
    if 报上下文 {
        match &属性正文 {
            None => tracing::warn!(
                "引擎回执：/props 取不到 —— **这不代表上下文不对**，只代表这次没核对成"
            ),
            Some(正文) => match crate::model_sidecar::解析上下文回执(正文) {
                Err(错误) => tracing::warn!(
                    错误 = %错误,
                    "引擎回执：属性回执里读不到 n_ctx，接管到的服务按什么上下文在跑**未核对**"
                ),
                Ok(上下文) => tracing::info!(
                    n_ctx = 上下文,
                    "引擎回执：接管到的服务自报的上下文长度 —— 接管**不重算**启动参数；\
                     这个值若明显偏小（4096 / 2048 = 兜底或保守参数），点「停止」再「启动」按当前显存重算"
                ),
            },
        }
    }

    // ── 第 ②b 层：接管到的服务，**权重真的进显存了吗**（P2-15）──────────────
    //
    // 第 ② 层报了「按什么上下文在跑」，但引擎**报不出 `n_gpu_layers`**
    // （`/props` 没有这个字段）⇒「ctx 对、层数错」在这一层仍然**同形**。
    // 补法不是再问引擎，而是从**它留下的痕迹**反推：见 `实测显存够装权重`。
    //
    // 只在接管态做（与第 ② 层同一个门）：
    // * 本次启动那条路径手里没有模型路径（调用点传 `None`，进程是我们亲手起的），
    //   拿不到「权重体积」这个比对基准；
    // * 而且那条路径已经有 `checked GPU offload … verified=true` 兜着（阈值 256 MiB
    //   —— 能抓「整卡没上」，但抓不到「只卸了几层」，那是另一条待办）。
    if 报上下文 {
        match 引擎自报模型.as_deref() {
            Some(自报路径) => 核对接管态显存(Path::new(自报路径)).await,
            None => tracing::info!(
                "引擎回执：没能从引擎自报里拿到模型文件路径，本次**未做**显存交叉验证"
            ),
        }
    }

    // 「对不上」的建议按场景区分。措辞必须准确：把「配置改过、进程还是旧的」
    // 说成「适配器没生效」，会把排查引到错的方向去。
    let 对不上的建议 = match 场景 {
        回执场景::本次启动 => "适配器可能没生效",
        回执场景::接管旧进程 => {
            "若刚改过启动配置，这是意料之中（进程还是旧的）—— 重启服务即可生效；否则适配器可能没生效"
        }
    };

    if !期望适配器.is_empty() {
        match 取回执(&客户端, &format!("{基址}/lora-adapters")).await {
            None => tracing::warn!(
                期望 = %crate::model_sidecar::描述适配器(&期望适配器),
                "引擎回执：/lora-adapters 取不到（端点未就绪或版本不支持）—— \
                 **这不代表适配器没挂上**，只代表这次没核对成"
            ),
            Some(正文) => match crate::model_sidecar::解析适配器回执(&正文) {
                Err(错误) => tracing::warn!(
                    错误 = %错误,
                    正文 = %正文.chars().take(200).collect::<String>(),
                    "引擎回执：适配器回执看不懂，本次不核对"
                ),
                Ok(实际) => {
                    let 问题 = crate::model_sidecar::核对适配器(&期望适配器, &实际);
                    if 问题.is_empty() {
                        tracing::info!(
                            适配器 = %crate::model_sidecar::描述适配器(&实际),
                            "引擎回执：适配器已确认加载（命令行与引擎自报一致）"
                        );
                    } else {
                        tracing::warn!(
                            期望 = %crate::model_sidecar::描述适配器(&期望适配器),
                            引擎自报 = %crate::model_sidecar::描述适配器(&实际),
                            问题 = ?问题,
                            建议 = %对不上的建议,
                            "引擎回执：**命令行传了，但引擎侧对不上**"
                        );
                    }
                }
            },
        }
    }

    if let Some(模板路径) = 期望模板文件 {
        let 期望内容 = match std::fs::read_to_string(&模板路径) {
            Ok(内容) => 内容,
            Err(错误) => {
                // 启动前 `读取旁车参数` 已经确认过文件存在，走到这里读不到属异常情况，
                // 但**不许静默** —— 否则「模板没换」会以「核对通过」的名义蒙混过去。
                tracing::warn!(模板 = %模板路径, 错误 = %错误, "引擎回执：模板文件读不出来，本次不核对");
                return;
            }
        };
        // 复用第 ② 层取回的属性回执，不重复请求（`期望模板文件.is_some()` ⇒ `需要属性` 必为真）。
        match &属性正文 {
            None => tracing::warn!(
                模板 = %模板路径,
                "引擎回执：/props 取不到 —— **这不代表模板没生效**，只代表这次没核对成"
            ),
            Some(正文) => match crate::model_sidecar::解析模板回执(正文) {
                Err(错误) => {
                    tracing::warn!(错误 = %错误, "引擎回执：属性回执里没有模板，本次不核对");
                }
                Ok(引擎模板) => {
                    let 问题 = crate::model_sidecar::核对对话模板(&期望内容, &引擎模板);
                    if 问题.is_empty() {
                        tracing::info!(
                            模板 = %模板路径,
                            字符数 = 引擎模板.chars().count(),
                            "引擎回执：对话模板已确认生效（与文件一致）"
                        );
                    } else {
                        // 🔴 这一条是「部署模板与训练契约不一致」的直接判据，
                        //    实测同一份适配器的增益会从 +39.1 pp 塌到 +4.3 pp。
                        tracing::warn!(
                            模板 = %模板路径,
                            问题 = ?问题,
                            建议 = %对不上的建议,
                            "引擎回执：**模板与文件对不上** —— 微调增益可能大幅缩水"
                        );
                    }
                }
            },
        }
    }
}

/// 取一个只读端点，失败时**有限重试**。
///
/// 为什么必须重试：服务刚过健康检查时，这两个端点**未必已经就绪**，
/// 一次失败就下结论会把「还没准备好」误报成「引擎没接」。
/// 重试用尽仍失败 ⇒ 由调用方记成「取不到」，**不许**写成「没生效」。
async fn 取回执(客户端: &reqwest::Client, 地址: &str) -> Option<String> {
    for 第几次 in 0..3u32 {
        if 第几次 > 0 {
            tokio::time::sleep(Duration::from_millis(800)).await;
        }
        match 客户端.get(地址).send().await {
            Ok(响应) if 响应.status().is_success() => match 响应.text().await {
                Ok(正文) => return Some(正文),
                Err(错误) => {
                    tracing::debug!(错误 = %错误, "引擎回执：读正文失败，重试");
                }
            },
            Ok(响应) => {
                tracing::debug!(状态 = %响应.status(), "引擎回执：状态码不是 2xx，重试");
            }
            Err(错误) => {
                tracing::debug!(错误 = %错误, "引擎回执：请求失败，重试");
            }
        }
    }
    None
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
        //
        // 注意借用顺序：先把退出码取成普通值，`try_wait` 的 &mut 借用随之结束，
        // 之后才能把 `child` 再借给读取 stderr 的函数（需要 &mut 去 take 管道）。
        let 已退出 = match child.try_wait() {
            Ok(Some(exit_status)) => Some(exit_status.code().unwrap_or(-1)),
            _ => None,
        };
        if let Some(code) = 已退出 {
            // 把 llama-server 的 stderr 读回来附在错误里。这是排障的关键：
            // 只给一个退出码，用户和我们都只能猜；llama.cpp 在真正失败前
            // 通常已经打印了原因（CUDA 初始化失败、架构不支持、显存不足...）。
            let 诊断 = 读取子进程诊断输出(child).await;
            return Err(AppError::Message(格式化启动失败(code, &诊断)));
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

/// 读回已退出子进程的 stderr，取最后若干行作为诊断信息。
///
/// llama.cpp 在真正失败前通常已经打印了原因（CUDA 初始化失败、架构不支持、
/// 显存不足、模型文件损坏...）。进程已退出，管道写入端关闭，读到 EOF 即结束。
/// 设一个短超时兜底：万一子进程留下了孙进程仍占着管道，不能让这里挂住。
async fn 读取子进程诊断输出(child: &mut Child) -> String {
    let Some(mut stderr) = child.stderr.take() else {
        return String::new();
    };
    let Ok(读取结果) = tokio::time::timeout(
        Duration::from_secs(3),
        tokio::task::spawn_blocking(move || {
            use std::io::Read;
            let mut 缓冲 = String::new();
            let _ = stderr.read_to_string(&mut 缓冲);
            缓冲
        }),
    )
    .await
    else {
        return String::new();
    };
    let Ok(文本) = 读取结果 else {
        return String::new();
    };

    // 取最后 12 行非空内容：崩溃原因基本都在尾部，前面多是加载进度。
    let 尾部: Vec<&str> = 文本
        .lines()
        .filter(|l| !l.trim().is_empty())
        .rev()
        .take(12)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    尾部.join("\n")
}

/// 把退出码翻译成人能行动的中文提示，并附上 llama.cpp 的原始输出。
///
/// **判据优先看引擎输出**（2026-09-29 虚拟机实测教训）：同一个退出码
/// （如 0xC0000409）可能由多种原因触发，只看码会指错方向。当时的真实案例是
/// 退出码指向「CPU 型号过精简」，但引擎输出里明摆着
/// `CUDA error: invalid device function` —— 真正的根因是 **CUDA kernel 的
/// 目标架构与显卡不匹配**，跟 CPU 毫无关系。所以这里先用引擎输出做归因，
/// 归因不出来才退回按退出码给通用指引。
fn 格式化启动失败(code: i32, 诊断: &str) -> String {
    let mut 文案 = format!("AI服务进程启动后异常退出（退出码 {}）。", code);

    if let Some(说明) = 从引擎输出推断原因(诊断) {
        文案.push('\n');
        文案.push_str(&说明);
    } else {
        // 引擎输出没能给出线索，退回按 Windows NTSTATUS 退出码给通用指引。
        文案.push('\n');
        文案.push_str(按退出码给指引(code));
    }

    if !诊断.is_empty() {
        文案.push_str("\n\n运行引擎的输出：\n");
        文案.push_str(诊断);
    }
    文案
}

/// 从 llama.cpp 的 stderr 里认出已知的失败特征，给出针对性的处置建议。
///
/// 返回 `None` 表示没认出已知特征 —— 调用方会退回按退出码给通用指引。
fn 从引擎输出推断原因(诊断: &str) -> Option<String> {
    // CUDA kernel 架构不匹配：引擎编入的设备码里没有这张卡的目标架构。
    if 诊断.contains("invalid device function") || 诊断.contains("no kernel image is available") {
        return Some(说明显卡架构不匹配());
    }

    // CUDA 运行时/驱动缺失。
    if 诊断.contains("CUDA error") && 诊断.contains("not found")
        || 诊断.contains("cudart64")
        || 诊断.contains("cublas")
    {
        return Some(
            "运行引擎缺少 CUDA 运行时库，或显卡驱动不可用。\
             请在虚拟机里确认已安装 NVIDIA 驱动（命令 nvidia-smi 能正常输出），\
             或改用 CPU 安全模式运行。"
                .to_string(),
        );
    }

    // 显存不足。
    if 诊断.contains("out of memory") || 诊断.contains("CUDA_ERROR_OUT_OF_MEMORY") {
        return Some(
            "显卡显存不足，模型加载失败。请在设置里调小上下文长度，\
             或把「GPU 层数」降低后再试。"
                .to_string(),
        );
    }

    None
}

/// 「CUDA 内核不是为这张卡编的」——按**本机实际显卡**生成说明。
///
/// 2026-09-30 修正：上一版文案写死了「RTX 50 系（sm_120）需要 CUDA 12.8」，
/// 并建议「改用 Vulkan」。在 P100（Pascal `sm_60`）上这两句**都是错的**：
/// ① 卡是 `sm_60` 不是 `sm_120`，会把用户往错方向带；
/// ② 该机处于 TCC 模式，Vulkan 根本用不了 —— 那条建议等于白说。
///
/// 现在改成：**只陈述可核实的事实**（引擎编入的架构档位 + 本机卡型号与算力），
/// 处置按「换另一种加速方式 -> CPU -> 升级驱动 / 换引擎包」给，不再指定 Vulkan。
fn 说明显卡架构不匹配() -> String {
    let 本机卡 = match (
        crate::hardware::primary_nvidia_name(),
        crate::hardware::nvidia_compute_cap(),
    ) {
        (Some(名), Some(算力)) => format!("你的显卡是 {}（算力 {}）", 名, 算力),
        (Some(名), None) => format!("你的显卡是 {}", 名),
        _ => "当前显卡".to_string(),
    };
    format!(
        "运行引擎里的 CUDA 内核不是为当前显卡编译的（CUDA error: invalid device function）。\n\
         原因：当前引擎（CUDA 12.4 预编译包）编入的真实架构是 sm_86（RTX 30 系）\
         与 sm_89（RTX 40 系），{}不在其中；引擎自带的 PTX 也没能被当前驱动\
         即时编译（驱动偏旧时会拒绝编译较新版本的 PTX）。\n\
         处置：① 先在设置里换用另一种显卡加速方式（若本机可用）；\
         ② 或先用 CPU 安全模式；③ 长期解决需要升级显卡驱动，\
         或换用包含你这张卡架构的引擎包。",
        本机卡
    )
}

/// 引擎输出没给出线索时，按 Windows NTSTATUS 快速失败码给通用指引。
fn 按退出码给指引(code: i32) -> &'static str {
    match code as u32 {
        0xC0000409 => {
            "运行引擎触发了保护性中止（STATUS_STACK_BUFFER_OVERRUN）。\
             常见原因有三类：①显卡驱动与引擎不匹配；②CPU 指令集不足\
             （虚拟机上如 PVE 默认的 qemu64/kvm64）；③运行库文件损坏。\
             建议先改用 CPU 安全模式确认程序与模型本身正常，再排查显卡。"
        }
        0xC000001D => {
            "运行引擎使用了当前 CPU 不支持的指令集（STATUS_ILLEGAL_INSTRUCTION）。\
             建议在虚拟机设置里把 CPU 类型改为 host。"
        }
        0xC0000005 => {
            "运行引擎访问了非法内存地址（STATUS_ACCESS_VIOLATION）。\
             常见于显卡驱动异常或引擎文件不完整。"
        }
        0xC00000FD => "运行引擎栈溢出（STATUS_STACK_OVERFLOW）。",
        _ => "可能是运行引擎与硬件不兼容或模型文件损坏。",
    }
}

/// Single health probe used by startup recovery to decide whether a persisted
/// service is still alive.
///
/// 🔴 **两次探测 + 间隔**，不是一次：这条路失败的后果不是「再试一次」，
/// 而是 `recover_service` 立刻 `kill_pid` 掉一个**可能正健康的服务**
/// （2026-10-19 P100：关 APP 再重开就撞上「端口被占用」，正是这条判据太脆的
/// 一种走法）。代价不对称 —— 多等 0.8 秒的代价是零，误杀的代价是用户重载模型。
///
/// 仍然保留「快失败」的语义：真死了的进程第一探就连不上，第二探只是补一次
/// 引擎正忙（显存吃紧、刚被唤醒）时的瞬时不应答。
async fn health_check_once(port: u16) -> bool {
    let Ok(client) = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
    else {
        return false;
    };
    let url = format!("http://127.0.0.1:{}/v1/models", port);
    for 次 in 0..2 {
        let 活着 = client
            .get(&url)
            .send()
            .await
            .map(|response| response.status().is_success())
            .unwrap_or(false);
        if 活着 {
            return true;
        }
        if 次 == 0 {
            tokio::time::sleep(Duration::from_millis(800)).await;
        }
    }
    false
}

/// 问一句「占着这个端口的是不是**我方引擎**」。
///
/// 判据是**协议级**的：`GET /v1/models` 能回一份带 `id` 的模型列表。
/// 只看「端口在听」不够 —— 任何程序（IDE 的内置服务、另一个工具的 Web UI）
/// 都能让端口在听，那样会把别人的服务误认成引擎、甚至接管过来。
///
/// 返回引擎**自报**的模型（路径或别名），由调用方决定「是不是我要的那个」。
/// 拿不到（连不上 / 非 200 / 回执解析不了）一律返回 `None`，按「不是引擎」处理：
/// 这条路只在端口已经确认被占用时走，宁可报「请换端口」也不能误接管。
async fn 识别端口上的引擎(host: &str, port: u16) -> Option<String> {
    let 客户端 = reqwest::Client::builder()
        // 8 秒：这是**本地回环**，慢到 8 秒还没答的进程基本是卡死的；
        // 但也确实见过显存吃紧时引擎短暂不应答，所以不取 2 秒那么短，
        // 免得把「还活着只是喘」误判成「不是引擎」。
        .timeout(Duration::from_secs(8))
        .build()
        .ok()?;
    let 基址 = format!("http://{}:{port}", 探测主机(host));
    let 正文 = 取回执(&客户端, &format!("{基址}/v1/models")).await?;
    crate::model_sidecar::解析模型回执(&正文).ok()
}

/// 端口类错误的统一前缀。
///
/// 两件事都靠它：`是端口占用类错误` 用它判「不该重试」，文案本身也以它开头。
/// 改文案时**必须**同步这里 —— 有一条单测钉住「三处构造点都以它开头」。
const 端口类错误前缀: &str = "接口端口";

/// 这条错误是不是「端口被别的引擎占着」—— 是的话**不许重试**。
///
/// `start_service_core` 的失败分支会先用保守参数再起一次、再用 CPU 兜底起一次。
/// 端口被占着时那两次注定同样失败（都绑不上端口），只是把真正的错因埋进两条
/// `GPU launch failed; retrying…` 的 WARN 里，最后才由 CPU 分支报出来。
fn 是端口占用类错误(消息: &str) -> bool {
    消息.starts_with(端口类错误前缀)
}

/// 「端口上监听的是不是刚 spawn 的那个进程」的三种结论。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum 端口归属 {
    是本次进程,
    /// 端口上是**别的**进程。
    是别的进程(u32),
    /// 查不到端口上的 pid（netstat 不可用 / 非 Windows）⇒ 没有反证。
    无法确认,
}

fn 判端口归属(端口pid: Option<u32>, 本次pid: u32) -> 端口归属 {
    match 端口pid {
        Some(pid) if pid == 本次pid => 端口归属::是本次进程,
        Some(pid) => 端口归属::是别的进程(pid),
        None => 端口归属::无法确认,
    }
}

/// 一次启动「为什么没能全量卸到显卡上」的归因。
///
/// 🔴 这个枚举存在的理由（**0.1.7-1**，2026-10-20）：启动参数按**可用**显存算，
/// 卡上还有别的程序时规划器会「合理地」降档 —— 每一步都对，界面写着「启动成功」，
/// 用户只知道慢、不知道慢在哪、更不知道关掉那个程序就能好。部分卸载本身是
/// 合法态，所以**不能拦**；能做的只有把「是谁的锅」说清楚。
#[derive(Debug, Clone, Copy, PartialEq)]
enum 降档 {
    /// 全量卸载，没有降档。此时无话可说。
    未降档,
    /// 降档了，但**把整张卡都给它也全量不了** —— 模型本来就比这张卡大
    /// （27B-Q3 放 8G 卡）。这是选购问题，不是故障，启动时不提示。
    与他人无关,
    /// 降档了，且**空卡本来全量得下** ⇒ 差额是被别的进程占走的。
    /// 这是唯一值得在界面上说的那种。
    卡被占 {
        /// 扣掉该驱动模式的空载基线之后，仍被别人占着的量（GiB）。
        /// **只写进日志**：基线是估计值，把它当精确数字念给用户是过度自信。
        他人_gib: f64,
        /// 规划时读到的可用 / 总量（GiB）。提示里只念这两个 ——
        /// 它们是用户能在任务管理器/`nvidia-smi` 里自己对上的事实。
        可用_gib: f64,
        总量_gib: f64,
    },
    /// 拿不到显存读数（非 N 卡 / `nvidia-smi` 不可用），或参数走了兜底档。
    无从判断,
}

/// 空载时「本来就该占着」的显存（GiB）—— 归因之前必须先扣掉它。
///
/// 🔴 没有这个数，判据会在**常态机器上恒亮**（0.1.7-1 的现场）：
/// 2026-10-20 本机 RTX 5060 Laptop / 8G，总量 7.960、空闲 5.760 ⇒ 桌面合成占
/// 2.20 GiB，`可用/总量 = 0.724`。旧的「可用 < 总量×0.80 就算被占」于是**每次
/// 启动都报**「卡上有别的进程占着」，而卡上只有桌面。报警在常态下亮着，用户
/// 迟早学会无视它 —— 真被占的那次就没人看了，护栏自己失效。
///
/// 取**绝对值**而不是比例：桌面合成的开销与卡多大基本无关（2~3 GiB 量级），
/// 用比例会让 24G 卡上的同样开销被判成「没占」、8G 卡上被判成「占了」。
const 空载基线_TCC_GIB: f64 = 1.0; // P100 实测空载 0.1~0.2 GiB
const 空载基线_WDDM_GIB: f64 = 3.0; // 笔记本实测 2.20 GiB，留多显示器/浏览器硬解的余量
const 空载基线_未知_GIB: f64 = 1.5; // 认不出驱动模式时的折中

/// 按驱动模式取空载基线。TCC 是计算专用模式（**没有桌面合成**），WDDM 有。
///
/// 纯函数：驱动模式由调用方拿到后传进来。放进函数里现查会让这条判据
/// 变成「测不了」的 —— 而测不了的判据等于没有。
fn 空载基线_gib(驱动模式: Option<&str>) -> f64 {
    match 驱动模式
        .map(|值| 值.trim().to_ascii_uppercase())
        .as_deref()
    {
        Some("TCC") => 空载基线_TCC_GIB,
        Some("WDDM") => 空载基线_WDDM_GIB,
        _ => 空载基线_未知_GIB,
    }
}

/// 归因：这次的降档该不该算到「别的进程头上」。
///
/// 判据只有一条 —— **反事实**：把整张卡都给这个模型，能全量吗？
///
/// * 能（`空卡计划` 全量）而现在不能 ⇒ 差额只能是被占走的（扣掉空载基线）；
/// * 空卡也不能 ⇒ 模型本身就装不下这张卡，跟别人没关系。
///
/// 这一条替代了旧的「可用 < 总量×0.80」。旧判据只看**比例**，于是
/// ①WDDM 笔记本上桌面合成那 2.2 GiB 让它在常态下恒亮；②27B 放 8G 卡上
/// 报「卡被占」，而其实是模型比卡大 —— 两个方向都错，且错法不同。
fn 判定降档(
    总量: f64,
    可用: f64,
    空载基线: f64,
    计划: LaunchParams,
    空卡计划: LaunchParams,
) -> 降档 {
    if 总量 <= 0.0 || 可用 <= 0.0 {
        return 降档::无从判断;
    }
    if 计划全量卸载(计划.n_gpu_layers) {
        return 降档::未降档;
    }
    if !计划全量卸载(空卡计划.n_gpu_layers) {
        return 降档::与他人无关;
    }
    let 他人_gib = 总量 - 可用 - 空载基线;
    if 他人_gib <= 0.0 {
        // 空卡本来全量得下，现在不行，但被占的量没有超过「该有的空载开销」——
        // 差额来自常态占用（桌面/驱动/多显示器），不是某个可关掉的程序。
        // 说「卡被占」只会让用户去关浏览器，关完还是这样。
        return 降档::与他人无关;
    }
    降档::卡被占 {
        他人_gib,
        可用_gib: 可用,
        总量_gib: 总量,
    }
}

/// 把归因写成一句**人能照着做**的话。返回 `None` = 没什么可说的。
///
/// 三条硬要求（都来自踩过的坑）：
/// 1. **不阻断**：部分卸载是合法态，这句话只能出现在启动成功之后；
/// 2. **只念用户能核对的事实**（可用 / 总量 / 层数 / 上下文），
///    不念估算出来的「别人占了多少」；
/// 3. **给出下一步**：否则用户知道慢了也没法改变。
fn 降档提示(归因: 降档, 计划: LaunchParams, 主干层数: Option<u32>) -> Option<String> {
    let 降档::卡被占 {
        可用_gib, 总量_gib, ..
    } = 归因
    else {
        return None;
    };
    let 层数 = match (计划.n_gpu_layers, 主干层数) {
        (Some(层), Some(总)) if 层 < 总 => format!("{层}/{总} 层"),
        (Some(层), _) => format!("{层} 层"),
        // CPU 启动不会走到这里（`卡被占` 只在 GPU 计划里判），留个兜底不撒谎。
        (None, _) => "一部分层".to_string(),
    };
    Some(format!(
        "显卡上还有别的程序占着显存（当前可用 {可用_gib:.1} GB，整卡 {总量_gib:.1} GB），\
         所以这次只把 {层数} 放到了显卡上、上下文压到 {} —— 生成会明显偏慢。\
         关掉占用显卡的程序后，点「停止」再「启动」会重新按余量分配。",
        计划.ctx_size
    ))
}

/// 健康检查通过**只证明这个端口上有东西在答**，不证明答的是本次起的进程。
///
/// 🔴 2026-10-19 P100（`07:42:10`）就是被这一点骗了：旧引擎
/// （`qwen3_1_7b_light`，管理员权限那一次起的、`taskkill` 拒绝访问）还占着 8000
/// 且 `/v1/models` 正常应答；新起的 Ornith 进程绑不上端口、却也活了一会儿 ——
/// 健康探测连续两次 200，于是日志写下
/// `llama-server running model_id=Some("local-Ornith-1.5-9B-Q8_0")`，
/// 界面显示切到了 9B，**实际跑的还是 1.7B**。
///
/// 判据用**进程身份**（端口 pid 与 `child.id()` 比对），它比模型名更硬：这个窗口内
/// pid 不会被复用，而模型名要绕过一次 `/v1/models` 解析才拿得到。只有查不到端口
/// pid 时才退回模型身份核对；两者都拿不到时**放行**（打 WARN）——
/// 「无法确认」不等于「确认有问题」，拿它拦启动会在 netstat 被封的机器上变成死路。
async fn 确认本次进程在服务(child_pid: u32, port: u16, 期望模型: &Path) -> AppResult<()> {
    let 期望 = 期望模型.display().to_string();
    match 判端口归属(端口上正在监听的进程(port), child_pid) {
        端口归属::是本次进程 => {
            // 🔴 成功路径**也必须留一行**。2026-10-19 P100 那份 23 分钟的干净日志
            // （`08:21`~`08:44`）里，这条判据跑过十几次，却一个字节都看不见 ——
            // 「核对通过」与「这段代码根本没被调用」在日志里同形。本项目已经栽过
            // 「钩子永不触发，全绿但现场照旧」，所以把结论打出来。
            tracing::info!(
                port,
                本次pid = child_pid,
                "端口归属已核对：监听端口的就是本次起的进程（不是上一个引擎在替它应答）"
            );
            Ok(())
        }
        端口归属::是别的进程(占用) => {
            let 自报 = 识别端口上的引擎("127.0.0.1", port).await;
            // 自报对得上 = 端口上跑的正是这个模型（可能是同一棵进程树里的另一个
            // pid，例如瘦启动器把真正监听的进程另起了一个），这种情况放行。
            if let Some(自报) = &自报 {
                if crate::model_sidecar::核对模型(&期望, 自报).is_empty() {
                    tracing::debug!(
                        port,
                        端口pid = 占用,
                        本次pid = child_pid,
                        "端口上监听的不是本次 spawn 的那个 pid，但引擎自报的模型一致 ⇒ 视为已在服务"
                    );
                    return Ok(());
                }
            }
            let 是谁 = 自报.unwrap_or_else(|| "另一个引擎".to_string());
            Err(AppError::Message(format!(
                "{端口类错误前缀} {port} 上现在是**另一个引擎**在服务（{是谁}，进程 {占用}），\
                 不是这次启动的「{期望}」—— 新起的进程（{child_pid}）没能绑上端口。\
                 多半是上一个引擎没被彻底结束（它若以管理员权限启动，普通权限的本程序结束不掉）。\
                 请右键「以管理员身份运行」本程序后再切换，或到任务管理器结束进程 {占用}。"
            )))
        }
        端口归属::无法确认 => match 识别端口上的引擎("127.0.0.1", port).await {
            Some(自报) if crate::model_sidecar::核对模型(&期望, &自报).is_empty() => Ok(()),
            Some(自报) => Err(AppError::Message(format!(
                "{端口类错误前缀} {port} 上跑的是「{自报}」，不是这次启动的「{期望}」—— \
                 新起的进程（{child_pid}）没能绑上端口。\
                 请先停掉端口上的服务，或到设置里换一个端口。"
            ))),
            None => {
                tracing::warn!(
                    port,
                    本次pid = child_pid,
                    "查不到端口归属于哪个进程，也问不出引擎自报的模型 ⇒ 本次**无法确认**\
                     端口上服务的是不是刚起的进程；按启动成功处理，但这不算证据"
                );
                Ok(())
            }
        },
    }
}

/// 端口上已经有一份**我方引擎**在跑时，直接接管，而不是报「端口被占用」。
///
/// 这条路径补的是 2026-10-19 P100 实测的那个死角：`recover_service` 手里的
/// 记录一旦缺失（换过安装包 / 数据目录），重开 APP 就**再也接不上**自己上一次
/// 起的服务，而前端自动启动又必然撞上它 —— 用户被卡在「既接管不了、也起不来」。
///
/// 接管与「本次启动」的区别只在**进程归属**：参数、上下文、旁车配置一律沿用
/// 进程里已有的（接管不重算，见 `respawn` 之外的注释）；因此这里**不推导附加参数**、
/// 不隔离 DLL、不动显存 —— 那些动作的前提都是「要新起一个进程」。
fn 接管遗留引擎(
    app: &AppHandle,
    state: &AppState,
    model_id: &str,
    host: &str,
    port: u16,
    引擎自报: &str,
) -> ServiceInfo {
    // pid 只是为了让「停止」和持久化记录有东西可写。查不到就写 0，
    // **不阻断接管**：进程确实在服务，拿不到 pid 只是以后不能按 pid 杀它，
    // 按端口总能再找回来。`kill_pid` 那边要因此跳过 0。
    let pid = 端口上正在监听的进程(port).unwrap_or(0);
    let info = {
        let mut inner = state.inner.lock().expect("state lock poisoned");
        let 安全模式 = inner.service.safe_mode;
        inner.service.model_id = Some(model_id.to_string());
        inner.service.port = Some(port);
        inner.service.host = Some(host.to_string());
        inner.service.pid = if pid == 0 { None } else { Some(pid) };
        inner.service.status = ServiceStatus::Running;
        // 关键：`process = None` ⇒ 这是**接管态**。看门狗据此跳过 `try_wait`，
        // 「停止」据此走 `kill_pid` 而不是 `child.kill()`。
        inner.service.process = None;
        // 接管的是**别人起的**进程，计划不可知 ⇒ 提示留空（未知，不是正常）。
        inner.service.warning = None;
        let persisted = PersistedServiceState {
            model_id: model_id.to_string(),
            port,
            pid,
            safe_mode: 安全模式,
            host: host.to_string(),
        };
        // 补写记录：下次重开就能走正常的 `recover_service`，不必再靠这条兜底。
        let _ = save_service_state(&inner.db_path, &persisted);
        ServiceInfo {
            model_id: Some(model_id.to_string()),
            port: Some(port),
            host: Some(host.to_string()),
            status: ServiceStatus::Running,
            safe_mode: 安全模式,
            message: format!(
                "接口端口 {port} 上已有上次运行的本地AI服务（{引擎自报}），已直接接管，没有重复启动。"
            ),
            warning: None,
        }
    };
    tracing::info!(
        port,
        pid,
        model_id,
        引擎自报 = %引擎自报,
        "端口上已有本机引擎在服务 ⇒ 接管它（没有重复启动，也没有让用户换端口）"
    );
    let _ = app.emit("service_status_changed", &info);
    spawn_service_watchdog(app.clone(), state.clone());
    info
}

/// 按端口反查正在 LISTENING 的进程 pid（Windows `netstat -ano`）。
///
/// 为什么不用名字找：APP 起的是 `llama-server-cuda.exe` / `llama-server-vulkan.exe`
/// （瘦启动器，后端起什么由同目录 DLL 决定），按 `llama-server.exe` 精确匹配会
/// **一个都匹配不上**——2026-10-19 就因此漏杀，服务仍占着显存，OOM 被误读成
/// 「模型不支持部分卸载」。端口是唯一稳的锚点。
#[cfg(windows)]
fn 端口上正在监听的进程(port: u16) -> Option<u32> {
    let mut command = Command::new("netstat");
    command.args(["-ano", "-p", "tcp"]);
    crate::proc::hide_console(&mut command);
    let 输出 = command.output().ok()?;
    let 文本 = String::from_utf8_lossy(&输出.stdout);
    从netstat里取pid(&文本, port)
}

#[cfg(not(windows))]
fn 端口上正在监听的进程(_port: u16) -> Option<u32> {
    // 本项目只发 Windows 包；别的平台按「查不到」处理，接管仍能成立（pid 写 0）。
    None
}

/// 从 `netstat -ano` 的文本里取出监听某个端口的 pid。
///
/// 抽成纯函数是为了能测：这段代码的核心是一个**后缀匹配**，而「后缀匹配」
/// 正是本项目栽过的那一类（键名用后缀匹配把 `x.y` 认成 `y`）。必须钉住
/// 「`:8000` 不命中 `:18000`」「ESTABLISHED 不算监听」两条。
fn 从netstat里取pid(文本: &str, port: u16) -> Option<u32> {
    let 后缀 = format!(":{port}");
    for 行 in 文本.lines() {
        let 列: Vec<&str> = 行.split_whitespace().collect();
        // 典型行：`  TCP    127.0.0.1:8000    0.0.0.0:0    LISTENING    12345`
        if 列.len() < 5 || !列[0].eq_ignore_ascii_case("TCP") {
            continue;
        }
        if !列[3].eq_ignore_ascii_case("LISTENING") {
            continue;
        }
        // `ends_with(":8000")` 不会把 `:18000` 认成同一个（末尾五个字符对不上）。
        if 列[1].ends_with(&后缀) {
            if let Ok(pid) = 列[4].parse::<u32>() {
                return Some(pid);
            }
        }
    }
    None
}

/// 结束当前服务进程 —— 并且**失败要留日志**。
///
/// 🔴 2026-10-19 P100 实测：`taskkill` 回 **`拒绝访问`**（`pid=8868` 的子进程
/// `11748` 杀不掉），而这条路径当时写的是 `let _ = child.kill()` —— 静默。
/// 后果是状态里写着「已停止」、系统里那个进程仍占着 8000 与 6.4 GiB 显存，
/// 下一次启动就表现为「端口被占用」或「正在运行另一个模型」，
/// 而排查方向会被引到端口配置上。
///
/// 另外**必须补一次 `kill_pid`**：`Child::kill` 只结束直接子进程。实测那次
/// 残留的 `11748` 是子进程，父进程没了它照样活着占端口。
///
/// 🔴 **接管态不许按「记录里的 pid」杀**。那条 pid 属于**上一个会话**，
/// 中间隔着重开 APP 的时间窗，完全可能已被系统复用 —— 而 `taskkill /T /F`
/// 会连带杀掉它的整棵子进程树。2026-10-19 那次 `拒绝访问` 反而帮了忙：
/// 若 `8868` 当时被复用成 `services.exe` 之类，后果不是「服务没起来」而是系统出事。
/// 唯一可信的实体是**现在占着那个端口的进程**，所以接管态一律按端口反查，
/// 端口上没人听就**不动手**（进程已经退出，记录只是过期）。
fn 结束服务进程(process: Option<Child>, pid: Option<u32>, port: Option<u16>) -> 终止结果 {
    match process {
        Some(mut child) => {
            let 子pid = child.id();
            // 本次会话亲手 spawn 的：`Child` 句柄还在，pid 在这个窗口内不会被复用。
            let 句柄成功 = match child.kill() {
                Ok(()) => {
                    tracing::info!(pid = 子pid, "已请求结束服务子进程");
                    true
                }
                Err(错误) => {
                    tracing::warn!(
                        pid = 子pid,
                        错误 = %错误,
                        "结束服务子进程失败 —— 它可能仍然占着端口与显存（下一步会按 pid 再试一次）"
                    );
                    false
                }
            };
            // 🔴 两条终止路径的**权限路径不同**，必须都试：
            // `Child::kill` 走 `TerminateProcess(建进程时的句柄)`，不受完整性级别限制；
            // `taskkill` 要**新开**一个进程句柄，目标级别比我们高就 `拒绝访问`。
            // 本次会话亲手起的进程两条都能成；接管来的（没有 `Child`）只有后者。
            let 结果 = kill_pid(子pid);
            // 🔴 句柄已经把它终止了，随后 `taskkill` 报的「没有此实例」是**必然**的
            // （进程刚被我们杀掉）。这不叫「本来就没有」—— 那是「本来就没在跑」，
            // 会让排查者以为这个服务从没起来过。2026-10-19 P100（`07:39:32`）
            // 日志里就写着 `终止结果=本来就没有`，而两秒前它还 `llama-server running`。
            if 句柄成功 && 结果 == 终止结果::本来就没有 {
                终止结果::已结束
            } else {
                结果
            }
        }
        None => match (port.and_then(端口上正在监听的进程), pid) {
            (Some(端口pid), _) => {
                tracing::info!(端口pid, 记录pid = ?pid, "接管态：按端口找到真正的监听进程并终止");
                kill_pid(端口pid)
            }
            (None, Some(pid)) => {
                tracing::info!(
                    pid,
                    "端口上已无监听 ⇒ 进程已退出；**不**按记录里的 pid 终止（可能已被系统复用）"
                );
                终止结果::本来就没有
            }
            (None, None) => 终止结果::未尝试,
        },
    }
}

/// 当前进程**实际**有没有管理员权限。
///
/// 不看「人在不在管理员组」：UAC 下非提权进程虽然在管理员组里，令牌完整性级别
/// 仍只有「中」（`S-1-16-8192`），照样杀不掉高完整性进程。
/// `whoami /groups` 里 `Mandatory Label` 那一行的 SID 是**语言无关**的，
/// 所以按 SID 判、不按「管理员」这种本地化文本判（本机是中文 Windows，
/// 解析中文标签在别的语言包上必崩）。
#[cfg(windows)]
fn 当前是否提权() -> Option<bool> {
    let mut command = Command::new("whoami");
    command.args(["/groups"]);
    crate::proc::hide_console(&mut command);
    let 输出 = command.output().ok()?;
    let 文本 = String::from_utf8_lossy(&输出.stdout);
    // 12288 = 高完整性（提权）；16384 = 系统完整性。中完整性是 8192，不在此列。
    Some(文本.contains("S-1-16-12288") || 文本.contains("S-1-16-16384"))
}

#[cfg(not(windows))]
fn 当前是否提权() -> Option<bool> {
    None
}

/// 把「本程序以什么权限在跑」写进日志 —— 这是「引擎杀不掉」那条的**唯一前置判据**。
///
/// 🔴 2026-10-19 P100 的整轮排查卡在这里：日志里只有一句 `拒绝访问`，
/// 谁也不知道本程序是不是提权跑的、那个引擎又是谁起的。缺了这条，
/// 「权限不足」与「进程被别的程序保护」在日志里**同形**。
///
/// 只打一次（`Once`）：每次启动都打会变成噪声。
fn 提示提权状态() {
    static 已提示: std::sync::Once = std::sync::Once::new();
    已提示.call_once(|| match 当前是否提权() {
        Some(true) => tracing::warn!(
            "本程序以管理员身份运行：它起的引擎继承高完整性，退出 APP 后仍会继续跑，\
             但下次用**普通权限**打开本程序将杀不掉它（taskkill 拒绝访问）。\
             若打算关掉 APP 让服务继续跑，请固定用同一种权限启动"
        ),
        Some(false) => tracing::info!("本程序以普通权限运行（UAC 未提权）"),
        None => tracing::debug!("查不到当前权限级别（whoami 不可用）"),
    });
}

/// 一次终止尝试的**结果**。
///
/// 🔴 它只回答「为什么没成」，**不**回答「停没停下」。判「停没停下」只看现场
/// （端口是否还归那个 pid）—— 沿用本项目那条：日志只记发生过的事，
/// 状态要看现场。`taskkill` 报成功也只说明命令没报错。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum 终止结果 {
    /// 没发终止：手上没有进程，或 pid 是 0 的占位。
    未尝试,
    /// `taskkill` 报成功。
    已结束,
    /// `taskkill` 说「没有此进程的实例」—— 进程本来就已经退出，属正常收尾。
    本来就没有,
    /// `taskkill` 说「拒绝访问」：目标的完整性级别高于本进程（典型：上一次以
    /// **管理员身份**运行 APP，它起的引擎继承高完整性；这次普通权限启动就杀不掉）。
    拒绝访问,
    /// 其它失败。
    其他,
}

/// `taskkill` 退出码 → 结果。
///
/// 抽成纯函数是为了能钉住映射：这两个码（128 / 255）是 2026-10-19 P100 实测
/// 出来的，**不是** MSDN 上的文档值 —— 一旦上游行为变了，要能靠单测发现。
fn 由退出码判终止结果(码: Option<i32>) -> 终止结果 {
    match 码 {
        Some(0) => 终止结果::已结束,
        // 255 = 「没有此进程的实例」：进程在 taskkill 打开它之前就已经退出。
        // 这是**正常收尾**（本次会话刚 `child.kill()` 过的就是这种），不是故障。
        Some(255) => 终止结果::本来就没有,
        // 128 = 「拒绝访问」。❗这条必须与「没有此实例」分开：2026-10-19 P100 的
        // 日志里 9 条 WARN 有 8 条是「没有此实例」，唯一那条「拒绝访问」被埋住了，
        // 结果把「杀不掉」误判成「端口被占」，排查方向整整偏了一轮。
        Some(128) => 终止结果::拒绝访问,
        _ => 终止结果::其他,
    }
}

/// Best-effort termination of a process (and its children) by pid. Used to stop
/// a reattached server or clean up a stale/zombie one on Windows.
fn kill_pid(pid: u32) -> 终止结果 {
    // pid 0 = 「接管时没查到 pid」的占位（见 `接管遗留引擎`）。`taskkill /PID 0`
    // 会去杀 Idle 进程并报错，语义上没有任何意义 —— 直接跳过，别留一条注定
    // 失败的调用去污染日志。
    if pid == 0 {
        tracing::debug!("kill_pid：pid 为 0（接管时没查到），跳过");
        return 终止结果::未尝试;
    }
    let mut command = Command::new("taskkill");
    command.args(["/PID", &pid.to_string(), "/T", "/F"]);
    crate::proc::hide_console(&mut command);
    // 结果**必须**记下来：此前是 `let _ = command.output();`，杀失败了也一声不响，
    // 于是「服务没停掉、端口还占着」在下一次启动时表现为「端口被占用」，
    // 排查方向直接被带偏（2026-10-19 P100）。
    match command.output() {
        Ok(输出) => {
            let 码 = 输出.status.code();
            let 结果 = 由退出码判终止结果(if 输出.status.success() { Some(0) } else { 码 });
            match 结果 {
                终止结果::已结束 => tracing::info!(pid, "kill_pid：已结束该进程（含子进程）"),
                // 「进程已不在」不是故障，降到 DEBUG：否则日志里一片 WARN，
                // 真正那条「拒绝访问」会被淹掉。
                终止结果::本来就没有 => tracing::debug!(pid, "kill_pid：进程已不在，无需终止"),
                终止结果::拒绝访问 => tracing::warn!(
                    pid,
                    状态 = ?码,
                    输出 = %String::from_utf8_lossy(&输出.stderr).trim(),
                    "kill_pid：拒绝访问 —— 目标进程的权限级别高于本程序，杀不掉\
                     （多半是上一次以管理员身份运行 APP 时起的引擎）。\
                     需要以管理员身份重新运行本程序，或到任务管理器结束该进程"
                ),
                _ => tracing::warn!(
                    pid,
                    状态 = ?码,
                    输出 = %String::from_utf8_lossy(&输出.stderr).trim(),
                    "kill_pid：taskkill 未能结束该进程"
                ),
            }
            结果
        }
        Err(错误) => {
            tracing::warn!(pid, 错误 = %错误, "kill_pid：taskkill 没能执行");
            终止结果::其他
        }
    }
}

fn set_failed(app: &AppHandle, state: &AppState, message: String) -> ServiceInfo {
    tracing::error!(reason = %message, "llama-server failed to start");
    let info = {
        let mut inner = state.inner.lock().expect("state lock poisoned");
        let 失败端口 = inner.service.port;
        结束服务进程(inner.service.process.take(), inner.service.pid, 失败端口);
        inner.service.pid = None;
        let _ = clear_service_state(&inner.db_path);
        inner.service.status = ServiceStatus::Failed;
        inner.service.port = None;
        // 没起来就没有「降档」可言；留着旧提示会让人以为服务在跑。
        inner.service.warning = None;
        ServiceInfo {
            model_id: inner.service.model_id.clone(),
            port: inner.service.port,
            host: inner.service.host.clone(),
            status: inner.service.status.clone(),
            safe_mode: inner.service.safe_mode,
            message,
            warning: None,
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

/// 等端口**真正**释放，最多 `上限`（每 500 ms 探一次）。
///
/// 与 `wait_for_port_available` 的区别只在用途：那个是「启动前预检」的短窗口
/// （3 秒，吸收正常退出的延迟）；这个是**已经确认占用者是我方引擎、且刚发过
/// 终止**之后的长窗口 —— 2026-10-19 P100 实测那次残留了 7 秒才走干净。
///
/// 不合并成一个函数：两处的「等多久」来自完全不同的判据，合并就等于让预检
/// 也为最坏情况买单（每次端口冲突都白等 10 秒）。
async fn 等端口释放(host: &str, port: u16, 上限: Duration) -> bool {
    let 截止 = Instant::now() + 上限;
    loop {
        if is_port_available(host, port) {
            return true;
        }
        if Instant::now() >= 截止 {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
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

    /// 退出码必须翻译成人能行动的中文，而不是只甩一个负数。
    ///
    /// **这条护栏是 2026-09-29 虚拟机实测换来的**：当时 llama-server 以
    /// 0xC0000409 退出，而错误信息只有「可能是运行引擎与硬件不兼容或模型
    /// 文件损坏」—— 用户完全无从下手。
    #[test]
    fn exit_code_is_translated_into_actionable_chinese() {
        // 0xC0000409 -> -1073740791，本次虚拟机遇到的真实退出码。
        // 没有引擎输出时，退回按退出码给通用指引。
        let 文案 = 格式化启动失败(-1073740791, "");
        assert!(文案.contains("-1073740791"), "应保留原始退出码：{文案}");
        assert!(
            !文案.contains("可能是运行引擎与硬件不兼容"),
            "已知退出码不应退化成含糊文案：{文案}"
        );
        assert!(
            文案.contains("保护性中止"),
            "应说明 0xC0000409 的含义：{文案}"
        );

        // 0xC000001D -> -1073741795，非法指令集
        let 文案 = 格式化启动失败(-1073741795, "");
        assert!(文案.contains("指令集"), "应指出指令集问题：{文案}");

        // 未知退出码走通用文案，但也要带上码
        let 文案 = 格式化启动失败(-1, "");
        assert!(文案.contains("-1"), "未知码也要保留：{文案}");
    }

    /// **核心归因逻辑**：引擎输出里明摆着是 CUDA 架构不匹配时，
    /// 就不能再按退出码去怪 CPU —— 这是虚拟机实测踩过的坑。
    ///
    /// 当时 0xC0000409 的通用文案指向「CPU 型号过精简（qemu64/kvm64）」，
    /// 但日志里明明写着 `CUDA error: invalid device function`，
    /// CPU 侧早已正常初始化到 `n_threads = 8`。**按码归因会把方向带偏。**
    #[test]
    fn engine_output_beats_exit_code_when_attributing_cause() {
        let 诊断 = "ggml-cuda.cu:108: CUDA error\n\
                    E CUDA error: invalid device function\n\
                    E cudaFuncGetAttributes(&attr, kernel)";
        let 文案 = 格式化启动失败(-1073740791, 诊断);

        assert!(
            文案.contains("CUDA 内核不是为当前显卡编译"),
            "应认出 CUDA 架构不匹配：{文案}"
        );
        // 必须给出可行动的处置，但**不能**再指定「改用 Vulkan」。
        // 2026-09-30 纠错：P100 那台机处于 TCC 模式，Vulkan 枚举不到设备，
        // 那条建议等于把人往坑里带。现在按「换另一种加速方式 -> CPU ->
        // 升级驱动 / 换引擎包」给，不点名具体后端。
        assert!(
            文案.contains("另一种显卡加速方式") && 文案.contains("CPU 安全模式"),
            "应给出可行动的处置建议：{文案}"
        );
        assert!(
            !文案.contains("改用 Vulkan"),
            "TCC 模式下 Vulkan 不可用，不能给「改用 Vulkan」这条错误建议：{文案}"
        );
        assert!(
            !文案.contains("qemu64"),
            "已经定位到 CUDA 架构问题时，不应再去误导用户改 CPU 类型：{文案}"
        );
    }

    /// 引擎输出识别不出来时，必须退回按退出码给指引 —— 不能什么都不给。
    #[test]
    fn unrecognised_output_falls_back_to_exit_code_guidance() {
        let 诊断 = "some totally unknown failure line";
        let 文案 = 格式化启动失败(-1073740791, 诊断);
        assert!(
            文案.contains("保护性中止"),
            "识别不出特征时应退回按退出码归因：{文案}"
        );
        assert!(文案.contains("qemu64") || 文案.contains("CPU 指令集"), "通用指引应保留：{文案}");
    }

    /// llama.cpp 的原始输出必须被附在错误里 —— 这是排障的唯一线索。
    #[test]
    fn engine_output_is_appended_to_the_error() {
        let 诊断 = "ggml_cuda_init: failed to initialize CUDA\n\
                    error: device 0 not found";
        let 文案 = 格式化启动失败(-1073740791, 诊断);
        assert!(文案.contains("ggml_cuda_init"), "应附上引擎原始输出：{文案}");
        assert!(文案.contains("device 0 not found"), "应附上引擎原始输出：{文案}");
    }

    /// 没有引擎输出时不应留下空标题，否则界面会出现「运行引擎的输出：」后面没内容。
    #[test]
    fn empty_diagnostics_do_not_leave_a_dangling_header() {
        let 文案 = 格式化启动失败(-1, "");
        assert!(!文案.contains("运行引擎的输出"), "空诊断不应有标题：{文案}");
    }

    /// 显存不足与运行时缺失也要能认出来，给对应的建议。
    #[test]
    fn other_cuda_failures_are_also_recognised() {
        let 文案 = 格式化启动失败(-1073740791, "CUDA error: out of memory");
        assert!(文案.contains("显存不足"), "应认出显存不足：{文案}");

        let 文案 = 格式化启动失败(-1073740791, "error while loading cudart64_12.dll");
        assert!(文案.contains("CUDA 运行时库"), "应认出运行时缺失：{文案}");
    }

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
        // 8G 卡跑 Qwen3-1.7B：权重 1.71 + 保留 1.84 = 3.55，余 4.45 GiB。
        // fp16 KV（112 KB/token）够约 41.6K，16K 目标档轻松放下 → 走 fp16 最快。
        let p = cuda_launch_params(8.0, W1_7B, KV1_7B, NO_KV_FIXED, 16384);
        assert_eq!(p.n_gpu_layers, Some(999));
        assert_eq!(p.ctx_size, 16384);
        assert_eq!(p.parallel, 1);
        assert!(!p.kv_quant, "fp16 should hold the default context easily");
    }

    #[test]
    fn qwen3_8b_needs_q8_to_reach_default_context() {
        // 8G 卡跑 Qwen3-8B Q4_K_M 的真实情形：权重 4.68 + 保留 1.84 = 6.52，
        // 只剩约 1.48 GiB。fp16 KV（144 KB/token）只够约 10.8K，差一点点不到
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
        // Qwen3-8B 权重 4.68 + 保留 1.84 = 6.52：可用显存低于 6.52 的卡全量装不进，
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
        //   budget = 8.0 − 1.71 − 1.84 = 4.45 GiB，fp16 每 token 112 KB
        //   → 够约 41.6K，足以放满原生上下文。
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
        // 容忍带机制本身：fp16 的物理上限比目标只差一点点时，不为那点长度
        // 把整段 KV 降成 q8（q8 每 token 多一次反量化，短对话反而更慢）。
        //
        // 🔴 目标上下文 **26,624 是构造值**，不是哪个模型的原生长度 —— 容忍带
        // 只有 5% 宽，真实组合很难正好落进去：8G 卡 usable 实测 6.33 GiB，
        // 1.7B 在其上只够 fp16 26.0K，离原生 40,960 差 36%，根本不在带内。
        // 用构造的 target 只为把这个分支覆盖到；真机口径见下一条
        // `overhead_matches_measured_footprint`。
        //   可用 6.33 − 1.71 − 1.84 = 2.78 GiB → fp16 约 26.0K
        //   26.0K × 1.05 = 27.3K ≥ 26,624 ⇒ 落在容忍带内
        let p = cuda_launch_params(6.33, W1_7B, KV1_7B, NO_KV_FIXED, 26_624);
        assert!(!p.kv_quant, "差距在容忍带内就该保留 fp16");
        assert!(
            p.ctx_size < 26_624 && p.ctx_size >= 24_576,
            "应开 fp16 的物理上限（而非降到 2048 档），实际 {}",
            p.ctx_size
        );
    }

    #[test]
    fn overhead_matches_measured_footprint() {
        // 把预留量钉在实测上，而不是钉在「上一版写了多少」。
        //
        // Qwen3-1.7B Q8_0、ctx=40960、KV q8 —— 2026-10-18 本机三次
        // `nvidia-smi` 读数（GiB）。规划口径必须能反推出这三次，否则就是又
        // 滑回「算得比现实乐观」：旧值 1.3 时这里差 −0.54，正是「计划放得下、
        // 实际 KV 溢出到共享内存」的成因，而 llama.cpp 全程不报错。
        const 实测_GIB: [f64; 3] = [5.7354, 5.7344, 5.7197];
        let kv_q8 = KV1_7B * 40_960.0 * 0.5;
        let 预估 = W1_7B + kv_q8 + GPU_OVERHEAD_GB;
        for 实测 in 实测_GIB {
            let 差 = 预估 - 实测;
            assert!(
                差.abs() <= 0.05,
                "规划口径反推实测失败：预估 {预估:.4} vs 实测 {实测:.4}（差 {差:+.4}）"
            );
        }
    }

    #[test]
    fn spark_reaches_far_past_16k_on_an_8g_card() {
        // Spark-X2.5-4B Q6_K 在 8G 卡上的真实情形 —— 放开上限的主要受益者。
        //   budget = 7.2（假设空闲；本机实测空载 6.33，见下）− 2.93（权重）
        //          − 1.84（预留）− 0.05（滑窗固定） = 2.38 GiB
        //   fp16 每 token 只有 36 KB（9 层全注意力），够约 69K，仍远不到原生 1M，
        //   于是走 q8 取物理最大。关键是这个结果远大于过去硬编码的 16384。
        //   ⚠️ 输入 7.2 是本条与
        //   `sliding_window_fixed_kv_is_deducted_from_the_budget` 共用的**基准值**，
        //   不是本机实测（本机空载 usable 为 6.33）。若换成 6.33，q8 只到约 85K，
        //   过不了这里 131,072 的下限 —— 但那不影响本条要证明的事：**4B 模型在
        //   8G 卡上能开到的上下文远大于旧的硬编码 16384**。
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
        // 54 MiB 只占预算的 2.2%，所以差距应该是几千 token 量级，而不是翻倍。
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

    /// **后端隔离的核心契约**：选定某个后端后，`backends/` 根目录下
    /// 必须**没有**另一个 GPU 后端的 DLL。
    ///
    /// 这条测试源自 2026-09-29 虚拟机实测：选了 Vulkan，但 `backends/`
    /// 里同时躺着 `ggml-cuda.dll`，而 llama.cpp 的 `ggml_backend_load_all()`
    /// **不看可执行文件名**，照样加载并初始化 CUDA → 在 CUDA 不适配的卡上
    /// 直接崩，报 `CUDA error: invalid device function`。
    #[test]
    fn isolating_vulkan_evicts_cuda_dlls() {
        let 沙盘 = 建临时引擎目录();
        let backends = 沙盘.join("engine").join("backends");

        let 移动数 = isolate_backend_dlls(&沙盘, EngineBackend::Vulkan);

        assert!(移动数 > 0, "应当有 CUDA 的 DLL 被移走");
        assert!(
            backends.join("ggml-vulkan.dll").exists(),
            "Vulkan 自己的 DLL 必须留在根目录"
        );
        assert!(
            !backends.join("ggml-cuda.dll").exists(),
            "CUDA 的 DLL 必须被移出根目录，否则仍会被加载"
        );
        assert!(
            backends.join("ggml.dll").exists() && backends.join("llama.dll").exists(),
            "共用底座 DLL 不能被误移"
        );
        assert!(
            backends.join("ggml-cpu-x64.dll").exists(),
            "CPU 变体必须常驻（它是最终兜底）"
        );

        清理临时目录(&沙盘);
    }

    /// 反向：选定 CUDA 时，Vulkan 的 DLL 要被移走，CUDA 的留在根下。
    #[test]
    fn isolating_cuda_evicts_vulkan_dlls() {
        let 沙盘 = 建临时引擎目录();
        let backends = 沙盘.join("engine").join("backends");

        isolate_backend_dlls(&沙盘, EngineBackend::Cuda);

        assert!(backends.join("ggml-cuda.dll").exists(), "CUDA 的 DLL 应留在根目录");
        assert!(
            !backends.join("ggml-vulkan.dll").exists(),
            "Vulkan 的 DLL 应被移出根目录"
        );

        清理临时目录(&沙盘);
    }

    /// **CPU 安全模式必须清光两个 GPU 后端** —— 这是最容易被忽略的一档。
    ///
    /// 理由：CUDA/Vulkan 的 `ggml_backend_init` 在不适配的机器上**本身就会崩**。
    /// 若 CPU 兜底时还留着它们，等于兜底路径也走不通（虚拟机实测正是如此）。
    #[test]
    fn isolating_cpu_evicts_both_gpu_backends() {
        let 沙盘 = 建临时引擎目录();
        let backends = 沙盘.join("engine").join("backends");

        isolate_backend_dlls(&沙盘, EngineBackend::Cpu);

        assert!(
            !backends.join("ggml-cuda.dll").exists(),
            "CPU 模式不能留下 CUDA 的 DLL"
        );
        assert!(
            !backends.join("ggml-vulkan.dll").exists(),
            "CPU 模式不能留下 Vulkan 的 DLL"
        );
        assert!(
            backends.join("ggml-cpu-x64.dll").exists(),
            "CPU 自己的 DLL 当然要保留"
        );

        清理临时目录(&沙盘);
    }

    /// **来回切换后端必须能切回来** —— 池子不能只进不出。
    ///
    /// 先用 Vulkan（CUDA 进池子），再切 CUDA：若第二步「从池子里搬回」
    /// 缺失，CUDA 的 DLL 会永远出不来，切后端等于单向操作。
    #[test]
    fn switching_backends_round_trips() {
        let 沙盘 = 建临时引擎目录();
        let backends = 沙盘.join("engine").join("backends");

        isolate_backend_dlls(&沙盘, EngineBackend::Vulkan);
        assert!(!backends.join("ggml-cuda.dll").exists());

        isolate_backend_dlls(&沙盘, EngineBackend::Cuda);
        assert!(
            backends.join("ggml-cuda.dll").exists(),
            "切回 CUDA 后，CUDA 的 DLL 必须从池子里被搬回来"
        );
        assert!(
            !backends.join("ggml-vulkan.dll").exists(),
            "切回 CUDA 后，Vulkan 的 DLL 应进池子"
        );

        清理临时目录(&沙盘);
    }

    /// 幂等：连续两次隔离同一后端，结果一致且不报错。
    #[test]
    fn isolation_is_idempotent() {
        let 沙盘 = 建临时引擎目录();
        let backends = 沙盘.join("engine").join("backends");

        isolate_backend_dlls(&沙盘, EngineBackend::Vulkan);
        let 第二次 = isolate_backend_dlls(&沙盘, EngineBackend::Vulkan);

        assert_eq!(第二次, 0, "第二次调用应当无事可做");
        assert!(backends.join("ggml-vulkan.dll").exists());
        assert!(!backends.join("ggml-cuda.dll").exists());

        清理临时目录(&沙盘);
    }

    /// `restore_backend_dlls` 要把所有被隔离的 DLL 还原，供卸载/校验使用。
    #[test]
    fn restore_puts_everything_back() {
        let 沙盘 = 建临时引擎目录();
        let backends = 沙盘.join("engine").join("backends");

        isolate_backend_dlls(&沙盘, EngineBackend::Vulkan);
        let 还原数 = restore_backend_dlls(&backends);

        assert!(还原数 > 0, "应当有文件被还原");
        assert!(backends.join("ggml-cuda.dll").exists(), "CUDA 的 DLL 应被还原");
        assert!(backends.join("ggml-vulkan.dll").exists());
        assert!(
            !backends.join(后端池目录名).exists(),
            "池子目录应当被清理干净"
        );

        清理临时目录(&沙盘);
    }

    /// 文件名归属判定：前缀匹配要能覆盖多变体，且不能误判共用底座。
    #[test]
    fn backend_file_kind_matches_by_prefix() {
        assert_eq!(
            文件名属于哪个后端("ggml-cuda.dll"),
            Some(BackendFileKind::Cuda)
        );
        assert_eq!(
            文件名属于哪个后端("ggml-vulkan.dll"),
            Some(BackendFileKind::Vulkan)
        );
        // CPU 有一整套微架构变体，都必须归到 CPU
        for 名 in [
            "ggml-cpu-alderlake.dll",
            "ggml-cpu-haswell.dll",
            "ggml-cpu-x64.dll",
            "ggml-cpu-sse42.dll",
        ] {
            assert_eq!(文件名属于哪个后端(名), Some(BackendFileKind::Cpu), "{名}");
        }
        // 共用底座不属于任何后端，绝不能被隔离走
        for 名 in [
            "ggml.dll",
            "ggml-base.dll",
            "llama.dll",
            "llama-server-impl.dll",
            "libomp.dll",
            "cudart64_12.dll",
        ] {
            assert_eq!(文件名属于哪个后端(名), None, "{名} 不该被归到某个后端");
        }
    }

    /// 搭建一个假的引擎目录，含齐三个后端的代表性 DLL 与共用底座。
    fn 建临时引擎目录() -> PathBuf {
        let 根 = std::env::temp_dir().join(format!(
            "localai-隔离测试-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let backends = 根.join("engine").join("backends");
        std::fs::create_dir_all(&backends).unwrap();
        for 名 in [
            "ggml-cuda.dll",
            "ggml-vulkan.dll",
            "ggml-cpu-x64.dll",
            "ggml-cpu-haswell.dll",
            "ggml.dll",
            "ggml-base.dll",
            "llama.dll",
            "llama-server-impl.dll",
            "llama-server-cuda.exe",
            "llama-server-vulkan.exe",
        ] {
            std::fs::write(backends.join(名), b"stub").unwrap();
        }
        根
    }

    fn 清理临时目录(根: &Path) {
        let _ = std::fs::remove_dir_all(根);
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

    /// **P1-8 护栏**：`effective_gpu_backend` 必须按**目录里实际存在的 DLL**
    /// 判后端，而不是按任何标签 —— 它是「两轮是否真的切过后端」的唯一实测依据。
    #[test]
    fn effective_gpu_backend_reads_the_actual_directory() {
        let 根 = 建临时引擎目录();
        let backends = 根.join("engine").join("backends");

        // 初始：cuda 与 vulkan 的 DLL 都在 → 按引擎加载序取 CUDA（cuda 先注册）。
        assert_eq!(effective_gpu_backend(&根), EngineBackend::Cuda);

        // 模拟「切到 Vulkan 真的生效了」：cuda DLL 被隔离走 → Vulkan。
        std::fs::remove_file(backends.join("ggml-cuda.dll")).unwrap();
        assert_eq!(effective_gpu_backend(&根), EngineBackend::Vulkan);

        // 两个 GPU 后端都没了（只剩共用底座）→ CPU。
        std::fs::remove_file(backends.join("ggml-vulkan.dll")).unwrap();
        assert_eq!(effective_gpu_backend(&根), EngineBackend::Cpu);

        清理临时目录(&根);
    }

    /// **P1-1b 护栏**：TCC 卡上请求 Vulkan → 实际隔离到 CUDA（四项分支全覆盖）。
    ///
    /// 为什么必须靠单测：这条分支的进入条件是
    /// `backend == Vulkan && 计算专用模式`，而 `backend` 由
    /// `ensure_engine_ready` 按 `hardware.backend` 覆写、`select_backend`
    /// 只看驱动版本（≥ 551.78 即判 CUDA，不看 TCC）。升过驱动的 P100 上
    /// `backend` 恒为 `Cuda` → 保底分支**真机不可达**，2026-10-08 在虚机上
    /// 按 0.1.4 清单复现时正是卡在这里。真机测不到的分支，只能由这里兜底。
    ///
    /// ⚠️ 反向断言同样重要：**非 TCC 时绝不能把 Vulkan 改成 CUDA** ——
    /// WDDM 卡上 Vulkan 是正常路径，误改会让「切后端」静默失效。
    #[test]
    fn tcc_turns_vulkan_isolation_into_cuda() {
        // ① TCC + Vulkan → 保底成 CUDA。
        assert_eq!(
            选择实际隔离后端(EngineBackend::Vulkan, true),
            EngineBackend::Cuda,
            "TCC 卡上 Vulkan 枚举不到设备，必须改用 CUDA 隔离"
        );
        // ② 非 TCC + Vulkan → 原样，不许改写。
        assert_eq!(
            选择实际隔离后端(EngineBackend::Vulkan, false),
            EngineBackend::Vulkan,
            "WDDM 卡上 Vulkan 是正常路径，不得改成 CUDA"
        );
        // ③ TCC + CUDA → 原样（本来就是对的）。
        assert_eq!(
            选择实际隔离后端(EngineBackend::Cuda, true),
            EngineBackend::Cuda
        );
        // ④ TCC + CPU → 原样：CPU 安全模式本来就要清掉所有 GPU 后端，
        //    不能被保底逻辑改成「留 CUDA」。
        assert_eq!(
            选择实际隔离后端(EngineBackend::Cpu, true),
            EngineBackend::Cpu,
            "CPU 安全模式必须绕开 GPU 路径，保底逻辑不能干预"
        );
    }

    /// `选择实际隔离后端` 只在「请求 Vulkan 且确知 TCC」时改变结果 ——
    /// 这条不变式保证 `start_service_core` 里「`实际隔离后端 != backend`
    /// 就打 WARN」等价于「只有保底分支打 WARN」。
    #[test]
    fn isolation_backend_changes_only_on_tcc_vulkan() {
        for 请求 in [EngineBackend::Cpu, EngineBackend::Cuda, EngineBackend::Vulkan] {
            for tcc in [false, true] {
                let 实际 = 选择实际隔离后端(请求, tcc);
                let 应改变 = 请求 == EngineBackend::Vulkan && tcc;
                assert_eq!(
                    实际 != 请求,
                    应改变,
                    "请求 {请求:?} / TCC={tcc}：是否改写与预期不符（实际 {实际:?}）"
                );
            }
        }
    }

    /// **P1-7 护栏**：内置模型（`local_path` 恒为 `None`）也必须能定位到真实
    /// GGUF —— 候选路径要按 `install_dir/models/<id>/<清单文件名>` 补齐。
    ///
    /// 原先只读 `local_path`，导致**所有内置模型**都走回落经验公式
    /// （实测 Qwen3-1.7B 的 KV 低估 7.8×），8G 卡会按错误预算开出过大上下文 → OOM。
    #[test]
    fn builtin_model_resolves_kv_from_install_location() {
        let 模型 = crate::models::builtin_models()
            .into_iter()
            .find(|m| m.id == "qwen3_1_7b_light")
            .expect("内置清单里有 Qwen3-1.7B");
        assert!(
            模型.local_path.is_none(),
            "内置模型的 local_path 本就为空 —— 这正是 P1-7 的前提"
        );

        let 安装目录 = PathBuf::from("/tmp/localai-install");
        let 候选 = kv_profile_candidate_paths(&模型, &安装目录);
        let 期望 = crate::models::model_install_path(&安装目录, &模型);
        assert_eq!(
            候选.first(),
            Some(&期望),
            "local_path 为空时，第一个候选必须是内置模型的标准落点"
        );
        assert!(
            期望.ends_with("qwen3_1_7b_light/Qwen3-1.7B-Q8_0.gguf"),
            "落点不对：{}",
            期望.display()
        );
    }

    // ── 引擎回执：取回执的三态 ────────────────────────────────────────────
    //
    // 🔴 这里钉的是**整套三态设计所依赖的那一个属性**：失败必须返回 `None`，
    //    **绝不允许**返回空正文或 `"[]"` —— 后者会被解析成「引擎报 0 个适配器」，
    //    于是「端点还没就绪」被读成「适配器白挂」，**把结论带反**。

    /// 起一个只按固定响应作答的极简 HTTP 服务，返回端口。
    ///
    /// 用 `std::net` 而不是 `tokio::net`：本仓库的 `tokio` 没开 `net` 特性，
    /// 为测试去改生产依赖的特性表不划算。绑 `:0` 由系统分配空闲端口 ——
    /// 这是本项目「自起服务必须挑空闲端口，被占当场失败」那条铁律的应用。
    fn 起假引擎(响应: String) -> u16 {
        use std::io::{Read, Write};
        let 监听 = std::net::TcpListener::bind("127.0.0.1:0").expect("空闲端口应能绑定");
        let 端口 = 监听.local_addr().expect("拿到本地地址").port();
        std::thread::spawn(move || {
            for 连接 in 监听.incoming() {
                let Ok(mut 套) = 连接 else { break };
                // 必须**先把请求读掉再回**：不读就写，客户端可能收不到（连接被 RST）。
                let mut 缓冲 = [0u8; 4096];
                let _ = 套.read(&mut 缓冲);
                let _ = 套.write_all(响应.as_bytes());
                let _ = 套.flush();
                // 连接在本次迭代结束时关闭。响应里也写了 `Connection: close`，
                // 免得 reqwest 复用连接、把「3 次重试」变成「1 次请求」。
            }
        });
        端口
    }

    fn 回执响应(正文: &str) -> String {
        // `Content-Length` 用**真实字节数**算，不手写 —— 写错会让客户端挂住或截断，
        // 而那种失败看起来像「引擎不答」，与要测的东西完全是两回事。
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
             Connection: close\r\nContent-Length: {}\r\n\r\n{}",
            正文.len(),
            正文
        )
    }

    #[tokio::test]
    async fn 取回执在成功时返回正文() {
        let 端口 = 起假引擎(回执响应(r#"[{"id":0,"path":"/m/lora.gguf","scale":1.0}]"#));
        let 客户端 = reqwest::Client::new();
        let 正文 = 取回执(&客户端, &format!("http://127.0.0.1:{端口}/lora-adapters"))
            .await
            .expect("200 应当拿到正文");
        let 回执 = crate::model_sidecar::解析适配器回执(&正文).expect("正文应能解析");
        assert_eq!(回执.len(), 1);
    }

    #[tokio::test]
    async fn 取回执在非2xx时返回None而不是空列表() {
        // 500 × 3 次重试用尽 ⇒ 必须是 `None`。若这里返回 `Some("[]")`，
        // 调用方会走「条数不符」分支报出**假警报**（明明是端点没就绪）。
        let 端口 = 起假引擎(
            "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\
             Connection: close\r\n\r\n"
                .to_string(),
        );
        let 客户端 = reqwest::Client::new();
        let 结果 = 取回执(&客户端, &format!("http://127.0.0.1:{端口}/lora-adapters")).await;
        assert!(结果.is_none(), "取不到必须是 None，实际：{结果:?}");
    }

    #[tokio::test]
    async fn 取回执在端口不通时返回None() {
        // 挑一个几乎不可能有人在听的端口；连不上必须立刻是 None，而不是 panic。
        let 客户端 = reqwest::Client::builder()
            .timeout(Duration::from_millis(300))
            .build()
            .expect("客户端应能建立");
        let 结果 = 取回执(&客户端, "http://127.0.0.1:1/lora-adapters").await;
        assert!(结果.is_none());
    }

    // ── 接管路径：模型身份那一层 ──────────────────────────────────────────
    //
    // 2026-10-16 老李重启 APP，日志里是 `reattached to previous llama-server` ——
    // 那条路径**不进 `set_running`**，上一版加的回执核对一次都没跑。
    // 这里钉住修好后的行为：接管必须问身份；本次启动不必问（省去日常噪声）。

    #[test]
    fn 接管且知道模型路径时要核对模型身份() {
        assert!(应当核对模型身份(回执场景::接管旧进程, true));
    }

    #[test]
    fn 本次启动时不核对模型身份() {
        // 进程是我们按 `model_path` 亲手起的，启动日志已经打了 model_id ⇒
        // 每次启动都多问一次是白噪声。
        assert!(!应当核对模型身份(回执场景::本次启动, true));
    }

    #[test]
    fn 接管但不知道模型路径时不核对() {
        // 没有期望值就无从核对 —— 但**不许**编一个（那会报出假警报）。
        assert!(!应当核对模型身份(回执场景::接管旧进程, false));
    }

    // ── P2-14：接管态必须报「引擎自报的上下文」──────────────────────────
    //
    // 2026-10-16 P100：14:35 那次接管到的其实是 07:18 用 24 层 / 4096 起的进程，
    // 而回执只说了「跑的确实是这个模型」⇒ 用户误判成「上一轮的修复没生效」。
    // 这里钉住修好后的行为：接管必报；本次启动不报（有三重记录，再报是白噪声）。

    #[test]
    fn 接管态要报引擎自报的上下文() {
        assert!(应当报告引擎上下文(回执场景::接管旧进程));
    }

    #[test]
    fn 本次启动不报引擎上下文() {
        assert!(!应当报告引擎上下文(回执场景::本次启动));
    }

    #[tokio::test]
    async fn 模型身份三层能从假引擎端到端走通() {
        // 取回执 → 解析模型回执 → 核对模型，三层缝合一次走完。
        let 端口 = 起假引擎(回执响应(
            r#"{"object":"list","data":[{"id":"D:\\m\\base-Q4_K_M.gguf"}]}"#,
        ));
        let 客户端 = reqwest::Client::new();
        let 正文 = 取回执(&客户端, &format!("http://127.0.0.1:{端口}/v1/models"))
            .await
            .expect("200 应当拿到正文");
        let 自报 = crate::model_sidecar::解析模型回执(&正文).expect("正文应能解析");
        assert_eq!(自报, r"D:\m\base-Q4_K_M.gguf");
        // 与期望一致 ⇒ 没有问题；分隔符/大小写不同也算一致。
        assert!(crate::model_sidecar::核对模型(r"D:\m\base-Q4_K_M.gguf", &自报).is_empty());
        assert!(crate::model_sidecar::核对模型(r"d:/m/base-q4_k_m.gguf", &自报).is_empty());
    }

    // ── 2026-10-19 P100：端口被自己上一轮的引擎占着 ⇒ 接管，不是「请换端口」 ──

    #[test]
    fn 从netstat里取pid_只认监听且端口精确匹配() {
        // 真实 `netstat -ano -p tcp` 的形状（表头两行也要能跳过）。
        // 🔴 `:18000` 那行**刻意放在前面**：真实 netstat 的输出顺序是不定的，
        // 而这里要钉住的正是「查 8000 时不许被 18000 抢先命中」。若去掉冒号
        // 只比数字后缀，第一条命中的就是 18000 那一行 —— 断言当场变红。
        let 文本 = "\r\n活动连接\r\n\r\n  协议  本地地址          外部地址        状态           PID\r\n\
                    TCP    127.0.0.1:18000        0.0.0.0:0              LISTENING       9999\r\n\
                    TCP    127.0.0.1:8000         0.0.0.0:0              LISTENING       4321\r\n\
                    TCP    127.0.0.1:8000         127.0.0.1:51234        ESTABLISHED     4321\r\n";
        assert_eq!(从netstat里取pid(文本, 8000), Some(4321), "命中的应是监听那一行");
        // 🔴 后缀匹配的经典坑：`:8000` 不许把 `:18000` 认成自己。
        assert_eq!(从netstat里取pid(文本, 18000), Some(9999));
        assert_eq!(从netstat里取pid(文本, 800), None, "没这个端口就是 None");
        // 只有 ESTABLISHED、没有 LISTENING 时**不能**算 —— 那不是「在提供服务」。
        let 只有连接 = "  TCP    127.0.0.1:8000    127.0.0.1:51234    ESTABLISHED    4321\r\n";
        assert_eq!(从netstat里取pid(只有连接, 8000), None);
    }

    /// `taskkill` 的退出码必须分得开「拒绝访问」与「本来就没有」。
    ///
    /// 2026-10-19 P100：这两类失败在旧代码里都打成同一句 WARN，日志里 9 条 WARN
    /// 有 8 条是「没有此实例」，唯一那条「拒绝访问」被埋掉 —— 于是「杀不掉」被
    /// 误读成「端口被占」，整轮排查方向都错了。
    #[test]
    fn taskkill退出码要分开拒绝访问与本来就没有() {
        assert_eq!(由退出码判终止结果(Some(0)), 终止结果::已结束);
        assert_eq!(由退出码判终止结果(Some(255)), 终止结果::本来就没有);
        assert_eq!(由退出码判终止结果(Some(128)), 终止结果::拒绝访问);
        assert_eq!(由退出码判终止结果(Some(1)), 终止结果::其他);
        assert_eq!(由退出码判终止结果(None), 终止结果::其他);
        // 「本来就没有」不许和「拒绝访问」同形 —— 这一条是上面那段事故的正向钉子。
        assert_ne!(由退出码判终止结果(Some(255)), 由退出码判终止结果(Some(128)));
    }

    /// 「停止」之后到底停没停，只认现场：端口上还是不是那个 pid。
    #[test]
    fn 停止后仍是同一个进程_只认端口上现在的pid() {
        // 端口还在、pid 一模一样 ⇒ 没停下（2026-10-19 P100 的 07:16:04 就是这种）。
        assert!(停止后仍是同一个进程(Some(2468), Some(2468)));
        // 端口还在、但我们没有 pid 可比 ⇒ 同样不许宣称停掉了。
        assert!(停止后仍是同一个进程(Some(2468), None));
        // 端口空了 ⇒ 停了。
        assert!(!停止后仍是同一个进程(None, Some(2468)));
        // 端口被**别的**进程占了 ⇒ 旧那个确实退了，认作已停（新占用者由启动路径处置）。
        assert!(!停止后仍是同一个进程(Some(999), Some(2468)));
    }

    /// 启动前「停旧服务没生效」的判据：只认两条明确反证。
    ///
    /// 2026-10-19 P100（`07:42:02`）：切模型时 `taskkill` 拒绝访问，旧引擎继续占着
    /// 8000。旧代码把 `结束服务进程` 的返回值丢掉，于是照常启动 —— 新引擎绑不上
    /// 端口，健康探测被旧引擎的 200 骗过，日志写下「Ornith 已运行」，实际跑的是 1.7B。
    #[test]
    fn 停止没能生效_只认明确反证() {
        // 端口上还是刚才那个 pid ⇒ 没停下。
        assert!(停止没能生效(终止结果::已结束, Some(8892), Some(8892)));
        // `taskkill` 明说拒绝访问 ⇒ 进程一定还在，即使这一下 netstat 没查到。
        assert!(停止没能生效(终止结果::拒绝访问, None, Some(8892)));
        assert!(停止没能生效(终止结果::拒绝访问, Some(8892), Some(8892)));
        // 端口空了 ⇒ 停了。
        assert!(!停止没能生效(终止结果::已结束, None, Some(8892)));
        // 🔴 端口有人在听、但我们本来就没有 pid ⇒ **不算**没停下：
        // 那多半是别人的服务，要交给端口分支去识别/接管，不能在这里拦成「无法启动」。
        assert!(!停止没能生效(终止结果::未尝试, Some(4321), None));
        // 旧的那个确实退了、端口被别的进程占了 ⇒ 也放行（后面按端口识别）。
        assert!(!停止没能生效(终止结果::已结束, Some(999), Some(8892)));
    }

    /// 等待循环里的那个条件：端口上还是不是刚才那个 pid。
    ///
    /// 2026-10-20 本机：杀掉 `29328` 之后 **43 ms** 内 netstat 仍报它在监听，
    /// 于是「停旧服务失败 ⇒ 拒绝切模型」；实际上它已经在退。这条判据就是那次的钉子：
    /// **「同号」与「查不到 pid」必须区别对待**，前者要等，后者不能等。
    #[test]
    fn 端口已让出_同号才算没让出() {
        // 还是刚才那个 pid ⇒ 没让出，继续等。
        assert!(!端口已让出(Some(29328), Some(29328)));
        // 端口空了 ⇒ 让出了。
        assert!(端口已让出(None, Some(29328)));
        // 换成了别的 pid ⇒ 旧那个确实退了（新占用者交给端口分支处置）。
        assert!(端口已让出(Some(999), Some(29328)));
        // 没有 pid 可比 ⇒ 不能把「有人在听」当成「没让出」，否则会白等满预算
        // 并把「别人的服务占着端口」错判成「我们没停下」。
        assert!(端口已让出(Some(4321), None));
        assert!(端口已让出(None, None));
    }

    /// 🔴 建议必须与事实一致：非权限原因时**不许**让用户去提权。
    ///
    /// 2026-10-20 那次的真实代价：用户本来就是管理员，提示让他「右键以管理员身份
    /// 运行」，他照做了、问题照旧，还白关了一次 APP。建议跑偏比没有建议更糟。
    #[test]
    fn 停止失败建议_只在权限问题时才提权() {
        let 权限 = 停止失败建议(终止结果::拒绝访问, 29328);
        assert!(权限.contains("以管理员身份"), "{权限}");
        for 非权限 in [终止结果::已结束, 终止结果::其他, 终止结果::未尝试] {
            let 文案 = 停止失败建议(非权限, 29328);
            assert!(
                !文案.contains("管理员"),
                "{非权限:?} 不该建议提权，否则建议与事实相反：{文案}"
            );
            // 无论哪种原因，都要给出**可执行的下一步**，否则用户只知道失败。
            assert!(文案.contains("29328"), "{文案}");
            assert!(文案.contains("任务管理器"), "{文案}");
        }
    }

    /// 等待循环的两条行为：旧 pid 还在时必须**重试**；预算用完必须**照实报**。
    ///
    /// 这一条盯的是 2026-10-20 那次事故的修法本身 —— 只把「查一次」换成「等一会儿」
    /// 是不够的：如果等成死循环，「真没停下」会被永久挂住；如果根本不重试，
    /// 40 ms 的 socket 时间差照样能骗过它。
    #[test]
    fn 等端口让出_会重试_且预算到了照样下结论() {
        // 前两次仍报旧 pid，第三次才让出 ⇒ 必须重试到让出为止，且能报出等了多久。
        let 序列 = vec![Some(29328), Some(29328), None];
        let mut 计数 = 0;
        let (最后, 等了ms) = 等端口让出_直到(
            || {
                let 值 = 序列[计数.min(序列.len() - 1)];
                计数 += 1;
                值
            },
            Some(29328),
            Duration::from_millis(600),
        );
        assert_eq!(最后, None);
        assert_eq!(计数, 3, "旧 pid 还在时必须继续查，查一次就收手等于没修");
        assert!(等了ms >= 100, "两次 50 ms 间隔应当留下痕迹，实际 {等了ms} ms");

        // 一直不让出 ⇒ 预算内返回**最后一次的真实读数**，不许改口说「让出了」。
        let mut 计数2 = 0;
        let (最后2, 等了2) = 等端口让出_直到(
            || {
                计数2 += 1;
                Some(29328)
            },
            Some(29328),
            Duration::from_millis(160),
        );
        assert_eq!(最后2, Some(29328), "没让出就是没让出");
        assert!(等了2 >= 160, "预算到了才收手，实际 {等了2} ms");
        assert!(计数2 >= 2, "预算内应当反复查，实际查了 {计数2} 次");

        // 端口本来就空（旧 pid 早退了）⇒ 一次都不用等。
        let mut 计数3 = 0;
        let (最后3, 等了3) = 等端口让出_直到(
            || {
                计数3 += 1;
                None
            },
            Some(29328),
            Duration::from_millis(600),
        );
        assert_eq!((最后3, 计数3), (None, 1));
        assert!(等了3 < 50, "本来就空着不该白等，实际 {等了3} ms");
    }

    #[test]
    fn 判端口归属_三分法() {
        assert_eq!(判端口归属(Some(7148), 7148), 端口归属::是本次进程);
        assert_eq!(判端口归属(Some(8892), 7148), 端口归属::是别的进程(8892));
        assert_eq!(判端口归属(None, 7148), 端口归属::无法确认);
    }

    fn 全量计划() -> LaunchParams {
        LaunchParams {
            n_gpu_layers: Some(999),
            ctx_size: 40960,
            parallel: 1,
            kv_quant: false,
        }
    }

    fn 部分计划(层: u32, ctx: u32) -> LaunchParams {
        LaunchParams {
            n_gpu_layers: Some(层),
            ctx_size: ctx,
            parallel: 1,
            kv_quant: true,
        }
    }

    /// 归因判据认的是**反事实**（空卡能不能全量），不是「可用占总量的比例」。
    ///
    /// 五个数全部取自实测日志：P100 空卡 `15.898/16.0`、被旧引擎占着时
    /// `9.448/16.0`、本机笔记本 `5.760/7.960`（桌面合成占 2.20）、
    /// 27B-Q3 放 8G 卡（空卡也全量不了）。
    ///
    /// 🔴 后两个是**回归钉子**：旧判据（可用 < 总量×0.80）在这两种情形下都会报
    /// 「卡被占」，而真相分别是「桌面合成」和「模型比卡大」—— 前者让报警在常态
    /// 机器上恒亮（0.1.7-1 现场），后者是冤枉别人。
    #[test]
    fn 降档归因认的是反事实不是比例() {
        // P100 空卡：全量、无话可说
        assert_eq!(
            判定降档(16.0, 15.8984375, 空载基线_TCC_GIB, 全量计划(), 全量计划()),
            降档::未降档
        );
        // 2026-10-19 `07:42:03`：旧引擎占着 6.4 GiB，9.448/16.0 ⇒ 必须归到别人头上
        match 判定降档(
            16.0,
            9.4482421875,
            空载基线_TCC_GIB,
            部分计划(20, 3072),
            全量计划(),
        ) {
            降档::卡被占 { 他人_gib, .. } => {
                // 16.0 - 9.4482421875 - 1.0(TCC 空载基线) = 5.5517578125
                assert!(
                    (他人_gib - 5.5517578125).abs() < 1e-6,
                    "他人_gib={他人_gib}"
                );
            }
            其他 => panic!("P100 被占那次必须判成卡被占，实得 {其他:?}"),
        }
        // 🔴 本机常态（2026-10-20 `06:19` 实测 5.760/7.960）：占着卡的是桌面合成，
        // 扣掉 WDDM 基线 3.0 之后没有「别人」⇒ 不许报。旧判据在这里**天天报**。
        assert_eq!(
            判定降档(7.96, 5.76, 空载基线_WDDM_GIB, 部分计划(20, 3072), 全量计划()),
            降档::与他人无关,
            "笔记本上桌面合成占 2.2 GiB 不是「卡被占」，报了就等于报警恒亮"
        );
        // 27B-Q3（12.244 GiB）放 8G 卡：空卡也只给部分层 ⇒ 是模型比卡大，
        // 不是别人的锅。旧判据在这里同样会报。
        assert_eq!(
            判定降档(7.96, 7.2, 空载基线_WDDM_GIB, 部分计划(12, 2048), 部分计划(12, 2048)),
            降档::与他人无关
        );
        // 显存读数两边任一拿不到 ⇒ 无从判断（不许编原因）
        assert_eq!(
            判定降档(0.0, 5.0, 空载基线_TCC_GIB, 部分计划(20, 3072), 全量计划()),
            降档::无从判断
        );
        assert_eq!(
            判定降档(16.0, 0.0, 空载基线_TCC_GIB, 部分计划(20, 3072), 全量计划()),
            降档::无从判断
        );
    }

    /// 空载基线按驱动模式分档 —— TCC 没有桌面合成，WDDM 有。
    /// 大小写/空白都要能吃下（`nvidia-smi` 在不同驱动版本上输出格式不一）。
    #[test]
    fn 空载基线按驱动模式分档() {
        assert_eq!(空载基线_gib(Some("TCC")), 空载基线_TCC_GIB);
        assert_eq!(空载基线_gib(Some("tcc")), 空载基线_TCC_GIB);
        assert_eq!(空载基线_gib(Some(" WDDM ")), 空载基线_WDDM_GIB);
        // 查不到/认不出 ⇒ 折中值，不许退化成 0（0 会把常态占用全算到别人头上）
        assert_eq!(空载基线_gib(None), 空载基线_未知_GIB);
        assert_eq!(空载基线_gib(Some("N/A")), 空载基线_未知_GIB);
        assert!(空载基线_未知_GIB > 0.0);
    }

    /// 提示只在这种情形出现：**降档确实是别人造成的**。
    /// 其余三态一律 `None` —— 否则又变成「每天都说一遍」。
    #[test]
    fn 降档提示只在卡被占时说并且念得出数字() {
        let 占着 = 降档::卡被占 {
            他人_gib: 5.45,
            可用_gib: 5.8,
            总量_gib: 8.0,
        };
        let 文案 = 降档提示(占着, 部分计划(20, 3072), Some(32)).expect("卡被占必须给提示");
        assert!(文案.contains("5.8"), "{文案}");
        assert!(文案.contains("8.0"), "{文案}");
        // 「只上了一半」必须看得出来：念层数与分母，不念抽象的「部分卸载」
        assert!(文案.contains("20/32 层"), "{文案}");
        assert!(文案.contains("3072"), "{文案}");
        // 必须给下一步动作，否则用户知道慢了也没法改
        assert!(文案.contains("停止") && 文案.contains("启动"), "{文案}");

        // 主干层数读不到时不许编分母
        let 无层数 = 降档提示(占着, 部分计划(20, 3072), None).unwrap();
        assert!(无层数.contains("20 层") && !无层数.contains("20/"), "{无层数}");

        for 无关 in [降档::未降档, 降档::与他人无关, 降档::无从判断] {
            assert_eq!(降档提示(无关, 部分计划(20, 3072), Some(32)), None, "{无关:?}");
        }
    }

    /// `是端口占用类错误` 靠文案前缀判断，前缀必须与三处构造点一致。
    ///
    /// 三处分别是：端口被别人占（"…已被占用，请换端口"）、端口上是另一个模型、
    /// 端口上是高阶权限的引擎。漏掉任何一处，对应的错误就会被当成「显卡没起来」
    /// 去重试两次 —— 白等，还把真正的错因埋掉。
    #[test]
    fn 端口类错误前缀与实际文案对齐() {
        assert_eq!(端口类错误前缀, "接口端口");
        let 三处 = [
            format!("{端口类错误前缀} 8000 已被占用，请在设置里换一个端口。"),
            format!(
                "{端口类错误前缀} 8000 上正在运行另一个模型（D:\\m\\a.gguf，进程 8892），\
                 不是这次要启动的「D:\\m\\b.gguf」。请先在那边停止服务，或到设置里换一个端口。"
            ),
            format!(
                "{端口类错误前缀} 8000 上的引擎（进程 8892）还在运行，但本程序结束不掉它 —— \
                 Windows 拒绝访问，说明它是用**管理员权限**启动的。"
            ),
        ];
        for 文案 in 三处 {
            assert!(是端口占用类错误(&文案), "端口类错误必须被识别出来：{文案}");
        }
        // 反例：显卡/超时类错误绝不能被当成端口问题 —— 否则会白白跳过重试。
        assert!(!是端口占用类错误("显卡加速没能生效，当前用的是普通（CPU）速度"));
        assert!(!是端口占用类错误("AI服务启动超时，请稍后重试或换一个更小的模型。"));
    }

    /// 健康检查 200 ≠ 本次起的进程在服务 —— 判据是**端口上那个 pid**。
    ///
    /// 这条直接对应 07:42 的现场：旧引擎（1.7B）在 8000 上正常应答，新起的
    /// Ornith 进程没绑上端口，却照样被判成「启动成功」。
    #[tokio::test]
    async fn 确认本次进程在服务_端口上是别人时必须拒绝() {
        let 端口 = 起假引擎(回执响应(
            r#"{"object":"list","data":[{"id":"D:\\m\\qwen3_1_7b_light\\Qwen3-1.7B-Q8_0.gguf"}]}"#,
        ));
        let 我 = std::process::id();
        let 旧 = Path::new(r"D:\m\qwen3_1_7b_light\Qwen3-1.7B-Q8_0.gguf");
        let 新 = Path::new(r"D:\m\Ornith-1.5-9B-Q8_0\Ornith-1.5-9B-Q8_0.gguf");

        // 端口上就是本进程（假装是「我们刚 spawn 的」）⇒ 通过。
        确认本次进程在服务(我, 端口, 旧)
            .await
            .expect("端口归属对得上就该通过");

        // 端口上是本进程，但调用方声称自己 spawn 的是另一个 pid，期望模型也是新的：
        // 这正是 07:42 那次 —— 旧引擎在服务、新进程没绑上端口。
        let 错误 = 确认本次进程在服务(我.wrapping_add(1), 端口, 新)
            .await
            .expect_err("端口上是别的引擎且模型对不上，必须拒绝");
        let 文案 = 错误.to_string();
        assert!(文案.contains("另一个引擎"), "应说清是别的引擎在服务：{文案}");
        assert!(
            文案.contains("Qwen3-1.7B-Q8_0.gguf"),
            "应报出占位引擎自报的模型，否则用户不知道该去停哪个：{文案}"
        );
        assert!(
            是端口占用类错误(&文案),
            "端口类错误必须走前缀判据（不许再重试两次）：{文案}"
        );
    }

    #[tokio::test]
    async fn 识别端口上的引擎_能从假引擎拿到自报模型() {
        let 端口 = 起假引擎(回执响应(
            r#"{"object":"list","data":[{"id":"D:\\m\\qwen3-1.7b-q8_0.gguf"}]}"#,
        ));
        assert_eq!(
            识别端口上的引擎("127.0.0.1", 端口).await.as_deref(),
            Some(r"D:\m\qwen3-1.7b-q8_0.gguf"),
            "能答出模型列表 ⇒ 判定为我方引擎"
        );
    }

    #[tokio::test]
    async fn 识别端口上的引擎_对占了端口但不是引擎的程序返回None() {
        // ★ 反例：端口在听、也回了 HTTP，但答的不是引擎格式。
        // 少了这条，任何占着 8000 的 Web 服务都会被「接管」过来。
        let 端口 = 起假引擎(回执响应("I am not llama.cpp"));
        assert_eq!(
            识别端口上的引擎("127.0.0.1", 端口).await,
            None,
            "端口在听 ≠ 是我方引擎，必须落回「请换端口」"
        );
    }

    #[tokio::test]
    async fn 等端口释放_占着就超时_放开才返回true() {
        use std::net::TcpListener;
        let 监听 = TcpListener::bind("127.0.0.1:0").expect("应能绑定空闲端口");
        let 端口 = 监听.local_addr().expect("拿到本地地址").port();
        // ① 还占着：必须在窗口结束时报 false（否则「等端口」等于没有，
        //    切模型时仍会撞上没退干净的旧进程）。
        assert!(
            !等端口释放("127.0.0.1", 端口, Duration::from_millis(300)).await,
            "端口仍被监听时不许返回 true"
        );
        // ② 放开后：同一个函数要能等到 true（判据不是「恒 false」）。
        drop(监听);
        assert!(
            等端口释放("127.0.0.1", 端口, Duration::from_secs(3)).await,
            "端口释放后应当等到 true"
        );
    }

    #[tokio::test]
    async fn 模型身份在接管到别的模型时会报出两边名字() {        let 端口 = 起假引擎(回执响应(
            r#"{"object":"list","data":[{"id":"D:\\m\\Spark-X2.5-4B-Q6_K.gguf"}]}"#,
        ));
        let 客户端 = reqwest::Client::new();
        let 正文 = 取回执(&客户端, &format!("http://127.0.0.1:{端口}/v1/models"))
            .await
            .expect("200 应当拿到正文");
        let 自报 = crate::model_sidecar::解析模型回执(&正文).expect("正文应能解析");
        let 问题 = crate::model_sidecar::核对模型(r"D:\m\base-Q4_K_M.gguf", &自报);
        assert_eq!(问题.len(), 1, "接管到别的模型必须报出来");
        assert!(问题[0].contains("base-Q4_K_M.gguf"), "{问题:?}");
        assert!(问题[0].contains("Spark-X2.5-4B-Q6_K.gguf"), "{问题:?}");
    }

    #[tokio::test]
    async fn 接管态的上下文两层能从假引擎端到端取到() {
        // 取回执 → 解析上下文回执，两层缝合一次走完。
        // 测这条是因为**取那个函数与解析那个函数各自绿过、接起来错**是本项目踩过的坑
        // （同一格式三份实现的老问题），而且这条链路只在接管态跑，单测是唯一兜底。
        let 端口 = 起假引擎(回执响应(
            r#"{"default_generation_settings":{"n_ctx":4096},"total_slots":1}"#,
        ));
        let 客户端 = reqwest::Client::new();
        let 正文 = 取回执(&客户端, &format!("http://127.0.0.1:{端口}/props"))
            .await
            .expect("200 应当拿到正文");
        assert_eq!(
            crate::model_sidecar::解析上下文回执(&正文).expect("正文应能解析"),
            4096,
            "事故现场那个值：引擎自报 4096，而它其实能开 262144"
        );
    }

    // ── P2-15：接管态的**显存交叉验证** ────────────────────────────────
    //
    // 引擎不报 `n_gpu_layers`（`/props` 没有这个字段）⇒「ctx 对、层数错」在回执
    // 层面**同形**。判据改用一条**单边**下界：实测整卡 used 必须装得下模型文件。
    // 下面钉住它的三条性质 —— 能分辨（不放过坏态）、不误报（单边）、边界可解释。

    #[test]
    fn 实测显存装不下权重时要报出来() {
        // P100 现场原值（§零·5 的 `07:35` 那段）：接管到「24 层 / 4096」的旧进程，
        // 整卡 used 6424 MiB；而 Ornith-1.5-9B-Q8_0 的文件 9787 MB ≈ 9334 MiB。缺口 31%。
        let 权重_gib = 9787.0_f64 * 1e6 / (1024.0 * 1024.0 * 1024.0); // ≈ 9.115
        let 实测_gib = 6424.0 / 1024.0; // ≈ 6.273
        assert!(
            !实测显存够装权重(权重_gib, 实测_gib),
            "权重 {权重_gib:.2} GiB 而整卡只有 {实测_gib:.2} GiB ⇒ 层数铁定没上去"
        );
    }

    #[test]
    fn 实测显存装得下权重时判通过() {
        // 同一台机器、同一个模型，重算参数后（999 层 / 262144）：整卡 14095 MiB。
        let 权重_gib = 9787.0_f64 * 1e6 / (1024.0 * 1024.0 * 1024.0);
        let 实测_gib = 14095.0 / 1024.0; // ≈ 13.765
        assert!(实测显存够装权重(权重_gib, 实测_gib));
    }

    #[test]
    fn 别的进程占着显存也不会被误报() {
        // 单边性的直接证明：接管态整卡 used 里含**别的进程**（P100 上常驻 ComfyUI），
        // 而它们只会把 used **推高** ⇒ 这条判据**不可能**因此误报（只会漏报）。
        let 权重_gib = 1.708; // Qwen3-1.7B-Q8_0
        assert!(实测显存够装权重(权重_gib, 30.0), "占得越多越该通过");
    }

    #[test]
    fn 余量边界压在九成上() {
        // 口径必须可解释：GGUF 文件体积含元数据（比纯权重略大），故留 10% 余量；
        // 而真出事时缺口远大于 10%（现场 31%）⇒ 不会因此漏掉。
        let 权重_gib = 10.0_f64;
        assert!(实测显存够装权重(权重_gib, 9.0), "恰好九成：通过");
        assert!(!实测显存够装权重(权重_gib, 8.99), "差一点点：不通过");
    }

    #[tokio::test]
    async fn 接管态能从引擎自报的路径算出权重体积并判定() {
        // 缝一次「引擎自报路径 → stat → 权重 GiB → 判据」：这条路只在接管态跑，
        // 单测是唯一兜底（本项目踩过「取的那一半与解析的那一半各自绿、接起来错」）。
        let 目录 = std::env::temp_dir().join("本地AI-P2-15-单测");
        std::fs::create_dir_all(&目录).expect("建临时目录");
        let 文件 = 目录.join("假模型-Q8_0.gguf");
        // 稀疏文件：只设长度，不真写 1 GiB 数据。
        std::fs::File::create(&文件)
            .expect("建文件")
            .set_len(1024 * 1024 * 1024)
            .expect("设长度");

        let 端口 = 起假引擎(回执响应(&format!(
            r#"{{"object":"list","data":[{{"id":"{}"}}]}}"#,
            // Windows 路径含反斜杠，进 JSON 前必须转义。
            文件.display().to_string().replace('\\', "\\\\")
        )));
        let 客户端 = reqwest::Client::new();
        let 正文 = 取回执(&客户端, &format!("http://127.0.0.1:{端口}/v1/models"))
            .await
            .expect("200 应当拿到正文");
        let 自报 = crate::model_sidecar::解析模型回执(&正文).expect("正文应能解析");
        let 权重_gib =
            std::fs::metadata(&自报).expect("自报路径应能 stat 到").len() as f64
                / (1024.0 * 1024.0 * 1024.0);
        assert!((权重_gib - 1.0).abs() < 0.01, "权重体积应约 1 GiB，实际 {权重_gib}");
        // 整卡 used 给一个不合理的低值 ⇒ 应当判「装不下」。
        assert!(!实测显存够装权重(权重_gib, 0.25));
        let _ = std::fs::remove_file(&文件);
    }

    #[tokio::test]
    async fn 模型文件读不到时安静返回而不是算通过() {
        // 「读不到」必须与「通过」「不通过」都不同。这里断言它**安静返回**、
        // 不 panic、也不误判 —— 日志里那条 INFO 会写明「本次**未做**」。
        let 不存在 = std::env::temp_dir().join("本地AI-绝不存在的模型-单测用.gguf");
        let _ = std::fs::remove_file(&不存在);
        核对接管态显存(&不存在).await;
    }

    // ── B-1：**计划全量 ⇏ 真上去了**（启动路径的档 2）────────────────────
    //
    // 档 1（`gpu_offload_verified`）阈值固定 256 MiB，只抓「完全没上卡」。
    // 显存不足时 llama.cpp 会静默少卸几层：启动成功、健康检查 200、`/props` 有 n_ctx，
    // 只有速度掉一半。下面钉住四件事 —— 只对全量计划生效、装不下才报、
    // **部分卸载装不下也不许报**（那是合法状态）、拿不到层数不猜。

    #[test]
    fn 计划全量只认规划器的哨兵值() {
        // 999 是 `cuda_launch_params` 写的「全量」哨兵（`--n-gpu-layers 999`）。
        assert!(计划全量卸载(Some(999)));
        assert!(计划全量卸载(Some(u32::MAX)), "比哨兵更大也只能是全量");
        // 按层数算出来的档位与保守固定档，都是**已经认了**部分卸载。
        for 层 in [24, 20, 12, 8] {
            assert!(!计划全量卸载(Some(层)), "{层} 层不是全量计划");
        }
        // CPU 模式（`None`）没有「上卡」可言。
        assert!(!计划全量卸载(None));
    }

    #[test]
    fn 计划全量却装不下时要报出来() {
        // P100 现场原值（§零·5）：Ornith-9B 文件 9787 MB ≈ 9.115 GiB，
        // 计划 999 层而整卡只有 6424 MiB ≈ 6.273 GiB —— 缺口 31%。
        let 权重_gib = 9787.0_f64 * 1e6 / (1024.0 * 1024.0 * 1024.0);
        let 实测_gib = 6424.0 / 1024.0;
        assert!(
            全量卸载却没上卡(Some(999), 权重_gib, 实测_gib),
            "声称全量、实测装不下 ⇒ 必须报（这是本条判据存在的理由）"
        );
    }

    #[test]
    fn 计划全量且装得下时不报() {
        // 同一台机器重算参数后（999 层 / 262144）：整卡 14095 MiB ≈ 13.765 GiB。
        let 权重_gib = 9787.0_f64 * 1e6 / (1024.0 * 1024.0 * 1024.0);
        assert!(!全量卸载却没上卡(Some(999), 权重_gib, 14095.0 / 1024.0));
        // 边界沿 P2-15：恰好九成通过，差一点点不通过。
        assert!(!全量卸载却没上卡(Some(999), 10.0, 9.0));
        assert!(全量卸载却没上卡(Some(999), 10.0, 8.99));
    }

    #[test]
    fn 部分卸载装不下属于合法状态不许报() {
        // 🔴 关键反例：16 GB 卡装 27B-Q3（12.244 GiB）本来就装不满，
        // 规划器已按余量给出 20/12/8 层 —— 那是「能用但慢」的**正常工作状态**。
        // 拿全量的尺子量它，会把每一次正常的部分卸载都报成故障（必然误报）。
        assert!(!全量卸载却没上卡(Some(20), 12.244, 8.0));
        assert!(!全量卸载却没上卡(None, 12.244, 0.2), "CPU 模式也不许报");
    }

    #[tokio::test]
    async fn 采样用同一把尺子() {
        // 接管态与启动态共用 `采样权重与显存`。钉住 GiB 换算（1024³，不是十进制 GB）：
        // 稀疏文件 1 GiB，采样必须在 1.00 附近；差 7.4% 就说明混进了 10⁹ 口径。
        let 目录 = std::env::temp_dir().join("本地AI-B1-采样单测");
        std::fs::create_dir_all(&目录).expect("建临时目录");
        let 文件 = 目录.join("假模型-1GiB.gguf");
        std::fs::File::create(&文件)
            .expect("建文件")
            .set_len(1024 * 1024 * 1024)
            .expect("设长度");
        match 采样权重与显存(&文件).await {
            Ok((权重_gib, _实测)) => {
                assert!((权重_gib - 1.0).abs() < 0.01, "应约 1 GiB，实际 {权重_gib}")
            }
            // 非 N 卡 / 无 nvidia-smi 的机器上会走这一支 —— 但**必须带出已读到的权重**，
            // 否则调用方连「文件是好的、只是读不到显存」都说不出来。
            Err(采样缺失::显存 { 权重_gib }) => {
                assert!((权重_gib - 1.0).abs() < 0.01, "显存缺失时权重仍须有效")
            }
            Err(采样缺失::文件(错误)) => panic!("文件明明在，不该读不到：{错误}"),
        }
        let _ = std::fs::remove_file(&文件);
    }

    #[tokio::test]
    async fn 全量判据文件读不到时不判定() {
        // 与接管态同一条纪律：读不到 → INFO「本次未做」，不是「通过」也不是「失败」。
        let 不存在 = std::env::temp_dir().join("本地AI-B1-绝不存在的模型.gguf");
        let _ = std::fs::remove_file(&不存在);
        核对计划全量卸载(&不存在, Some(999), EngineBackend::Cuda).await;
        // 非全量计划直接跳过，连文件都不用碰 —— 这条在真实机型上不产生任何 WARN。
        核对计划全量卸载(&不存在, Some(20), EngineBackend::Cuda).await;
    }

    #[test]
    fn 启动路径真的调用了全量判据() {
        // 🔴 上面五条只证明「判据算得对」，**不证明有人在调用它** ——
        // 本项目踩过「钩子永不触发」（配置写好了、代码路径走不到，全绿但现场照旧）。
        // 这里用源码断言兜住「调用点被删 / 被挪进死分支」。needle 用 concat 拼，
        // 避免把测试自身这段字面量也数进去。
        let 源码 = include_str!("service.rs");
        let 调用 = concat!("核对计划全量", "卸载(&model_path");
        assert_eq!(
            源码.matches(调用).count(),
            2,
            "启动路径有两条分支（激进一次成功 / 保守参数重试）都要核对；少一条就有一半启动路径不受保护"
        );
        // ⚠️ 这条**不**证明运行时真的走到了那两条分支 —— 那只能靠实机日志（判据 G-5）。
    }

    /// 判据存在 ≠ 判据被调用 —— 这两条钉住**调用点**。
    ///
    /// 2026-10-19 P100（`07:42`）的病根就是这个：`结束服务进程` 写得没问题，
    /// 但调用处是裸调用 `结束服务进程(…);`，返回值直接丢掉，于是「拒绝访问」
    /// 只留一条 WARN、启动照走。同理，`wait_for_health` 通过之后若不核对端口归属，
    /// 旧引擎的 200 会把新进程「绑不上端口」这件事完全盖住。
    #[test]
    fn 启动路径真的检查了停旧服务的结果与端口归属() {
        let 源码 = include_str!("service.rs");
        assert!(
            源码.contains(concat!("let 终止 = 结束服务", "进程(")),
            "启动前停旧服务必须把结果接住；裸调用会重现 07:42 那次「停不掉还照常启动」"
        );
        assert!(
            源码.contains(concat!("if 停止没能生", "效(")),
            "接住结果之后必须据此中止启动（没停就不许拉起新进程）"
        );
        assert!(
            源码.contains(concat!("确认本次进程在服务(child", ".id()")),
            "健康检查之后必须核对端口归属，否则旧引擎的 200 会把新进程的失败盖掉"
        );
    }

    /// 启动失败必须把状态复位 —— 否则界面永久卡在「正在启动」，用户只能重开 APP。
    ///
    /// 2026-10-20 P100 实测现场：`launch.json` 写坏 ⇒ 启动失败 ⇒ 状态停在
    /// `Starting`。此时**启动与停止两个按钮同时变灰**，而弹窗说的是「配置不合法」，
    /// 用户改好配置再点启动，得到的是「AI服务正在启动中，请稍候」—— 那个「启动中」
    /// 永远不会结束。四条旁车配置反例（C-13~C-16）每条都逼着重开一次 APP。
    #[test]
    fn 启动失败必须把状态复位_不能卡在正在启动() {
        let (目录, 库) = 建临时数据目录();
        let state = AppState::new(
            crate::settings::AppSettings::default(),
            目录,
            库,
            Default::default(),
        );
        // 复刻 guard 把状态翻成 Starting 之后的样子：pid / port / model_id 都已被
        // 新值覆盖。这些字段若留着，界面会显示「正在运行 qwen3_1_7b_light」。
        {
            let mut inner = state.inner.lock().expect("state lock poisoned");
            inner.service.status = ServiceStatus::Starting;
            inner.service.model_id = Some("qwen3_1_7b_light".to_string());
            inner.service.port = Some(8000);
            inner.service.host = Some("0.0.0.0".to_string());
            inner.service.pid = Some(1234);
            inner.service.safe_mode = true;
        }

        let info = 复位为已停止(&state, "启动失败，服务已复位为未启动：配置不合法");

        assert_eq!(info.status, ServiceStatus::Stopped);
        assert!(info.model_id.is_none(), "模型名留着会让界面继续显示它");
        assert!(info.port.is_none());
        assert!(
            info.message.contains("已复位"),
            "文案要能让人看出这不是「用户点了停止」，否则两种结局同形"
        );

        let inner = state.inner.lock().expect("state lock poisoned");
        assert_ne!(
            inner.service.status,
            ServiceStatus::Starting,
            "停在 Starting 会让启动与停止两个按钮同时变灰，用户只能重开 APP"
        );
        assert_eq!(inner.service.status, ServiceStatus::Stopped);
        assert!(
            inner.service.pid.is_none(),
            "pid 留着会让「停止」去杀一个根本没起来的进程"
        );
        assert!(inner.service.model_id.is_none());
        assert!(!inner.service.safe_mode, "安全模式标志属于那次没成功的启动");
    }

    /// 判据存在 ≠ 判据被调用（同 `启动路径真的检查了停旧服务的结果与端口归属` 的理由）。
    ///
    /// 上面那条只证明「复位函数本身对」，**不证明有人在失败路径上调它**。这里钉
    /// 的是「旁车配置解析失败时不许裸 `?` 早退」—— 那个 `?` 正是 2026-10-20 界面
    /// 卡死的直接原因。
    #[test]
    fn 旁车配置失败不能裸问号早退() {
        let 源码 = include_str!("service.rs");
        assert!(
            !源码.contains(concat!("推导附加参数(state, &model_path)", "?;")),
            "裸 `?` 早退会让状态永久停在 Starting（2026-10-20 实测：四条反例每条都要重开 APP）"
        );
        assert!(
            源码.contains(concat!("match 推导附加参数", "(state, &model_path)")),
            "旁车配置解析失败必须被接住并复位，而不是把错误原样抛回去"
        );
        assert!(
            源码.contains(concat!("let info = 复位为已停", "止(state, &format!")),
            "复位之后要把新状态 emit 给前端；只改内存里的值，界面仍然卡着"
        );
    }

    // ── 旁车项并入：它决定「已并入本次启动命令行」那句日志的**真值** ──────────
    //
    // 2026-10-16 实机踩到「日志宣称了没发生的动作」，修法之一就是把日志挪到
    // 真正动手的地方；这几条测试钉的就是**挪过去之后那句话是否还有依据**。

    #[test]
    fn 没有同名参数时旁车项全部并入并报出生效() {
        let mut 参数 = vec!["--threads".to_string(), "8".to_string()];
        let 旁车 = vec![
            ("--chat-template-file".to_string(), "tpl.jinja".to_string()),
            ("--lora".to_string(), "lora.gguf".to_string()),
        ];
        let 生效 = 并入旁车项(&mut 参数, 旁车);
        assert_eq!(生效.len(), 2, "两项都该生效：{生效:?}");
        // 原参数在前、旁车按书写顺序追加 —— 日志按生效项打印，顺序须与命令行一致。
        assert_eq!(
            参数.join(" "),
            "--threads 8 --chat-template-file tpl.jinja --lora lora.gguf"
        );
    }

    #[test]
    fn 用户手写的同名标志会把旁车那一项挡在命令行外() {
        // 这套「用户优先」规则的地基：被挡掉的**不许**算「已并入」，
        // 否则那句日志又会变成假情报（工具修了半天，换了个地方接着说假话）。
        let mut 参数 = vec!["--lora".to_string(), "手写.gguf".to_string()];
        let 旁车 = vec![
            ("--chat-template-file".to_string(), "tpl.jinja".to_string()),
            ("--lora".to_string(), "旁车.gguf".to_string()),
        ];
        let 生效 = 并入旁车项(&mut 参数, 旁车);
        assert_eq!(生效.len(), 1, "只有模板该生效：{生效:?}");
        assert_eq!(生效[0].0, "--chat-template-file");
        assert_eq!(参数.iter().filter(|词| *词 == "--lora").count(), 1);
        assert!(参数.contains(&"手写.gguf".to_string()));
        assert!(!参数.contains(&"旁车.gguf".to_string()));
    }

    #[test]
    fn 等号写法的同名标志也算用户手写() {
        let mut 参数 = vec!["--lora=D:/手写.gguf".to_string()];
        let 旁车 = vec![("--lora".to_string(), "D:/旁车.gguf".to_string())];
        let 生效 = 并入旁车项(&mut 参数, 旁车);
        assert!(生效.is_empty(), "等号写法同样该挡住旁车项：{生效:?}");
        assert_eq!(参数.len(), 1);
    }

    #[test]
    fn 前缀相同的另一个标志不挡旁车项() {
        // `--lora-scaled` 不以 `--lora=` 开头 ⇒ 按当前口径**不算**「用户已指定」。
        // 这条钉的是**现状**（已知边界）：两族并存时引擎会挂两次适配器；
        // 要改得先立「适配器参数族」的口径，不许在「修日志措辞」里顺手改语义。
        let mut 参数 = vec!["--lora-scaled".to_string(), "D:/x.gguf:0".to_string()];
        let 旁车 = vec![("--lora".to_string(), "D:/旁车.gguf".to_string())];
        let 生效 = 并入旁车项(&mut 参数, 旁车);
        assert_eq!(生效.len(), 1, "按当前口径应当仍并入：{生效:?}");
        assert!(参数.contains(&"D:/旁车.gguf".to_string()));
    }

    #[test]
    fn 旁车项为空时什么也不做() {
        let mut 参数 = vec!["--threads".to_string(), "8".to_string()];
        let 生效 = 并入旁车项(&mut 参数, Vec::new());
        assert!(生效.is_empty());
        assert_eq!(参数.join(" "), "--threads 8");
    }

    // ── P2-11：启动时的模型元数据来源（清单 → 数据库两级查找）──────────────
    //
    // 🔴 判据两条，缺一不可：
    //    ① 导入模型（id 形如 `local-…`，清单里**天生没有**）必须能从数据库
    //       **回落**取到完整元数据；
    //    ② 内置模型（清单里查得到）的结果必须**一字不变** —— 回落是兜底，
    //       不是改口径；改错了会让本来调好的内置模型参数一起跑偏。
    //    再加一条「改前会怎样」的反推（第四条），否则修的到底是什么说不清。

    fn 建临时数据目录() -> (PathBuf, PathBuf) {
        // 目录名带 uuid：验证环境必须是自己造的（本项目铁律）。复用固定路径
        // 会把上一次跑剩的库读进来，而那种「串味」在断言失败时极难分辨。
        let 目录 = std::env::temp_dir().join(format!("localai-p211-{}", uuid::Uuid::new_v4()));
        let 库 = crate::database::init_database(&目录).expect("建临时库");
        (目录, 库)
    }

    #[test]
    fn 导入模型查不到清单时从数据库回落() {
        // 🔴 复刻**老库的真实样子**，别自己造一个干净前提（2026-10-16 实测教训）：
        //    行是旧版导入流程写的 `imported-…`，而调用方（UI / 文件扫描侧）手里
        //    是 `local-…`（P100 日志：`switched current model
        //    model_id=local-Ornith-1.5-9B-Q8_0 source=imported`）。
        //    上一轮这条测试自己造了一个 `local-…` 的行 —— 真实代码从不会写出那种
        //    行，于是三条测试全绿、两台实机照旧（测试通过只证明「实现与判据一致」，
        //    不证明判据是对的）。
        const 主名: &str = "Ornith-1.5-9B-Q8_0";
        let 扫描侧id = crate::models::导入模型标识(主名);
        assert_eq!(
            扫描侧id, "local-Ornith-1.5-9B-Q8_0",
            "扫描侧的 id 规则改了，本测试的判据要跟着改"
        );
        assert_ne!(
            扫描侧id,
            format!("imported-{主名}"),
            "两种写法必须真的不同，否则这条测的是空气"
        );

        let (数据目录, 库) = 建临时数据目录();
        crate::model_repo::upsert_imported_model(
            &库,
            &format!("imported-{主名}"),
            "Ornith-1.5-9B-Q8_0",
            &数据目录.join("Ornith-1.5-9B-Q8_0.gguf"),
            9785,
            11.0,
            None,
        )
        .expect("写入导入模型");

        // 先把「前提」验掉：验证一条被测性质之前，先说清它成立的前提。
        // 若哪天清单开始包含导入模型，这条会先红，提示本测试已失去意义。
        assert!(
            crate::catalog::effective_models(&数据目录)
                .iter()
                .all(|m| m.id != 扫描侧id),
            "前提不成立：导入模型居然出现在清单里，本测试证明不了任何事"
        );

        let 命中 = 查找启动模型(&数据目录, &库, &数据目录, None, &扫描侧id)
            .expect("应能从数据库回落取到");
        assert_eq!(命中.file_size_mb, 9785);
        assert_eq!(
            命中.id, 扫描侧id,
            "命中历史写法后 id 要归成调用方给的那个 —— 否则日志里会突然换名字"
        );
        assert!(
            命中.local_path.is_some(),
            "回落取到的行必须带真实文件路径，否则 KV 层结构照样读不到"
        );

        let _ = std::fs::remove_dir_all(&数据目录);
    }

    #[test]
    fn 内置模型仍只走清单且结果一字不变() {
        let (数据目录, 库) = 建临时数据目录();
        let 期望 = crate::models::builtin_models()
            .into_iter()
            .find(|m| m.id == "qwen3_1_7b_light")
            .expect("内置清单里有 Qwen3-1.7B");
        let 实际 = 查找启动模型(&数据目录, &库, &数据目录, None, "qwen3_1_7b_light")
            .expect("内置模型必命中");

        // 逐字段比，不图省事 —— 回落逻辑一动就可能顺手改了内置路径的取值，
        // 而那种偏差在日志里只表现为「-ngl 跟以前不一样」，看不出来是回归。
        assert_eq!(实际.id, 期望.id);
        assert_eq!(实际.name, 期望.name);
        assert_eq!(实际.file_size_mb, 期望.file_size_mb);
        assert_eq!(实际.parameters_raw, 期望.parameters_raw);
        assert_eq!(实际.active_parameters, 期望.active_parameters);
        assert_eq!(实际.context_length, 期望.context_length);
        assert_eq!(实际.quant_bpp, 期望.quant_bpp);
        assert_eq!(实际.architecture, 期望.architecture);
        assert_eq!(实际.local_path, 期望.local_path);

        let _ = std::fs::remove_dir_all(&数据目录);
    }

    #[test]
    fn 库里也没有的模型返回未知而不是报错() {
        let (数据目录, 库) = 建临时数据目录();
        assert!(
            查找启动模型(&数据目录, &库, &数据目录, None, "压根不存在的模型").is_none(),
            "清单、库、磁盘三处都没有时不许乱猜，必须老实返回 None"
        );
        let _ = std::fs::remove_dir_all(&数据目录);
    }

    #[test]
    fn 库里没有行时改从磁盘兜底() {
        // 🔴 P2-11 续二的**判据本体**（2026-10-16 P100 实机抓出）：
        //    模型压根没进过库 —— 真实用法是把 gguf 拷进 `models/<目录>/`
        //    让界面扫描发现它（`switch_installed_model` 只设路径、不落库）。
        //    上一版只查到第二级为止，此处会**静默**返回 None，实机表现为
        //    「日志里一句话都不说，参数还是 24 层 / 4096」。
        const 主名: &str = "兜底模型";
        let (数据目录, 库) = 建临时数据目录();
        // 故意**不**调 `upsert_imported_model`：库里就是没有这一行。
        let 模型目录 = 数据目录.join("models").join(主名);
        std::fs::create_dir_all(&模型目录).expect("建模型目录");
        let 文件 = 模型目录.join(format!("{主名}.gguf"));
        // 写占位内容即可：本测试只验「取到了、取的哪一份」，
        // GGUF 头解析失败时 `imported_metadata` 会给出兜底的上下文档，
        // 那不是这条判据要管的事（真机上的解析结果由 §九 的实机清单覆盖）。
        std::fs::write(&文件, vec![0u8; 3_000_000]).expect("写占位模型文件");

        let 期望id = crate::models::导入模型标识(主名);
        let 命中 = 查找启动模型(&数据目录, &库, &数据目录, None, &期望id)
            .expect("文件就在磁盘上，库有没有行不该决定这件事");

        assert_eq!(命中.id, 期望id, "id 要用调用方给的那个，不能自己改名");
        assert_eq!(
            命中.file_size_mb, 3,
            "文件大小按十进制向上取整（3,000,000 B ⇒ 3 MB），口径要与扫描侧一致"
        );
        assert!(
            命中.local_path
                .as_deref()
                .is_some_and(|路径| 路径.ends_with("兜底模型.gguf")),
            "必须把真正用到的那个文件路径带回来 —— 它是 KV 层结构解析的入口"
        );

        let _ = std::fs::remove_dir_all(&数据目录);
    }

    #[test]
    fn 磁盘兜底只认主名相同的文件() {
        // 防止「修过头」：第三级若见 gguf 就用，同一个目录里放着别的模型时
        // 会张冠李戴，把别人的层结构算进本次启动 —— 比原来的兜底值更坏，
        // 因为它**看起来是对的**。
        const 主名: &str = "要的模型";
        let (数据目录, 库) = 建临时数据目录();
        let 目录 = 数据目录.join("models").join("某个目录");
        std::fs::create_dir_all(&目录).unwrap();
        std::fs::write(目录.join("别的模型.gguf"), b"GGUF").unwrap();

        assert!(
            查找启动模型(
                &数据目录,
                &库,
                &数据目录,
                None,
                &crate::models::导入模型标识(主名)
            )
            .is_none(),
            "目录里只有别的模型时，不许被兜底捡走"
        );

        let _ = std::fs::remove_dir_all(&数据目录);
    }

    #[test]
    fn 磁盘兜底优先用当前正在用的那个文件() {
        // 同一个模型名可能在两处各有一份（不同量化档）。用户界面上选的是
        // `local_model_path` 指向的那一份，参数就该按那一份算 —— 兜底路径
        // 里的排序不能把它的优先级冲掉。
        const 主名: &str = "双份模型";
        let (数据目录, 库) = 建临时数据目录();
        let id = crate::models::导入模型标识(主名);

        // 托管目录里一份（1 MB）
        let 托管目录 = 数据目录.join("models").join(主名);
        std::fs::create_dir_all(&托管目录).unwrap();
        std::fs::write(托管目录.join(format!("{主名}.gguf")), vec![0u8; 1_000_000]).unwrap();

        // 「当前正在用」的那一份在别处（5 MB），靠大小区分是哪一份被采纳了。
        // 路径里带中文，顺带钉住「排序不会因为编码顺序把它挤到后面」。
        let 当前文件 = 数据目录.join("别处").join(format!("{主名}.gguf"));
        std::fs::create_dir_all(当前文件.parent().unwrap()).unwrap();
        std::fs::write(&当前文件, vec![0u8; 5_000_000]).unwrap();

        let 命中 = 查找启动模型(&数据目录, &库, &数据目录, Some(&当前文件), &id)
            .expect("应能从磁盘兜底取到");
        assert_eq!(
            命中.file_size_mb, 5,
            "应当优先采信「当前正在用」的那份（5 MB），而不是目录里扫到的（1 MB）"
        );

        let _ = std::fs::remove_dir_all(&数据目录);
    }

    #[test]
    fn 回落之后启动参数不再落到兜底默认值() {
        // 用 P100 上的**真实数字**反推：Ornith-1.5-9B Q8_0（9.114 GiB ⇒ 十进制
        // 9785 MB）、可用显存 15.898 GiB。
        //
        // 修复前：模型元数据恒为 None ⇒ `cuda_launch_params` 根本没被调用，
        // 参数恒为 DEFAULT_CUDA_PARAMS（24 层 / 4096），**与显存多少无关**
        // （本机 8G 与虚机 16G 算出同一组，正是这一点暴露的 P2-11）。
        // 修复后：权重与原生上下文都拿得到 ⇒ 必须走反解。
        //
        // ⚠️ KV 那一项这里故意用**旧的参数量经验系数**（最保守的输入），
        // 不是真实 GGUF 层结构 —— 所以下面算出的上下文是**下限**，真机只会更长。
        // 行按**老库的真实写法**写（`imported-…`），查询用**UI 手里的那个 id**
        // （`local-…`）；两个字符串都由 `导入模型标识` 派生，不手抄字面量。
        const 主名: &str = "Ornith-1.5-9B-Q8_0";
        let (数据目录, 库) = 建临时数据目录();
        crate::model_repo::upsert_imported_model(
            &库,
            &format!("imported-{主名}"),
            "Ornith-1.5-9B-Q8_0",
            &数据目录.join("Ornith-1.5-9B-Q8_0.gguf"),
            9785,
            11.0,
            Some(&crate::gguf::GgufMetadata {
                parameter_count: 9_197_000_000,
                context_length: Some(262_144),
                ..Default::default()
            }),
        )
        .expect("写入导入模型");

        let 模型 = 查找启动模型(
            &数据目录,
            &库,
            &数据目录,
            None,
            &crate::models::导入模型标识(主名),
        )
        .expect("应能回落取到");
        assert_eq!(
            模型.context_length, 262_144,
            "原生上下文必须带回来 —— 否则就算权重读到了，上下文仍被钉在 4096"
        );

        let 参数 = cuda_launch_params(
            15.8984375,
            resolve_weight_gb(&模型),
            LEGACY_KV_GIB_PER_TOKEN_PER_BILLION * 9.197,
            0.0,
            模型.context_length as u32,
        );

        assert_ne!(参数, DEFAULT_CUDA_PARAMS, "回落生效后不该再是兜底默认值");
        assert_eq!(
            参数.n_gpu_layers,
            Some(999),
            "16 GB 卡放得下 9.1 GiB 权重，应当全量卸载（实际：{参数:?}）"
        );
        assert!(
            参数.ctx_size > DEFAULT_CUDA_PARAMS.ctx_size,
            "上下文应当明显大于 4096（实际：{}）",
            参数.ctx_size
        );

        let _ = std::fs::remove_dir_all(&数据目录);
    }

    // ── P2-12：回执探测用的地址（监听地址 ≠ 可连接地址）────────────────────
    //
    // 这条在 P100 上暴露：那台机开了「局域网可访问」⇒ 监听 `0.0.0.0`，
    // 而回执探测照着监听地址拼 URL，于是 `/v1/models` 永远连接失败
    // （日志原文：`error sending request for url (http://0.0.0.0:8000/v1/models)`），
    // 「接管到的服务跑的是不是这个模型」在这台机上**从来没核对成过**。
    // 本机的默认值是 `127.0.0.1`，所以只看本机永远看不到。
    #[test]
    fn 探测主机_只把通配监听地址换成本机回环() {
        assert_eq!(探测主机("0.0.0.0"), "127.0.0.1", "通配地址不可连接");
        assert_eq!(探测主机("::"), "127.0.0.1", "IPv6 通配同样不可连接");
        assert_eq!(探测主机(" 0.0.0.0 "), "127.0.0.1", "带空白的也要认出来");
        assert_eq!(探测主机(""), "127.0.0.1", "没配就按本机回环探");
        // 非通配**必须原样**：一律改回环会在「只监听某块网卡」时反而连不上。
        assert_eq!(探测主机("192.168.1.5"), "192.168.1.5");
        assert_eq!(探测主机("127.0.0.1"), "127.0.0.1");
    }

    // ── P2-13：接管与「正在启动」的竞态 ────────────────────────────────────
    //
    // P100 日志里那次 `reattached` 出现在本会话**正在起服务**的当口：
    // 它把一个刚起的进程认成「上次留下的旧进程」，把状态里的 model_id 写回旧的，
    // 并把 `process` 置回 None（子进程句柄还在启动任务手里）。
    // 这条判据（可不可达分支由单测兜底）：只有本会话手里没有服务时，才去读旧记录。
    #[test]
    fn 应当尝试接管_本会话正在管服务时不接管() {
        assert!(应当尝试接管(&ServiceStatus::Stopped));
        assert!(
            应当尝试接管(&ServiceStatus::Failed),
            "起步失败 ⇒ 本会话手里没有服务，接管真正的残留进程仍是原语义"
        );
        assert!(
            !应当尝试接管(&ServiceStatus::Starting),
            "正在起服务时接管 = 把新起的进程认成旧进程（P2-13）"
        );
        assert!(
            !应当尝试接管(&ServiceStatus::Running),
            "本会话已有服务，不必再读上个会话的记录"
        );
    }
}
