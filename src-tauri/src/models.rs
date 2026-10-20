use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::{
    app_state::AppState,
    error::{AppError, AppResult},
    settings::save_settings,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportedModelInfo {
    pub model_id: String,
    pub name: String,
    pub local_path: String,
    pub file_size_mb: u64,
    pub estimated_vram_gb: f64,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstalledModelInfo {
    pub model_id: String,
    pub name: String,
    pub local_path: String,
    pub file_size_mb: u64,
    pub source: String,
    pub is_current: bool,
    /// True only for a builtin model whose recorded version marker differs from
    /// the current catalog `sha256`. Unknown (no marker) counts as up to date.
    pub update_available: bool,
    /// 该模型是否自带**微调包**（旁车配置 `<主名>.launch.json`，见
    /// `model_sidecar.rs`）。为真表示启动时会自动挂上 LoRA 适配器并套用
    /// 与训练一致的对话模板 —— 界面据此打「微调包」角标，让「能力存在」
    /// 同时也是「看得见」。只判断文件在不在，不解析内容（写坏了由启动时报错）。
    #[serde(rename = "has_finetune_bundle")]
    pub has_finetune_bundle: bool,
}

/// Sidecar written next to a downloaded model recording which catalog version is
/// on disk, so we can later tell whether an update is available.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstalledVersion {
    pub sha256: String,
    pub file_name: String,
    pub installed_at: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub id: String,
    pub name: String,
    pub author: String,
    pub base_model: String,
    pub display_compression: String,
    pub quant_level: String,
    pub quant_type: String,
    pub quant_bpp: f64,
    pub parameter_count: String,
    pub parameters_raw: u64,
    pub active_parameters: Option<u64>,
    pub is_moe: bool,
    pub vram_required_gb: f64,
    pub file_size_mb: u64,
    pub context_length: u64,
    pub recommended_scene: Vec<Scene>,
    pub chinese_score: u8,
    pub speed_rating: u8,
    pub architecture: String,
    pub model_format: String,
    pub release_date: String,
    pub download_url: String,
    pub mirror_urls: Vec<String>,
    pub sha256: String,
    pub supported_gpu: Vec<String>,
    pub license: String,
    pub is_installed: bool,
    pub local_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Scene {
    /// Casual chat and Q&A: fast, lightweight, low memory.
    Chat,
    /// Long-form Chinese writing: reports, articles, structured output.
    Writing,
    /// Code generation, explanation, debugging.
    Coding,
    /// Catch-all when no specific scene fits; falls back to general-purpose scoring.
    General,
}

impl Scene {
    pub fn label(&self) -> &'static str {
        match self {
            Scene::Chat => "日常聊天问答",
            Scene::Writing => "写文案/写报告",
            Scene::Coding => "写代码",
            Scene::General => "通用场景",
        }
    }
}

#[tauri::command]
pub fn list_installed_models(
    state: tauri::State<'_, AppState>,
) -> AppResult<Vec<InstalledModelInfo>> {
    let (install_dir, current_path, db_path) = {
        let inner = state.inner.lock().expect("state lock poisoned");
        (
            inner.settings.install_dir.clone(),
            inner.settings.local_model_path.clone(),
            inner.db_path.clone(),
        )
    };

    let mut seen = HashSet::new();
    let mut models = Vec::new();
    let models_dir = install_dir.join("models");
    collect_installed_models(&install_dir, &models_dir, &current_path, &mut seen, &mut models)?;

    if let Some(path) = current_path.as_ref() {
        if path.exists()
            && path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("gguf"))
                .unwrap_or(false)
        {
            push_installed_model(&install_dir, path, &current_path, &mut seen, &mut models)?;
        }
    }

    // Imported models tracked in SQLite that the file scan misses (they live
    // outside install_dir). Skip any whose file has since been removed.
    for row in crate::model_repo::list_imported_models(&db_path).unwrap_or_default() {
        let path = PathBuf::from(&row.local_path);
        if !path.exists() {
            continue;
        }
        if !seen.insert(path_key(&path)) {
            continue;
        }
        let is_current = current_path
            .as_ref()
            .map(|current| path_key(current) == path_key(&path))
            .unwrap_or(false);
        let has_finetune_bundle = crate::model_sidecar::有旁车配置(&path);
        models.push(InstalledModelInfo {
            model_id: row.id,
            name: row.name,
            local_path: row.local_path,
            file_size_mb: row.file_size_mb,
            source: "imported".to_string(),
            is_current,
            update_available: false,
            has_finetune_bundle,
        });
    }

    models.sort_by(|a, b| {
        b.is_current
            .cmp(&a.is_current)
            .then_with(|| a.name.cmp(&b.name))
    });
    Ok(models)
}

#[tauri::command]
pub fn switch_installed_model(
    path: String,
    state: tauri::State<'_, AppState>,
) -> AppResult<InstalledModelInfo> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err(AppError::Message("请先选择要切换的模型。".to_string()));
    }

    let model_path = PathBuf::from(trimmed);
    validate_model_file(&model_path)?;

    let mut inner = state.inner.lock().expect("state lock poisoned");
    inner.settings.local_model_path = Some(model_path.clone());
    save_settings(&inner.data_dir, &inner.settings)?;
    let install_dir = inner.settings.install_dir.clone();
    let info = installed_model_info(&install_dir, &model_path, &inner.settings.local_model_path)?;
    tracing::info!(model_id = %info.model_id, source = %info.source, "switched current model");
    Ok(info)
}

fn collect_installed_models(
    install_dir: &Path,
    dir: &Path,
    current_path: &Option<PathBuf>,
    seen: &mut HashSet<String>,
    models: &mut Vec<InstalledModelInfo>,
) -> AppResult<()> {
    if !dir.exists() {
        return Ok(());
    }

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_installed_models(install_dir, &path, current_path, seen, models)?;
        } else if path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| extension.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false)
        {
            push_installed_model(install_dir, &path, current_path, seen, models)?;
        }
    }
    Ok(())
}

fn push_installed_model(
    install_dir: &Path,
    path: &Path,
    current_path: &Option<PathBuf>,
    seen: &mut HashSet<String>,
    models: &mut Vec<InstalledModelInfo>,
) -> AppResult<()> {
    let key = path_key(path);
    if !seen.insert(key) {
        return Ok(());
    }
    models.push(installed_model_info(install_dir, path, current_path)?);
    Ok(())
}

/// **导入模型的唯一 id 规则**（本项目「唯一事实来源」在命名上的落点）。
///
/// 🔴 **为什么必须只剩这一份**（P2-11 续，2026-10-16 实机抓出）：这条规则此前
/// **有两份** —— 文件扫描侧（`installed_model_info`，也是 UI 真正拿到的那个 id）
/// 写 `local-<主名>`，导入写库侧（`import_model_file`）写 `imported-<主名>`。
/// 两边看着都对，合起来就是：**UI 说 `local-Ornith-1.5-9B-Q8_0`，库里存
/// `imported-Ornith-1.5-9B-Q8_0`** ⇒ 「按 UI 给的 id 回库查元数据」必然落空
/// （P100 日志铁证：`switched current model model_id=local-Ornith-1.5-9B-Q8_0
/// source=imported`，而 `model_metadata` 里那一行是 `imported-…`）。
///
/// 代价不是「报个错」，而是**静默失效**：模型元数据恒为 `None` ⇒ 显存反解与
/// KV 精度在导入模型上一次都不生效 ⇒ 启动参数永远是兜底的 `24 层 / 4096`，
/// 与显存多少无关。P100 的 16 GB 上只用到 6424 MiB 就是这么来的。
///
/// 两份规则的典型后果就是「改一处、另一处不报错、只给错值」—— 所以两侧都改成
/// 调本函数，谁再想内联一个 `format!` 时先看一眼这里。
pub fn 导入模型标识(文件主名: &str) -> String {
    format!("local-{}", 文件主名.replace(' ', "-"))
}

fn installed_model_info(
    install_dir: &Path,
    path: &Path,
    current_path: &Option<PathBuf>,
) -> AppResult<InstalledModelInfo> {
    validate_model_file(path)?;
    let metadata = std::fs::metadata(path)?;
    let file_size_mb = (metadata.len() + 999_999) / 1_000_000;
    let matched = match_builtin_model(path);
    let fallback_name = path
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("本地模型文件")
        .to_string();
    let model_id = matched
        .as_ref()
        .map(|model| model.id.clone())
        .unwrap_or_else(|| 导入模型标识(&fallback_name));
    let name = matched
        .as_ref()
        .map(|model| model.name.clone())
        .unwrap_or(fallback_name);
    let source = if matched.is_some() {
        "builtin"
    } else {
        "imported"
    }
    .to_string();
    let update_available = matched
        .as_ref()
        .map(|model| is_update_available(install_dir, model))
        .unwrap_or(false);

    Ok(InstalledModelInfo {
        model_id,
        name,
        local_path: path.to_string_lossy().to_string(),
        file_size_mb,
        source,
        is_current: current_path
            .as_ref()
            .map(|current| path_key(current) == path_key(path))
            .unwrap_or(false),
        update_available,
        has_finetune_bundle: crate::model_sidecar::有旁车配置(path),
    })
}

fn validate_model_file(path: &Path) -> AppResult<()> {
    if !path.exists() || !path.is_file() {
        return Err(AppError::Message(
            "模型文件暂时找不到，请重新选择。".to_string(),
        ));
    }
    let is_gguf = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.eq_ignore_ascii_case("gguf"))
        .unwrap_or(false);
    if !is_gguf {
        return Err(AppError::Message(
            "当前只支持切换模型文件，请选择 .gguf 文件。".to_string(),
        ));
    }
    Ok(())
}

fn match_builtin_model(path: &Path) -> Option<ModelMetadata> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    let parent_name = path
        .parent()
        .and_then(|parent| parent.file_name())
        .and_then(|name| name.to_str())
        .unwrap_or("");

    builtin_models().into_iter().find(|model| {
        model.id == parent_name
            || model
                .download_url
                .split('?')
                .next()
                .and_then(|clean| clean.rsplit('/').next())
                .map(|name| name.eq_ignore_ascii_case(file_name))
                .unwrap_or(false)
    })
}

fn path_key(path: &Path) -> String {
    path.canonicalize()
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .to_ascii_lowercase()
}

#[tauri::command]
pub fn import_local_model(
    path: String,
    state: tauri::State<'_, AppState>,
) -> AppResult<ImportedModelInfo> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err(AppError::Message("请先选择本地模型文件。".to_string()));
    }

    let model_path = PathBuf::from(trimmed);
    validate_model_file(&model_path)?;

    let file_bytes = std::fs::metadata(&model_path)?.len();
    let file_size_mb = (file_bytes + 999_999) / 1_000_000;

    // 解析 GGUF 头部获取真实架构参数（层数 / KV 头数 / 原生上下文 / 参数量 / 量化档）。
    // 失败不阻断导入 —— 异常文件仍按保守估算处理，只是会丢掉这些信息。
    let gguf_metadata = match crate::gguf::parse_file(&model_path) {
        Ok(meta) => {
            tracing::info!(
                architecture = %meta.architecture,
                parameters = meta.parameter_count,
                context_length = ?meta.context_length,
                "parsed gguf metadata"
            );
            Some(meta)
        }
        Err(error) => {
            tracing::warn!(
                error = %error,
                "failed to parse gguf metadata; falling back to conservative estimate"
            );
            None
        }
    };

    // 导入模型这条路上没有缓存过的驱动模式，现查一次（nvidia-smi 一次调用，
    // 相比解析 GGUF 的耗时可以忽略）。查不到就按 WDDM 档 —— `显存开销_gib` 兜底。
    let 驱动模式 = crate::hardware::nvidia_driver_model();
    let estimated_vram_gb = estimate_vram_gb(
        file_bytes,
        gguf_metadata.as_ref(),
        crate::service::显存开销_gib(驱动模式.as_deref()),
    );
    let name = model_path
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("本地模型文件")
        .to_string();

    let mut inner = state.inner.lock().expect("state lock poisoned");
    inner.settings.local_model_path = Some(model_path.clone());
    save_settings(&inner.data_dir, &inner.settings)?;

    // 与文件扫描侧共用同一条规则（`导入模型标识`）：以前这里写 `imported-…`，
    // 而 UI 侧算出来的是 `local-…` ⇒ 按 UI 的 id 回库查不到这一行，
    // 启动参数就退回兜底的 24 层 / 4096（P2-11 续，2026-10-16 实机抓出）。
    let model_id = 导入模型标识(&name);
    let _ = crate::model_repo::upsert_imported_model(
        &inner.db_path,
        &model_id,
        &name,
        &model_path,
        file_size_mb,
        estimated_vram_gb,
        gguf_metadata.as_ref(),
    );
    tracing::info!(model_id = %model_id, file_size_mb, "imported local model file");

    Ok(ImportedModelInfo {
        model_id,
        name,
        local_path: model_path.to_string_lossy().to_string(),
        file_size_mb,
        estimated_vram_gb,
        message: match &gguf_metadata {
            Some(meta) => format!(
                "已导入本地模型（{}，原生上下文 {} tokens），预计需要约 {:.1} GB 显存。",
                meta.parameter_label().unwrap_or_else(|| "参数未知".to_string()),
                meta.context_length.unwrap_or(4096),
                estimated_vram_gb
            ),
            None => format!(
                "已导入本地模型文件，预计需要约 {:.1} GB 资源。",
                estimated_vram_gb
            ),
        },
    })
}

/// 估算模型跑起来需要多少显存（GB）：权重 + 运行时开销 + 默认上下文下的 KV。
///
/// 此前只有文件大小时用的是 `文件大小 × 1.1 + 0.5`：这会低估未量化模型的权重，
/// 又会把混合注意力模型的 KV 高估数倍。有 GGUF 结构后按真实层结构计算。
///
/// `开销_gib` 由调用方按驱动模式给出（`service::显存开销_gib`）—— TCC 卡不跑
/// 桌面，开销比 WDDM 小 1.2 GiB，写死一个数会让某一端多报/少报。
fn estimate_vram_gb(file_bytes: u64, gguf: Option<&crate::gguf::GgufMetadata>, 开销_gib: f64) -> f64 {
    // 显存一律按 GiB（1024³）计，与 `nvidia-smi` 的口径一致；用 10⁹ 会与
    // 可用显存相减时引入约 7% 的系统偏差。
    let weight_gb = file_bytes as f64 / crate::gguf::每GiB字节;
    // 计算缓冲 + 驱动与运行库占用。与启动规划共用同一个来源 —— 这里曾经
    // 也硬编码了一份 1.3，两处会各自漂移（实测需 1.84，反推过程见
    // `service.rs` 的 `GPU_OVERHEAD_GB` 注释）。
    let overhead_gb = 开销_gib;
    // 以 8K 上下文作为「能不能跑」的门槛；启动时再按可用显存求最大上下文。
    let kv_gb = gguf.map(|meta| meta.kv_gb(8192, 2.0)).unwrap_or(0.5);
    let total = weight_gb + overhead_gb + kv_gb;
    (total * 10.0).round() / 10.0
}

/// 内置模型清单（下载来源）。
///
/// `parameters_raw` / `file_size_mb` / `vram_required_gb` / `quant_bpp` 都是
/// **下载前的估算**，用于推荐过滤与「能不能跑」的门槛；模型下载完成后会被
/// 真实文件大小与 GGUF 元数据覆盖（见 `model_repo::upsert_imported_model`）。
///
/// 数值来源（2026-09-24 校准，实测表见 CLAUDE.md）：
/// - **Qwen3-8B / Qwen3-1.7B** —— 本机真实文件实测，字段按实测校准。
/// - **Qwen3-14B / Qwen3-4B** —— 本机没有文件。`quant_bpp` 取同架构同量化档
///   （Q4_K_M）的实测值 0.614；`context_length` 取 Qwen3 系列实测的 40960。
/// - **Ornith-1.0-9B** —— 本机只有 1.5 版，版本与架构都不同，**不可外推**，
///   因此保持原值，等拿到 1.0 的文件再校准。
///
/// 注意：每条的 `sha256` 必须与 `manifests/signed_manifest.json` 里的记录
/// 一致，改错会让下载校验失败 —— 动这些字段前先确认两者同源。
pub fn builtin_models() -> Vec<ModelMetadata> {
    vec![
        ModelMetadata {
            id: "qwen3_14b_chat".to_string(),
            name: "Qwen3 14B 全能旗舰版".to_string(),
            author: "Qwen".to_string(),
            base_model: "Qwen3-14B".to_string(),
            display_compression: "普通压缩版".to_string(),
            quant_level: "Q4_K_M".to_string(),
            quant_type: "GGUF/Q4_K_M".to_string(),
            quant_bpp: 0.614,
            parameter_count: "14B".to_string(),
            parameters_raw: 14_800_000_000,
            active_parameters: None,
            is_moe: false,
            vram_required_gb: 10.5,
            file_size_mb: 9002,
            context_length: 40960,
            recommended_scene: vec![Scene::Coding, Scene::Writing, Scene::General],
            chinese_score: 95,
            speed_rating: 2,
            architecture: "Qwen3".to_string(),
            model_format: "gguf".to_string(),
            release_date: "2025-04-29".to_string(),
            download_url: "https://www.modelscope.cn/models/Qwen/Qwen3-14B-GGUF/resolve/master/Qwen3-14B-Q4_K_M.gguf".to_string(),
            mirror_urls: vec!["https://hf-mirror.com/Qwen/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q4_K_M.gguf".to_string()],
            sha256: "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0".to_string(),
            supported_gpu: vec!["NVIDIA".to_string(), "CPU".to_string()],
            license: "Apache-2.0".to_string(),
            is_installed: false,
            local_path: None,
        },
        ModelMetadata {
            id: "qwen3_8b_chat".to_string(),
            name: "Qwen3 8B 均衡版".to_string(),
            author: "Qwen".to_string(),
            base_model: "Qwen3-8B".to_string(),
            display_compression: "普通压缩版".to_string(),
            quant_level: "Q4_K_M".to_string(),
            quant_type: "GGUF/Q4_K_M".to_string(),
            // 实测：文件 5,027,783,488 字节 / 8.2B 参数 → bpp 0.614（原记 0.4，
            // 是 Q4_K_M 的理论值，按它算权重会低估 1.53×）。
            quant_bpp: 0.614,
            parameter_count: "8B".to_string(),
            parameters_raw: 8_200_000_000,
            active_parameters: None,
            is_moe: false,
            // 4.68(权重) + 1.3(保留) + 1.13(KV@8K) ≈ 7.1 GiB
            vram_required_gb: 7.1,
            file_size_mb: 5028,
            context_length: 40960,
            recommended_scene: vec![Scene::Writing, Scene::General, Scene::Chat],
            chinese_score: 93,
            speed_rating: 3,
            architecture: "Qwen3".to_string(),
            model_format: "gguf".to_string(),
            release_date: "2025-04-29".to_string(),
            download_url: "https://www.modelscope.cn/models/Qwen/Qwen3-8B-GGUF/resolve/master/Qwen3-8B-Q4_K_M.gguf".to_string(),
            mirror_urls: vec!["https://hf-mirror.com/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf".to_string()],
            sha256: "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785".to_string(),
            supported_gpu: vec!["NVIDIA".to_string(), "CPU".to_string()],
            license: "Apache-2.0".to_string(),
            is_installed: false,
            local_path: None,
        },
        ModelMetadata {
            id: "qwen3_4b_chat".to_string(),
            name: "Qwen3 4B 轻快版".to_string(),
            author: "Qwen".to_string(),
            base_model: "Qwen3-4B".to_string(),
            display_compression: "普通压缩版".to_string(),
            quant_level: "Q4_K_M".to_string(),
            quant_type: "GGUF/Q4_K_M".to_string(),
            quant_bpp: 0.614,
            parameter_count: "4B".to_string(),
            parameters_raw: 4_000_000_000,
            active_parameters: None,
            is_moe: false,
            vram_required_gb: 3.4,
            file_size_mb: 2497,
            context_length: 40960,
            recommended_scene: vec![Scene::Chat, Scene::General],
            chinese_score: 90,
            speed_rating: 4,
            architecture: "Qwen3".to_string(),
            model_format: "gguf".to_string(),
            release_date: "2025-04-29".to_string(),
            download_url: "https://www.modelscope.cn/models/Qwen/Qwen3-4B-GGUF/resolve/master/Qwen3-4B-Q4_K_M.gguf".to_string(),
            mirror_urls: vec!["https://hf-mirror.com/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf".to_string()],
            sha256: "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5".to_string(),
            supported_gpu: vec!["NVIDIA".to_string(), "CPU".to_string()],
            license: "Apache-2.0".to_string(),
            is_installed: false,
            local_path: None,
        },
        ModelMetadata {
            id: "qwen3_1_7b_light".to_string(),
            name: "Qwen3 1.7B 迷你版".to_string(),
            author: "Qwen".to_string(),
            base_model: "Qwen3-1.7B".to_string(),
            display_compression: "高质量压缩版".to_string(),
            quant_level: "Q8_0".to_string(),
            quant_type: "GGUF/Q8_0".to_string(),
            // 实测：文件 1834 MB / 1.721B 参数 → bpp 1.066（Q8_0）。
            quant_bpp: 1.066,
            parameter_count: "1.7B".to_string(),
            parameters_raw: 1_720_000_000,
            active_parameters: None,
            is_moe: false,
            // 1.71(权重) + 1.3(保留) + 0.88(KV@8K) ≈ 3.9 GiB
            vram_required_gb: 3.9,
            file_size_mb: 1834,
            context_length: 40960,
            recommended_scene: vec![Scene::Chat],
            chinese_score: 84,
            speed_rating: 5,
            architecture: "Qwen3".to_string(),
            model_format: "gguf".to_string(),
            release_date: "2025-04-29".to_string(),
            download_url: "https://www.modelscope.cn/models/Qwen/Qwen3-1.7B-GGUF/resolve/master/Qwen3-1.7B-Q8_0.gguf".to_string(),
            mirror_urls: vec!["https://hf-mirror.com/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf".to_string()],
            sha256: "061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a".to_string(),
            supported_gpu: vec!["NVIDIA".to_string(), "CPU".to_string()],
            license: "Apache-2.0".to_string(),
            is_installed: false,
            local_path: None,
        },
        ModelMetadata {
            id: "ornith_1_0_9b".to_string(),
            name: "Ornith-1.0 9B 增强版".to_string(),
            author: "Ornith".to_string(),
            base_model: "Ornith-1.0-9B".to_string(),
            display_compression: "普通压缩版".to_string(),
            quant_level: "Q4_K_M".to_string(),
            quant_type: "GGUF/Q4_K_M".to_string(),
            quant_bpp: 0.5,
            parameter_count: "9B".to_string(),
            parameters_raw: 9_500_000_000,
            active_parameters: None,
            is_moe: false,
            vram_required_gb: 5.5,
            file_size_mb: 5367,
            context_length: 32768,
            // Ornith is a Qwen3-8B fine-tune positioned as a general-purpose
            // chat/write/coding model. Slot it between 8B (light) and 14B
            // (top) on the 8GB tier — fills the mid-range gap.
            recommended_scene: vec![Scene::General, Scene::Writing, Scene::Coding],
            chinese_score: 91,
            speed_rating: 3,
            architecture: "Qwen3".to_string(),
            model_format: "gguf".to_string(),
            release_date: "2025-07-01".to_string(),
            download_url: "https://www.modelscope.cn/models/deepreinforce-ai/Ornith-1.0-9B-GGUF/file/view/master/ornith-1.0-9b-Q4_K_M.gguf?status=2".to_string(),
            mirror_urls: vec!["https://hf-mirror.com/deepreinforce-ai/Ornith-1.0-9B-GGUF/resolve/main/ornith-1.0-9b-Q4_K_M.gguf".to_string()],
            sha256: "5720d1f671b4996481274fffe01868c3c36e87c135cc8538471cc7bd6087b106".to_string(),
            supported_gpu: vec!["NVIDIA".to_string(), "CPU".to_string()],
            license: "Apache-2.0".to_string(),
            is_installed: false,
            local_path: None,
        },
    ]
}

/// Local file name a model downloads to, derived from its download URL.
pub fn model_file_name(model: &ModelMetadata) -> String {
    url_file_name(&model.download_url).unwrap_or_else(|| format!("{}.gguf", model.id))
}

pub fn url_file_name(url: &str) -> Option<String> {
    url.split('?')
        .next()
        .and_then(|clean| clean.rsplit('/').next())
        .filter(|name| !name.is_empty())
        .map(|name| name.to_string())
}

/// Where a builtin model is stored once downloaded:
/// `install_dir/models/{id}/{file_name}`.
pub fn model_install_path(install_dir: &Path, model: &ModelMetadata) -> PathBuf {
    install_dir
        .join("models")
        .join(&model.id)
        .join(model_file_name(model))
}

/// A model counts as installed only when its file exists and is non-empty.
pub fn is_model_installed(install_dir: &Path, model: &ModelMetadata) -> bool {
    std::fs::metadata(model_install_path(install_dir, model))
        .map(|meta| meta.is_file() && meta.len() > 0)
        .unwrap_or(false)
}

/// Where the version marker sidecar lives: alongside the model file.
pub fn version_marker_path(install_dir: &Path, model_id: &str) -> PathBuf {
    install_dir.join("models").join(model_id).join("version.json")
}

/// Record which catalog version is installed on disk. Called after a successful
/// download or update so `update_available` can be computed later.
pub fn write_installed_version(install_dir: &Path, model: &ModelMetadata) -> AppResult<()> {
    write_version_marker(&install_dir.join("models").join(&model.id), model)
}

/// Write the version marker into an already-known model directory (the parent of
/// the `.gguf` file). Used by the download path where the dir is already at hand.
pub fn write_version_marker(model_dir: &Path, model: &ModelMetadata) -> AppResult<()> {
    std::fs::create_dir_all(model_dir)?;
    let marker = InstalledVersion {
        sha256: model.sha256.clone(),
        file_name: model_file_name(model),
        installed_at: chrono::Utc::now().to_rfc3339(),
    };
    std::fs::write(
        model_dir.join("version.json"),
        serde_json::to_string_pretty(&marker)?,
    )?;
    Ok(())
}

pub fn read_installed_version(install_dir: &Path, model_id: &str) -> Option<InstalledVersion> {
    let content = std::fs::read_to_string(version_marker_path(install_dir, model_id)).ok()?;
    serde_json::from_str(&content).ok()
}

/// True when a recorded marker exists and its hash differs from the catalog's.
/// Missing marker = unknown provenance, treated as up to date to avoid nagging.
pub fn is_update_available(install_dir: &Path, model: &ModelMetadata) -> bool {
    read_installed_version(install_dir, &model.id)
        .map(|marker| !crate::hash::matches(&model.sha256, &marker.sha256))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn catalog_does_not_hardcode_install_state() {
        assert!(builtin_models().iter().all(|model| !model.is_installed));
    }

    /// **跨语言边界护栏**：`InstalledModelInfo` 的字段名就是前端读的键名。
    /// 这条断言把「模型是否自带微调包」这个新字段的**线上名字**钉死 ——
    /// 本项目踩过「改了字段名、前端一个分支都进不去却完全不报错」，所以
    /// 跨边界的键名必须有字面量断言兜着。
    #[test]
    fn installed_model_info_wire_keys_are_pinned() {
        let 信息 = InstalledModelInfo {
            model_id: "m".to_string(),
            name: "名字".to_string(),
            local_path: "C:/x.gguf".to_string(),
            file_size_mb: 1,
            source: "imported".to_string(),
            is_current: false,
            update_available: false,
            has_finetune_bundle: true,
        };
        let 文本 = serde_json::to_string(&信息).expect("结构体应能序列化");
        assert!(
            文本.contains("\"has_finetune_bundle\":true"),
            "「微调包」角标靠这个键名，改了就静默失效。实际：{文本}"
        );
        assert!(文本.contains("\"is_current\":false"), "实际：{文本}");
        assert!(文本.contains("\"update_available\":false"), "实际：{文本}");
    }

    #[test]
    fn url_file_name_extracts_basename() {
        assert_eq!(
            url_file_name("https://host/path/model-q4.gguf").as_deref(),
            Some("model-q4.gguf")
        );
        assert_eq!(
            url_file_name("https://host/path/model.gguf?download=1").as_deref(),
            Some("model.gguf")
        );
        assert_eq!(url_file_name("https://host/path/"), None);
    }

    #[test]
    fn install_detection_requires_nonempty_file() {
        let install_dir =
            std::env::temp_dir().join(format!("localai-models-test-{}", uuid_like()));
        let model = builtin_models().into_iter().next().unwrap();

        assert!(!is_model_installed(&install_dir, &model));

        let path = model_install_path(&install_dir, &model);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        // Empty file must NOT count as installed.
        std::fs::File::create(&path).unwrap();
        assert!(!is_model_installed(&install_dir, &model));

        // Non-empty file counts as installed.
        {
            let mut file = std::fs::File::create(&path).unwrap();
            file.write_all(b"gguf-bytes").unwrap();
        }
        assert!(is_model_installed(&install_dir, &model));

        let _ = std::fs::remove_dir_all(&install_dir);
    }

    #[test]
    fn version_marker_round_trips_and_flags_updates() {
        let install_dir =
            std::env::temp_dir().join(format!("localai-version-test-{}", uuid_like()));
        let model = builtin_models().into_iter().next().unwrap();

        // No marker yet: unknown provenance is treated as up to date.
        assert!(read_installed_version(&install_dir, &model.id).is_none());
        assert!(!is_update_available(&install_dir, &model));

        // After recording the current version, no update is available.
        write_installed_version(&install_dir, &model).unwrap();
        let marker = read_installed_version(&install_dir, &model.id).unwrap();
        assert_eq!(marker.sha256, model.sha256);
        assert!(!is_update_available(&install_dir, &model));

        // A catalog entry with a different hash makes the on-disk copy stale.
        let mut newer = model.clone();
        newer.sha256 =
            "0000000000000000000000000000000000000000000000000000000000000000".to_string();
        assert!(is_update_available(&install_dir, &newer));

        let _ = std::fs::remove_dir_all(&install_dir);
    }

    /// **id 规则唯一性护栏**（P2-11 续，2026-10-16）。
    ///
    /// 文件扫描侧算出来的 id（也就是 UI 真正拿到的那个）必须**等于**
    /// `导入模型标识` 的输出 —— 导入写库侧走的是同一个函数，两侧一致，
    /// 「按 UI 的 id 回库查元数据」才可能命中。
    ///
    /// 这条测试存在的唯一理由：上一轮两侧各写各的（`local-…` vs `imported-…`），
    /// 所有单测全绿、两台实机上却静默失效（启动参数永远是 24 层 / 4096）。
    #[test]
    fn 扫描侧与导入侧共用同一条id规则() {
        // ① 规则本身：真实机器上的名字（P100 的 Ornith-1.5-9B-Q8_0）与含空格的写法
        assert_eq!(
            导入模型标识("Ornith-1.5-9B-Q8_0"),
            "local-Ornith-1.5-9B-Q8_0"
        );
        assert_eq!(导入模型标识("我的 模型"), "local-我的-模型");

        // ② 扫描侧真的调了它（不只是一个漂亮函数的单元测试）
        let install_dir = std::env::temp_dir().join(format!("localai-idrule-{}", uuid_like()));
        let model_dir = install_dir.join("models").join("我的自定义目录");
        std::fs::create_dir_all(&model_dir).unwrap();
        let path = model_dir.join("我的 模型.gguf");
        std::fs::write(&path, b"GGUF").unwrap();

        let 信息 = installed_model_info(&install_dir, &path, &None).unwrap();
        assert_eq!(信息.model_id, 导入模型标识("我的 模型"), "扫描侧没走唯一规则");
        assert_eq!(信息.model_id, "local-我的-模型");
        assert_eq!(信息.source, "imported");

        let _ = std::fs::remove_dir_all(&install_dir);
    }

    fn uuid_like() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    }
}
