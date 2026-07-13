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
    Chat,
    Writing,
    Coding,
    Knowledge,
    Roleplay,
}

impl Scene {
    pub fn label(&self) -> &'static str {
        match self {
            Scene::Chat => "日常聊天问答",
            Scene::Writing => "写文案/写报告",
            Scene::Coding => "写代码",
            Scene::Knowledge => "本地知识库",
            Scene::Roleplay => "角色扮演",
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
        models.push(InstalledModelInfo {
            model_id: row.id,
            name: row.name,
            local_path: row.local_path,
            file_size_mb: row.file_size_mb,
            source: "imported".to_string(),
            is_current,
            update_available: false,
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
        .unwrap_or_else(|| format!("local-{}", fallback_name.replace(' ', "-")));
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

    let metadata = std::fs::metadata(&model_path)?;
    let file_size_mb = (metadata.len() + 999_999) / 1_000_000;
    let estimated_vram_gb = ((file_size_mb as f64 / 1000.0) * 1.1 + 0.5) * 10.0;
    let estimated_vram_gb = estimated_vram_gb.round() / 10.0;
    let name = model_path
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("本地模型文件")
        .to_string();

    let mut inner = state.inner.lock().expect("state lock poisoned");
    inner.settings.local_model_path = Some(model_path.clone());
    save_settings(&inner.data_dir, &inner.settings)?;

    let model_id = format!("imported-{}", name.replace(' ', "-"));
    let _ = crate::model_repo::upsert_imported_model(
        &inner.db_path,
        &model_id,
        &name,
        &model_path,
        file_size_mb,
        estimated_vram_gb,
    );
    tracing::info!(model_id = %model_id, file_size_mb, "imported local model file");

    Ok(ImportedModelInfo {
        model_id,
        name,
        local_path: model_path.to_string_lossy().to_string(),
        file_size_mb,
        estimated_vram_gb,
        message: format!(
            "已导入本地模型文件，预计需要约 {:.1} GB 资源。",
            estimated_vram_gb
        ),
    })
}

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
            quant_bpp: 0.4,
            parameter_count: "14B".to_string(),
            parameters_raw: 14_800_000_000,
            active_parameters: None,
            is_moe: false,
            vram_required_gb: 10.5,
            file_size_mb: 9002,
            context_length: 32768,
            recommended_scene: vec![Scene::Chat, Scene::Writing, Scene::Knowledge, Scene::Roleplay, Scene::Coding],
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
            quant_bpp: 0.4,
            parameter_count: "8B".to_string(),
            parameters_raw: 8_200_000_000,
            active_parameters: None,
            is_moe: false,
            vram_required_gb: 6.0,
            file_size_mb: 5028,
            context_length: 32768,
            recommended_scene: vec![Scene::Chat, Scene::Writing, Scene::Knowledge, Scene::Roleplay, Scene::Coding],
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
            quant_bpp: 0.4,
            parameter_count: "4B".to_string(),
            parameters_raw: 4_000_000_000,
            active_parameters: None,
            is_moe: false,
            vram_required_gb: 3.4,
            file_size_mb: 2497,
            context_length: 32768,
            recommended_scene: vec![Scene::Chat, Scene::Writing, Scene::Roleplay, Scene::Coding],
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
            quant_bpp: 0.9,
            parameter_count: "1.7B".to_string(),
            parameters_raw: 1_720_000_000,
            active_parameters: None,
            is_moe: false,
            vram_required_gb: 2.4,
            file_size_mb: 1834,
            context_length: 32768,
            recommended_scene: vec![Scene::Chat, Scene::Roleplay],
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

    fn uuid_like() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    }
}
