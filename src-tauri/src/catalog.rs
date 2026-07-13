use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{
    app_state::AppState,
    error::AppResult,
    models::{builtin_models, ModelMetadata},
};

/// Remote model manifest: a monotonically increasing `version` plus the full
/// model list. Distributed as a signed-in-future JSON; for now HTTPS + version
/// gating is the integrity story, matching the bundled-fallback design.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteCatalog {
    pub version: u32,
    pub models: Vec<ModelMetadata>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CatalogRefreshResult {
    pub updated: bool,
    pub version: u32,
    pub model_count: usize,
    pub message: String,
}

/// Where the online manifest is fetched from. Left as an unconfigured placeholder
/// until a real signed endpoint is published; `refresh_model_catalog` no-ops
/// gracefully while it points here (same pattern as unconfigured download URLs).
const CATALOG_URL: &str = "https://catalog-not-configured.local/models.json";

fn cache_path(data_dir: &Path) -> PathBuf {
    data_dir.join("catalog.json")
}

fn is_configured(url: &str) -> bool {
    url.starts_with("https://") && !url.contains("not-configured")
}

/// The catalog the app should actually use: the cached online manifest merged
/// over the bundled models, or just the bundled models when no valid cache
/// exists. Never fails — falls back to bundled on any read/parse error.
pub fn effective_models(data_dir: &Path) -> Vec<ModelMetadata> {
    let builtin = builtin_models();
    match read_cached(data_dir) {
        Some(remote) if !remote.models.is_empty() => merge_catalogs(builtin, remote.models),
        _ => builtin,
    }
}

fn read_cached(data_dir: &Path) -> Option<RemoteCatalog> {
    let content = std::fs::read_to_string(cache_path(data_dir)).ok()?;
    serde_json::from_str::<RemoteCatalog>(&content).ok()
}

/// Merge a remote catalog over the bundled one: remote entries override bundled
/// entries with the same id (in place), remote-only entries are appended, and
/// bundled-only entries are preserved. Order is stable for the UI.
fn merge_catalogs(builtin: Vec<ModelMetadata>, remote: Vec<ModelMetadata>) -> Vec<ModelMetadata> {
    let mut merged: Vec<ModelMetadata> = builtin
        .into_iter()
        .map(|model| {
            remote
                .iter()
                .find(|candidate| candidate.id == model.id)
                .cloned()
                .unwrap_or(model)
        })
        .collect();
    for candidate in remote {
        if !merged.iter().any(|model| model.id == candidate.id) {
            merged.push(candidate);
        }
    }
    merged
}

/// Fetch the online manifest, and cache it when it's newer than what we have.
/// Best-effort: returns a friendly message (never a hard error) so a missing
/// endpoint or offline machine can't block startup.
#[tauri::command]
pub async fn refresh_model_catalog(
    state: tauri::State<'_, AppState>,
) -> AppResult<CatalogRefreshResult> {
    let (data_dir, db_path) = {
        let inner = state.inner.lock().expect("state lock poisoned");
        (inner.data_dir.clone(), inner.db_path.clone())
    };

    if !is_configured(CATALOG_URL) {
        let current = read_cached(&data_dir).map(|c| c.version).unwrap_or(0);
        return Ok(CatalogRefreshResult {
            updated: false,
            version: current,
            model_count: builtin_models().len(),
            message: "在线模型清单地址尚未配置，当前使用随软件内置的清单。".to_string(),
        });
    }

    let cached_version = read_cached(&data_dir).map(|c| c.version).unwrap_or(0);
    match fetch_remote(CATALOG_URL).await {
        Ok(remote) if remote.version > cached_version && !remote.models.is_empty() => {
            let count = remote.models.len();
            let version = remote.version;
            write_cache(&data_dir, &remote)?;
            // Re-seed the DB catalog so downstream reads see the new models.
            let _ = crate::model_repo::seed_catalog(&db_path, &effective_models(&data_dir));
            tracing::info!(version, model_count = count, "online model catalog updated");
            Ok(CatalogRefreshResult {
                updated: true,
                version,
                model_count: count,
                message: format!("已更新到最新模型清单（含 {} 个模型）。", count),
            })
        }
        Ok(remote) => Ok(CatalogRefreshResult {
            updated: false,
            version: cached_version.max(remote.version),
            model_count: effective_models(&data_dir).len(),
            message: "模型清单已是最新。".to_string(),
        }),
        Err(_) => {
            tracing::warn!("failed to fetch online model catalog; continuing with local catalog");
            Ok(CatalogRefreshResult {
                updated: false,
                version: cached_version,
                model_count: effective_models(&data_dir).len(),
                message: "暂时没能获取在线模型清单，继续使用本地清单。".to_string(),
            })
        }
    }
}
async fn fetch_remote(url: &str) -> AppResult<RemoteCatalog> {
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(8))
        .timeout(std::time::Duration::from_secs(20))
        .build()
        .map_err(|_| crate::error::AppError::Message("网络组件暂时不可用。".to_string()))?;
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|_| crate::error::AppError::Message("无法连接在线模型清单。".to_string()))?;
    if !response.status().is_success() {
        return Err(crate::error::AppError::Message(
            "在线模型清单暂时不可用。".to_string(),
        ));
    }
    let text = response
        .text()
        .await
        .map_err(|_| crate::error::AppError::Message("无法读取在线模型清单内容。".to_string()))?;
    let catalog = serde_json::from_str::<RemoteCatalog>(&text)
        .map_err(|_| crate::error::AppError::Message("在线模型清单格式无法识别。".to_string()))?;
    Ok(catalog)
}

fn write_cache(data_dir: &Path, catalog: &RemoteCatalog) -> AppResult<()> {
    std::fs::create_dir_all(data_dir)?;
    let content = serde_json::to_string_pretty(catalog)?;
    std::fs::write(cache_path(data_dir), content)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_overrides_and_appends_by_id() {
        let builtin = builtin_models();
        let first_id = builtin[0].id.clone();

        // A remote entry overriding the first builtin model plus a brand-new one.
        let mut overridden = builtin[0].clone();
        overridden.name = "改名后的模型".to_string();
        let mut newcomer = builtin[0].clone();
        newcomer.id = "remote-only-model".to_string();
        newcomer.name = "全新在线模型".to_string();

        let merged = merge_catalogs(builtin.clone(), vec![overridden, newcomer]);

        // Same count + 1 new; the override replaced in place, new one appended.
        assert_eq!(merged.len(), builtin.len() + 1);
        assert_eq!(
            merged.iter().find(|m| m.id == first_id).unwrap().name,
            "改名后的模型"
        );
        assert_eq!(merged.last().unwrap().id, "remote-only-model");
    }

    #[test]
    fn placeholder_url_is_not_configured() {
        assert!(!is_configured(CATALOG_URL));
        assert!(!is_configured("http://example.com/x.json"));
        assert!(is_configured("https://real.example.com/models.json"));
    }
}
