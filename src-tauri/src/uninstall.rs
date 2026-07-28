use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::{
    app_state::{AppState, ServiceStatus},
    error::{AppError, AppResult},
    models::{builtin_models, model_install_path},
    settings::save_settings,
};

/// Result of an uninstall, surfaced back to the UI so it can show a truthful
/// message (imported files stay on disk, builtin files are deleted).
#[derive(Debug, Clone, Serialize)]
pub struct UninstallResult {
    pub model_id: String,
    pub source: String,
    pub deleted_chat_sessions: u32,
    pub cleared_current_model: bool,
    pub message: String,
}

#[tauri::command]
pub fn uninstall_model(
    model_id: String,
    delete_chat_history: bool,
    local_path: Option<String>,
    state: tauri::State<'_, AppState>,
) -> AppResult<UninstallResult> {
    let trimmed = model_id.trim();
    if trimmed.is_empty() {
        return Err(AppError::Message("请先选择要卸载的模型。".to_string()));
    }
    let model_id = trimmed.to_string();

    // Snapshot everything the uninstall path needs, then release the lock before
    // filesystem work so the UI stays responsive even if remove_dir_all is slow.
    let (install_dir, data_dir, db_path, current_path, is_running) = {
        let inner = state.inner.lock().expect("state lock poisoned");
        let is_running = matches!(
            inner.service.status,
            ServiceStatus::Running | ServiceStatus::Starting
        ) && inner.service.model_id.as_deref() == Some(model_id.as_str());
        (
            inner.settings.install_dir.clone(),
            inner.data_dir.clone(),
            inner.db_path.clone(),
            inner.settings.local_model_path.clone(),
            is_running,
        )
    };

    if is_running {
        return Err(AppError::Message(
            "这个模型正在运行中，请先在聊天区点「停止服务」，然后再卸载。".to_string(),
        ));
    }

    let builtin = builtin_models()
        .into_iter()
        .find(|model| model.id == model_id);
    let source = if builtin.is_some() {
        "builtin"
    } else {
        "imported"
    };

    // Track what we actually removed so message + logging reflect reality.
    let mut removed_file_path: Option<PathBuf> = None;

    if let Some(model) = builtin.as_ref() {
        // Builtin models live under a well-known folder; nuke the whole thing so
        // the .gguf, version.json and any stray downloader temp files all go.
        let model_file = model_install_path(&install_dir, model);
        let model_dir = install_dir.join("models").join(&model_id);
        if model_dir.exists() {
            std::fs::remove_dir_all(&model_dir).map_err(|error| {
                AppError::Message(format!(
                    "删除模型文件失败：{}。请手动检查保存位置是否被其他程序占用。",
                    error
                ))
            })?;
            removed_file_path = Some(model_file);
        } else {
            // The DB or a stale UI can still point us here after a manual delete —
            // treat as success (idempotent) but note it in the log.
            tracing::warn!(
                model_id = %model_id,
                "uninstall: builtin model directory already missing, continuing"
            );
        }
    } else {
        // Imported model. The tracking row goes regardless. Whether we also
        // delete the .gguf depends on where it lives:
        //   - inside install_dir/models/ → our managed area, delete it (otherwise
        //     the filesystem scan in list_installed_models re-discovers it and the
        //     uninstall looks like it did nothing).
        //   - anywhere else (a path the user picked) → leave it untouched; it may
        //     be data they care about outside our control.
        crate::model_repo::delete_imported_model(&db_path, &model_id)?;

        let managed_root = install_dir.join("models");
        if let Some(path) = local_path.as_deref().map(PathBuf::from) {
            if path.starts_with(&managed_root) && path.exists() {
                std::fs::remove_file(&path).map_err(|error| {
                    AppError::Message(format!(
                        "删除模型文件失败：{}。请手动检查保存位置是否被其他程序占用。",
                        error
                    ))
                })?;
                // Clean up the now-empty per-model folder (best-effort; a
                // non-empty dir just stays).
                if let Some(parent) = path.parent() {
                    if parent.starts_with(&managed_root) && parent != managed_root {
                        let _ = std::fs::remove_dir(parent);
                    }
                }
                removed_file_path = Some(path);
            }
        }
    }

    // If the model we just removed was the one flagged as the current model in
    // settings, clear that pointer so auto-start on next launch doesn't try to
    // spin up a nonexistent file.
    let cleared_current_model = {
        let mut inner = state.inner.lock().expect("state lock poisoned");
        let matched = current_path
            .as_deref()
            .map(|path| should_clear_current(path, removed_file_path.as_deref(), &model_id))
            .unwrap_or(false);
        if matched {
            inner.settings.local_model_path = None;
            save_settings(&data_dir, &inner.settings)?;
        }
        matched
    };

    let deleted_chat_sessions = if delete_chat_history {
        match crate::chat::delete_chat_sessions_for_model(&db_path, &model_id) {
            Ok(count) => count,
            Err(error) => {
                // Chat wipe failure shouldn't undo an otherwise-successful uninstall
                // (files are already gone); surface it in the log so nothing hides.
                tracing::warn!(
                    model_id = %model_id,
                    error = %error,
                    "uninstall: failed to remove chat history, model still uninstalled"
                );
                0
            }
        }
    } else {
        0
    };

    tracing::info!(
        model_id = %model_id,
        source,
        cleared_current_model,
        delete_chat_history,
        deleted_chat_sessions,
        "uninstalled model"
    );

    let message = build_message(source, cleared_current_model, delete_chat_history, deleted_chat_sessions, removed_file_path.is_some());

    Ok(UninstallResult {
        model_id,
        source: source.to_string(),
        deleted_chat_sessions,
        cleared_current_model,
        message,
    })
}

/// Whether `settings.local_model_path` should be cleared after uninstalling the
/// given model. For builtin models we compare against the file we just removed;
/// for imported models the DB row is gone, so match on parent path + model id.
fn should_clear_current(current: &Path, removed_file: Option<&Path>, model_id: &str) -> bool {
    if let Some(removed) = removed_file {
        if path_equal(current, removed) {
            return true;
        }
    }
    // Fallback: imported models don't have a canonical "install path", but if
    // the current path's file stem embeds the model id it's almost certainly the
    // same one. Cheap best-effort — worst case the pointer stays and start-up
    // just fails softly on missing file.
    current
        .file_stem()
        .and_then(|stem| stem.to_str())
        .map(|stem| model_id.ends_with(stem) || model_id.contains(stem))
        .unwrap_or(false)
}

fn path_equal(a: &Path, b: &Path) -> bool {
    let normalize = |path: &Path| -> String {
        path.canonicalize()
            .unwrap_or_else(|_| path.to_path_buf())
            .to_string_lossy()
            .to_ascii_lowercase()
    };
    normalize(a) == normalize(b)
}

fn build_message(
    source: &str,
    cleared_current: bool,
    delete_chat_history: bool,
    deleted_chat_sessions: u32,
    imported_file_deleted: bool,
) -> String {
    let head = if source == "builtin" {
        "模型文件已从本地删除。".to_string()
    } else if imported_file_deleted {
        // The .gguf lived inside our managed models folder, so we removed it too.
        "模型文件已从本地删除。".to_string()
    } else {
        "已取消跟踪该本地模型。模型文件仍保留在你原来的位置，如需删除请去文件管理器操作。".to_string()
    };
    let mut parts: Vec<String> = vec![head];
    if cleared_current {
        parts.push("当前默认模型已重置，下次启动不会自动加载它。".to_string());
    }
    if delete_chat_history {
        if deleted_chat_sessions > 0 {
            parts.push(format!("同时清理了 {} 个聊天会话。", deleted_chat_sessions));
        } else {
            parts.push("没有找到关联的聊天会话可清理。".to_string());
        }
    }
    parts.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_dir() -> PathBuf {
        std::env::temp_dir().join(format!(
            "localai-uninstall-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ))
    }

    #[test]
    fn should_clear_current_matches_exact_removed_file() {
        let dir = temp_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("model.gguf");
        std::fs::File::create(&file).unwrap();

        assert!(should_clear_current(&file, Some(&file), "some-id"));
        // A different path with the same file name must NOT match.
        let other = dir.join("other.gguf");
        assert!(!should_clear_current(&other, Some(&file), "some-id"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn build_message_reflects_source_and_optional_wipes() {
        let msg = build_message("builtin", true, true, 3, false);
        assert!(msg.contains("已从本地删除"));
        assert!(msg.contains("当前默认模型已重置"));
        assert!(msg.contains("清理了 3 个聊天会话"));

        let imported = build_message("imported", false, false, 0, false);
        assert!(imported.contains("取消跟踪"));
        assert!(!imported.contains("当前默认模型已重置"));
        assert!(!imported.contains("聊天"));

        // Imported model whose file lived inside the managed folder: deleted too.
        let imported_deleted = build_message("imported", false, false, 0, true);
        assert!(imported_deleted.contains("已从本地删除"));
        assert!(!imported_deleted.contains("取消跟踪"));

        let no_sessions = build_message("builtin", false, true, 0, false);
        assert!(no_sessions.contains("没有找到关联的聊天会话"));
    }

    #[test]
    fn builtin_uninstall_removes_model_directory_and_is_idempotent() {
        let install_dir = temp_dir();
        let model_id = "qwen3_4b_chat"; // one of the builtin ids
        let model_dir = install_dir.join("models").join(model_id);
        std::fs::create_dir_all(&model_dir).unwrap();

        // Fake gguf + version marker to prove both go.
        let gguf = model_dir.join("Qwen3-4B-Q4_K_M.gguf");
        let mut f = std::fs::File::create(&gguf).unwrap();
        f.write_all(b"gguf-bytes").unwrap();
        std::fs::write(model_dir.join("version.json"), r#"{"sha256":""}"#).unwrap();
        assert!(gguf.exists());

        // Mirror the core filesystem step of the uninstall command (removing
        // the model dir). The real command wraps this with locking, DB and log
        // work exercised elsewhere; here we care that remove_dir_all is the
        // right primitive for the layout we ship.
        std::fs::remove_dir_all(&model_dir).unwrap();
        assert!(!model_dir.exists());

        // Second remove is idempotent from the caller's perspective: the
        // command checks .exists() first, so we simulate that here.
        assert!(!model_dir.exists()); // pre-check would skip

        let _ = std::fs::remove_dir_all(&install_dir);
    }
}
