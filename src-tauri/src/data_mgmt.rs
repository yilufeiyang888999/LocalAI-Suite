use std::path::{Path, PathBuf};

use serde::Serialize;
use tauri::AppHandle;

use crate::{
    app_state::{AppState, ServiceStatus},
    error::{AppError, AppResult},
};

/// Filesystem locations exposed to the UI so the "open data folder" and
/// "open logs folder" buttons don't hard-code paths.
#[derive(Debug, Clone, Serialize)]
pub struct DataPaths {
    pub data_dir: String,
    pub install_dir: String,
    pub logs_dir: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClearChatResult {
    pub deleted_sessions: u32,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct UninstallAllResult {
    pub removed_count: u32,
    pub failed_count: u32,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResetResult {
    pub message: String,
}

const RESET_SENTINEL: &str = ".reset-pending";

#[tauri::command]
pub fn get_data_paths(state: tauri::State<'_, AppState>) -> AppResult<DataPaths> {
    let inner = state.inner.lock().expect("state lock poisoned");
    Ok(DataPaths {
        data_dir: inner.data_dir.to_string_lossy().to_string(),
        install_dir: inner.settings.install_dir.to_string_lossy().to_string(),
        logs_dir: inner.data_dir.join("logs").to_string_lossy().to_string(),
    })
}

/// Wipe every chat session and every message. Independent from uninstall so a
/// user can start over conversation-wise without losing their models.
#[tauri::command]
pub fn clear_all_chat_records(state: tauri::State<'_, AppState>) -> AppResult<ClearChatResult> {
    let db_path = {
        let inner = state.inner.lock().expect("state lock poisoned");
        inner.db_path.clone()
    };
    let connection = rusqlite::Connection::open(&db_path)?;
    // Messages first so any failure between the two statements can't leave
    // orphan messages hanging around bloating the DB.
    connection.execute("DELETE FROM chat_messages", [])?;
    let deleted = connection.execute("DELETE FROM chat_sessions", [])?;
    tracing::info!(
        deleted_sessions = deleted,
        "cleared all chat records"
    );
    Ok(ClearChatResult {
        deleted_sessions: deleted as u32,
        message: if deleted > 0 {
            format!("已清空全部聊天记录（{} 个会话）。", deleted)
        } else {
            "本来就没有聊天记录可以清理。".to_string()
        },
    })
}

/// Uninstall every model in the installed list. Refuses when the service is
/// running because letting the batch operation stop the current model behind
/// the user's back is more surprising than helpful — the single-model uninstall
/// makes the same choice.
#[tauri::command]
pub fn uninstall_all_models(
    state: tauri::State<'_, AppState>,
) -> AppResult<UninstallAllResult> {
    // Snapshot service state under the lock so we can drop it before doing IO.
    let service_running = {
        let inner = state.inner.lock().expect("state lock poisoned");
        matches!(
            inner.service.status,
            ServiceStatus::Running | ServiceStatus::Starting
        )
    };
    if service_running {
        return Err(AppError::Message(
            "本地AI服务正在运行，请先在聊天区点「停止服务」，再一键卸载。".to_string(),
        ));
    }

    // Reuse the single-model uninstall path so file cleanup, DB row removal
    // and settings pointer clearing all follow the exact same code as A1.
    let models = crate::models::list_installed_models(state.clone())?;
    let mut removed = 0u32;
    let mut failed = 0u32;
    for model in &models {
        match crate::uninstall::uninstall_model(
            model.model_id.clone(),
            false,
            Some(model.local_path.clone()),
            state.clone(),
        ) {
            Ok(_) => removed += 1,
            Err(error) => {
                tracing::warn!(
                    model_id = %model.model_id,
                    error = %error,
                    "batch uninstall: failed to uninstall a model, continuing"
                );
                failed += 1;
            }
        }
    }

    let message = if failed == 0 {
        format!("已卸载全部 {} 个模型。", removed)
    } else {
        format!(
            "共卸载 {} 个模型，{} 个卸载失败（详情见日志）。",
            removed, failed
        )
    };
    tracing::info!(removed, failed, "batch uninstall completed");
    Ok(UninstallAllResult {
        removed_count: removed,
        failed_count: failed,
        message,
    })
}

/// Schedule a full factory reset. Rather than deleting live-in-use files (DB
/// handles, tracing appender, engine .exe processes) which invites file-lock
/// races on Windows, we write a sentinel file with the install_dir path inside
/// and let the next startup do the wipe from a clean slate. See
/// `perform_pending_reset`.
#[tauri::command]
pub fn reset_to_factory(
    app: AppHandle,
    state: tauri::State<'_, AppState>,
) -> AppResult<ResetResult> {
    // Try to stop the service first. Failure here isn't fatal — the sentinel
    // still gets written and startup will delete the engine dir anyway.
    let _ = crate::service::stop_service_core(&app, &state);

    let (data_dir, install_dir) = {
        let inner = state.inner.lock().expect("state lock poisoned");
        (inner.data_dir.clone(), inner.settings.install_dir.clone())
    };

    std::fs::create_dir_all(&data_dir)?;
    // Write install_dir inside the sentinel so `perform_pending_reset` can find
    // it after config.json is gone; empty content means "use default install dir".
    std::fs::write(
        data_dir.join(RESET_SENTINEL),
        install_dir.to_string_lossy().as_bytes(),
    )?;

    tracing::warn!(
        install_dir = %install_dir.to_string_lossy(),
        "factory reset scheduled; will run on next startup"
    );

    Ok(ResetResult {
        message: "已标记出厂重置。请关闭本软件后重新打开，重启时会完成清理。".to_string(),
    })
}

/// Called from `lib.rs::run` BEFORE settings/logging/database are initialised.
/// If the sentinel is present, wipe every persistent artefact — models, engine,
/// database (incl. its WAL/SHM sidecars), settings, catalog cache, logs — then
/// remove the sentinel so subsequent startups run normally. Everything is
/// best-effort: a missing path is not an error, a locked file that somehow
/// slips through is logged to stderr (tracing isn't up yet).
pub fn perform_pending_reset(data_dir: &Path) {
    let sentinel = data_dir.join(RESET_SENTINEL);
    if !sentinel.exists() {
        return;
    }

    let install_dir_hint = std::fs::read_to_string(&sentinel).unwrap_or_default();
    let install_dir: PathBuf = if install_dir_hint.trim().is_empty() {
        // Fallback: assume default install_dir. Reasonable because the sentinel
        // is only ever written by `reset_to_factory`, which always fills it.
        default_install_dir()
    } else {
        PathBuf::from(install_dir_hint.trim())
    };

    delete_all(&install_dir, data_dir);

    if let Err(error) = std::fs::remove_file(&sentinel) {
        // If we can't delete the sentinel, the next startup will loop through
        // reset again — probably fine, but log it so we notice.
        eprintln!(
            "[localai] warning: reset sentinel could not be removed: {}",
            error
        );
    } else {
        eprintln!("[localai] factory reset applied on startup");
    }
}

/// Best-effort filesystem wipe. Kept separate so `perform_pending_reset` stays
/// short and this list of what a "clean slate" means is one obvious place.
fn delete_all(install_dir: &Path, data_dir: &Path) {
    // Model + engine binaries (biggest disk footprint by far).
    let _ = std::fs::remove_dir_all(install_dir.join("models"));
    let _ = std::fs::remove_dir_all(install_dir.join("engine"));
    // App-owned state under data_dir.
    let _ = std::fs::remove_file(data_dir.join("config.json"));
    let _ = std::fs::remove_file(data_dir.join("app.db"));
    // SQLite may leave WAL/SHM sidecars if the previous run crashed.
    let _ = std::fs::remove_file(data_dir.join("app.db-wal"));
    let _ = std::fs::remove_file(data_dir.join("app.db-shm"));
    // Cached online model catalog.
    let _ = std::fs::remove_file(data_dir.join("catalog.json"));
    // Rolling logs. Safe to nuke before tracing is initialised.
    let _ = std::fs::remove_dir_all(data_dir.join("logs"));
}

/// Mirrors the default in `lib.rs::run` so a sentinel with no hint can still
/// find the default install dir.
fn default_install_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("LocalAI-Installer")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_dir(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "localai-data-mgmt-{}-{}",
            tag,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ))
    }

    #[test]
    fn delete_all_wipes_expected_artefacts_and_ignores_missing() {
        let data_dir = temp_dir("data");
        let install_dir = temp_dir("install");

        std::fs::create_dir_all(data_dir.join("logs")).unwrap();
        std::fs::create_dir_all(install_dir.join("models").join("qwen3_4b_chat")).unwrap();
        std::fs::create_dir_all(install_dir.join("engine").join("backends")).unwrap();

        let mut f = std::fs::File::create(data_dir.join("app.db")).unwrap();
        f.write_all(b"sqlite").unwrap();
        std::fs::write(data_dir.join("app.db-wal"), b"w").unwrap();
        std::fs::write(data_dir.join("config.json"), b"{}").unwrap();
        std::fs::write(data_dir.join("catalog.json"), b"{}").unwrap();
        std::fs::write(data_dir.join("logs").join("today.log"), b"log").unwrap();
        std::fs::write(
            install_dir.join("models").join("qwen3_4b_chat").join("f.gguf"),
            b"gguf",
        )
        .unwrap();
        std::fs::write(
            install_dir.join("engine").join("backends").join("llama-server-cpu.exe"),
            b"exe",
        )
        .unwrap();

        delete_all(&install_dir, &data_dir);

        // Everything scheduled for deletion is gone.
        assert!(!data_dir.join("app.db").exists());
        assert!(!data_dir.join("app.db-wal").exists());
        assert!(!data_dir.join("config.json").exists());
        assert!(!data_dir.join("catalog.json").exists());
        assert!(!data_dir.join("logs").exists());
        assert!(!install_dir.join("models").exists());
        assert!(!install_dir.join("engine").exists());

        // Re-running on already-clean dirs is a no-op, not an error.
        delete_all(&install_dir, &data_dir);

        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&install_dir);
    }

    #[test]
    fn perform_pending_reset_is_no_op_without_sentinel() {
        let data_dir = temp_dir("no-sentinel");
        std::fs::create_dir_all(&data_dir).unwrap();
        std::fs::write(data_dir.join("config.json"), b"{}").unwrap();

        perform_pending_reset(&data_dir);

        // Without the sentinel, nothing gets touched.
        assert!(data_dir.join("config.json").exists());

        let _ = std::fs::remove_dir_all(&data_dir);
    }

    #[test]
    fn perform_pending_reset_reads_install_dir_from_sentinel() {
        let data_dir = temp_dir("reset-data");
        let install_dir = temp_dir("reset-install");
        std::fs::create_dir_all(&data_dir).unwrap();
        std::fs::create_dir_all(install_dir.join("models")).unwrap();
        std::fs::write(install_dir.join("models").join("x.gguf"), b"gguf").unwrap();
        std::fs::write(data_dir.join("config.json"), b"{}").unwrap();
        // Sentinel points at our test install_dir, not the default.
        std::fs::write(
            data_dir.join(RESET_SENTINEL),
            install_dir.to_string_lossy().as_bytes(),
        )
        .unwrap();

        perform_pending_reset(&data_dir);

        assert!(!data_dir.join("config.json").exists(), "config.json should be gone");
        assert!(!install_dir.join("models").exists(), "install_dir/models should be gone");
        assert!(
            !data_dir.join(RESET_SENTINEL).exists(),
            "sentinel should be removed after reset"
        );

        let _ = std::fs::remove_dir_all(&data_dir);
        let _ = std::fs::remove_dir_all(&install_dir);
    }
}
