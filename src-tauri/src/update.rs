use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use futures_util::StreamExt;
use reqwest::Client;
use serde::Serialize;
use tauri::{AppHandle, Emitter};
use tokio::{fs, io::AsyncWriteExt};

use crate::{
    app_state::{AppState, ServiceStatus},
    error::{AppError, AppResult},
    models::{model_install_path, write_installed_version, ModelMetadata},
    service::{self, ServiceInfo},
    settings::save_settings,
};

/// Progress for a model update, emitted on the `model_update_progress` channel.
/// Kept separate from `download_progress` so the install flow never mistakes an
/// update for a fresh install (which would double-trigger engine start).
#[derive(Debug, Clone, Serialize)]
pub struct UpdateProgress {
    pub model_id: String,
    pub phase: String,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub message: String,
}

/// Snapshot of the service before an update, so a failed update can bring the
/// previously running model back up.
struct RunningSnapshot {
    running: bool,
    model_id: Option<String>,
}

#[tauri::command]
pub async fn update_installed_model(
    model_id: String,
    app: AppHandle,
    state: tauri::State<'_, AppState>,
) -> AppResult<ServiceInfo> {
    tracing::info!(%model_id, "model update requested");
    let data_dir = {
        let inner = state.inner.lock().expect("state lock poisoned");
        inner.data_dir.clone()
    };
    let model = crate::catalog::effective_models(&data_dir)
        .into_iter()
        .find(|model| model.id == model_id)
        .ok_or_else(|| AppError::Message("暂时没找到这个模型，无法更新。".to_string()))?;

    let sources = update_sources(&model);
    if sources.is_empty()
        || sources
            .iter()
            .any(|url| url.contains("placeholder") || !url.starts_with("https://"))
    {
        return Err(AppError::Message(
            "模型下载源尚未配置，暂时无法更新。".to_string(),
        ));
    }

    let (data_dir, install_dir, prev_local_model_path) = {
        let inner = state.inner.lock().expect("state lock poisoned");
        (
            inner.data_dir.clone(),
            inner.settings.install_dir.clone(),
            inner.settings.local_model_path.clone(),
        )
    };

    let save_path = model_install_path(&install_dir, &model);
    if !save_path.exists() {
        return Err(AppError::Message(
            "这个模型还没安装，无法更新，请先完成安装。".to_string(),
        ));
    }

    let prev_service = {
        let inner = state.inner.lock().expect("state lock poisoned");
        RunningSnapshot {
            running: matches!(inner.service.status, ServiceStatus::Running),
            model_id: inner.service.model_id.clone(),
        }
    };

    // 1) Download the new version to a dedicated temp file. The live model file
    //    keeps serving until we have a verified replacement.
    let new_tmp = update_tmp_path(&save_path);
    let _ = fs::remove_file(&new_tmp).await;
    let total_hint = model.file_size_mb * 1_000_000;
    emit_update(&app, &model_id, "downloading", 0, total_hint, "正在下载新版本模型文件");
    if let Err(error) = download_new_version(&app, &model, &sources, &new_tmp).await {
        let _ = fs::remove_file(&new_tmp).await;
        emit_update(&app, &model_id, "failed", 0, 0, &error.to_string());
        return Err(error);
    }

    // 2) Verify integrity + signed manifest before touching the live file.
    emit_update(&app, &model_id, "verifying", 0, 0, "正在检查新版本是否完整");
    if let Err(error) = verify_new_version(&new_tmp, &model).await {
        let _ = fs::remove_file(&new_tmp).await;
        emit_update(&app, &model_id, "failed", 0, 0, &error.to_string());
        return Err(error);
    }

    // 3) Stop the old service so the file is no longer memory-mapped (required
    //    before we can replace it on Windows).
    emit_update(&app, &model_id, "switching", 0, 0, "正在停止旧版本并切换到新版本");
    let _ = service::stop_service_core(&app, &state);

    // 4) Back up the old file and promote the new one.
    let bak_path = match promote_with_backup(&save_path, &new_tmp) {
        Ok(bak) => bak,
        Err(error) => {
            let _ = fs::remove_file(&new_tmp).await;
            // Nothing was swapped; try to bring the previous model back up.
            restart_previous(&app, &state, &prev_service).await;
            emit_update(&app, &model_id, "failed", 0, 0, &error.to_string());
            return Err(error);
        }
    };

    // 5) Make the updated model current and start it.
    {
        let mut inner = state.inner.lock().expect("state lock poisoned");
        inner.settings.local_model_path = Some(save_path.clone());
        save_settings(&data_dir, &inner.settings)?;
    }
    emit_update(&app, &model_id, "starting", 0, 0, "正在启动新版本");

    match service::start_service_core(&model_id, &app, &state).await {
        Ok(info) => {
            let _ = fs::remove_file(&bak_path).await;
            let _ = write_installed_version(&install_dir, &model);
            emit_update(&app, &model_id, "completed", 0, 0, "新版本已更新并成功启动。");
            Ok(info)
        }
        Err(start_error) => {
            // Roll back the file, restore the previous default, and bring the
            // previously running model back. Chat records and settings are
            // untouched because the model id and file path never changed.
            rollback_from_backup(&save_path, &bak_path);
            {
                let mut inner = state.inner.lock().expect("state lock poisoned");
                inner.settings.local_model_path = prev_local_model_path.clone();
                let _ = save_settings(&data_dir, &inner.settings);
            }
            restart_previous(&app, &state, &prev_service).await;
            let message = format!(
                "新版本启动失败，已回滚到旧版本，聊天记录和设置保持不变。原因：{}",
                start_error
            );
            emit_update(&app, &model_id, "rolled_back", 0, 0, &message);
            Err(AppError::Message(message))
        }
    }
}

async fn restart_previous(app: &AppHandle, state: &AppState, snapshot: &RunningSnapshot) {
    if !snapshot.running {
        return;
    }
    if let Some(model_id) = snapshot.model_id.clone() {
        let _ = service::start_service_core(&model_id, app, state).await;
    }
}

fn emit_update(
    app: &AppHandle,
    model_id: &str,
    phase: &str,
    downloaded: u64,
    total: u64,
    message: &str,
) {
    match phase {
        // "downloading" is emitted per-chunk; skip it to avoid flooding the log.
        "downloading" => {}
        "failed" => tracing::error!(%model_id, phase, "model update failed: {message}"),
        "rolled_back" => tracing::warn!(%model_id, phase, "model update rolled back: {message}"),
        _ => tracing::info!(%model_id, phase, "model update: {message}"),
    }
    let _ = app.emit(
        "model_update_progress",
        UpdateProgress {
            model_id: model_id.to_string(),
            phase: phase.to_string(),
            downloaded_bytes: downloaded,
            total_bytes: total,
            message: message.to_string(),
        },
    );
}

fn update_sources(model: &ModelMetadata) -> Vec<String> {
    let mut sources = vec![model.download_url.clone()];
    sources.extend(model.mirror_urls.clone());
    sources
}

async fn download_new_version(
    app: &AppHandle,
    model: &ModelMetadata,
    sources: &[String],
    target: &Path,
) -> AppResult<()> {
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(60 * 60))
        .build()
        .map_err(|_| AppError::Message("下载组件暂时没能启动，请稍后再试。".to_string()))?;

    let total_hint = model.file_size_mb * 1_000_000;
    let mut last_error =
        "当前网络无法连接模型下载源，请稍后重试或使用高级离线导入。".to_string();

    for source in sources {
        for attempt in 0..3 {
            match stream_source(&client, source, target, app, &model.id, total_hint).await {
                Ok(()) => return Ok(()),
                Err(error) => {
                    last_error = error.to_string();
                    if attempt < 2 {
                        let delay = 2_u64.pow(attempt + 1);
                        emit_update(
                            app,
                            &model.id,
                            "downloading",
                            0,
                            total_hint,
                            &format!("当前线路不稳定，{} 秒后自动重试", delay),
                        );
                        tokio::time::sleep(Duration::from_secs(delay)).await;
                    }
                }
            }
        }
    }

    Err(AppError::Message(last_error))
}

async fn stream_source(
    client: &Client,
    source: &str,
    target: &Path,
    app: &AppHandle,
    model_id: &str,
    total_hint: u64,
) -> AppResult<()> {
    let response = client
        .get(source)
        .send()
        .await
        .map_err(|_| AppError::Message("当前下载线路连接失败，正在尝试备用线路。".to_string()))?;
    if !response.status().is_success() {
        return Err(AppError::Message(
            "当前下载线路不可用，正在尝试备用线路。".to_string(),
        ));
    }

    let total = response.content_length().unwrap_or(total_hint);
    let mut file = fs::File::create(target).await?;
    let mut downloaded = 0u64;
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk
            .map_err(|_| AppError::Message("下载中断了，正在重试。".to_string()))?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        emit_update(app, model_id, "downloading", downloaded, total, "正在下载新版本模型文件");
    }

    file.flush().await?;
    Ok(())
}

async fn verify_new_version(path: &Path, model: &ModelMetadata) -> AppResult<()> {
    if !crate::hash::is_configured(&model.sha256) {
        return Err(AppError::Message(
            "模型文件校验信息尚未配置，请先补充真实校验值。".to_string(),
        ));
    }
    let actual = crate::hash::file_sha256_hex(path).await?;
    if !crate::hash::matches(&model.sha256, &actual) {
        return Err(AppError::Message(
            "新版本模型文件检查不通过，请稍后重试。".to_string(),
        ));
    }
    crate::manifest::verify_file(&format!("model:{}", model.id), path).await?;
    Ok(())
}

fn update_tmp_path(save_path: &Path) -> PathBuf {
    with_suffix(save_path, "upd")
}

fn backup_path(save_path: &Path) -> PathBuf {
    with_suffix(save_path, "bak")
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".");
    name.push(suffix);
    PathBuf::from(name)
}

/// Move `new_tmp` into `save_path`, backing up the existing file first. Returns
/// the backup path so a later start failure can restore it. If the promote step
/// fails, the backup is restored before returning an error.
fn promote_with_backup(save_path: &Path, new_tmp: &Path) -> AppResult<PathBuf> {
    let bak = backup_path(save_path);
    let _ = std::fs::remove_file(&bak);
    std::fs::rename(save_path, &bak).map_err(|_| {
        AppError::Message("切换新版本前备份旧版本失败，已保留旧版本。".to_string())
    })?;
    if std::fs::rename(new_tmp, save_path).is_err() {
        let _ = std::fs::rename(&bak, save_path);
        return Err(AppError::Message(
            "切换到新版本失败，已恢复旧版本。".to_string(),
        ));
    }
    Ok(bak)
}

/// Restore the backed-up old file, discarding the broken new file in place.
fn rollback_from_backup(save_path: &Path, bak: &Path) {
    let _ = std::fs::remove_file(save_path);
    let _ = std::fs::rename(bak, save_path);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_dir() -> PathBuf {
        std::env::temp_dir().join(format!("localai-update-test-{}", uuid::Uuid::new_v4()))
    }

    fn write_file(path: &Path, contents: &[u8]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(contents).unwrap();
    }

    #[test]
    fn tmp_and_backup_paths_are_distinct_and_stable() {
        let save = PathBuf::from("C:/models/id/model.gguf");
        assert_eq!(update_tmp_path(&save), PathBuf::from("C:/models/id/model.gguf.upd"));
        assert_eq!(backup_path(&save), PathBuf::from("C:/models/id/model.gguf.bak"));
    }

    #[test]
    fn promote_replaces_file_and_keeps_backup() {
        let dir = temp_dir();
        let save = dir.join("model.gguf");
        let new_tmp = update_tmp_path(&save);
        write_file(&save, b"old-version");
        write_file(&new_tmp, b"new-version");

        let bak = promote_with_backup(&save, &new_tmp).unwrap();

        // New content is live, old content preserved in the backup, tmp consumed.
        assert_eq!(std::fs::read(&save).unwrap(), b"new-version");
        assert_eq!(std::fs::read(&bak).unwrap(), b"old-version");
        assert!(!new_tmp.exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rollback_restores_old_version() {
        let dir = temp_dir();
        let save = dir.join("model.gguf");
        let new_tmp = update_tmp_path(&save);
        write_file(&save, b"old-version");
        write_file(&new_tmp, b"broken-new-version");

        let bak = promote_with_backup(&save, &new_tmp).unwrap();
        assert_eq!(std::fs::read(&save).unwrap(), b"broken-new-version");

        // Simulate the new version failing to start.
        rollback_from_backup(&save, &bak);

        assert_eq!(std::fs::read(&save).unwrap(), b"old-version");
        assert!(!bak.exists());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
