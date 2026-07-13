use std::{
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU8, Ordering},
        Arc, Mutex,
    },
    time::Instant,
};

use chrono::Utc;
use futures_util::StreamExt;
use reqwest::{header, Client, StatusCode};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter};
use tokio::{fs, io::AsyncWriteExt};
use uuid::Uuid;

use crate::{
    app_state::{AppState, AppStateInner},
    database::save_download_task,
    error::{AppError, AppResult},
    models::ModelMetadata,
    settings::{save_settings, MirrorSource},
};

/// Cooperative control flag shared with an in-flight download loop.
const CTRL_RUNNING: u8 = 0;
const CTRL_PAUSE: u8 = 1;
const CTRL_CANCEL: u8 = 2;

/// How a single source attempt ended once control signals are considered.
enum SourceOutcome {
    Completed,
    Paused,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadTask {
    pub id: String,
    pub model_id: String,
    pub save_path: String,
    pub total_bytes: u64,
    pub downloaded_bytes: u64,
    pub status: DownloadStatus,
    pub current_source: String,
    pub source_history: Vec<String>,
    pub create_time: String,
    pub update_time: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DownloadStatus {
    Pending,
    Downloading,
    Paused,
    Verifying,
    Failed,
    Completed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub task_id: String,
    pub model_id: String,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub speed_bps: u64,
    pub status: DownloadStatus,
    pub current_step: String,
}

#[tauri::command]
pub fn start_download(
    model_id: String,
    app: AppHandle,
    state: tauri::State<'_, AppState>,
) -> AppResult<String> {
    tracing::info!(%model_id, "download requested");
    let data_dir = {
        let inner = state.inner.lock().expect("state lock poisoned");
        inner.data_dir.clone()
    };
    let model = crate::catalog::effective_models(&data_dir)
        .into_iter()
        .find(|model| model.id == model_id)
        .ok_or_else(|| AppError::Message("暂时没找到这个模型，请重新选择。".to_string()))?;

    let sources = download_sources(&model);
    if sources.is_empty()
        || sources
            .iter()
            .any(|url| is_placeholder_url(url) || !url.starts_with("https://"))
    {
        return Err(AppError::Message(
            "模型下载源尚未配置，请先在模型清单中填入 ModelScope 或 hf-mirror 的 HTTPS 下载地址。"
                .to_string(),
        ));
    }

    let (data_dir, install_dir) = {
        let inner = state.inner.lock().expect("state lock poisoned");
        (inner.data_dir.clone(), inner.settings.install_dir.clone())
    };
    let save_path = crate::models::model_install_path(&install_dir, &model);
    if save_path.exists() {
        let file_size = std::fs::metadata(&save_path)
            .map(|meta| meta.len())
            .unwrap_or(model.file_size_mb * 1_000_000);
        {
            let mut inner = state.inner.lock().expect("state lock poisoned");
            inner.settings.local_model_path = Some(save_path.clone());
            save_settings(&data_dir, &inner.settings)?;
        }
        if let Some(model_dir) = save_path.parent() {
            let _ = crate::models::write_version_marker(model_dir, &model);
        }
        let now = Utc::now().to_rfc3339();
        let task = DownloadTask {
            id: Uuid::new_v4().to_string(),
            model_id: model.id.clone(),
            save_path: save_path.to_string_lossy().to_string(),
            total_bytes: file_size,
            downloaded_bytes: file_size,
            status: DownloadStatus::Completed,
            current_source: sources[0].clone(),
            source_history: Vec::new(),
            create_time: now.clone(),
            update_time: now,
        };
        let task_id = task.id.clone();
        let progress = progress_from_task(&task, 0);
        {
            let mut inner = state.inner.lock().expect("state lock poisoned");
            let _ = save_download_task(&inner.db_path, &task);
            inner.downloads.insert(task_id.clone(), task);
        }
        let _ = app.emit("download_progress", progress);
        tracing::info!(%model_id, "model file already present; marked completed");
        return Ok(task_id);
    }
    let now = Utc::now().to_rfc3339();
    let task = DownloadTask {
        id: Uuid::new_v4().to_string(),
        model_id: model.id.clone(),
        save_path: save_path.to_string_lossy().to_string(),
        total_bytes: model.file_size_mb * 1_000_000,
        downloaded_bytes: 0,
        status: DownloadStatus::Pending,
        current_source: sources[0].clone(),
        source_history: Vec::new(),
        create_time: now.clone(),
        update_time: now,
    };
    let task_id = task.id.clone();

    {
        let mut inner = state.inner.lock().expect("state lock poisoned");
        let _ = save_download_task(&inner.db_path, &task);
        inner.downloads.insert(task_id.clone(), task);
        inner
            .download_controls
            .insert(task_id.clone(), Arc::new(AtomicU8::new(CTRL_RUNNING)));
    }

    let cloned_state = state.inner.clone();
    let cloned_task_id = task_id.clone();
    tauri::async_runtime::spawn(async move {
        let result = download_model_file(
            app.clone(),
            cloned_state.clone(),
            cloned_task_id.clone(),
            model,
            save_path,
            data_dir,
        )
        .await;
        if let Err(error) = result {
            mark_failed(&app, &cloned_state, &cloned_task_id, error.to_string()).await;
        }
        clear_control(&cloned_state, &cloned_task_id);
    });

    Ok(task_id)
}

#[tauri::command]
pub fn pause_download(task_id: String, app: AppHandle, state: tauri::State<'_, AppState>) -> AppResult<()> {
    tracing::info!(%task_id, "pause download requested");
    let mut inner = state.inner.lock().expect("state lock poisoned");
    if let Some(control) = inner.download_controls.get(&task_id) {
        // A loop is running: ask it to stop and let it persist the Paused state.
        control.store(CTRL_PAUSE, Ordering::SeqCst);
        return Ok(());
    }
    // No active loop: flip the stored task to Paused directly.
    let db_path = inner.db_path.clone();
    let Some(task) = inner.downloads.get_mut(&task_id) else {
        return Err(AppError::Message("暂时没找到这个下载任务。".to_string()));
    };
    task.status = DownloadStatus::Paused;
    task.update_time = Utc::now().to_rfc3339();
    let progress = progress_from_task(task, 0);
    let task = task.clone();
    let _ = save_download_task(&db_path, &task);
    let _ = app.emit("download_progress", progress);
    Ok(())
}

#[tauri::command]
pub fn cancel_download(task_id: String, app: AppHandle, state: tauri::State<'_, AppState>) -> AppResult<()> {
    tracing::info!(%task_id, "cancel download requested");
    let (running, db_path, tmp_path, progress) = {
        let mut inner = state.inner.lock().expect("state lock poisoned");
        let running = inner.download_controls.get(&task_id).is_some();
        if let Some(control) = inner.download_controls.get(&task_id) {
            control.store(CTRL_CANCEL, Ordering::SeqCst);
        }
        let db_path = inner.db_path.clone();
        let Some(task) = inner.downloads.get_mut(&task_id) else {
            return Err(AppError::Message("暂时没找到这个下载任务。".to_string()));
        };
        let tmp_path = PathBuf::from(&task.save_path).with_extension("gguf.tmp");
        if !running {
            task.status = DownloadStatus::Cancelled;
            task.update_time = Utc::now().to_rfc3339();
        }
        let progress = progress_from_task(task, 0);
        (running, db_path, tmp_path, progress)
    };

    // When a loop is running it owns the .tmp cleanup and status persistence.
    if !running {
        let _ = std::fs::remove_file(&tmp_path);
        let task = {
            let inner = state.inner.lock().expect("state lock poisoned");
            inner.downloads.get(&task_id).cloned()
        };
        if let Some(task) = task {
            let _ = save_download_task(&db_path, &task);
        }
        let _ = app.emit("download_progress", progress);
    }
    Ok(())
}

#[tauri::command]
pub fn resume_download(
    task_id: String,
    app: AppHandle,
    state: tauri::State<'_, AppState>,
) -> AppResult<()> {
    tracing::info!(%task_id, "resume download requested");
    let (model, save_path, data_dir) = {
        let mut inner = state.inner.lock().expect("state lock poisoned");
        if inner.download_controls.contains_key(&task_id) {
            // Already running, nothing to resume.
            return Ok(());
        }
        let Some(task) = inner.downloads.get(&task_id).cloned() else {
            return Err(AppError::Message("暂时没找到这个下载任务。".to_string()));
        };
        let model = crate::catalog::effective_models(&inner.data_dir)
            .into_iter()
            .find(|model| model.id == task.model_id)
            .ok_or_else(|| AppError::Message("暂时没找到这个模型，请重新选择。".to_string()))?;
        let save_path = PathBuf::from(&task.save_path);
        let data_dir = inner.data_dir.clone();
        inner
            .download_controls
            .insert(task_id.clone(), Arc::new(AtomicU8::new(CTRL_RUNNING)));
        (model, save_path, data_dir)
    };

    let cloned_state = state.inner.clone();
    let cloned_task_id = task_id.clone();
    tauri::async_runtime::spawn(async move {
        let result = download_model_file(
            app.clone(),
            cloned_state.clone(),
            cloned_task_id.clone(),
            model,
            save_path,
            data_dir,
        )
        .await;
        if let Err(error) = result {
            mark_failed(&app, &cloned_state, &cloned_task_id, error.to_string()).await;
        }
        clear_control(&cloned_state, &cloned_task_id);
    });

    Ok(())
}

fn clear_control(state: &Arc<Mutex<AppStateInner>>, task_id: &str) {
    let mut inner = state.lock().expect("state lock poisoned");
    inner.download_controls.remove(task_id);
}

fn control_handle(state: &Arc<Mutex<AppStateInner>>, task_id: &str) -> Arc<AtomicU8> {
    let inner = state.lock().expect("state lock poisoned");
    inner
        .download_controls
        .get(task_id)
        .cloned()
        .unwrap_or_else(|| Arc::new(AtomicU8::new(CTRL_RUNNING)))
}

#[tauri::command]
pub fn get_download_progress(
    task_id: String,
    state: tauri::State<'_, AppState>,
) -> AppResult<DownloadProgress> {
    let inner = state.inner.lock().expect("state lock poisoned");
    let task = inner
        .downloads
        .get(&task_id)
        .ok_or_else(|| AppError::Message("暂时没找到这个下载任务。".to_string()))?;
    Ok(progress_from_task(task, 0))
}

async fn download_model_file(
    app: AppHandle,
    state: std::sync::Arc<std::sync::Mutex<crate::app_state::AppStateInner>>,
    task_id: String,
    model: ModelMetadata,
    save_path: PathBuf,
    data_dir: PathBuf,
) -> AppResult<()> {
    let sources = download_sources(&model);
    let tmp_path = save_path.with_extension("gguf.tmp");
    if let Some(parent) = save_path.parent() {
        fs::create_dir_all(parent).await?;
    }

    let db_path = {
        let inner = state.lock().expect("state lock poisoned");
        inner.db_path.clone()
    };

    // Probe the mirrors and download from the most responsive one first. Falls
    // back to the original ModelScope-then-hf-mirror order when probes fail.
    let (proxy, speed_limit, mirror_source) = {
        let inner = state.lock().expect("state lock poisoned");
        (
            inner.settings.proxy.clone(),
            inner.settings.download_speed_limit,
            inner.settings.mirror_source.clone(),
        )
    };

    let sources = match &mirror_source {
        MirrorSource::Auto if sources.len() > 1 => {
            set_task_status(
                &app,
                &state,
                &db_path,
                &task_id,
                DownloadStatus::Downloading,
                "正在测速挑选最快的下载线路",
                0,
            )
            .await;
            order_sources_by_speed(sources, proxy.as_deref()).await
        }
        MirrorSource::Auto => sources,
        preference => apply_mirror_preference(sources, preference),
    };

    let client = build_download_client(proxy.as_deref())?;
    tracing::info!(
        model_id = %model.id,
        source_count = sources.len(),
        first_source = sources.first().map(String::as_str).unwrap_or(""),
        "starting model download"
    );

    let mut last_error = "当前网络无法连接模型下载源，请稍后重试或使用高级离线导入。".to_string();
    for source in sources {
        update_source(&app, &state, &db_path, &task_id, &source, "正在选择下载线路").await;
        match try_download_source_with_retry(&client, &app, &state, &db_path, &task_id, &source, &tmp_path, speed_limit).await {
            Ok(SourceOutcome::Completed) => {
                set_task_status(
                    &app,
                    &state,
                    &db_path,
                    &task_id,
                    DownloadStatus::Verifying,
                    "正在检查文件是否完整",
                    0,
                )
                .await;
                verify_sha256(&tmp_path, &model.sha256).await?;
                crate::manifest::verify_file(&format!("model:{}", model.id), &tmp_path).await?;
                tracing::info!(model_id = %model.id, "model integrity + signature verified");
                if save_path.exists() {
                    fs::remove_file(&save_path).await?;
                }
                fs::rename(&tmp_path, &save_path).await?;
                if let Some(model_dir) = save_path.parent() {
                    let _ = crate::models::write_version_marker(model_dir, &model);
                }
                {
                    let mut inner = state.lock().expect("state lock poisoned");
                    inner.settings.local_model_path = Some(save_path.clone());
                    save_settings(&data_dir, &inner.settings)?;
                }
                set_task_status(
                    &app,
                    &state,
                    &db_path,
                    &task_id,
                    DownloadStatus::Completed,
                    "模型文件已下载完成",
                    0,
                )
                .await;
                return Ok(());
            }
            Ok(SourceOutcome::Paused) => {
                set_task_status(
                    &app,
                    &state,
                    &db_path,
                    &task_id,
                    DownloadStatus::Paused,
                    "已暂停下载，可随时继续",
                    0,
                )
                .await;
                return Ok(());
            }
            Ok(SourceOutcome::Cancelled) => {
                let _ = fs::remove_file(&tmp_path).await;
                set_task_status(
                    &app,
                    &state,
                    &db_path,
                    &task_id,
                    DownloadStatus::Cancelled,
                    "已取消下载",
                    0,
                )
                .await;
                return Ok(());
            }
            Err(error) => {
                last_error = error.to_string();
            }
        }
    }

    Err(AppError::Message(last_error))
}

async fn try_download_source_with_retry(
    client: &Client,
    app: &AppHandle,
    state: &Arc<Mutex<AppStateInner>>,
    db_path: &Path,
    task_id: &str,
    source: &str,
    tmp_path: &Path,
    speed_limit: Option<u64>,
) -> AppResult<SourceOutcome> {
    let mut last_error = "当前下载线路连接失败，正在尝试备用线路。".to_string();
    for attempt in 0..3 {
        match try_download_source(client, app, state, db_path, task_id, source, tmp_path, speed_limit).await {
            Ok(outcome) => return Ok(outcome),
            Err(error) => {
                last_error = error.to_string();
                if attempt < 2 {
                    let delay_secs = 2_u64.pow(attempt + 1);
                    tracing::warn!(%task_id, attempt = attempt + 1, delay_secs, "download attempt failed, retrying");
                    update_retry_step(app, state, db_path, task_id, delay_secs).await;
                    tokio::time::sleep(std::time::Duration::from_secs(delay_secs)).await;
                }
            }
        }
    }
    Err(AppError::Message(last_error))
}

async fn try_download_source(
    client: &Client,
    app: &AppHandle,
    state: &Arc<Mutex<AppStateInner>>,
    db_path: &Path,
    task_id: &str,
    source: &str,
    tmp_path: &Path,
    speed_limit: Option<u64>,
) -> AppResult<SourceOutcome> {
    let control = control_handle(state, task_id);
    match control.load(Ordering::SeqCst) {
        CTRL_PAUSE => return Ok(SourceOutcome::Paused),
        CTRL_CANCEL => return Ok(SourceOutcome::Cancelled),
        _ => {}
    }

    let existing = fs::metadata(tmp_path)
        .await
        .map(|meta| meta.len())
        .unwrap_or(0);
    let mut request = client.get(source);
    if existing > 0 {
        request = request.header(header::RANGE, format!("bytes={}-", existing));
    }

    let response = request
        .send()
        .await
        .map_err(|_| AppError::Message("当前下载线路连接失败，正在尝试备用线路。".to_string()))?;

    if !(response.status().is_success() || response.status() == StatusCode::PARTIAL_CONTENT) {
        return Err(AppError::Message(
            "当前下载线路不可用，正在尝试备用线路。".to_string(),
        ));
    }

    let content_length = response.content_length().unwrap_or(0);
    let total = if response.status() == StatusCode::PARTIAL_CONTENT {
        existing + content_length
    } else {
        content_length
    };
    let mut downloaded = if response.status() == StatusCode::PARTIAL_CONTENT {
        existing
    } else {
        0
    };
    let mut file = if downloaded > 0 {
        fs::OpenOptions::new().append(true).open(tmp_path).await?
    } else {
        fs::File::create(tmp_path).await?
    };
    let started = Instant::now();
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        match control.load(Ordering::SeqCst) {
            CTRL_PAUSE => {
                file.flush().await?;
                return Ok(SourceOutcome::Paused);
            }
            CTRL_CANCEL => {
                let _ = file.flush().await;
                return Ok(SourceOutcome::Cancelled);
            }
            _ => {}
        }
        let chunk = chunk
            .map_err(|_| AppError::Message("下载中断了，稍后会从已下载部分继续。".to_string()))?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        let elapsed_ms = started.elapsed().as_millis() as u64;
        // Honour a user-set speed cap by pacing against bytes read this session.
        if let Some(limit) = speed_limit.filter(|value| *value > 0) {
            let session_bytes = downloaded.saturating_sub(existing);
            let delay = throttle_delay_ms(session_bytes, limit, elapsed_ms);
            if delay > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
            }
        }
        let elapsed = started.elapsed().as_secs().max(1);
        let speed = downloaded.saturating_sub(existing) / elapsed;
        update_progress(
            app,
            state,
            db_path,
            task_id,
            downloaded,
            total,
            speed,
            "正在下载模型文件",
        )
        .await;
    }

    file.flush().await?;
    Ok(SourceOutcome::Completed)
}

async fn verify_sha256(path: &Path, expected: &str) -> AppResult<()> {
    if !crate::hash::is_configured(expected) {
        return Err(AppError::Message(
            "模型文件校验信息尚未配置，请先补充真实校验值。".to_string(),
        ));
    }

    let actual = crate::hash::file_sha256_hex(path).await?;
    if !crate::hash::matches(expected, &actual) {
        return Err(AppError::Message(
            "模型文件检查不通过，请重新下载。".to_string(),
        ));
    }
    Ok(())
}

async fn update_source(
    app: &AppHandle,
    state: &std::sync::Arc<std::sync::Mutex<crate::app_state::AppStateInner>>,
    db_path: &Path,
    task_id: &str,
    source: &str,
    step: &str,
) {
    let (progress, task) = {
        let mut inner = state.lock().expect("state lock poisoned");
        let Some(task) = inner.downloads.get_mut(task_id) else {
            return;
        };
        task.current_source = source.to_string();
        task.source_history.push(source.to_string());
        task.status = DownloadStatus::Downloading;
        task.update_time = Utc::now().to_rfc3339();
        let mut progress = progress_from_task(task, 0);
        progress.current_step = step.to_string();
        (progress, task.clone())
    };
    let _ = save_download_task(db_path, &task);
    let _ = app.emit("download_progress", progress);
}

async fn update_progress(
    app: &AppHandle,
    state: &std::sync::Arc<std::sync::Mutex<crate::app_state::AppStateInner>>,
    db_path: &Path,
    task_id: &str,
    downloaded: u64,
    total: u64,
    speed_bps: u64,
    step: &str,
) {
    let (progress, task) = {
        let mut inner = state.lock().expect("state lock poisoned");
        let Some(task) = inner.downloads.get_mut(task_id) else {
            return;
        };
        task.status = DownloadStatus::Downloading;
        task.downloaded_bytes = downloaded;
        if total > 0 {
            task.total_bytes = total;
        }
        task.update_time = Utc::now().to_rfc3339();
        let mut progress = progress_from_task(task, speed_bps);
        progress.current_step = step.to_string();
        (progress, task.clone())
    };
    let _ = save_download_task(db_path, &task);
    let _ = app.emit("download_progress", progress);
}

async fn update_retry_step(
    app: &AppHandle,
    state: &std::sync::Arc<std::sync::Mutex<crate::app_state::AppStateInner>>,
    db_path: &Path,
    task_id: &str,
    delay_secs: u64,
) {
    let (progress, task) = {
        let mut inner = state.lock().expect("state lock poisoned");
        let Some(task) = inner.downloads.get_mut(task_id) else {
            return;
        };
        task.update_time = Utc::now().to_rfc3339();
        let mut progress = progress_from_task(task, 0);
        progress.current_step = format!("当前线路不稳定，{} 秒后自动重试", delay_secs);
        (progress, task.clone())
    };
    let _ = save_download_task(db_path, &task);
    let _ = app.emit("download_progress", progress);
}

async fn mark_failed(
    app: &AppHandle,
    state: &std::sync::Arc<std::sync::Mutex<crate::app_state::AppStateInner>>,
    task_id: &str,
    message: String,
) {
    let db_path = {
        let inner = state.lock().expect("state lock poisoned");
        inner.db_path.clone()
    };
    set_task_status(app, state, &db_path, task_id, DownloadStatus::Failed, &message, 0).await;
}

async fn set_task_status(
    app: &AppHandle,
    state: &std::sync::Arc<std::sync::Mutex<crate::app_state::AppStateInner>>,
    db_path: &Path,
    task_id: &str,
    status: DownloadStatus,
    step: &str,
    speed_bps: u64,
) {
    if status == DownloadStatus::Failed {
        tracing::warn!(%task_id, "download failed: {step}");
    } else {
        tracing::info!(%task_id, status = ?status, "download status: {step}");
    }
    let (progress, task) = {
        let mut inner = state.lock().expect("state lock poisoned");
        let Some(task) = inner.downloads.get_mut(task_id) else {
            return;
        };
        task.status = status;
        if matches!(
            task.status,
            DownloadStatus::Completed | DownloadStatus::Verifying
        ) {
            task.downloaded_bytes = task.total_bytes;
        }
        task.update_time = Utc::now().to_rfc3339();
        let mut progress = progress_from_task(task, speed_bps);
        progress.current_step = step.to_string();
        (progress, task.clone())
    };
    let _ = save_download_task(db_path, &task);
    let _ = app.emit("download_progress", progress);
}

fn progress_from_task(task: &DownloadTask, speed_bps: u64) -> DownloadProgress {
    let current_step = match task.status {
        DownloadStatus::Pending => "等待开始".to_string(),
        DownloadStatus::Downloading => "正在下载模型文件".to_string(),
        DownloadStatus::Paused => "已暂停".to_string(),
        DownloadStatus::Verifying => "正在检查文件是否完整".to_string(),
        DownloadStatus::Failed => "下载遇到问题".to_string(),
        DownloadStatus::Completed => "模型文件已下载完成".to_string(),
        DownloadStatus::Cancelled => "已取消".to_string(),
    };

    DownloadProgress {
        task_id: task.id.clone(),
        model_id: task.model_id.clone(),
        downloaded_bytes: task.downloaded_bytes,
        total_bytes: task.total_bytes,
        speed_bps,
        status: task.status.clone(),
        current_step,
    }
}

fn download_sources(model: &ModelMetadata) -> Vec<String> {
    let mut sources = vec![model.download_url.clone()];
    sources.extend(model.mirror_urls.clone());
    sources
}

/// Build the download HTTP client, routing through `proxy` when configured.
fn build_download_client(proxy: Option<&str>) -> AppResult<Client> {
    let mut builder = Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .timeout(std::time::Duration::from_secs(60 * 60));
    if let Some(url) = proxy.map(str::trim).filter(|value| !value.is_empty()) {
        let proxy = reqwest::Proxy::all(url)
            .map_err(|_| AppError::Message("代理地址无效，请检查网络设置。".to_string()))?;
        builder = builder.proxy(proxy);
    }
    builder
        .build()
        .map_err(|_| AppError::Message("下载组件暂时没能启动，请稍后再试。".to_string()))
}

/// Reorder sources so the user's chosen mirror is tried first. Non-matching
/// sources keep their relative order behind the preferred ones, so the download
/// still works if the preferred mirror happens to be down.
fn apply_mirror_preference(sources: Vec<String>, mirror: &MirrorSource) -> Vec<String> {
    let needle = match mirror {
        MirrorSource::ModelScope => "modelscope".to_string(),
        MirrorSource::HfMirror => "hf-mirror".to_string(),
        MirrorSource::Custom(host) => host.trim().to_ascii_lowercase(),
        MirrorSource::Auto => return sources,
    };
    if needle.is_empty() {
        return sources;
    }
    let (mut preferred, mut rest): (Vec<_>, Vec<_>) = sources
        .into_iter()
        .partition(|url| url.to_ascii_lowercase().contains(&needle));
    preferred.append(&mut rest);
    preferred
}

/// Milliseconds to sleep to keep the session's average rate at or below
/// `limit_bps`. Zero when we're already within budget.
fn throttle_delay_ms(session_bytes: u64, limit_bps: u64, elapsed_ms: u64) -> u64 {
    if limit_bps == 0 {
        return 0;
    }
    let expected_ms = session_bytes.saturating_mul(1000) / limit_bps;
    expected_ms.saturating_sub(elapsed_ms)
}

/// Reorder sources fastest-first by probing each concurrently. Sources that fail
/// to respond keep their original relative order after the responsive ones, so a
/// total probe failure leaves the ModelScope-first order unchanged.
async fn order_sources_by_speed(sources: Vec<String>, proxy: Option<&str>) -> Vec<String> {
    if sources.len() <= 1 {
        return sources;
    }
    let mut builder = Client::builder()
        .connect_timeout(std::time::Duration::from_secs(5))
        .timeout(std::time::Duration::from_secs(6));
    if let Some(url) = proxy.map(str::trim).filter(|value| !value.is_empty()) {
        if let Ok(proxy) = reqwest::Proxy::all(url) {
            builder = builder.proxy(proxy);
        }
    }
    let Ok(client) = builder.build() else {
        return sources;
    };

    let probes = sources.into_iter().enumerate().map(|(index, url)| {
        let client = client.clone();
        async move {
            let latency = probe_source(&client, &url).await;
            (index, url, latency)
        }
    });
    let probed = futures_util::future::join_all(probes).await;
    rank_sources(probed)
}

/// Measure how quickly a source responds. Prefers a cheap `HEAD`; some mirrors
/// reject it, so it falls back to a single-byte ranged `GET`. Returns the
/// round-trip time in milliseconds, or `None` when the source is unreachable.
async fn probe_source(client: &Client, url: &str) -> Option<u128> {
    let started = Instant::now();
    if let Ok(response) = client.head(url).send().await {
        if response.status().is_success() {
            return Some(started.elapsed().as_millis());
        }
    }
    let started = Instant::now();
    if let Ok(response) = client
        .get(url)
        .header(header::RANGE, "bytes=0-0")
        .send()
        .await
    {
        if response.status().is_success() || response.status() == StatusCode::PARTIAL_CONTENT {
            return Some(started.elapsed().as_millis());
        }
    }
    None
}

/// Order probed sources: responsive ones ascending by latency, unreachable ones
/// last; original index breaks ties so equal/failed probes keep priority order.
fn rank_sources(mut probed: Vec<(usize, String, Option<u128>)>) -> Vec<String> {
    probed.sort_by(|a, b| match (a.2, b.2) {
        (Some(x), Some(y)) => x.cmp(&y).then(a.0.cmp(&b.0)),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => a.0.cmp(&b.0),
    });
    probed.into_iter().map(|(_, url, _)| url).collect()
}

fn is_placeholder_url(url: &str) -> bool {
    url.contains("download-placeholder.local") || url.contains("placeholder")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(index: usize, url: &str, latency: Option<u128>) -> (usize, String, Option<u128>) {
        (index, url.to_string(), latency)
    }

    #[test]
    fn ranks_fastest_source_first() {
        let probed = vec![
            probe(0, "https://modelscope/model.gguf", Some(300)),
            probe(1, "https://hf-mirror/model.gguf", Some(120)),
        ];
        assert_eq!(
            rank_sources(probed),
            vec![
                "https://hf-mirror/model.gguf".to_string(),
                "https://modelscope/model.gguf".to_string(),
            ]
        );
    }

    #[test]
    fn unreachable_sources_sink_to_the_end() {
        let probed = vec![
            probe(0, "https://a/model.gguf", None),
            probe(1, "https://b/model.gguf", Some(200)),
            probe(2, "https://c/model.gguf", None),
        ];
        // Responsive source first; failed ones keep their original relative order.
        assert_eq!(
            rank_sources(probed),
            vec![
                "https://b/model.gguf".to_string(),
                "https://a/model.gguf".to_string(),
                "https://c/model.gguf".to_string(),
            ]
        );
    }

    #[test]
    fn all_failures_preserve_priority_order() {
        let probed = vec![
            probe(0, "https://first/model.gguf", None),
            probe(1, "https://second/model.gguf", None),
        ];
        assert_eq!(
            rank_sources(probed),
            vec![
                "https://first/model.gguf".to_string(),
                "https://second/model.gguf".to_string(),
            ]
        );
    }

    #[test]
    fn equal_latency_keeps_priority_order() {
        let probed = vec![
            probe(0, "https://modelscope/model.gguf", Some(150)),
            probe(1, "https://hf-mirror/model.gguf", Some(150)),
        ];
        // Tie broken by original index: ModelScope stays first.
        assert_eq!(
            rank_sources(probed),
            vec![
                "https://modelscope/model.gguf".to_string(),
                "https://hf-mirror/model.gguf".to_string(),
            ]
        );
    }

    #[test]
    fn mirror_preference_moves_chosen_source_first() {
        let sources = vec![
            "https://www.modelscope.cn/models/x/model.gguf".to_string(),
            "https://hf-mirror.com/x/model.gguf".to_string(),
        ];
        // Preferring hf-mirror pulls it ahead of ModelScope.
        assert_eq!(
            apply_mirror_preference(sources.clone(), &MirrorSource::HfMirror),
            vec![sources[1].clone(), sources[0].clone()]
        );
        // Auto leaves the order untouched.
        assert_eq!(
            apply_mirror_preference(sources.clone(), &MirrorSource::Auto),
            sources
        );
    }

    #[test]
    fn mirror_preference_keeps_all_sources_when_none_match() {
        let sources = vec!["https://a/model.gguf".to_string(), "https://b/model.gguf".to_string()];
        // A custom host that matches nothing must not drop any source.
        assert_eq!(
            apply_mirror_preference(sources.clone(), &MirrorSource::Custom("nowhere".to_string())),
            sources
        );
    }

    #[test]
    fn throttle_delay_paces_against_limit() {
        // 2MB in 0ms against a 1MB/s cap → must wait ~2000ms.
        assert_eq!(throttle_delay_ms(2_000_000, 1_000_000, 0), 2000);
        // Already behind schedule → no extra wait.
        assert_eq!(throttle_delay_ms(1_000_000, 1_000_000, 3000), 0);
        // No limit → never wait.
        assert_eq!(throttle_delay_ms(5_000_000, 0, 0), 0);
    }
}
