use std::path::{Path, PathBuf};

use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

use crate::{
    download::{DownloadStatus, DownloadTask},
    error::AppResult,
};

const MIGRATION_0001: &str = include_str!("../migrations/0001_init.sql");

const SERVICE_STATE_KEY: &str = "service_state";

/// Snapshot of a running local service, persisted so the app can reattach to a
/// still-alive `llama-server` after a restart instead of orphaning it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PersistedServiceState {
    pub model_id: String,
    pub port: u16,
    pub pid: u32,
    pub safe_mode: bool,
    #[serde(default = "default_persisted_host")]
    pub host: String,
}

fn default_persisted_host() -> String {
    "127.0.0.1".to_string()
}

pub fn database_path(data_dir: &Path) -> PathBuf {
    data_dir.join("app.db")
}

pub fn init_database(data_dir: &Path) -> AppResult<PathBuf> {
    std::fs::create_dir_all(data_dir)?;
    let path = database_path(data_dir);
    let connection = Connection::open(&path)?;
    connection.execute_batch(MIGRATION_0001)?;
    Ok(path)
}

pub fn load_download_tasks(db_path: &Path) -> AppResult<Vec<DownloadTask>> {
    let connection = Connection::open(db_path)?;
    let mut statement = connection.prepare(
        "SELECT id, model_id, save_path, total_bytes, downloaded_bytes, status, current_source, source_history, create_time, update_time
         FROM download_tasks
         ORDER BY create_time DESC",
    )?;
    let rows = statement.query_map([], |row| {
        let status: String = row.get(5)?;
        let source_history: String = row.get(7)?;
        Ok(DownloadTask {
            id: row.get(0)?,
            model_id: row.get(1)?,
            save_path: row.get(2)?,
            total_bytes: row.get::<_, i64>(3)?.max(0) as u64,
            downloaded_bytes: row.get::<_, i64>(4)?.max(0) as u64,
            status: download_status_from_db(&status),
            current_source: row.get(6)?,
            source_history: serde_json::from_str(&source_history).unwrap_or_default(),
            create_time: row.get(8)?,
            update_time: row.get(9)?,
        })
    })?;

    let mut tasks = Vec::new();
    for row in rows {
        tasks.push(row?);
    }
    Ok(tasks)
}

pub fn save_download_task(db_path: &Path, task: &DownloadTask) -> AppResult<()> {
    let connection = Connection::open(db_path)?;
    connection.execute(
        "INSERT INTO download_tasks (id, model_id, save_path, total_bytes, downloaded_bytes, status, current_source, source_history, create_time, update_time)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
         ON CONFLICT(id) DO UPDATE SET
           model_id = excluded.model_id,
           save_path = excluded.save_path,
           total_bytes = excluded.total_bytes,
           downloaded_bytes = excluded.downloaded_bytes,
           status = excluded.status,
           current_source = excluded.current_source,
           source_history = excluded.source_history,
           update_time = excluded.update_time",
        rusqlite::params![
            task.id,
            task.model_id,
            task.save_path,
            task.total_bytes as i64,
            task.downloaded_bytes as i64,
            download_status_to_db(&task.status),
            task.current_source,
            serde_json::to_string(&task.source_history)?,
            task.create_time,
            task.update_time,
        ],
    )?;
    Ok(())
}

fn download_status_to_db(status: &DownloadStatus) -> &'static str {
    match status {
        DownloadStatus::Pending => "pending",
        DownloadStatus::Downloading => "downloading",
        DownloadStatus::Paused => "paused",
        DownloadStatus::Verifying => "verifying",
        DownloadStatus::Failed => "failed",
        DownloadStatus::Completed => "completed",
        DownloadStatus::Cancelled => "cancelled",
    }
}

fn download_status_from_db(status: &str) -> DownloadStatus {
    match status {
        "downloading" => DownloadStatus::Downloading,
        "paused" => DownloadStatus::Paused,
        "verifying" => DownloadStatus::Verifying,
        "failed" => DownloadStatus::Failed,
        "completed" => DownloadStatus::Completed,
        "cancelled" => DownloadStatus::Cancelled,
        _ => DownloadStatus::Pending,
    }
}

pub fn save_service_state(db_path: &Path, state: &PersistedServiceState) -> AppResult<()> {
    let connection = Connection::open(db_path)?;
    connection.execute(
        "INSERT INTO system_config (key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        rusqlite::params![SERVICE_STATE_KEY, serde_json::to_string(state)?],
    )?;
    Ok(())
}

pub fn load_service_state(db_path: &Path) -> AppResult<Option<PersistedServiceState>> {
    let connection = Connection::open(db_path)?;
    let raw: Option<String> = connection
        .query_row(
            "SELECT value FROM system_config WHERE key = ?1",
            [SERVICE_STATE_KEY],
            |row| row.get(0),
        )
        .optional()?;
    Ok(raw.and_then(|value| serde_json::from_str(&value).ok()))
}

pub fn clear_service_state(db_path: &Path) -> AppResult<()> {
    let connection = Connection::open(db_path)?;
    connection.execute(
        "DELETE FROM system_config WHERE key = ?1",
        [SERVICE_STATE_KEY],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> PathBuf {
        std::env::temp_dir().join(format!("localai-db-test-{}", uuid::Uuid::new_v4()))
    }

    fn sample_task(id: &str, status: DownloadStatus) -> DownloadTask {
        DownloadTask {
            id: id.to_string(),
            model_id: "qwen2-1_5b".to_string(),
            save_path: "C:/models/qwen2-1_5b/model.gguf".to_string(),
            total_bytes: 1_000_000,
            downloaded_bytes: 250_000,
            status,
            current_source: "https://modelscope.cn/model.gguf".to_string(),
            source_history: vec!["https://modelscope.cn/model.gguf".to_string()],
            create_time: "2026-07-01T00:00:00Z".to_string(),
            update_time: "2026-07-01T00:01:00Z".to_string(),
        }
    }

    #[test]
    fn download_tasks_reload_after_restart() {
        let dir = temp_dir();
        let db = init_database(&dir).expect("init database");

        save_download_task(&db, &sample_task("task-a", DownloadStatus::Downloading)).unwrap();
        save_download_task(&db, &sample_task("task-b", DownloadStatus::Completed)).unwrap();

        // Upsert: saving the same id again must update, not duplicate.
        let mut updated = sample_task("task-a", DownloadStatus::Verifying);
        updated.downloaded_bytes = 900_000;
        save_download_task(&db, &updated).unwrap();

        // Simulate the startup reload path used by lib.rs.
        let tasks = load_download_tasks(&db).unwrap();
        assert_eq!(tasks.len(), 2);

        let task_a = tasks.iter().find(|t| t.id == "task-a").unwrap();
        assert_eq!(task_a.status, DownloadStatus::Verifying);
        assert_eq!(task_a.downloaded_bytes, 900_000);
        assert_eq!(task_a.source_history, vec!["https://modelscope.cn/model.gguf".to_string()]);

        let task_b = tasks.iter().find(|t| t.id == "task-b").unwrap();
        assert_eq!(task_b.status, DownloadStatus::Completed);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn service_state_round_trips_and_clears() {
        let dir = temp_dir();
        let db = init_database(&dir).expect("init database");

        assert_eq!(load_service_state(&db).unwrap(), None);

        let state = PersistedServiceState {
            model_id: "qwen2_5_1_5b_chat_compact".to_string(),
            port: 8123,
            pid: 4242,
            safe_mode: true,
            host: "127.0.0.1".to_string(),
        };
        save_service_state(&db, &state).unwrap();
        assert_eq!(load_service_state(&db).unwrap(), Some(state.clone()));

        // Upsert: saving again with a different port replaces, not duplicates.
        let mut updated = state.clone();
        updated.port = 9000;
        save_service_state(&db, &updated).unwrap();
        assert_eq!(load_service_state(&db).unwrap(), Some(updated));

        clear_service_state(&db).unwrap();
        assert_eq!(load_service_state(&db).unwrap(), None);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
