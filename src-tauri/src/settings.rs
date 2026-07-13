use std::{fs, path::PathBuf};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{AppError, AppResult};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSettings {
    pub install_dir: PathBuf,
    pub download_speed_limit: Option<u64>,
    pub mirror_source: MirrorSource,
    pub proxy: Option<String>,
    pub log_level: String,
    #[serde(default)]
    pub autostart: bool,
    #[serde(default)]
    pub extra_launch_args: Option<String>,
    /// Bind address for the local AI service. "127.0.0.1" = local-only (default),
    /// "0.0.0.0" = reachable from the LAN (user opt-in, no auth — see UI warning).
    #[serde(default = "default_service_host")]
    pub service_host: String,
    /// Fixed listen port so external clients don't need to reconfigure each launch.
    #[serde(default = "default_service_port")]
    pub service_port: u16,
    pub llama_server_path: Option<PathBuf>,
    pub local_model_path: Option<PathBuf>,
    pub instance_id: String,
}

fn default_service_host() -> String {
    "127.0.0.1".to_string()
}

fn default_service_port() -> u16 {
    8080
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MirrorSource {
    Auto,
    ModelScope,
    HfMirror,
    Custom(String),
}

impl Default for AppSettings {
    fn default() -> Self {
        let base = dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("LocalAI-Installer");

        Self {
            install_dir: base,
            download_speed_limit: None,
            mirror_source: MirrorSource::Auto,
            proxy: None,
            log_level: "INFO".to_string(),
            autostart: false,
            extra_launch_args: None,
            service_host: default_service_host(),
            service_port: default_service_port(),
            llama_server_path: None,
            local_model_path: None,
            instance_id: Uuid::new_v4().to_string(),
        }
    }
}

pub fn settings_path(data_dir: &PathBuf) -> PathBuf {
    data_dir.join("config.json")
}

pub fn load_settings(data_dir: &PathBuf) -> AppResult<AppSettings> {
    fs::create_dir_all(data_dir)?;
    let path = settings_path(data_dir);
    if !path.exists() {
        let settings = AppSettings::default();
        save_settings(data_dir, &settings)?;
        return Ok(settings);
    }

    let content = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&content)?)
}

pub fn save_settings(data_dir: &PathBuf, settings: &AppSettings) -> AppResult<()> {
    fs::create_dir_all(data_dir)?;
    let content = serde_json::to_string_pretty(settings)?;
    fs::write(settings_path(data_dir), content)?;
    Ok(())
}

#[tauri::command]
pub fn get_settings(state: tauri::State<'_, crate::app_state::AppState>) -> AppResult<AppSettings> {
    let inner = state.inner.lock().expect("state lock poisoned");
    Ok(inner.settings.clone())
}

#[tauri::command]
pub fn update_settings(
    settings: AppSettings,
    state: tauri::State<'_, crate::app_state::AppState>,
) -> AppResult<AppSettings> {
    let mut inner = state.inner.lock().expect("state lock poisoned");
    let autostart_changed = inner.settings.autostart != settings.autostart;
    save_settings(&inner.data_dir, &settings)?;
    if autostart_changed {
        // Best-effort: a failed registry write must not block saving settings.
        let _ = apply_autostart(settings.autostart);
    }
    tracing::info!(
        autostart = settings.autostart,
        autostart_changed,
        mirror = ?settings.mirror_source,
        log_level = %settings.log_level,
        speed_limit = ?settings.download_speed_limit,
        service_host = %settings.service_host,
        service_port = settings.service_port,
        "settings updated"
    );
    inner.settings = settings.clone();
    Ok(settings)
}

/// Register or remove this app from Windows per-user startup (HKCU Run key) via
/// `reg.exe`, avoiding a new dependency. No-op / error-free on other platforms.
#[cfg(windows)]
fn apply_autostart(enable: bool) -> AppResult<()> {
    const RUN_KEY: &str = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run";
    const VALUE_NAME: &str = "LocalAIInstaller";
    if enable {
        let exe = std::env::current_exe()
            .map_err(|_| AppError::Message("无法确定程序路径，开机自启设置失败。".to_string()))?;
        let mut command = std::process::Command::new("reg");
        command.args([
            "add", RUN_KEY, "/v", VALUE_NAME, "/t", "REG_SZ", "/d",
            &exe.to_string_lossy(), "/f",
        ]);
        crate::proc::hide_console(&mut command);
        command
            .output()
            .map_err(|_| AppError::Message("开机自启设置失败。".to_string()))?;
    } else {
        // Deleting a missing value returns a non-zero code; ignore it.
        let mut command = std::process::Command::new("reg");
        command.args(["delete", RUN_KEY, "/v", VALUE_NAME, "/f"]);
        crate::proc::hide_console(&mut command);
        let _ = command.output();
    }
    Ok(())
}

#[cfg(not(windows))]
fn apply_autostart(_enable: bool) -> AppResult<()> {
    Ok(())
}
