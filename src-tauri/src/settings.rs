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
    /// User-chosen name the AI reports for itself. Empty/None → derive from the
    /// model id in the frontend. Prevents finetunes from hallucinating an
    /// upstream vendor identity.
    #[serde(default)]
    pub ai_display_name: Option<String>,
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
            ai_display_name: None,
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
    // The frontend's `SettingsPanel.save()` spreads its cached `settings`
    // object and forwards every field, but `local_model_path` and
    // `llama_server_path` are owned by dedicated commands
    // (`switch_installed_model` / `import_local_model` /
    // `ensure_engine_ready` / `start_download` / `update_installed_model`).
    // Letting the frontend's stale copy overwrite them here would silently
    // revert those pointers — e.g. a settings change right after switching
    // models would re-point the config at the previous model. Sanitize first.
    let sanitized = sanitize_settings_update(&inner.settings, settings);
    save_settings(&inner.data_dir, &sanitized)?;
    if autostart_changed {
        // Best-effort: a failed registry write must not block saving settings.
        let _ = apply_autostart(sanitized.autostart);
    }
    tracing::info!(
        autostart = sanitized.autostart,
        autostart_changed,
        mirror = ?sanitized.mirror_source,
        log_level = %sanitized.log_level,
        speed_limit = ?sanitized.download_speed_limit,
        service_host = %sanitized.service_host,
        service_port = sanitized.service_port,
        "settings updated"
    );
    inner.settings = sanitized.clone();
    Ok(sanitized)
}

/// Keep the two engine/model pointers from being overwritten by a stale
/// frontend snapshot passed into `update_settings`. All other fields are
/// accepted as-is — they are the ones the user can actually edit from
/// SettingsPanel.
pub fn sanitize_settings_update(current: &AppSettings, incoming: AppSettings) -> AppSettings {
    AppSettings {
        local_model_path: current.local_model_path.clone(),
        llama_server_path: current.llama_server_path.clone(),
        ..incoming
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_preserves_current_engine_and_model_paths() {
        // Simulate a stale frontend snapshot that still references the previous
        // model/engine, sent right after the user switched models. Without
        // sanitization, this would silently revert the config to the old paths.
        let mut current = AppSettings::default();
        current.local_model_path = Some(PathBuf::from("/install/models/qwen3_8b_chat/Qwen3-8B-Q4_K_M.gguf"));
        current.llama_server_path = Some(PathBuf::from("/install/engine/backends/llama-server-cuda.exe"));

        let mut incoming = current.clone();
        incoming.local_model_path = Some(PathBuf::from("/install/models/ornith-1.0-9b-Q4/ornith-1.0-9b-Q4_K_M.gguf"));
        incoming.llama_server_path = Some(PathBuf::from("/install/engine/backends/llama-server-vulkan.exe"));
        // User also changed the listen port — that change MUST survive.
        incoming.service_port = 9000;

        let sanitized = sanitize_settings_update(&current, incoming);

        assert_eq!(
            sanitized.local_model_path,
            Some(PathBuf::from("/install/models/qwen3_8b_chat/Qwen3-8B-Q4_K_M.gguf")),
            "stale local_model_path from the frontend must not overwrite the current pointer"
        );
        assert_eq!(
            sanitized.llama_server_path,
            Some(PathBuf::from("/install/engine/backends/llama-server-cuda.exe")),
            "stale llama_server_path from the frontend must not overwrite the current pointer"
        );
        assert_eq!(
            sanitized.service_port, 9000,
            "user-edited fields must still go through"
        );
    }

    #[test]
    fn sanitize_passes_through_all_other_fields() {
        // Everything the user can actually edit in SettingsPanel must reach
        // config.json unchanged — sanitize is a filter, not a rewriter.
        let current = AppSettings::default();
        let mut incoming = current.clone();
        incoming.mirror_source = MirrorSource::HfMirror;
        incoming.log_level = "DEBUG".to_string();
        incoming.ai_display_name = Some("小助手".to_string());
        incoming.download_speed_limit = Some(5_000_000);
        incoming.proxy = Some("http://127.0.0.1:7890".to_string());
        incoming.autostart = true;
        incoming.extra_launch_args = Some("--ctx-size 8192".to_string());
        incoming.service_host = "0.0.0.0".to_string();

        let sanitized = sanitize_settings_update(&current, incoming);

        assert!(matches!(sanitized.mirror_source, MirrorSource::HfMirror));
        assert_eq!(sanitized.log_level, "DEBUG");
        assert_eq!(sanitized.ai_display_name.as_deref(), Some("小助手"));
        assert_eq!(sanitized.download_speed_limit, Some(5_000_000));
        assert_eq!(sanitized.proxy.as_deref(), Some("http://127.0.0.1:7890"));
        assert!(sanitized.autostart);
        assert_eq!(sanitized.extra_launch_args.as_deref(), Some("--ctx-size 8192"));
        assert_eq!(sanitized.service_host, "0.0.0.0");
        // And the two protected pointers still come from `current`.
        assert_eq!(sanitized.local_model_path, current.local_model_path);
        assert_eq!(sanitized.llama_server_path, current.llama_server_path);
    }
}
