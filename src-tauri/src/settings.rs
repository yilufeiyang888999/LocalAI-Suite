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
    /// 知识库问答服务的地址（作品一，**独立的 Python 进程**，见 RAG-Embedding 仓库）。
    ///
    /// 🔴 APP **只做探活与调用，不负责拉起那个进程** —— 它需要 Python 运行时和
    /// 独立的模型/索引，属于另一条交付链（跨进程边界，见 CLAUDE.md「对接方式」）。
    /// 服务未启动时，前端给出启动引导而不是静默失败。
    ///
    /// 默认 8092 必须与 `RAG-Embedding/scripts/知识库服务.py` 的 `监听端口` 一致；
    /// 改那边端口时要同步这里。
    #[serde(default = "default_kb_base_url")]
    pub kb_base_url: String,
    /// 智能问数 + 动作执行服务的地址（作品二 · ChatBI，**独立的 Python 进程**，见 ChatBI 仓库）。
    ///
    /// 🔴 与 `kb_base_url` 同一条边界：APP **只探活与调用，不负责拉起那个进程**。
    /// 它是 8096「编排服务」—— 一个入口三条链路（知识问答 / 数据问数 / 动作执行），
    /// 其中知识问答那条链路是**HTTP 转发**给 8092 的（作品一），所以这个地址是可用的
    /// 前提是 8092 也在跑。服务未启动时前端给出启动引导，而不是静默失败。
    ///
    /// 默认 8096 必须与 `ChatBI/scripts/编排服务.py` 的 `默认端口` 一致；
    /// 改那边端口时要同步这里。
    #[serde(default = "default_chatbi_base_url")]
    pub chatbi_base_url: String,
    pub instance_id: String,
}

fn default_kb_base_url() -> String {
    "http://127.0.0.1:8092".to_string()
}

fn default_chatbi_base_url() -> String {
    "http://127.0.0.1:8096".to_string()
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
            kb_base_url: default_kb_base_url(),
            chatbi_base_url: default_chatbi_base_url(),
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
        incoming.kb_base_url = "http://127.0.0.1:9999".to_string();

        let sanitized = sanitize_settings_update(&current, incoming);

        assert!(matches!(sanitized.mirror_source, MirrorSource::HfMirror));
        assert_eq!(sanitized.log_level, "DEBUG");
        assert_eq!(sanitized.ai_display_name.as_deref(), Some("小助手"));
        assert_eq!(sanitized.download_speed_limit, Some(5_000_000));
        assert_eq!(sanitized.proxy.as_deref(), Some("http://127.0.0.1:7890"));
        assert!(sanitized.autostart);
        assert_eq!(sanitized.extra_launch_args.as_deref(), Some("--ctx-size 8192"));
        assert_eq!(sanitized.service_host, "0.0.0.0");
        assert_eq!(sanitized.kb_base_url, "http://127.0.0.1:9999");
        // And the two protected pointers still come from `current`.
        assert_eq!(sanitized.local_model_path, current.local_model_path);
        assert_eq!(sanitized.llama_server_path, current.llama_server_path);
    }

    /// 旧 `config.json` 里没有 `kb_base_url` / `chatbi_base_url`（0.1.5 及更早写的）。
    /// 若 `#[serde(default = ...)]` 没生效，升级后**读取设置会整体失败**
    /// ——`load_settings` 用的是 `serde_json::from_str(...)?`，是 `?` 不是 `unwrap_or_default`，
    /// 所以少一个字段就会让 APP 起不来。这条护栏专门盯这个。
    ///
    /// 🔴 下面这份 JSON **照抄自真机的 `data/config.json`**（2026-10-09 取，其 mtime 早于
    ///    本次改动、文件里确实没有这两个字段，是一份天然的真实旧配置），**不是手写近似值**。
    ///    第一版凭想象写了两处：`"mirror_source": "Auto"`（serde 是
    ///    `rename_all = "snake_case"`，序列化名是小写 `"auto"`）、`"extra_launch_args": null`
    ///    （真实旧配置里是 `"--ctx-size 16384"`）。**测试数据一错，这条护栏就从没真正验证过。**
    #[test]
    fn legacy_config_without_kb_base_url_still_loads() {
        let 旧配置 = r#"{
  "install_dir": "D:\\Ivan-File\\10-LocalAI-Installer",
  "download_speed_limit": null,
  "mirror_source": "auto",
  "proxy": null,
  "log_level": "INFO",
  "autostart": false,
  "extra_launch_args": "--ctx-size 16384",
  "service_host": "127.0.0.1",
  "service_port": 8000,
  "llama_server_path": "D:\\Ivan-File\\10-LocalAI-Installer\\engine\\backends\\llama-server-cuda.exe",
  "local_model_path": "D:\\Ivan-File\\10-LocalAI-Installer\\models\\Spark-X2.5-4B-Q6_K\\Spark-X2.5-4B-Q6_K.gguf",
  "ai_display_name": null,
  "instance_id": "11111111-2222-3333-4444-555555555555"
}"#;

        let 设置: AppSettings =
            serde_json::from_str(旧配置).expect("缺 kb_base_url 的旧配置必须仍能反序列化");

        // 补默认值（这条护栏要证明的核心）
        assert_eq!(设置.kb_base_url, "http://127.0.0.1:8092");
        assert_eq!(设置.chatbi_base_url, "http://127.0.0.1:8096");
        // 其余字段必须**原样读回来**，不能被默认值顶掉
        assert_eq!(设置.service_port, 8000);
        assert_eq!(设置.extra_launch_args.as_deref(), Some("--ctx-size 16384"));
        assert_eq!(设置.instance_id, "11111111-2222-3333-4444-555555555555");
    }
}
