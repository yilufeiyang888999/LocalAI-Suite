mod app_state;
mod backend_check;
mod catalog;
mod chat;
mod data_mgmt;
mod database;
mod diagnose;
mod download;
mod engine;
mod error;
mod fix;
mod gguf;
mod hardware;
mod hash;
mod logging;
pub mod manifest;
mod model_repo;
mod model_sidecar;
mod models;
mod proc;
mod recommend;
mod service;
mod settings;
mod uninstall;
mod update;

use std::collections::HashMap;

use app_state::AppState;
use database::{init_database, load_download_tasks};
use settings::load_settings;

/// 返回二进制的内部版本号（取自 `Cargo.toml` 的 `CARGO_PKG_VERSION`）。
///
/// 刻意 **不**用 `tauri.conf.json` 的 version、也不用前端 `package.json` 的 version：
/// 那两个文件与 `Cargo.toml` 可能被分别改动而不同步（本项目踩过：包名 0.1.1、
/// 二进制内部却是 0.1.0）。此命令与日志首行的 `version=` 字段**同源**，
/// 因此「关于」里显示的版本可以和日志逐字对上 —— 一眼看出是否装错了包。
#[tauri::command]
fn app_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

pub fn run() {
    let data_dir = dirs::data_local_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("LocalAI-Installer")
        .join("data");
    // Handle a pending factory reset BEFORE any settings/logging/DB init reads
    // or opens the files we're about to delete. Best-effort: never panics.
    data_mgmt::perform_pending_reset(&data_dir);
    let settings = load_settings(&data_dir).unwrap_or_default();
    let _log_guard = logging::init(&data_dir, &settings.log_level);
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        log_level = %settings.log_level,
        "LocalAI-Installer backend starting"
    );
    let db_path = init_database(&data_dir).unwrap_or_else(|error| {
        tracing::warn!("failed to initialize sqlite database: {}", error);
        database::database_path(&data_dir)
    });
    if let Err(error) = model_repo::seed_catalog(&db_path, &catalog::effective_models(&data_dir)) {
        tracing::warn!("failed to seed model catalog: {}", error);
    }
    let downloads = load_download_tasks(&db_path)
        .map(|tasks| {
            tasks
                .into_iter()
                .map(|task| (task.id.clone(), task))
                .collect::<HashMap<_, _>>()
        })
        .unwrap_or_else(|error| {
            tracing::warn!("failed to load download tasks: {}", error);
            HashMap::new()
        });
    let state = AppState::new(settings, data_dir, db_path, downloads);
    tracing::info!("backend state initialized; launching desktop shell");

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(state)
        .invoke_handler(tauri::generate_handler![
            hardware::get_hardware_info,
            hardware::local_ip,
            recommend::get_recommend_models,
            download::start_download,
            download::get_download_progress,
            download::pause_download,
            download::resume_download,
            download::cancel_download,
            engine::ensure_engine_ready,
            models::list_installed_models,
            models::switch_installed_model,
            models::import_local_model,
            update::update_installed_model,
            uninstall::uninstall_model,
            service::start_model_service,
            service::stop_model_service,
            service::get_service_status,
            service::recover_service,
            backend_check::get_backend_check_availability,
            backend_check::run_backend_consistency_check,
            diagnose::run_diagnose,
            fix::apply_low_vram_fix,
            catalog::refresh_model_catalog,
            logging::export_logs,
            app_version,
            data_mgmt::get_data_paths,
            data_mgmt::clear_all_chat_records,
            data_mgmt::uninstall_all_models,
            data_mgmt::reset_to_factory,
            chat::create_chat_session,
            chat::get_chat_sessions,
            chat::get_chat_messages,
            chat::save_chat_message,
            chat::clear_chat_context,
            chat::rename_chat_session,
            chat::delete_chat_session,
            chat::update_session_role,
            settings::get_settings,
            settings::update_settings
        ])
        .run(tauri::generate_context!())
        .expect("failed to run tauri app");
}
