use std::{
    collections::HashMap,
    path::PathBuf,
    process::Child,
    sync::{atomic::{AtomicBool, AtomicU8}, Arc, Mutex},
};

use serde::{Deserialize, Serialize};

use crate::{download::DownloadTask, settings::AppSettings};

#[derive(Clone)]
pub struct AppState {
    pub inner: Arc<Mutex<AppStateInner>>,
}

pub struct AppStateInner {
    pub settings: AppSettings,
    pub downloads: HashMap<String, DownloadTask>,
    pub download_controls: HashMap<String, Arc<AtomicU8>>,
    pub service: ServiceState,
    pub data_dir: PathBuf,
    pub db_path: PathBuf,
    /// Stop flag for the background "crashed llama-server? auto-restart"
    /// watchdog. The flag is set to `true` on user-initiated stop / model
    /// switch so the loop exits on its next iteration. The field is `None`
    /// when no watchdog is currently running.
    pub watchdog_stop: Option<Arc<AtomicBool>>,
    /// 上一次「后端一致性自检被强制放行」时看到的来源（`None` = 上一次没有放行）。
    ///
    /// **P3-5**：可用性结论每刷新一次就会重算一次，而放行 WARN 原先每次都打，
    /// 同一个状态能刷出好几条一样的日志。这里记下上次状态，**只在状态变化时打**。
    /// 但两个方向的变化都要留痕 —— 放行 → WARN，解除 → INFO，**绝不能静默**。
    pub 强制放行上次来源: Option<String>,
}

#[derive(Debug, Default)]
pub struct ServiceState {
    pub model_id: Option<String>,
    pub port: Option<u16>,
    /// Bind address the running service is listening on (e.g. "127.0.0.1" or
    /// "0.0.0.0"). Lets the frontend show the right call address for LAN mode.
    pub host: Option<String>,
    pub status: ServiceStatus,
    pub process: Option<Child>,
    /// OS pid of the running server. Set for both processes we spawned and
    /// processes we reattached to after a restart (where `process` is None).
    pub pid: Option<u32>,
    pub safe_mode: bool,
    /// 启动成功但性能打折的原因（见 `ServiceInfo::warning`）。
    /// 存在状态里而不是只放在返回结构里：状态会被 `get_service_status`
    /// 反复重建，只放返回值的话，前端一次轮询就能把提示抹掉。
    pub warning: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ServiceStatus {
    #[default]
    Stopped,
    Starting,
    Running,
    Failed,
}

impl AppState {
    pub fn new(settings: AppSettings, data_dir: PathBuf, db_path: PathBuf, downloads: HashMap<String, DownloadTask>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(AppStateInner {
                settings,
                downloads,
                download_controls: HashMap::new(),
                service: ServiceState::default(),
                data_dir,
                db_path,
                watchdog_stop: None,
                强制放行上次来源: None,
            })),
        }
    }
}
