use std::{
    collections::HashMap,
    path::PathBuf,
    process::Child,
    sync::{atomic::AtomicU8, Arc, Mutex},
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
            })),
        }
    }
}
