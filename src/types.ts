export type BackendType = 'CUDA' | 'Vulkan' | 'CPUx86';

export interface HardwareInfo {
  gpu_list: GpuInfo[];
  gpu_name: string;
  gpu_vram_gb: number;
  gpu_count: number;
  has_gpu: boolean;
  cpu: CpuInfo;
  memory: MemoryInfo;
  disk: DiskInfo;
  backend: BackendType;
  needs_manual_tier: boolean;
  summary: string;
}

export interface GpuInfo {
  index: number;
  vendor: string;
  name: string;
  vram_gb: number;
  driver_version: string;
  bandwidth_gbs?: number | null;
}

export interface CpuInfo {
  name: string;
  cores: number;
  threads: number;
  arch: string;
}

export interface MemoryInfo {
  total_gb: number;
  available_gb: number;
}

export interface DiskInfo {
  install_path: string;
  available_gb: number;
}

export type Scene = 'chat' | 'writing' | 'coding' | 'general';

export interface ModelMetadata {
  id: string;
  name: string;
  author: string;
  base_model: string;
  display_compression: string;
  vram_required_gb: number;
  file_size_mb: number;
  context_length: number;
  recommended_scene: Scene[];
  chinese_score: number;
  speed_rating: number;
  architecture: string;
  license: string;
  is_installed: boolean;
  local_path?: string | null;
}

export interface RecommendedModel {
  model: ModelMetadata;
  rank: 'primary' | 'alternative';
  score: number;
  reason: string;
  speed_label: string;
  /** 预估解码速度（字/秒），按显存带宽与权重体积推算。显卡不在带宽表里时为 null。 */
  estimated_tps: number | null;
  quality_label: string;
}

export type DownloadStatus = 'pending' | 'downloading' | 'paused' | 'verifying' | 'failed' | 'completed' | 'cancelled';

/** 后端一致性自检：能不能跑。 */
export interface BackendCheckAvailability {
  available: boolean;
  /** 当前正在使用的后端：CUDA / Vulkan / CPU。 */
  current_backend: string;
  /** 可以拿来对比的后端名列表。 */
  comparable_backends: string[];
  /** 不可用时的原因。 */
  reason: string | null;
}

/** 一个后端的一次生成结果。 */
export interface BackendRun {
  backend: string;
  text: string;
  token_count: number;
  /** limit = 打满 n_predict；stop = 命中 EOS。 */
  stop_type: string;
  elapsed_sec: number;
  tokens_per_sec: number;
}

/** 比对结论。 */
export type ConsistencyVerdict = 'identical' | 'length_only' | 'diverged';

export interface BackendComparison {
  verdict: ConsistencyVerdict;
  common_prefix_len: number;
  first_divergence: number | null;
  match_ratio: number;
  summary: string;
}

export interface BackendCheckResult {
  first: BackendRun;
  second: BackendRun;
  comparison: BackendComparison;
  /** 自检结束后恢复到的后端。 */
  restored_to: string;
}

export interface DownloadProgress {
  task_id: string;
  model_id: string;
  downloaded_bytes: number;
  total_bytes: number;
  speed_bps: number;
  status: DownloadStatus;
  current_step: string;
}

export interface EngineInfo {
  backend: BackendType;
  version: string;
  executable_path: string;
  message: string;
}

export type MirrorSource = 'auto' | 'model_scope' | 'hf_mirror' | { custom: string };

export interface AppSettings {
  install_dir: string;
  download_speed_limit?: number | null;
  mirror_source: MirrorSource;
  proxy?: string | null;
  log_level: string;
  autostart?: boolean;
  extra_launch_args?: string | null;
  service_host?: string;
  service_port?: number;
  llama_server_path?: string | null;
  local_model_path?: string | null;
  ai_display_name?: string | null;
  instance_id: string;
}

export interface InstalledModelInfo {
  model_id: string;
  name: string;
  local_path: string;
  file_size_mb: number;
  source: 'builtin' | 'imported' | string;
  is_current: boolean;
  update_available: boolean;
}

export interface UpdateProgress {
  model_id: string;
  phase: 'downloading' | 'verifying' | 'switching' | 'starting' | 'completed' | 'rolled_back' | 'failed' | string;
  downloaded_bytes: number;
  total_bytes: number;
  message: string;
}

export interface ImportedModelInfo {
  model_id: string;
  name: string;
  local_path: string;
  file_size_mb: number;
  estimated_vram_gb: number;
  message: string;
}

export interface UninstallResult {
  model_id: string;
  source: 'builtin' | 'imported' | string;
  deleted_chat_sessions: number;
  cleared_current_model: boolean;
  message: string;
}

export interface DataPaths {
  data_dir: string;
  install_dir: string;
  logs_dir: string;
}

export interface ClearChatResult {
  deleted_sessions: number;
  message: string;
}

export interface UninstallAllResult {
  removed_count: number;
  failed_count: number;
  message: string;
}

export interface ResetResult {
  message: string;
}

export interface ServiceInfo {
  model_id?: string | null;
  port?: number | null;
  host?: string | null;
  status: 'stopped' | 'starting' | 'running' | 'failed';
  safe_mode: boolean;
  message: string;
}

export interface DiagnoseResult {
  issue_id: string;
  title: string;
  description: string;
  auto_level: 'full_auto' | 'semi_auto' | 'manual_guide';
  fixable: boolean;
  ok: boolean;
}

export interface LowVramFixResult {
  message: string;
  switched: boolean;
  suggested_model_id?: string | null;
  service?: ServiceInfo | null;
}

export interface CatalogRefreshResult {
  updated: boolean;
  version: number;
  model_count: number;
  message: string;
}

export type RolePreset = 'daily_assistant' | 'writing' | 'coding';

export interface ChatSession {
  id: string;
  name: string;
  model_id: string;
  role_preset: RolePreset;
  created_at: string;
  updated_at: string;
}

export interface ChatMessage {
  id?: string;
  session_id?: string;
  role: 'user' | 'assistant';
  content: string;
  created_at?: string;
}
