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
  /**
   * 预估解码速度（**token/秒**，不是「字/秒」：1 个中文汉字通常约占 1.5–2 个
   * token，把 token 数当字数会高估约 1.5 倍）。按显存带宽与权重体积推算；
   * 显卡不在带宽表里时为 null。
   */
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
  /** 本轮**请求**的后端（调用方标签）：CUDA / Vulkan。 */
  backend: string;
  /**
   * 本轮**引擎实际加载**的后端 —— 由后端扫描 `engine/backends/` 根目录里
   * 真正存在的 GPU DLL 得出，不是调用方标签。两轮此值相同即代表
   * **没能切换后端**，校验结论不成立（后端会直接报错）。
   */
  effective_backend: string;
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
  /** 知识库问答服务地址（独立 Python 进程）。后端保证非空，默认 http://127.0.0.1:8092。 */
  kb_base_url: string;
  /**
   * 智能问数 / 动作执行服务地址（作品二 · ChatBI，独立 Python 进程）。
   * 后端保证非空，默认 http://127.0.0.1:8096（编排服务，一个入口三条链路）。
   */
  chatbi_base_url: string;
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
  /** 模型自带微调包（旁车配置 `<主名>.launch.json`）：启动时会自动挂 LoRA
   *  适配器并套用与训练一致的对话模板。键名由 Rust 侧的护栏测试钉住。 */
  has_finetune_bundle: boolean;
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
  /**
   * 知识库模式专有：服务返回的结构化结果（引用卡片、拒答原因）。
   *
   * 🔴 **只在本次进程内有效**。`chat_messages` 表没有这一列，落库时只存
   * `content`（答案正文 + 文末的纯文本引用）。刷新后卡片降级为文本 ——
   * 这是本轮有意选择的取舍：项目当时**没有版本化迁移机制**
   * （`database.rs` 只 `execute_batch(CREATE TABLE IF NOT EXISTS)`，
   * 往已存在的库里加列不会生效），为一个 UI 展示引入迁移基建不划算。
   * 要把卡片做成可持久化时，这是一个独立任务：加迁移 + `ALTER TABLE` + 兼容分支。
   */
  kb?: KbMessageMeta;
  /**
   * 智能问数模式专有：服务返回的链路 / 轨迹 / 权限结论。
   *
   * 🔴 与 `kb` 同一条取舍：**只在本次进程内有效**。落库时摊平成纯文本
   * （见 `utils/chatbi.ts::chatbiMetaToPlainText`），刷新后轨迹降级为文字。
   * 理由同样是「没有版本化迁移机制」，为一个 UI 展示引入迁移基建不划算。
   */
  chatbi?: ChatbiMessageMeta;
}

/** 知识库服务 `POST /ask` 的返回。字段全 ASCII —— 与 Python 侧逐字对应。 */
export interface KbAnswer {
  question: string;
  /** 全库最高重排分（原始 logit，非概率）。低于阈值即拒答（类1）。 */
  top_score: number;
  citations: KbCitation[];
  /** 送进生成端的原文。前端做「依据」展开，也用于核对数值是否照抄。 */
  context: string;
  refused: boolean;
  /** `score_threshold` | `broken_citation_chain` | `model_refused` | `""` */
  refusal_type: string;
  refusal_reason: string;
  /** Top-1 块里出现、但不在库内的文档号（类2 的证据）。 */
  off_library_refs: string[];
  answer: string;
  /** 生成端自己说了「资料不足」（类3，兜底）。 */
  model_refused: boolean;
  timings: { retrieve?: number; generate?: number; total?: number };
}

export interface KbCitation {
  rank: number;
  doc_no: string;
  section: string;
  score: number;
  excerpt: string;
}

/** 每个助手消息随身带的知识库元信息。 */
export interface KbMessageMeta {
  citations: KbCitation[];
  refused: boolean;
  refusalType: string;
  refusalReason: string;
  offLibraryRefs: string[];
  topScore: number;
  timings?: { retrieve?: number; generate?: number; total?: number };
}

/** `GET /health` 的返回（仅取前端要用的字段）。 */
export interface KbHealth {
  status: string;
  docs: number;
  blocks: number;
  pool: number;
  top_k: number;
  class2_soft: boolean;
  chain: string;
  /**
   * 服务内部依赖的地址。
   *
   * 🔴 `generate` 必须和 APP 设置里的 `service_port` 对上。知识库服务默认写死
   * `http://127.0.0.1:8000`，而 APP 的生成端端口是用户可配的 —— 不一致时现象是
   * 「问什么都答资料不足 / 生成失败」，**不报错，只是没结果**。前端拿它做核对。
   */
  deps: { rerank: string; generate: string };
}

// ===========================================================================
// 作品二 · ChatBI（智能问数 / 动作执行）—— 对接类型
// ===========================================================================
/**
 * 🔴 这些字段名是**中文**，而且**故意不翻成英文**。
 *
 * 服务端（`ChatBI/scripts/编排.py::执行()` 的返回）用的就是中文键名，而这条链路
 * 是前端**直接 fetch** JSON（不经 Tauri IPC、不经过任何 Rust 结构体）。
 * 中间加一层「中文 → 英文」的映射 = 同一份事实两处定义 ——
 * 服务端改一个字段名，前端**不报错，只是那个字段永远是 undefined**。
 * 所以这里逐字对应服务端。
 *
 * （对比：`KbAnswer` 是 ASCII，因为作品一的服务端本来就用了 ASCII。）
 */
export interface ChatbiAnswer {
  问题: string;
  /** `知识问答` | `数据问数` | `动作执行` */
  类型: string;
  路由: ChatbiRoute;
  /** ReAct 轨迹。知识 / 数据链路是单步委派，这里通常为空。 */
  步骤: ChatbiStep[];
  /** 依据与问题不匹配（包 5 的护栏：单据数字对、但依据不支撑它）。 */
  依据不切题: boolean;
  权限: ChatbiPermission;
  /** 越权 → true。此时 `产物` 必为 null（硬拒，不留半成品）。 */
  被拒: boolean;
  产物: ChatbiArtifact | null;
  /** 各链路形状不同：知识=检索结果、数据=SQL+行列、动作=取数观察。按 `类型` 分支读。 */
  依据: Record<string, any> | null;
  错误: string | null;
  耗时: { 合计?: number };
  回答: string;
  总耗时秒: number;
}

export interface ChatbiRoute {
  链路: string;
  /** `确定性`（规则命中）| `模型`（兜底）| `默认` | `强制`。**兜底必须让用户看见**。 */
  依据: string;
  命中: Record<string, string[] | string | null>;
  模型用时: number | null;
  理由: string;
}

export interface ChatbiStep {
  步: number;
  动作: string;
  参数: Record<string, any>;
  思考: string;
  观察: string;
  模型原文?: string;
   /** 越权 = 硬拒：不换动作重试、不落盘。 */
  被拒?: boolean;
}

export interface ChatbiPermission {
  角色: string | null;
  被拒: boolean;
  /** 拒绝理由里必须写清「需要哪个角色」——**拒绝要给出路**。 */
  拒绝理由: string | null;
}

export interface ChatbiArtifact {
  路径?: string;
  动作?: string;
  [k: string]: any;
}

/** 每个助手消息随身带的 ChatBI 元信息。 */
export interface ChatbiMessageMeta {
  类型: string;
  路由依据: string;
  路由理由: string;
  步骤: ChatbiStep[];
  被拒: boolean;
  拒绝理由: string | null;
  角色: string | null;
  依据不切题: boolean;
  产物路径: string | null;
  耗时秒: number;
}

/** `GET /health` 的返回（仅取前端要用的字段）。 */
export interface ChatbiHealth {
  status: string;
  版本: string;
  端口: number;
  /**
   * 三条链路各自的下游状态。
   *
   * 🔴 「8096 起来了」**不等于**「三条链路都通」—— 知识问答那条是 HTTP 转发给
   * 8092（作品一）的，8092 没起时 8096 照样 200，只是那条链路不可达。
   * 所以界面必须逐条显示，不能只看服务在不在。
   */
  链路: Record<string, ChatbiChainStatus>;
  log_file: string | null;
  console_dropped: number;
  审计: ChatbiAuditStat;
  权限: { 默认角色: string; 角色: string[]; 白名单问题: string[] };
  路由确定性: Record<string, number>;
  动作: string[];
}

export interface ChatbiChainStatus {
  可达?: boolean;
  地址?: string;
  bytes?: number;
  产物目录?: string;
  可写?: boolean;
}

export interface ChatbiAuditStat {
  文件: string;
  总条数: number;
  被拒数: number;
  落盘数: number;
  /** 依据不切题的次数 —— **白名单只防「不许做的事」，它防「做了但做错的事」**。 */
  依据不切题数: number;
  按角色: Record<string, number>;
  按动作: Record<string, number>;
}
