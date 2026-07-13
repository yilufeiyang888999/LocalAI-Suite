CREATE TABLE IF NOT EXISTS model_metadata (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  author TEXT NOT NULL,
  base_model TEXT NOT NULL,
  quant_level TEXT NOT NULL,
  quant_type TEXT NOT NULL,
  quant_bpp REAL NOT NULL,
  parameter_count TEXT NOT NULL,
  parameters_raw INTEGER NOT NULL,
  active_parameters INTEGER,
  is_moe INTEGER NOT NULL DEFAULT 0,
  vram_required_gb REAL NOT NULL,
  file_size_mb INTEGER NOT NULL,
  context_length INTEGER NOT NULL,
  recommended_scene TEXT NOT NULL,
  chinese_score INTEGER NOT NULL,
  speed_rating INTEGER NOT NULL,
  architecture TEXT NOT NULL,
  model_format TEXT NOT NULL,
  release_date TEXT NOT NULL,
  download_url TEXT NOT NULL,
  mirror_urls TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  supported_gpu TEXT NOT NULL,
  license TEXT NOT NULL,
  is_imported INTEGER NOT NULL DEFAULT 0,
  local_path TEXT,
  supersedes_model_id TEXT
);

CREATE TABLE IF NOT EXISTS download_tasks (
  id TEXT PRIMARY KEY,
  model_id TEXT NOT NULL,
  save_path TEXT NOT NULL,
  total_bytes INTEGER NOT NULL,
  downloaded_bytes INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL,
  current_source TEXT NOT NULL,
  source_history TEXT NOT NULL,
  create_time TEXT NOT NULL,
  update_time TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_sessions (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  model_id TEXT NOT NULL,
  role_preset TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS system_config (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
