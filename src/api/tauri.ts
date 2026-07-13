import { invoke } from '@tauri-apps/api/core';
import type {
  AppSettings,
  CatalogRefreshResult,
  ChatMessage,
  ChatSession,
  DiagnoseResult,
  DownloadProgress,
  EngineInfo,
  HardwareInfo,
  ImportedModelInfo,
  InstalledModelInfo,
  LowVramFixResult,
  RecommendedModel,
  RolePreset,
  Scene,
  ServiceInfo
} from '../types';

export function getHardwareInfo() {
  return invoke<HardwareInfo>('get_hardware_info');
}

export function getLocalIp() {
  return invoke<string | null>('local_ip');
}

export function getRecommendModels(scene: Scene, hardware: HardwareInfo, manualVramGb?: number) {
  return invoke<RecommendedModel[]>('get_recommend_models', {
    request: {
      scene,
      hardware,
      manual_vram_gb: manualVramGb ?? null
    }
  });
}

export function startDownload(modelId: string) {
  return invoke<string>('start_download', { modelId });
}

export function getDownloadProgress(taskId: string) {
  return invoke<DownloadProgress>('get_download_progress', { taskId });
}

export function pauseDownload(taskId: string) {
  return invoke<void>('pause_download', { taskId });
}

export function resumeDownload(taskId: string) {
  return invoke<void>('resume_download', { taskId });
}

export function cancelDownload(taskId: string) {
  return invoke<void>('cancel_download', { taskId });
}

export function ensureEngineReady(backend: string) {
  return invoke<EngineInfo>('ensure_engine_ready', { request: { backend } });
}

export function startModelService(modelId: string) {
  return invoke<ServiceInfo>('start_model_service', { modelId });
}

export function stopModelService() {
  return invoke<ServiceInfo>('stop_model_service');
}

export function getServiceStatus() {
  return invoke<ServiceInfo>('get_service_status');
}

export function recoverService() {
  return invoke<ServiceInfo>('recover_service');
}

export function runDiagnose() {
  return invoke<DiagnoseResult[]>('run_diagnose');
}

export function applyLowVramFix() {
  return invoke<LowVramFixResult>('apply_low_vram_fix');
}

export function refreshModelCatalog() {
  return invoke<CatalogRefreshResult>('refresh_model_catalog');
}

export function exportLogs(destPath: string) {
  return invoke<string>('export_logs', { destPath });
}

export function getSettings() {
  return invoke<AppSettings>('get_settings');
}

export function listInstalledModels() {
  return invoke<InstalledModelInfo[]>('list_installed_models');
}

export function switchInstalledModel(path: string) {
  return invoke<InstalledModelInfo>('switch_installed_model', { path });
}

export function updateInstalledModel(modelId: string) {
  return invoke<ServiceInfo>('update_installed_model', { modelId });
}

export function importLocalModel(path: string) {
  return invoke<ImportedModelInfo>('import_local_model', { path });
}


export function updateSettings(settings: AppSettings) {
  return invoke<AppSettings>('update_settings', { settings });
}

export function createChatSession(modelId: string, rolePreset: RolePreset = 'daily_assistant') {
  return invoke<ChatSession>('create_chat_session', { modelId, rolePreset });
}

export function getChatSessions() {
  return invoke<ChatSession[]>('get_chat_sessions');
}

export function getChatMessages(sessionId: string) {
  return invoke<ChatMessage[]>('get_chat_messages', { sessionId });
}

export function saveChatMessage(sessionId: string, role: ChatMessage['role'], content: string) {
  return invoke<ChatMessage>('save_chat_message', {
    request: { session_id: sessionId, role, content }
  });
}

export function clearChatContext(sessionId: string) {
  return invoke<void>('clear_chat_context', { sessionId });
}

export function renameChatSession(sessionId: string, name: string) {
  return invoke<void>('rename_chat_session', { sessionId, name });
}

export function deleteChatSession(sessionId: string) {
  return invoke<void>('delete_chat_session', { sessionId });
}

export function updateSessionRole(sessionId: string, rolePreset: RolePreset) {
  return invoke<void>('update_session_role', { sessionId, rolePreset });
}
