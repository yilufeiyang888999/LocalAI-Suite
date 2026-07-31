<script setup lang="ts">
import { listen } from '@tauri-apps/api/event';
import { onMounted, ref } from 'vue';
import HardwareCheck from './components/HardwareCheck.vue';
import ScenarioSelect from './components/ScenarioSelect.vue';
import ModelRecommendation from './components/ModelRecommendation.vue';
import InstallProgress from './components/InstallProgress.vue';
import InstalledModels from './components/InstalledModels.vue';
import SettingsPanel from './components/SettingsPanel.vue';
import DiagnosticsPanel from './components/DiagnosticsPanel.vue';
import ChatView from './components/ChatView.vue';
import ThirdPartyConnect from './components/ThirdPartyConnect.vue';
import {
  ensureEngineReady,
  getHardwareInfo,
  getRecommendModels,
  getDownloadProgress,
  getServiceStatus,
  getSettings,
  importLocalModel,
  listInstalledModels,
  recoverService,
  runDiagnose,
  applyLowVramFix,
  refreshModelCatalog,
  startDownload,
  pauseDownload,
  resumeDownload,
  cancelDownload,
  startModelService,
  stopModelService,
  switchInstalledModel,
  uninstallModel,
  updateInstalledModel
} from './api/tauri';
import { openUrl } from '@tauri-apps/plugin-opener';
import { open as openFileDialog } from '@tauri-apps/plugin-dialog';
import type {
  AppSettings,
  DiagnoseResult,
  DownloadProgress,
  HardwareInfo,
  InstalledModelInfo,
  RecommendedModel,
  Scene,
  ServiceInfo,
  UpdateProgress
} from './types';

const hardware = ref<HardwareInfo | null>(null);
const hardwareLoading = ref(true);
const hardwareError = ref<string | null>(null);
const manualVramTier = ref<number | null>(null);
const selectedScene = ref<Scene>('chat');
const recommendations = ref<RecommendedModel[]>([]);
const recommendationLoading = ref(false);
const selectedModelId = ref<string | null>(null);
const progress = ref<DownloadProgress | null>(null);
const service = ref<ServiceInfo | null>(null);
const serviceMessage = ref<string | null>(null);
const settings = ref<AppSettings | null>(null);
const installedModels = ref<InstalledModelInfo[]>([]);
const installedModelsLoading = ref(false);
const switchingModelPath = ref<string | null>(null);
const updatingModelId = ref<string | null>(null);
const updateMessage = ref<string | null>(null);
const uninstallingModelId = ref<string | null>(null);
const importingLocalModel = ref(false);
const restoringGpu = ref(false);
const diagnoseResults = ref<DiagnoseResult[]>([]);
const diagnoseLoading = ref(false);
const fixingIssueId = ref<string | null>(null);

onMounted(async () => {
  await Promise.all([loadHardware(), loadSettings(), loadService(), loadInstalledModels()]);
  await tryAutoStartService();
  // Best-effort online catalog refresh; when it brings new models, refresh the
  // recommendation list. Runs off the critical path so a slow/absent endpoint
  // never delays startup.
  refreshCatalogQuietly();
  await listen<DownloadProgress>('download_progress', async (event) => {
    progress.value = event.payload;
    if (['completed', 'failed', 'cancelled'].includes(event.payload.status)) {
      selectedModelId.value = null;
    }
    if (event.payload.status === 'completed') {
      await handleModelReady(event.payload.model_id);
    }
  });
  await listen<ServiceInfo>('service_status_changed', (event) => {
    service.value = event.payload;
    serviceMessage.value = event.payload.message;
  });
  await listen<UpdateProgress>('model_update_progress', (event) => {
    if (updatingModelId.value === event.payload.model_id) {
      updateMessage.value = event.payload.message;
    }
  });
});

async function loadHardware() {
  hardwareLoading.value = true;
  hardwareError.value = null;
  try {
    hardware.value = await getHardwareInfo();
    await loadRecommendations();
  } catch (error) {
    hardwareError.value = String(error);
  } finally {
    hardwareLoading.value = false;
  }
}

async function loadSettings() {
  settings.value = await getSettings();
}


async function loadService() {
  // Reattach to a service left running by a previous session when possible;
  // otherwise this returns the current (stopped) status.
  service.value = await recoverService();
  serviceMessage.value = service.value.message;
}

async function tryAutoStartService() {
  if (!service.value || service.value.status !== 'stopped') return;
  if (!installedModels.value.length) return;
  
  const currentModel = installedModels.value.find(m => m.is_current);
  if (!currentModel) return;
  
  serviceMessage.value = `检测到上次使用的模型 ${currentModel.name}，正在自动启动服务。`;
  try {
    await prepareEngineAndStart(currentModel.model_id);
  } catch (error) {
    serviceMessage.value = String(error);
    service.value = await getServiceStatus();
  }
}

async function loadInstalledModels() {
  installedModelsLoading.value = true;
  try {
    installedModels.value = await listInstalledModels();
  } finally {
    installedModelsLoading.value = false;
  }
}

async function refreshCatalogQuietly() {
  try {
    const result = await refreshModelCatalog();
    if (result.updated) {
      await loadRecommendations();
      await loadInstalledModels();
    }
  } catch {
    // Offline / unconfigured endpoint: keep the bundled catalog silently.
  }
}

async function selectScene(scene: Scene) {
  selectedScene.value = scene;
  await loadRecommendations();
}

async function loadRecommendations() {
  if (!hardware.value) return;
  recommendationLoading.value = true;
  try {
    recommendations.value = await getRecommendModels(selectedScene.value, hardware.value, manualVramTier.value ?? undefined);
  } finally {
    recommendationLoading.value = false;
  }
}

async function selectManualTier(tier: number | null) {
  manualVramTier.value = tier;
  await loadRecommendations();
}

async function installModel(item: RecommendedModel) {
  await startInstall(item.model.id);
}

async function startInstall(modelId: string) {
  selectedModelId.value = modelId;
  serviceMessage.value = null;
  progress.value = null;
  try {
    const taskId = await startDownload(modelId);
    const currentProgress = await getDownloadProgress(taskId);
    progress.value = currentProgress;
    if (currentProgress.status === 'completed') {
      selectedModelId.value = null;
      await handleModelReady(currentProgress.model_id);
    } else if (['failed', 'cancelled'].includes(currentProgress.status)) {
      selectedModelId.value = null;
    }
  } catch (error) {
    selectedModelId.value = null;
    serviceMessage.value = String(error);
  }
}

async function pauseCurrentDownload(taskId: string) {
  try {
    await pauseDownload(taskId);
  } catch (error) {
    serviceMessage.value = String(error);
  }
}

async function resumeCurrentDownload(taskId: string) {
  try {
    await resumeDownload(taskId);
  } catch (error) {
    serviceMessage.value = String(error);
  }
}

async function cancelCurrentDownload(taskId: string) {
  try {
    await cancelDownload(taskId);
    selectedModelId.value = null;
  } catch (error) {
    serviceMessage.value = String(error);
  }
}

async function handleModelReady(modelId: string) {  serviceMessage.value = '模型文件已下载完成，正在准备本地AI运行引擎。';
  await prepareEngineAndStart(modelId);
  await loadInstalledModels();
}

async function prepareEngineAndStart(modelId: string) {
  try {
    const backend = hardware.value?.backend ?? 'CPUx86';
    try {
      await ensureEngineReady(backend);
    } catch (error) {
      if (backend === 'CPUx86') throw error;
      serviceMessage.value = '显卡加速暂时没准备好，正在切换到更稳妥的 CPU 模式。';
      await ensureEngineReady('CPUx86');
    }
    settings.value = await getSettings();
    serviceMessage.value = '运行引擎已准备好，正在启动本地AI服务。';
    await tryStartService(modelId);
  } catch (error) {
    serviceMessage.value = String(error);
    service.value = await getServiceStatus();
  }
}

async function switchModel(model: InstalledModelInfo) {
  if (model.is_current || switchingModelPath.value) return;
  switchingModelPath.value = model.local_path;
  serviceMessage.value = `正在切换到 ${model.name}。`;
  try {
    await switchInstalledModel(model.local_path);
    settings.value = await getSettings();
    await loadInstalledModels();
    await prepareEngineAndStart(model.model_id);
    await loadInstalledModels();
  } catch (error) {
    serviceMessage.value = String(error);
    service.value = await getServiceStatus();
  } finally {
    switchingModelPath.value = null;
  }
}

async function updateModel(model: InstalledModelInfo) {
  if (updatingModelId.value || switchingModelPath.value) return;
  updatingModelId.value = model.model_id;
  updateMessage.value = `正在检查并更新 ${model.name} 到最新版本。`;
  try {
    await updateInstalledModel(model.model_id);
  } catch (error) {
    updateMessage.value = String(error);
  } finally {
    updatingModelId.value = null;
    await loadInstalledModels();
    service.value = await getServiceStatus();
  }
}

async function stopService() {
  serviceMessage.value = '正在停止AI服务。';
  try {
    service.value = await stopModelService();
    serviceMessage.value = service.value.message;
  } catch (error) {
    serviceMessage.value = String(error);
    service.value = await getServiceStatus();
  }
}

async function startCurrentModel(model: InstalledModelInfo) {
  serviceMessage.value = `正在启动 ${model.name}。`;
  try {
    await prepareEngineAndStart(model.model_id);
    await loadInstalledModels();
  } catch (error) {
    serviceMessage.value = String(error);
    service.value = await getServiceStatus();
  }
}

async function uninstallModelAndRefresh(model: InstalledModelInfo, deleteChatHistory: boolean) {
  if (uninstallingModelId.value) return;
  uninstallingModelId.value = model.model_id;
  serviceMessage.value = `正在卸载 ${model.name}。`;
  try {
    const result = await uninstallModel(model.model_id, deleteChatHistory, model.local_path);
    serviceMessage.value = result.message;
    settings.value = await getSettings();
    await loadInstalledModels();
  } catch (error) {
    serviceMessage.value = String(error);
  } finally {
    uninstallingModelId.value = null;
  }
}

async function importLocalGguf() {
  if (importingLocalModel.value) return;
  const selected = await openFileDialog({
    multiple: false,
    filters: [{ name: '模型文件', extensions: ['gguf'] }]
  });
  if (typeof selected !== 'string') return;
  importingLocalModel.value = true;
  serviceMessage.value = '正在导入本地模型文件。';
  try {
    const imported = await importLocalModel(selected);
    serviceMessage.value = imported.message;
    settings.value = await getSettings();
    await loadInstalledModels();
  } catch (error) {
    serviceMessage.value = String(error);
  } finally {
    importingLocalModel.value = false;
  }
}

async function tryStartService(modelId: string) {
  try {
    service.value = await startModelService(modelId);
    serviceMessage.value = service.value.message;
  } catch (error) {
    serviceMessage.value = String(error);
    service.value = await getServiceStatus();
  }
}

async function restoreGpuAcceleration() {
  if (!service.value?.model_id || restoringGpu.value) return;
  restoringGpu.value = true;
  serviceMessage.value = '正在尝试恢复显卡加速，请稍候。';
  try {
    // start_model_service always attempts GPU (CUDA) first, so this retries
    // acceleration; if it still fails it falls back to CPU safe mode again.
    service.value = await startModelService(service.value.model_id);
    serviceMessage.value = service.value.message;
  } catch (error) {
    serviceMessage.value = String(error);
    service.value = await getServiceStatus();
  } finally {
    restoringGpu.value = false;
  }
}

async function restartService() {
  if (!service.value?.model_id) return;
  serviceMessage.value = '正在按新的接口设置重启AI服务，请稍候。';
  try {
    // start_model_service reads the current host/port settings and replaces the
    // running server, so this applies a changed接口地址/端口.
    service.value = await startModelService(service.value.model_id);
    serviceMessage.value = service.value.message;
  } catch (error) {
    serviceMessage.value = String(error);
    service.value = await getServiceStatus();
  }
}

async function handleDataChanged() {
  // After batch data-mgmt operations (clear all chat, uninstall all, reset)
  // refresh everything the UI reads from disk/DB so state doesn't go stale.
  settings.value = await getSettings();
  await loadInstalledModels();
  service.value = await getServiceStatus();
}

async function runChecks() {
  diagnoseLoading.value = true;
  try {
    diagnoseResults.value = await runDiagnose();
  } finally {
    diagnoseLoading.value = false;
  }
}

async function handleFix(issueId: string) {
  if (fixingIssueId.value) return;
  fixingIssueId.value = issueId;
  serviceMessage.value = null;
  try {
    if (issueId === 'vram') {
      const result = await applyLowVramFix();
      serviceMessage.value = result.message;
      if (result.service) service.value = result.service;
      if (result.suggested_model_id) {
        await startInstall(result.suggested_model_id);
      }
      await loadInstalledModels();
    } else if (issueId === 'runtime_libs') {
      const backend = hardware.value?.backend ?? 'CPUx86';
      await ensureEngineReady(backend);
      settings.value = await getSettings();
      serviceMessage.value = '已重新准备本地AI运行引擎。';
    } else if (issueId === 'model_file') {
      const current = installedModels.value.find((model) => model.is_current) ?? installedModels.value[0];
      if (current) {
        await startInstall(current.model_id);
      } else {
        serviceMessage.value = '还没有已安装的模型，请先在上方完成一次安装。';
      }
    } else if (issueId === 'gpu_driver') {
      await openUrl('https://www.nvidia.cn/geforce/drivers/');
      serviceMessage.value = '已打开显卡驱动下载页面，升级后重启本软件即可尝试显卡加速。';
    } else if (issueId === 'disk_space') {
      serviceMessage.value = '请在右侧「设置」中更换到空间更充足的 AI 文件保存位置。';
    }
  } catch (error) {
    serviceMessage.value = String(error);
  } finally {
    fixingIssueId.value = null;
    await runChecks();
  }
}
</script>

<template>
  <main class="app-shell">
    <header class="app-header">
      <div class="header-content">
        <div>
          <h1>本地AI模型一键部署助手</h1>
          <p>检测电脑配置，推荐合适模型，并引导启动本地AI聊天。</p>
        </div>
      </div>
    </header>

    <div class="layout">
      <div class="main-flow">
        <HardwareCheck
          :hardware="hardware"
          :loading="hardwareLoading"
          :error="hardwareError"
          :manual-tier="manualVramTier"
          @refresh="loadHardware"
          @select-tier="selectManualTier"
        />
        <ScenarioSelect :selected="selectedScene" @select="selectScene" />
        <ModelRecommendation
          :models="recommendations"
          :loading="recommendationLoading"
          :selected-id="selectedModelId"
          @install="installModel"
        />
        <InstallProgress
          :progress="progress"
          :service-message="serviceMessage"
          @pause="pauseCurrentDownload"
          @resume="resumeCurrentDownload"
          @cancel="cancelCurrentDownload"
        />
        <InstalledModels
          :models="installedModels"
          :loading="installedModelsLoading"
          :switching-path="switchingModelPath"
          :updating-id="updatingModelId"
          :update-message="updateMessage"
          :uninstalling-id="uninstallingModelId"
          :importing="importingLocalModel"
          :service="service"
          @refresh="loadInstalledModels"
          @switch="switchModel"
          @update="updateModel"
          @stop="stopService"
          @start="startCurrentModel"
          @uninstall="uninstallModelAndRefresh"
          @import="importLocalGguf"
        />
        <ChatView
          :service="service"
          :settings="settings"
          :restoring-gpu="restoringGpu"
          @restore-gpu="restoreGpuAcceleration"
          @restart-service="restartService"
        />
        <ThirdPartyConnect :service="service" :settings="settings" />
      </div>
      <aside class="side-flow">
        <SettingsPanel
          :settings="settings"
          :model-id="service?.model_id ?? null"
          @saved="settings = $event; loadInstalledModels()"
          @data-changed="handleDataChanged"
        />
        <DiagnosticsPanel
          :results="diagnoseResults"
          :loading="diagnoseLoading"
          :fixing-id="fixingIssueId"
          @run="runChecks"
          @fix="handleFix"
        />
      </aside>
    </div>
  </main>
</template>
