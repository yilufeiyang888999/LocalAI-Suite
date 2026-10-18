<script setup lang="ts">
import { open, confirm, message as dialogMessage } from '@tauri-apps/plugin-dialog';
import { openPath } from '@tauri-apps/plugin-opener';
import { computed, watch, ref, onMounted } from 'vue';
import type { AppSettings, MirrorSource } from '../types';
import { displayNameFromModelId } from '../utils/markdown';
import { DEFAULT_KB_BASE_URL, normalizeKbBaseUrl } from '../utils/kb';
import { DEFAULT_CHATBI_BASE_URL, normalizeChatbiBaseUrl } from '../utils/chatbi';
import {
  clearAllChatRecords,
  getAppVersion,
  getDataPaths,
  importLocalModel,
  resetToFactory,
  uninstallAllModels,
  updateSettings
} from '../api/tauri';

const props = defineProps<{ settings: AppSettings | null; modelId?: string | null }>();
const emit = defineEmits<{ saved: [settings: AppSettings]; 'data-changed': [] }>();

const saving = ref(false);
const importing = ref(false);
const dataMgmtBusy = ref<string | null>(null);
const message = ref<string | null>(null);
const installDir = ref('');
const llamaPath = ref('');
const modelPath = ref('');
// null = 不限速; otherwise MB/s (decimal, 1MB = 1,000,000 bytes).
const speedLimitMbs = ref<number | null>(null);
const mirror = ref<'auto' | 'model_scope' | 'hf_mirror'>('auto');
const proxy = ref('');
const autostart = ref(false);
const serviceHost = ref('127.0.0.1');
const servicePort = ref(8080);
const aiDisplayName = ref('');
const extraArgs = ref('');
const logLevel = ref('INFO');
const kbBaseUrl = ref(DEFAULT_KB_BASE_URL);
const chatbiBaseUrl = ref(DEFAULT_CHATBI_BASE_URL);
// 实际版本号从后端读（与日志首行 version= 同源），不再写死在模板里 ——
// 之前硬编码 "0.1.0（MVP）"，升级到 0.1.2 后界面仍显示 0.1.0，用户无法
// 从界面确认装的是哪个包。
const appVersion = ref('');

onMounted(async () => {
  try {
    appVersion.value = await getAppVersion();
  } catch {
    appVersion.value = '未知';
  }
});

function mirrorKey(source: MirrorSource | undefined): 'auto' | 'model_scope' | 'hf_mirror' {
  if (source === 'model_scope' || source === 'hf_mirror') return source;
  return 'auto';
}

// Placeholder hint for the AI-name box: shows what leaving it blank means.
const aiNamePlaceholder = computed(() =>
  props.modelId ? `留空 = 模型自己决定（如 ${displayNameFromModelId(props.modelId)}）` : '留空 = 模型自己决定'
);

function syncFromProps() {
  installDir.value = props.settings?.install_dir ?? '';
  llamaPath.value = props.settings?.llama_server_path ?? '';
  modelPath.value = props.settings?.local_model_path ?? '';
  const limit = props.settings?.download_speed_limit ?? null;
  speedLimitMbs.value = limit ? Math.round(limit / 1_000_000) : null;
  mirror.value = mirrorKey(props.settings?.mirror_source);
  proxy.value = props.settings?.proxy ?? '';
  autostart.value = props.settings?.autostart ?? false;
  serviceHost.value = props.settings?.service_host ?? '127.0.0.1';
  servicePort.value = props.settings?.service_port ?? 8080;
  aiDisplayName.value = props.settings?.ai_display_name ?? '';
  extraArgs.value = props.settings?.extra_launch_args ?? '';
  logLevel.value = props.settings?.log_level ?? 'INFO';
  kbBaseUrl.value = props.settings?.kb_base_url ?? DEFAULT_KB_BASE_URL;
  chatbiBaseUrl.value = props.settings?.chatbi_base_url ?? DEFAULT_CHATBI_BASE_URL;
}

async function chooseInstallDir() {
  const selected = await open({ multiple: false, directory: true });
  if (typeof selected === 'string') {
    installDir.value = selected;
    message.value = '已选择AI文件保存位置，点击保存设置后生效。';
  }
}

async function importModel() {
  if (!props.settings) return;
  const selected = await open({
    multiple: false,
    filters: [{ name: '模型文件', extensions: ['gguf'] }]
  });
  if (typeof selected !== 'string') return;

  importing.value = true;
  message.value = null;
  try {
    const imported = await importLocalModel(selected);
    modelPath.value = imported.local_path;
    emit('saved', { ...props.settings, local_model_path: imported.local_path });
    message.value = imported.message;
  } catch (error) {
    message.value = String(error);
  } finally {
    importing.value = false;
  }
}

async function save() {
  if (!props.settings) return;
  saving.value = true;
  message.value = null;
  const port = Math.min(65535, Math.max(1, Math.round(servicePort.value || 8080)));
  servicePort.value = port;
  try {
    const next = await updateSettings({
      ...props.settings,
      install_dir: installDir.value || props.settings.install_dir,
      download_speed_limit: speedLimitMbs.value ? speedLimitMbs.value * 1_000_000 : null,
      mirror_source: mirror.value,
      proxy: proxy.value.trim() || null,
      autostart: autostart.value,
      service_host: serviceHost.value,
      service_port: port,
      ai_display_name: aiDisplayName.value.trim() || null,
      extra_launch_args: extraArgs.value.trim() || null,
      log_level: logLevel.value,
      // 规范化后再存：容忍用户只填 `127.0.0.1:8092`，并去掉末尾斜杠
      kb_base_url: normalizeKbBaseUrl(kbBaseUrl.value),
      // 同一条纪律：规范化后再存（容忍 `127.0.0.1:8096` 简写、去掉末尾斜杠）
      chatbi_base_url: normalizeChatbiBaseUrl(chatbiBaseUrl.value),
      // 引擎/模型路径由后端自动管理（安装、导入、切换时自动设置），
      // 这里只透传现有值，不从只读展示框取值。
      llama_server_path: props.settings.llama_server_path ?? null,
      local_model_path: props.settings.local_model_path ?? null
    });
    emit('saved', next);
    message.value = '设置已保存。';
  } catch (error) {
    message.value = String(error);
  } finally {
    saving.value = false;
  }
}

watch(() => props.settings, syncFromProps, { immediate: true });

async function openDataFolder(kind: 'data' | 'logs') {
  if (dataMgmtBusy.value) return;
  dataMgmtBusy.value = `open-${kind}`;
  message.value = null;
  try {
    const paths = await getDataPaths();
    const target = kind === 'logs' ? paths.logs_dir : paths.data_dir;
    await openPath(target);
  } catch (error) {
    message.value = `打开文件夹失败：${String(error)}`;
  } finally {
    dataMgmtBusy.value = null;
  }
}

async function doClearAllChat() {
  if (dataMgmtBusy.value) return;
  if (!(await confirm('确定清空全部聊天记录？\n\n所有会话和消息都会被删除，模型和设置不受影响。此操作不可撤销。', { title: '清空聊天记录', kind: 'warning' }))) return;
  dataMgmtBusy.value = 'clear-chat';
  message.value = null;
  try {
    const result = await clearAllChatRecords();
    message.value = result.message;
    emit('data-changed');
  } catch (error) {
    message.value = String(error);
  } finally {
    dataMgmtBusy.value = null;
  }
}

async function doUninstallAll() {
  if (dataMgmtBusy.value) return;
  if (!(await confirm('确定一键卸载全部模型？\n\n内置模型的文件将从本地删除，本地导入的模型仅取消跟踪（文件保留）。运行引擎不动。此操作不可撤销。', { title: '卸载全部模型', kind: 'warning' }))) return;
  dataMgmtBusy.value = 'uninstall-all';
  message.value = null;
  try {
    const result = await uninstallAllModels();
    message.value = result.message;
    emit('data-changed');
  } catch (error) {
    message.value = String(error);
  } finally {
    dataMgmtBusy.value = null;
  }
}

async function doResetToFactory() {
  if (dataMgmtBusy.value) return;
  if (!(await confirm('【警告】重置到出厂设置会删除：\n· 所有模型文件\n· 运行引擎\n· 所有聊天记录\n· 所有设置\n· 日志\n\n此操作不可撤销。要继续吗？', { title: '重置到出厂设置', kind: 'warning' }))) return;
  if (!(await confirm('再次确认：所有本地数据都会在下次启动时被清空，相当于全新安装状态。真的要重置吗？', { title: '再次确认重置', kind: 'warning' }))) return;
  dataMgmtBusy.value = 'reset';
  message.value = null;
  try {
    const result = await resetToFactory();
    message.value = result.message;
    await dialogMessage(result.message, { title: '重置到出厂设置' });
  } catch (error) {
    message.value = String(error);
  } finally {
    dataMgmtBusy.value = null;
  }
}
</script>

<template>
  <section class="card side-card">
    <h2>设置 / 离线导入</h2>
    <p class="muted">模型文件和运行引擎都会自动放在保存位置里。空间不够时，建议选择 D 盘等大空间文件夹。</p>
    <label>
      AI文件保存位置
      <input v-model="installDir" placeholder="例如 D:\\LocalAI-Installer" />
    </label>
    <button class="secondary-button full-width" @click="chooseInstallDir">选择保存位置</button>

    <label>
      下载限速
      <select v-model.number="speedLimitMbs">
        <option :value="null">不限速</option>
        <option :value="1">1 MB/s</option>
        <option :value="5">5 MB/s</option>
        <option :value="10">10 MB/s</option>
      </select>
    </label>

    <label>
      下载线路
      <select v-model="mirror">
        <option value="auto">自动测速（推荐）</option>
        <option value="model_scope">ModelScope</option>
        <option value="hf_mirror">hf-mirror</option>
      </select>
    </label>

    <label class="checkbox-row">
      <input v-model="autostart" type="checkbox" />
      开机时自动启动本软件
    </label>

    <label>
      AI 自称名字
      <input v-model="aiDisplayName" :placeholder="aiNamePlaceholder" />
    </label>
    <p class="muted small-hint">只在你想让 AI 用固定名字自称时才填。留空则让模型自己决定怎么回答「你是谁」。</p>

    <div class="service-access">
      <label>
        谁可以调用本机AI
        <select v-model="serviceHost">
          <option value="127.0.0.1">仅本机访问（更安全）</option>
          <option value="0.0.0.0">局域网内其他设备也可访问</option>
        </select>
      </label>
      <label>
        接口端口
        <input v-model.number="servicePort" type="number" min="1" max="65535" placeholder="8080" />
      </label>
      <p v-if="serviceHost === '0.0.0.0'" class="notice warning">
        开启后，局域网内任何人都能直接调用本机AI，且没有密码保护。请仅在可信网络（如家庭/公司内网）开启。
      </p>
      <p class="muted">固定端口后，其他程序的调用地址就不会再变。修改后需在聊天区停止再启动AI服务才生效。</p>
    </div>

    <div class="field">
      <label>
        知识库服务地址
        <input v-model="kbBaseUrl" :placeholder="DEFAULT_KB_BASE_URL" />
      </label>
      <p class="muted">
        「聊天」页里「知识库问答」所依赖的独立本地服务，不随本应用启动 —— 需在 RAG-Embedding 目录下运行
        <code>python scripts/知识库服务.py</code>。留空则用 {{ DEFAULT_KB_BASE_URL }}。
      </p>
    </div>

    <div class="field">
      <label>
        智能问数服务地址
        <input v-model="chatbiBaseUrl" :placeholder="DEFAULT_CHATBI_BASE_URL" />
      </label>
      <p class="muted">
        「聊天」页里「智能问数」所依赖的独立本地服务（查制度 / 查数据 / 起草单据三合一），同样不随本应用启动
        —— 需在 ChatBI 目录下运行 <code>python scripts/启动-服务.py</code>（它管 8095 + 8096 两个端口）。
        它的「知识问答」链路会转发给上面的知识库服务，所以那条链路要求两者都开着。
        留空则用 {{ DEFAULT_CHATBI_BASE_URL }}。
      </p>
    </div>

    <details class="advanced-settings">
      <summary>网络与高级设置</summary>
      <p class="muted">普通用户无需填写下面内容。仅当自动下载不可用、需要代理，或你已有本地文件时才使用。</p>

      <label>
        代理地址（可选）
        <input v-model="proxy" placeholder="例如 http://127.0.0.1:7890" />
      </label>

      <label>
        日志详细程度
        <select v-model="logLevel">
          <option value="DEBUG">最详细（DEBUG）</option>
          <option value="INFO">常规（INFO）</option>
          <option value="WARN">仅警告（WARN）</option>
          <option value="ERROR">仅错误（ERROR）</option>
        </select>
      </label>

      <label>
        自定义运行参数（风险自负）
        <input v-model="extraArgs" placeholder="例如 --threads 8 --no-mmap" />
      </label>

      <p class="muted">模型文件和运行引擎都会自动放在上面的「AI文件保存位置」里，无需手动设置。下面仅供查看当前位置。</p>
      <label>
        本地AI运行引擎位置（自动）
        <input v-model="llamaPath" readonly class="readonly-path" placeholder="安装运行引擎后自动生成" />
      </label>
      <label>
        当前模型文件位置（自动）
        <input v-model="modelPath" readonly class="readonly-path" placeholder="安装或导入模型后自动生成" />
      </label>
      <button class="secondary-button full-width" :disabled="importing" @click="importModel">
        {{ importing ? '正在导入' : '导入本地模型文件' }}
      </button>
    </details>

    <button class="primary-button" :disabled="saving" @click="save">保存设置</button>
    <p v-if="message" class="notice">{{ message }}</p>

    <details class="data-mgmt-box">
      <summary>数据管理 / 清理</summary>
      <p class="muted">这里的操作会直接影响本地数据，请谨慎使用。</p>

      <div class="data-mgmt-row">
        <button class="secondary-button" :disabled="dataMgmtBusy !== null" @click="openDataFolder('data')">
          打开数据文件夹
        </button>
        <button class="secondary-button" :disabled="dataMgmtBusy !== null" @click="openDataFolder('logs')">
          打开日志文件夹
        </button>
      </div>

      <button class="secondary-button full-width" :disabled="dataMgmtBusy !== null" @click="doClearAllChat">
        {{ dataMgmtBusy === 'clear-chat' ? '正在清空聊天记录' : '一键清空所有聊天记录' }}
      </button>

      <button class="danger-button-full" :disabled="dataMgmtBusy !== null" @click="doUninstallAll">
        {{ dataMgmtBusy === 'uninstall-all' ? '正在卸载全部模型' : '一键卸载全部模型' }}
      </button>

      <button class="danger-button-full nuclear" :disabled="dataMgmtBusy !== null" @click="doResetToFactory">
        {{ dataMgmtBusy === 'reset' ? '正在标记重置' : '重置到出厂设置' }}
      </button>
      <p class="muted small">重置需要关闭软件后重新打开才会生效。</p>
    </details>

    <details class="about-box">
      <summary>关于</summary>
      <p class="muted">本地AI模型一键部署助手 · 版本 {{ appVersion || '读取中…' }}</p>
      <p class="muted">所有对话、设置与日志均保存在本机，不会上传云端。</p>
      <p class="muted">首次启动 AI 服务时，Windows 可能弹窗询问是否允许运行，请选择「仍要运行」。</p>
      <p class="muted">遇到问题可在「故障自查」面板点「导出诊断日志（已脱敏）」反馈给开发者。</p>
      <p class="muted small">日志在数据目录的 logs 文件夹里，文件名带日期后缀，例如 localai.log.2026-09-30（不是 localai.log）。</p>
    </details>
  </section>
</template>

<style scoped>
.checkbox-row {
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 8px;
}
.checkbox-row input[type='checkbox'] {
  margin: 0;
  width: 16px;
  height: 16px;
  flex-shrink: 0;
}
.about-box {
  margin-top: 8px;
  font-size: 12px;
}
.data-mgmt-box {
  margin-top: 12px;
  font-size: 13px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.data-mgmt-box summary {
  cursor: pointer;
  font-weight: 600;
}
.data-mgmt-row {
  display: flex;
  gap: 8px;
}
.data-mgmt-row .secondary-button {
  flex: 1;
}
.danger-button-full {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid #d64545;
  background: transparent;
  color: #d64545;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
}
.danger-button-full:hover:not(:disabled) {
  background: #d64545;
  color: #fff;
}
.danger-button-full:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.danger-button-full.nuclear {
  border-color: #a01414;
  color: #a01414;
}
.danger-button-full.nuclear:hover:not(:disabled) {
  background: #a01414;
}
.small {
  font-size: 11px;
  margin: -4px 0 0;
}
.small-hint {
  font-size: 12px;
  margin: -6px 0 4px;
}
.readonly-path {
  background: #f2f3f5;
  color: #6b7280;
  cursor: default;
}
</style>
