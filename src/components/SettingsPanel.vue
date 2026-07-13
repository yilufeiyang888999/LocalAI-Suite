<script setup lang="ts">
import { open } from '@tauri-apps/plugin-dialog';
import { watch, ref } from 'vue';
import type { AppSettings, MirrorSource } from '../types';
import { importLocalModel, updateSettings } from '../api/tauri';

const props = defineProps<{ settings: AppSettings | null }>();
const emit = defineEmits<{ saved: [settings: AppSettings] }>();

const saving = ref(false);
const importing = ref(false);
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
const extraArgs = ref('');
const logLevel = ref('INFO');

function mirrorKey(source: MirrorSource | undefined): 'auto' | 'model_scope' | 'hf_mirror' {
  if (source === 'model_scope' || source === 'hf_mirror') return source;
  return 'auto';
}

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
  extraArgs.value = props.settings?.extra_launch_args ?? '';
  logLevel.value = props.settings?.log_level ?? 'INFO';
}

async function chooseInstallDir() {
  const selected = await open({ multiple: false, directory: true });
  if (typeof selected === 'string') {
    installDir.value = selected;
    message.value = '已选择AI文件保存位置，点击保存设置后生效。';
  }
}

async function chooseEngine() {
  const selected = await open({
    multiple: false,
    filters: [{ name: '本地AI运行引擎', extensions: ['exe'] }]
  });
  if (typeof selected === 'string') {
    llamaPath.value = selected;
    message.value = '已选择本地AI运行引擎，点击保存设置后生效。';
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
      extra_launch_args: extraArgs.value.trim() || null,
      log_level: logLevel.value,
      llama_server_path: llamaPath.value || null,
      local_model_path: modelPath.value || null
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

      <label>
        本地AI运行引擎路径
        <input v-model="llamaPath" placeholder="例如 D:\\tools\\llama-server.exe" />
      </label>
      <button class="secondary-button full-width" @click="chooseEngine">选择运行引擎</button>
      <label>
        本地模型文件路径
        <input v-model="modelPath" placeholder="例如 D:\\models\\chat-model.gguf" />
      </label>
      <button class="secondary-button full-width" :disabled="importing" @click="importModel">
        {{ importing ? '正在导入' : '导入本地模型文件' }}
      </button>
    </details>

    <button class="primary-button" :disabled="saving" @click="save">保存设置</button>
    <p v-if="message" class="notice">{{ message }}</p>

    <details class="about-box">
      <summary>关于</summary>
      <p class="muted">本地AI模型一键部署助手 · 版本 0.1.0（MVP）</p>
      <p class="muted">所有对话、设置与日志均保存在本机，不会上传云端。</p>
      <p class="muted">遇到问题可导出诊断日志（已脱敏）反馈给开发者。</p>
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
</style>
