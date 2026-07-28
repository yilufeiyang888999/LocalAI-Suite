<script setup lang="ts">
import { confirm } from '@tauri-apps/plugin-dialog';
import type { InstalledModelInfo, ServiceInfo } from '../types';

const props = defineProps<{
  models: InstalledModelInfo[];
  loading: boolean;
  switchingPath: string | null;
  updatingId: string | null;
  updateMessage: string | null;
  uninstallingId: string | null;
  importing: boolean;
  service: ServiceInfo | null;
}>();

const emit = defineEmits<{
  refresh: [];
  switch: [model: InstalledModelInfo];
  update: [model: InstalledModelInfo];
  stop: [];
  start: [model: InstalledModelInfo];
  uninstall: [model: InstalledModelInfo, deleteChatHistory: boolean];
  import: [];
}>();

function sizeLabel(mb: number) {
  return `${(mb / 1000).toFixed(1)} GB`;
}

function sourceLabel(source: string) {
  return source === 'builtin' ? '自动安装' : '本地导入';
}

function updateLabel(model: InstalledModelInfo) {
  if (props.updatingId === model.model_id) return '正在更新';
  return model.update_available ? '更新到新版本' : '检查并更新';
}

function isServiceBusy() {
  return props.service?.status === 'starting';
}

function isCurrentRunning(model: InstalledModelInfo) {
  return model.is_current && (props.service?.status === 'running' || props.service?.status === 'starting');
}

async function requestUninstall(model: InstalledModelInfo) {
  const sourceHint = model.source === 'builtin'
    ? '将从本地删除模型文件（可再次下载安装）。'
    : '仅取消跟踪，模型文件本身会保留在原来的位置。';
  const confirmed = await confirm(`确定卸载模型「${model.name}」？\n\n${sourceHint}`, {
    title: '卸载模型',
    kind: 'warning'
  });
  if (!confirmed) return;
  const alsoDeleteChat = await confirm(
    '是否同时删除该模型的所有聊天记录？\n\n【确定】= 一起删除\n【取消】= 只卸载模型，保留聊天记录',
    { title: '聊天记录', kind: 'warning' }
  );
  emit('uninstall', model, alsoDeleteChat);
}
</script>

<template>
  <section class="card">
    <div class="section-title model-manager-title">
      <span class="step-badge">6</span>
      <div>
        <h2>已安装模型</h2>
        <p>下载过的模型会显示在这里，点击即可切换当前聊天使用的模型。</p>
      </div>
      <div class="model-manager-actions">
        <button class="secondary-button" :disabled="importing" @click="emit('import')">
          {{ importing ? '正在导入' : '导入本地模型' }}
        </button>
        <button class="secondary-button" :disabled="loading" @click="emit('refresh')">刷新</button>
      </div>
    </div>

    <div v-if="loading" class="loading-box">正在查找已安装模型...</div>
    <div v-else-if="models.length === 0" class="notice">还没有已安装模型，先点击推荐模型的开始安装，或用「导入本地模型」加入已有的 gguf 文件。</div>
    <div v-else class="installed-model-list">
      <article v-for="model in models" :key="model.local_path" class="installed-model-card" :class="{ current: model.is_current }">
        <div>
          <div class="installed-model-header">
            <h3>{{ model.name }}</h3>
            <span v-if="model.is_current" class="current-badge">当前使用中</span>
            <span v-if="model.update_available" class="update-badge">有新版本</span>
          </div>
          <p class="muted">{{ sourceLabel(model.source) }} · {{ sizeLabel(model.file_size_mb) }}</p>
          <p class="model-path">{{ model.local_path }}</p>
          <p v-if="updatingId === model.model_id && updateMessage" class="update-status">{{ updateMessage }}</p>
        </div>
        <div class="installed-model-actions">
          <!-- Current model: stop when running, start when stopped. -->
          <template v-if="model.is_current">
            <button
              v-if="isCurrentRunning(model)"
              class="secondary-button"
              :disabled="isServiceBusy() || uninstallingId !== null"
              @click="emit('stop')"
            >
              停止服务
            </button>
            <button
              v-else
              class="primary-button"
              :disabled="isServiceBusy() || uninstallingId !== null || updatingId !== null"
              @click="emit('start', model)"
            >
              启动这个模型
            </button>
          </template>
          <!-- Non-current models: switch + uninstall. -->
          <template v-else>
            <button
              class="primary-button"
              :disabled="switchingPath === model.local_path || switchingPath !== null || updatingId !== null || uninstallingId !== null || isServiceBusy()"
              @click="emit('switch', model)"
            >
              {{ switchingPath === model.local_path ? '正在切换' : '切换到这个模型' }}
            </button>
            <button
              class="danger-button"
              :disabled="uninstallingId === model.model_id || uninstallingId !== null || switchingPath !== null || updatingId !== null || isServiceBusy()"
              @click="requestUninstall(model)"
            >
              {{ uninstallingId === model.model_id ? '正在卸载' : '卸载' }}
            </button>
          </template>
          <!-- Updates still available for builtin models regardless of current state. -->
          <button
            v-if="model.source === 'builtin'"
            class="secondary-button"
            :disabled="updatingId !== null || switchingPath !== null || uninstallingId !== null || isServiceBusy()"
            @click="emit('update', model)"
          >
            {{ updateLabel(model) }}
          </button>
        </div>
      </article>
    </div>
  </section>
</template>

<style scoped>
.model-manager-actions {
  display: flex;
  gap: 8px;
  margin-left: auto;
}
.danger-button {
  border: 1px solid #d64545;
  background: transparent;
  color: #d64545;
  padding: 6px 12px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
}
.danger-button:hover:not(:disabled) {
  background: #d64545;
  color: #fff;
}
.danger-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
