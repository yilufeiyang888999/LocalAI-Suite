<script setup lang="ts">
import type { InstalledModelInfo, ServiceInfo } from '../types';

const props = defineProps<{
  models: InstalledModelInfo[];
  loading: boolean;
  switchingPath: string | null;
  updatingId: string | null;
  updateMessage: string | null;
  service: ServiceInfo | null;
}>();

const emit = defineEmits<{
  refresh: [];
  switch: [model: InstalledModelInfo];
  update: [model: InstalledModelInfo];
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
</script>

<template>
  <section class="card">
    <div class="section-title model-manager-title">
      <span class="step-badge">6</span>
      <div>
        <h2>已安装模型</h2>
        <p>下载过的模型会显示在这里，点击即可切换当前聊天使用的模型。</p>
      </div>
      <button class="secondary-button" :disabled="loading" @click="emit('refresh')">刷新</button>
    </div>

    <div v-if="loading" class="loading-box">正在查找已安装模型...</div>
    <div v-else-if="models.length === 0" class="notice">还没有已安装模型，先点击推荐模型的开始安装。</div>
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
          <button
            class="primary-button"
            :disabled="model.is_current || switchingPath === model.local_path || updatingId !== null || props.service?.status === 'starting'"
            @click="emit('switch', model)"
          >
            {{ model.is_current ? '正在使用' : switchingPath === model.local_path ? '正在切换' : '切换到这个模型' }}
          </button>
          <button
            v-if="model.source === 'builtin'"
            class="secondary-button"
            :disabled="updatingId !== null || switchingPath !== null || props.service?.status === 'starting'"
            @click="emit('update', model)"
          >
            {{ updateLabel(model) }}
          </button>
        </div>
      </article>
    </div>
  </section>
</template>
