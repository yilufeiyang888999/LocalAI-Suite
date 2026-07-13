<script setup lang="ts">
import type { DownloadProgress, DownloadStatus } from '../types';

defineProps<{ progress: DownloadProgress | null; serviceMessage: string | null }>();
const emit = defineEmits<{ pause: [string]; resume: [string]; cancel: [string] }>();

function percent(progress: DownloadProgress | null) {
  if (!progress || progress.total_bytes === 0) return 0;
  return Math.min(100, Math.round((progress.downloaded_bytes / progress.total_bytes) * 100));
}

const ACTIVE: DownloadStatus[] = ['pending', 'downloading', 'verifying'];

function statusText(status: DownloadStatus) {
  const labels: Record<DownloadStatus, string> = {
    pending: '等待开始',
    downloading: '正在下载模型文件',
    paused: '已暂停',
    verifying: '正在检查文件是否完整',
    failed: '安装遇到问题',
    completed: '模型文件已准备好',
    cancelled: '已取消'
  };
  return labels[status];
}
</script>

<template>
  <section class="card">
    <div class="section-title">
      <span class="step-badge">4</span>
      <div>
        <h2>安装进度</h2>
        <p>点击开始安装后，会优先从 ModelScope 下载模型文件，失败时自动尝试 hf-mirror。</p>
      </div>
    </div>
    <div v-if="progress" class="progress-wrap">
      <div class="progress-row">
        <strong>{{ progress.current_step }}</strong>
        <span>{{ percent(progress) }}%</span>
      </div>
      <div class="progress-bar">
        <div :style="{ width: `${percent(progress)}%` }"></div>
      </div>
      <p class="muted">当前状态：{{ statusText(progress.status) }}</p>
      <div class="download-actions">
        <button
          v-if="progress.status === 'downloading' || progress.status === 'pending'"
          type="button"
          @click="emit('pause', progress.task_id)"
        >
          暂停下载
        </button>
        <button
          v-if="progress.status === 'paused'"
          type="button"
          @click="emit('resume', progress.task_id)"
        >
          继续下载
        </button>
        <button
          v-if="ACTIVE.includes(progress.status) || progress.status === 'paused'"
          type="button"
          class="danger"
          @click="emit('cancel', progress.task_id)"
        >
          取消下载
        </button>
      </div>
    </div>
    <div v-else class="notice">选择模型后会显示下载、检查文件和启动服务进度。</div>
    <div v-if="serviceMessage" class="notice next-step">{{ serviceMessage }}</div>
  </section>
</template>
