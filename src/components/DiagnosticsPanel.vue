<script setup lang="ts">
import { save } from '@tauri-apps/plugin-dialog';
import { ref } from 'vue';
import type { DiagnoseResult } from '../types';
import { exportLogs } from '../api/tauri';

defineProps<{ results: DiagnoseResult[]; loading: boolean; fixingId: string | null }>();
const emit = defineEmits<{ run: []; fix: [issueId: string] }>();

const exporting = ref(false);
const exportMessage = ref<string | null>(null);

// Issues that have a meaningful one-click fix action wired in App.vue.
const FIXABLE_ISSUES = new Set(['vram', 'runtime_libs', 'model_file', 'gpu_driver', 'disk_space']);
function fixLabel(issueId: string): string {
  if (issueId === 'gpu_driver') return '查看驱动升级';
  if (issueId === 'disk_space') return '去更换保存位置';
  return '尝试修复';
}

async function doExportLogs() {
  const dest = await save({
    defaultPath: 'localai-logs.zip',
    filters: [{ name: '日志包', extensions: ['zip'] }]
  });
  if (typeof dest !== 'string') return;

  exporting.value = true;
  exportMessage.value = null;
  try {
    const path = await exportLogs(dest);
    exportMessage.value = `日志包已导出到：${path}`;
  } catch (error) {
    exportMessage.value = String(error);
  } finally {
    exporting.value = false;
  }
}
</script>

<template>
  <section class="card side-card">
    <div class="side-title">
      <h2>故障自查</h2>
      <button class="secondary-button" :disabled="loading" @click="emit('run')">开始检查</button>
    </div>
    <div v-if="loading" class="loading-box">正在检查常见问题...</div>
    <div v-else-if="results.length === 0" class="muted">点击开始检查，查看模型启动前的常见问题。</div>
    <ul v-else class="diagnose-list">
      <li v-for="item in results" :key="item.issue_id" :class="{ ok: item.ok }">
        <strong>{{ item.title }}</strong>
        <span>{{ item.description }}</span>
        <button
          v-if="!item.ok && item.fixable && FIXABLE_ISSUES.has(item.issue_id)"
          class="link-button fix-button"
          :disabled="fixingId === item.issue_id"
          @click="emit('fix', item.issue_id)"
        >
          {{ fixingId === item.issue_id ? '正在处理…' : fixLabel(item.issue_id) }}
        </button>
      </li>
    </ul>
    <button class="secondary-button full-width" :disabled="exporting" @click="doExportLogs">
      {{ exporting ? '正在导出日志' : '导出诊断日志（已脱敏）' }}
    </button>
    <p v-if="exportMessage" class="notice">{{ exportMessage }}</p>
  </section>
</template>
