<script setup lang="ts">
import type { HardwareInfo } from '../types';

defineProps<{
  hardware: HardwareInfo | null;
  loading: boolean;
  error: string | null;
  manualTier: number | null;
}>();
const emit = defineEmits<{
  refresh: [];
  selectTier: [tier: number | null];
}>();

const tiers = [
  { label: 'CPU模式', value: 3, desc: '无独显或只想简单问答' },
  { label: '8GB档', value: 8, desc: '常见入门显卡' },
  { label: '12-16GB档', value: 14, desc: '进阶写作和代码' },
  { label: '24GB+档', value: 24, desc: '高端配置' }
];
</script>

<template>
  <section class="card hero-card">
    <div class="section-title">
      <span class="step-badge">1</span>
      <div>
        <h2>硬件体检</h2>
        <p>先看看这台电脑适合运行哪类本地AI。</p>
      </div>
    </div>

    <div v-if="loading" class="loading-box">正在检测电脑配置...</div>
    <div v-else-if="error" class="notice warning">
      {{ error }}
      <button class="link-button" @click="emit('refresh')">重新检测</button>
    </div>
    <div v-else-if="hardware" class="hardware-grid">
      <div class="summary">
        <span class="mode-pill" :class="hardware.has_gpu ? 'gpu' : 'cpu'">
          {{ hardware.has_gpu ? '显卡加速模式' : 'CPU模式' }}
        </span>
        {{ hardware.summary }}
      </div>
      <div class="metric">
        <span>显卡</span>
        <strong>{{ hardware.gpu_name }}</strong>
      </div>
      <div class="metric">
        <span>显卡内存</span>
        <strong>{{ hardware.gpu_vram_gb.toFixed(1) }} GB</strong>
      </div>
      <div class="metric">
        <span>内存</span>
        <strong>{{ hardware.memory.total_gb.toFixed(1) }} GB</strong>
      </div>
      <div class="metric">
        <span>CPU</span>
        <strong>{{ hardware.cpu.name }}</strong>
      </div>
      <div v-if="hardware.needs_manual_tier" class="notice manual-tier">
        <strong>可以继续使用 CPU 模式。</strong>
        <span>如果你知道这台电脑的实际配置，也可以手动选择档位来调整推荐结果。</span>
        <div class="tier-grid">
          <button
            v-for="tier in tiers"
            :key="tier.label"
            class="tier-card"
            :class="{ active: manualTier === tier.value }"
            @click="emit('selectTier', tier.value)"
          >
            <strong>{{ tier.label }}</strong>
            <small>{{ tier.desc }}</small>
          </button>
          <button class="tier-card" :class="{ active: manualTier === null }" @click="emit('selectTier', null)">
            <strong>自动判断</strong>
            <small>按检测结果推荐</small>
          </button>
        </div>
      </div>
    </div>
  </section>
</template>
