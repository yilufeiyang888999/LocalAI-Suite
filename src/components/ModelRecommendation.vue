<script setup lang="ts">
import type { RecommendedModel } from '../types';

const props = defineProps<{ models: RecommendedModel[]; loading: boolean; selectedId: string | null }>();
const emit = defineEmits<{ install: [model: RecommendedModel] }>();

function sizeLabel(mb: number) {
  return `${(mb / 1000).toFixed(1)} GB`;
}

function buttonLabel(item: RecommendedModel) {
  if (props.selectedId === item.model.id) return '正在准备';
  return item.model.is_installed ? '直接使用' : '开始安装';
}
</script>

<template>
  <section class="card">
    <div class="section-title">
      <span class="step-badge">3</span>
      <div>
        <h2>推荐模型</h2>
        <p>最多展示三个选择，默认优先推荐最适合当前电脑和场景的模型。</p>
      </div>
    </div>
    <div v-if="loading" class="loading-box">正在挑选合适模型...</div>
    <div v-else class="model-list">
      <article v-for="item in models" :key="item.model.id" class="model-card" :class="{ primary: item.rank === 'primary' }">
        <div class="model-header">
          <div>
            <span class="tag">{{ item.rank === 'primary' ? '推荐首选' : '备选方案' }}</span>
            <span v-if="item.model.is_installed" class="tag installed">已安装</span>
            <h3>{{ item.model.name }}</h3>
          </div>
          <div class="score">{{ item.score }}</div>
        </div>
        <p>{{ item.reason }}</p>
        <div class="model-meta">
          <span>预计占用 {{ sizeLabel(item.model.file_size_mb) }}</span>
          <span>速度 {{ item.speed_label }}</span>
          <span>效果 {{ item.quality_label }}</span>
          <span>{{ item.model.display_compression }}</span>
        </div>
        <button class="primary-button" :disabled="selectedId === item.model.id" @click="emit('install', item)">
          {{ buttonLabel(item) }}
        </button>
      </article>
    </div>
  </section>
</template>
