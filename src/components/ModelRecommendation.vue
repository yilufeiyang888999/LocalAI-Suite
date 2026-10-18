<script setup lang="ts">
import type { RecommendedModel } from '../types';

const props = defineProps<{ models: RecommendedModel[]; loading: boolean; selectedId: string | null }>();
const emit = defineEmits<{ install: [model: RecommendedModel] }>();

function sizeLabel(mb: number) {
  return `${(mb / 1000).toFixed(1)} GB`;
}

/** 「速度」一栏：有带宽预估就给具体数字，否则退回定性标签。 */
function speedText(item: RecommendedModel) {
  if (item.estimated_tps === null || item.estimated_tps === undefined) {
    return `速度 ${item.speed_label}`;
  }
  return `速度 约 ${item.estimated_tps.toFixed(1)} token/秒`;
}

/** 预估来自显存带宽 ÷ 权重体积，鼠标悬停时说明口径，避免用户误以为是实测。 */
function speedHint(item: RecommendedModel) {
  if (item.estimated_tps === null || item.estimated_tps === undefined) {
    return '当前显卡未收录带宽数据，仅给出速度档位。';
  }
  return '按显存带宽与模型体积估算，实际速度会因显卡、驱动与上下文长度有所浮动。';
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
          <span :title="speedHint(item)">{{ speedText(item) }}</span>
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
