<script setup lang="ts">
import type { Scene } from '../types';

const props = defineProps<{ selected: Scene }>();
const emit = defineEmits<{ select: [scene: Scene] }>();

const scenes: Array<{ value: Scene; title: string; desc: string }> = [
  { value: 'chat', title: '日常聊天问答', desc: '适合先体验本地AI，回答问题和简单交流。' },
  { value: 'writing', title: '写文案/写报告', desc: '更重视中文表达和长文本整理。' },
  { value: 'coding', title: '写代码', desc: '更适合解释代码、生成代码和排查问题。' },
  { value: 'general', title: '通用', desc: '不限定具体场景，按模型综合表现推荐。' }
];
</script>

<template>
  <section class="card">
    <div class="section-title">
      <span class="step-badge">2</span>
      <div>
        <h2>选择使用场景</h2>
        <p>选择你最常用的用途，系统会自动推荐合适模型。</p>
      </div>
    </div>
    <div class="scene-grid">
      <button
        v-for="scene in scenes"
        :key="scene.value"
        class="scene-card"
        :class="{ active: props.selected === scene.value }"
        @click="emit('select', scene.value)"
      >
        <strong>{{ scene.title }}</strong>
        <span>{{ scene.desc }}</span>
      </button>
    </div>
  </section>
</template>
