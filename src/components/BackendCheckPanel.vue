<script setup lang="ts">
import { computed } from 'vue';
import type { BackendCheckAvailability, BackendCheckResult } from '../types';

const props = defineProps<{
  availability: BackendCheckAvailability | null;
  result: BackendCheckResult | null;
  running: boolean;
  error: string | null;
}>();

const emit = defineEmits<{ run: [] }>();

/** 结论对应的样式类，便于用颜色区分「安全」和「要留意」。 */
const verdictClass = computed(() => {
  const v = props.result?.comparison.verdict;
  if (v === 'identical') return 'verdict-ok';
  if (v === 'length_only') return 'verdict-warn';
  return 'verdict-bad';
});

const verdictTitle = computed(() => {
  const v = props.result?.comparison.verdict;
  if (v === 'identical') return '两个后端输出一致';
  if (v === 'length_only') return '两个后端内容一致、长度略有差异';
  if (v === 'diverged') return '两个后端在中间出现了分歧';
  return '';
});

/** 速度对比：把两次生成的真实读数并排给出。 */
const speedRows = computed(() => {
  if (!props.result) return [];
  return [props.result.first, props.result.second];
});

const canRun = computed(() => Boolean(props.availability?.available) && !props.running);
</script>

<template>
  <section class="card side-card">
    <div class="side-title">
      <h2>后端一致性校验</h2>
      <button class="secondary-button" :disabled="!canRun" @click="emit('run')">
        {{ running ? '校验中…' : '开始校验' }}
      </button>
    </div>

    <p class="muted hint">
      用同一段提示词在当前后端与另一个后端各生成一次，逐字比对输出。
      <strong>校验期间服务会重启三次，聊天暂时不可用（约 1–2 分钟）</strong>，
      结束后自动切回你原来的后端。
    </p>

    <div v-if="running" class="loading-box">
      正在分别用两个后端生成同一段内容，请稍候…
    </div>

    <div v-else-if="error" class="check-error">{{ error }}</div>

    <template v-else-if="result">
      <div class="verdict" :class="verdictClass">
        <strong>{{ verdictTitle }}</strong>
        <span>{{ result.comparison.summary }}</span>
      </div>

      <table class="run-table">
        <thead>
          <tr>
            <th>请求后端</th>
            <th>实际后端</th>
            <th>速度</th>
            <th>生成长度</th>
            <th>停止原因</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="run in speedRows" :key="run.backend">
            <td>{{ run.backend }}</td>
            <td>{{ run.effective_backend || '—' }}</td>
            <td>{{ run.tokens_per_sec.toFixed(2) }} token/秒</td>
            <td>{{ run.token_count }} token</td>
            <td>{{ run.stop_type === 'limit' ? '跑满长度' : '自然结束' }}</td>
          </tr>
        </tbody>
      </table>

      <p class="muted footnote">
        「实际后端」是引擎真正加载的那个（扫描引擎目录得出）。两轮这一列必须<strong>不同</strong>，
        校验才成立；若相同，本次校验会被直接判为无效。
      </p>

      <p class="muted footnote">
        已恢复到你原来的后端：{{ result.restored_to }}
      </p>
      <p class="muted footnote">
        提示：两个后端的速度通常接近，瓶颈多在显存带宽；但少数组合（例如新架构配
        Vulkan）可能相差数倍，那是后端实现效率的差异，不代表算错了。校验的意义是
        确认输出可靠，而不是挑选更快的后端。
      </p>
    </template>

    <div v-else-if="availability && !availability.available" class="muted">
      {{ availability.reason }}
    </div>
    <div v-else class="muted">
      点击开始校验，确认两个显卡后端的输出是否一致。
    </div>
  </section>
</template>

<style scoped>
.hint {
  margin: 0 0 12px;
  line-height: 1.6;
}

.verdict {
  border-radius: 8px;
  padding: 10px 12px;
  margin-bottom: 12px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  line-height: 1.5;
}

.verdict strong {
  font-size: 14px;
}

.verdict span {
  font-size: 13px;
}

.verdict-ok {
  background: rgba(46, 160, 67, 0.12);
  border: 1px solid rgba(46, 160, 67, 0.4);
}

.verdict-warn {
  background: rgba(210, 153, 34, 0.12);
  border: 1px solid rgba(210, 153, 34, 0.4);
}

.verdict-bad {
  background: rgba(207, 34, 46, 0.12);
  border: 1px solid rgba(207, 34, 46, 0.4);
}

.check-error {
  color: #cf222e;
  line-height: 1.6;
}

.run-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  margin-bottom: 10px;
}

.run-table th,
.run-table td {
  text-align: left;
  padding: 6px 8px;
  border-bottom: 1px solid rgba(128, 128, 128, 0.25);
}

.run-table th {
  font-weight: 600;
  opacity: 0.75;
}

.footnote {
  font-size: 12px;
  line-height: 1.6;
  margin: 6px 0 0;
}
</style>
