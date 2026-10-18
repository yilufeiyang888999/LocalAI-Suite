<script setup lang="ts">
import type { KbMessageMeta } from '../types';
import { refusalLabel } from '../utils/kb';

defineProps<{ meta: KbMessageMeta }>();

/** 重排分是**原始 logit**（非概率），量级 2~14 之间，保留 2 位即可看出排序依据。 */
function 分(n: number): string {
  return Number.isFinite(n) ? n.toFixed(2) : '—';
}

/** 秒级耗时，超过 1 秒才显示 —— 正常情况下检索+重排只有几百毫秒，不值得占位置。 */
function 秒(n: number | undefined): string {
  return typeof n === 'number' && n >= 1 ? `${n.toFixed(1)} 秒` : '';
}
</script>

<template>
  <div class="kb-meta">
    <!-- 拒答：单独一行、带左边框，与正常回答在视觉上一眼可分 -->
    <div v-if="meta.refused" class="kb-refusal">
      <span class="kb-refusal-title">{{ refusalLabel(meta.refusalType) }}</span>
      <span v-if="meta.refusalReason" class="kb-refusal-reason">{{ meta.refusalReason }}</span>
    </div>

    <!-- 类2 的线索：答案所需的信息指向了没入库的文件，这是「拒答」最有用的那部分信息 -->
    <div v-if="meta.offLibraryRefs.length" class="kb-offlib">
      线索指向未入库的文档：<code v-for="号 in meta.offLibraryRefs" :key="号">{{ 号 }}</code>
    </div>

    <details v-if="meta.citations.length" class="kb-cites">
      <summary>
        依据 {{ meta.citations.length }} 条
        <span v-if="秒(meta.timings?.total)" class="kb-timing">· {{ 秒(meta.timings?.total) }}</span>
      </summary>
      <ol>
        <li v-for="条 in meta.citations" :key="条.rank">
          <div class="kb-cite-head">
            <code class="kb-docno">{{ 条.doc_no }}</code>
            <span class="kb-section">{{ 条.section }}</span>
            <span class="kb-score" title="重排得分（原始 logit，越大越相关）">{{ 分(条.score) }}</span>
          </div>
          <p class="kb-excerpt">{{ 条.excerpt }}</p>
        </li>
      </ol>
    </details>
  </div>
</template>

<style scoped>
.kb-meta {
  margin-top: 8px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  font-size: 13px;
}
.kb-refusal {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 8px 10px;
  border-left: 3px solid rgba(217, 119, 6, 0.7);
  border-radius: 4px;
  background: rgba(251, 191, 36, 0.12);
}
.kb-refusal-title {
  font-weight: 600;
  color: #92400e;
}
.kb-refusal-reason {
  opacity: 0.85;
  color: #92400e;
}
.kb-offlib {
  padding: 6px 10px;
  border-radius: 4px;
  background: rgba(59, 130, 246, 0.1);
  color: #1e40af;
}
.kb-offlib code {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  margin-left: 4px;
  padding: 1px 5px;
  border-radius: 3px;
  background: rgba(30, 64, 175, 0.12);
}
.kb-cites summary {
  cursor: pointer;
  color: #4338ca;
  font-weight: 600;
  user-select: none;
}
.kb-timing {
  font-weight: 400;
  opacity: 0.7;
}
.kb-cites ol {
  margin: 6px 0 0;
  padding-left: 20px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.kb-cite-head {
  display: flex;
  align-items: baseline;
  gap: 8px;
  flex-wrap: wrap;
}
.kb-docno {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  padding: 1px 6px;
  border-radius: 3px;
  background: rgba(99, 102, 241, 0.14);
}
.kb-section {
  font-weight: 600;
}
.kb-score {
  margin-left: auto;
  font-variant-numeric: tabular-nums;
  opacity: 0.55;
  font-size: 12px;
}
.kb-excerpt {
  margin: 3px 0 0;
  opacity: 0.75;
  line-height: 1.5;
}
</style>
