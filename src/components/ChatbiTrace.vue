<script setup lang="ts">
import type { ChatbiMessageMeta } from '../types';
import { 链路标签, 路由依据标签 } from '../utils/chatbi';

defineProps<{ meta: ChatbiMessageMeta }>();
</script>

<template>
  <div class="cb-meta">
    <!--
      🔴 越权**单独一块红框、放在最前面** —— 与「依据不切题」同一个道理：
         这类问题的特点是「表面上没有任何异常」（回答文本看着正常），
         混在正文里就没人会注意到。
    -->
    <div v-if="meta.被拒" class="cb-deny">
      <div class="cb-deny-title">🚫 权限不足，未生成任何单据。</div>
      <div v-if="meta.拒绝理由" class="cb-deny-reason">{{ meta.拒绝理由 }}</div>
      <div class="cb-deny-hint">
        当前角色：{{ meta.角色 ?? '—' }}。换一个有此权限的角色再试，或改走只读链路先查数据。
        （这次调用已进审计。）
      </div>
    </div>

    <!-- 依据不切题：单据会照出、数字也是真的，但它不支撑这张单据 —— 不提示就没人会发现 -->
    <div v-if="meta.依据不切题" class="cb-offtopic">
      ⚠️ 本单据的依据可能不切题，请核对后再使用。
    </div>

    <div class="cb-line">
      <span class="cb-chain">{{ 链路标签(meta.类型) }}</span>
      <span class="cb-route" :class="{ fallback: meta.路由依据 === '模型' }">
        路由 {{ 路由依据标签(meta.路由依据) }}
      </span>
      <span v-if="meta.耗时秒" class="cb-time">{{ meta.耗时秒.toFixed(2) }} s</span>
    </div>

    <div v-if="meta.路由理由" class="cb-why">{{ meta.路由理由 }}</div>

    <!-- ReAct 轨迹：只有动作链路会有多步。知识 / 数据链路是单步委派，这里通常为空。 -->
    <details v-if="meta.步骤.length" class="cb-steps">
      <summary>执行轨迹 {{ meta.步骤.length }} 步</summary>
      <ol>
        <li v-for="步 in meta.步骤" :key="步.步" :class="{ denied: 步.被拒 }">
          <div class="cb-step-head">
            <code class="cb-step-no">{{ 步.步 }}</code>
            <span class="cb-step-action">{{ 步.动作 }}</span>
            <span v-if="步.被拒" class="cb-step-deny">被拒</span>
          </div>
          <p v-if="步.思考" class="cb-step-think">{{ 步.思考 }}</p>
          <p v-if="步.观察" class="cb-step-obs">{{ 步.观察 }}</p>
        </li>
      </ol>
    </details>

    <div v-if="meta.产物路径" class="cb-artifact">
      产物已落盘：<code>{{ meta.产物路径 }}</code>
    </div>
  </div>
</template>

<style scoped>
.cb-meta {
  margin-top: 8px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  font-size: 13px;
}
.cb-deny {
  display: flex;
  flex-direction: column;
  gap: 3px;
  padding: 8px 10px;
  border-left: 3px solid rgba(220, 38, 38, 0.75);
  border-radius: 4px;
  background: rgba(220, 38, 38, 0.1);
  color: #991b1b;
}
.cb-deny-title {
  font-weight: 700;
}
.cb-deny-reason {
  opacity: 0.92;
}
.cb-deny-hint {
  opacity: 0.8;
  font-size: 12px;
}
.cb-offtopic {
  padding: 6px 10px;
  border-left: 3px solid rgba(217, 119, 6, 0.7);
  border-radius: 4px;
  background: rgba(251, 191, 36, 0.12);
  color: #92400e;
}
.cb-line {
  display: flex;
  align-items: baseline;
  gap: 10px;
  flex-wrap: wrap;
}
.cb-chain {
  font-weight: 600;
}
.cb-route {
  opacity: 0.7;
  font-size: 12px;
}
/* 模型兜底 = 不可复现的判断，视觉上要和「确定性规则」分开 */
.cb-route.fallback {
  color: #92400e;
  font-weight: 600;
  opacity: 1;
}
.cb-time {
  margin-left: auto;
  font-variant-numeric: tabular-nums;
  opacity: 0.55;
  font-size: 12px;
}
.cb-why {
  opacity: 0.7;
  font-size: 12px;
}
.cb-steps summary {
  cursor: pointer;
  color: #4338ca;
  font-weight: 600;
  user-select: none;
}
.cb-steps ol {
  margin: 6px 0 0;
  padding-left: 20px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.cb-steps li.denied {
  opacity: 0.95;
}
.cb-step-head {
  display: flex;
  align-items: baseline;
  gap: 8px;
  flex-wrap: wrap;
}
.cb-step-no {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  padding: 1px 6px;
  border-radius: 3px;
  background: rgba(99, 102, 241, 0.14);
}
.cb-step-action {
  font-weight: 600;
}
.cb-step-deny {
  color: #991b1b;
  font-weight: 600;
}
.cb-step-think {
  margin: 3px 0 0;
  opacity: 0.7;
  line-height: 1.5;
}
.cb-step-obs {
  margin: 2px 0 0;
  opacity: 0.85;
  line-height: 1.5;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
}
.cb-artifact {
  padding: 6px 10px;
  border-radius: 4px;
  background: rgba(16, 185, 129, 0.1);
  color: #065f46;
}
.cb-artifact code {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  user-select: all;
}
</style>
