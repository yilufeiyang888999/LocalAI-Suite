<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { getLocalIp } from '../api/tauri';
import type { AppSettings, ServiceInfo } from '../types';

const props = defineProps<{
  service: ServiceInfo | null;
  settings: AppSettings | null;
}>();

const lanIp = ref<string | null>(null);
const copiedField = ref<string | null>(null);

onMounted(async () => {
  try {
    lanIp.value = await getLocalIp();
  } catch {
    lanIp.value = null;
  }
});

const running = computed(() => props.service?.status === 'running' && !!props.service?.port);

const port = computed(() => props.service?.port ?? props.settings?.service_port ?? 8080);
const runningHost = computed(() => props.service?.host ?? '127.0.0.1');
const lanMode = computed(() => runningHost.value === '0.0.0.0');

const localBaseUrl = computed(() => `http://127.0.0.1:${port.value}/v1`);
const lanBaseUrl = computed(() => (lanMode.value && lanIp.value ? `http://${lanIp.value}:${port.value}/v1` : null));
const modelId = computed(() => props.service?.model_id ?? '');
const apiKey = 'sk-no-key-required';

async function copy(fieldId: string, value: string) {
  try {
    await navigator.clipboard.writeText(value);
    copiedField.value = fieldId;
    setTimeout(() => {
      if (copiedField.value === fieldId) copiedField.value = null;
    }, 1500);
  } catch {
    copiedField.value = null;
  }
}

// A per-app template rendered in the accordion. `fields` covers the simple
// "fill three inputs" apps; `snippet` covers Continue.dev's config.yaml case.
interface AppTemplate {
  id: string;
  name: string;
  hint: string;
  steps: string[];
  fields?: { label: string; value: string }[];
  snippet?: { fileName: string; code: string };
  docs?: { url: string; label: string };
}

const templates = computed<AppTemplate[]>(() => {
  // Note: some clients require a non-empty model name string in requests; we
  // pass the current model_id, but any string works with llama.cpp.
  const model = modelId.value || '任意名称都可以';
  return [
    {
      id: 'cherry-studio',
      name: 'Cherry Studio',
      hint: '中文用户常用的桌面聊天客户端。',
      steps: [
        '打开 Cherry Studio → 左下角「设置」',
        '选择「模型服务」→ 点击底部「+ 添加」',
        '「提供商类型」选 OpenAI，然后填下面 3 项：'
      ],
      fields: [
        { label: 'API 地址（Base URL）', value: localBaseUrl.value },
        { label: 'API 密钥', value: apiKey },
        { label: '模型名', value: model }
      ],
      docs: { url: 'https://docs.cherry-ai.com', label: 'Cherry Studio 官方文档' }
    },
    {
      id: 'chatbox',
      name: 'Chatbox',
      hint: '跨平台聊天客户端，中文界面友好。',
      steps: [
        '打开 Chatbox → 左下角设置图标',
        '「AI 模型提供方」选择 OpenAI API',
        '把 API Host 改成下面的 Base URL，然后填 Key 和模型：'
      ],
      fields: [
        { label: 'API Host', value: localBaseUrl.value },
        { label: 'API Key', value: apiKey },
        { label: 'Model', value: model }
      ],
      docs: { url: 'https://chatboxai.app', label: 'Chatbox 官网' }
    },
    {
      id: 'cursor',
      name: 'Cursor',
      hint: '需要在 Cursor 设置里手动打开「Override OpenAI Base URL」。',
      steps: [
        '打开 Cursor → Settings（Ctrl/Cmd + ,）→ Models',
        '在 API Keys 分区把「OpenAI API Key」填成下面的占位',
        '打开「Override OpenAI Base URL」，粘贴 Base URL 后点 Verify',
        '在下面「Model Names」里添加一个自定义模型名（就用下面这个）'
      ],
      fields: [
        { label: 'OpenAI API Key', value: apiKey },
        { label: 'Override OpenAI Base URL', value: localBaseUrl.value },
        { label: 'Custom Model Name', value: model }
      ],
      docs: { url: 'https://docs.cursor.com/settings/api-keys', label: 'Cursor 官方文档' }
    },
    {
      id: 'continue',
      name: 'Continue.dev',
      hint: 'VSCode/JetBrains 插件，需要编辑 config.yaml。',
      steps: [
        '在 VSCode/JetBrains 里安装 Continue 插件',
        '按 Ctrl/Cmd + Shift + P，输入并执行「Continue: Open Config File」打开 config.yaml',
        '把下面这段替换进 config.yaml（顶部的 name/version/schema 是必填项，缺了插件会加载失败）：',
        'contextLength 已设为 16384，与本软件默认一致；若仍提示上下文超限，请在软件「设置 → 自定义运行参数」加 --ctx-size 16384 并重启服务（两边要一起调大）',
        'Windows 路径：%USERPROFILE%\\.continue\\config.yaml；保存后自动生效，未生效则执行「Reload Window」'
      ],
      snippet: {
        fileName: '~/.continue/config.yaml',
        code: [
          'name: Local Assistant',
          'version: 1.0.0',
          'schema: v1',
          'models:',
          '  - name: 本地AI',
          '    provider: openai',
          `    model: ${model}`,
          `    apiBase: ${localBaseUrl.value}`,
          `    apiKey: ${apiKey}`,
          '    roles:',
          '      - chat',
          '      - edit',
          '      - apply',
          '    defaultCompletionOptions:',
          '      contextLength: 16384'
        ].join('\n')
      },
      docs: { url: 'https://docs.continue.dev/reference', label: 'Continue.dev 官方文档' }
    },
    {
      id: 'generic',
      name: '通用（其他 OpenAI 兼容应用）',
      hint: '任何遵循 OpenAI API 规范的应用都可以按下面填。',
      steps: [
        '在应用中找到「OpenAI API」或「自定义 Provider」相关设置',
        '把 Base URL、API Key、模型名分别填成下面三项',
        '本地服务的可用接口：/v1/models、/v1/chat/completions、/v1/completions'
      ],
      fields: [
        { label: 'Base URL', value: localBaseUrl.value },
        { label: 'API Key', value: apiKey },
        { label: '模型名', value: model }
      ]
    }
  ];
});

function fieldButtonLabel(fieldId: string) {
  return copiedField.value === fieldId ? '✓ 已复制' : '复制';
}
</script>

<template>
  <section class="card">
    <div class="section-title">
      <span class="step-badge">7</span>
      <div>
        <h2>在其他应用中使用本地AI</h2>
        <p>本地AI服务兼容 OpenAI 接口，可以直接接入 Cherry Studio、Cursor、Continue.dev 等第三方应用。</p>
      </div>
    </div>

    <div v-if="!running" class="notice">
      本地AI服务尚未启动。启动一个模型后，这里会显示连接地址和各类应用的配置模板。
    </div>

    <div v-else class="third-party-body">
      <div class="connect-summary">
        <div class="summary-row">
          <label>本机调用地址</label>
          <code>{{ localBaseUrl }}</code>
          <button class="mini-button" @click="copy('local-url', localBaseUrl)">{{ fieldButtonLabel('local-url') }}</button>
        </div>
        <div v-if="lanBaseUrl" class="summary-row">
          <label>局域网调用地址</label>
          <code>{{ lanBaseUrl }}</code>
          <button class="mini-button" @click="copy('lan-url', lanBaseUrl ?? '')">{{ fieldButtonLabel('lan-url') }}</button>
        </div>
        <div v-else-if="lanMode" class="summary-row">
          <label>局域网调用地址</label>
          <span class="muted">正在检测本机局域网 IP...</span>
        </div>
        <div class="summary-row">
          <label>API Key（占位，本地服务不校验）</label>
          <code>{{ apiKey }}</code>
          <button class="mini-button" @click="copy('api-key', apiKey)">{{ fieldButtonLabel('api-key') }}</button>
        </div>
        <div v-if="modelId" class="summary-row">
          <label>当前模型名</label>
          <code>{{ modelId }}</code>
          <button class="mini-button" @click="copy('model-id', modelId)">{{ fieldButtonLabel('model-id') }}</button>
        </div>
        <p v-if="lanMode" class="muted lan-hint">
          已开启局域网访问：本机和同网段其他设备都能调用。请仅在可信网络（如家庭/公司内网）使用。
        </p>
      </div>

      <div class="app-list">
        <details v-for="tpl in templates" :key="tpl.id" class="app-template">
          <summary>
            <span class="app-name">{{ tpl.name }}</span>
            <span class="app-hint">{{ tpl.hint }}</span>
          </summary>
          <div class="app-body">
            <ol class="app-steps">
              <li v-for="(step, i) in tpl.steps" :key="i">{{ step }}</li>
            </ol>

            <div v-if="tpl.fields" class="app-fields">
              <div v-for="(field, i) in tpl.fields" :key="i" class="app-field">
                <label>{{ field.label }}</label>
                <code>{{ field.value }}</code>
                <button class="mini-button" @click="copy(`${tpl.id}-${i}`, field.value)">
                  {{ fieldButtonLabel(`${tpl.id}-${i}`) }}
                </button>
              </div>
            </div>

            <div v-if="tpl.snippet" class="app-snippet">
              <div class="snippet-header">
                <span class="muted">{{ tpl.snippet.fileName }}</span>
                <button class="mini-button" @click="copy(`${tpl.id}-snippet`, tpl.snippet.code)">
                  {{ fieldButtonLabel(`${tpl.id}-snippet`) }}
                </button>
              </div>
              <pre><code>{{ tpl.snippet.code }}</code></pre>
            </div>

            <p v-if="tpl.docs" class="app-docs">
              <a :href="tpl.docs.url" target="_blank" rel="noopener">→ {{ tpl.docs.label }}</a>
            </p>
          </div>
        </details>
      </div>
    </div>
  </section>
</template>

<style scoped>
.third-party-body {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.connect-summary {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 12px 14px;
  background: rgba(0, 0, 0, 0.03);
  border-radius: 8px;
}
.summary-row {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.summary-row label {
  min-width: 190px;
  font-weight: 600;
  font-size: 13px;
}
.summary-row code {
  flex: 1;
  padding: 4px 8px;
  background: rgba(0, 0, 0, 0.05);
  border-radius: 4px;
  font-size: 13px;
  word-break: break-all;
}
.lan-hint {
  margin: 4px 0 0;
  font-size: 12px;
}
.mini-button {
  padding: 4px 10px;
  font-size: 12px;
  border-radius: 4px;
  border: 1px solid #d0d0d0;
  background: #fff;
  cursor: pointer;
  white-space: nowrap;
}
.mini-button:hover {
  background: #f0f0f0;
}
.app-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.app-template {
  border: 1px solid #e5e5e5;
  border-radius: 6px;
  padding: 10px 14px;
}
.app-template summary {
  cursor: pointer;
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.app-template summary::-webkit-details-marker {
  display: none;
}
.app-name {
  font-weight: 600;
  font-size: 15px;
}
.app-hint {
  font-size: 12px;
  color: #666;
}
.app-body {
  padding-top: 10px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.app-steps {
  margin: 0;
  padding-left: 20px;
  font-size: 13px;
  line-height: 1.6;
}
.app-fields {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.app-field {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.app-field label {
  min-width: 200px;
  font-size: 13px;
}
.app-field code {
  flex: 1;
  padding: 4px 8px;
  background: rgba(0, 0, 0, 0.05);
  border-radius: 4px;
  font-size: 13px;
  word-break: break-all;
}
.app-snippet {
  background: rgba(0, 0, 0, 0.04);
  border-radius: 6px;
  padding: 10px 12px;
}
.snippet-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 6px;
}
.app-snippet pre {
  margin: 0;
  overflow-x: auto;
  font-size: 12px;
  line-height: 1.5;
}
.app-docs {
  margin: 0;
  font-size: 12px;
}
.app-docs a {
  color: #0066cc;
  text-decoration: none;
}
.app-docs a:hover {
  text-decoration: underline;
}
</style>
