<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue';
import { confirm } from '@tauri-apps/plugin-dialog';
import {
  clearChatContext,
  createChatSession,
  deleteChatSession,
  getChatMessages,
  getChatSessions,
  getLocalIp,
  renameChatSession,
  saveChatMessage,
  updateSessionRole
} from '../api/tauri';
import type { AppSettings, ChatMessage, ChatSession, RolePreset, ServiceInfo } from '../types';
import { parseMessage, systemPromptFor } from '../utils/markdown';

const props = defineProps<{
  service: ServiceInfo | null;
  settings: AppSettings | null;
  restoringGpu: boolean;
}>();

const emit = defineEmits<{ 'restore-gpu': []; 'restart-service': [] }>();

const input = ref('');
const messages = ref<ChatMessage[]>([]);
const sessions = ref<ChatSession[]>([]);
const currentSession = ref<ChatSession | null>(null);
const sending = ref(false);
const loadingHistory = ref(false);
const error = ref<string | null>(null);
const temperature = ref(0.7);
let abortController: AbortController | null = null;

// The AI's self-reported name — opt-in. Empty unless the user set a custom
// name in settings, in which case systemPromptFor prepends an identity clause.
// Left empty, no identity is injected and the model answers "who are you" on
// its own (correct for architecture-derived models like Qwen-based finetunes).
const aiIdentity = computed(() => props.settings?.ai_display_name?.trim() || '');

const lanIp = ref<string | null>(null);
onMounted(async () => {
  try {
    lanIp.value = await getLocalIp();
  } catch {
    lanIp.value = null;
  }
});

// Addresses external programs use to call the AI. Always shows the local
// address; adds the LAN address when LAN access is on. `port`/`host` come from
// the *running* service, so a changed setting only appears after a restart —
// which we surface via `pending`.
const apiEndpoint = computed(() => {
  const port = props.service?.port;
  if (!port || props.service?.status !== 'running') return null;
  const runningHost = props.service?.host ?? '127.0.0.1';
  const lanMode = runningHost === '0.0.0.0';
  const cfgHost = props.settings?.service_host ?? runningHost;
  const cfgPort = props.settings?.service_port ?? port;
  const pending = cfgHost !== runningHost || cfgPort !== port;
  return {
    localUrl: `http://127.0.0.1:${port}/v1`,
    lanUrl: lanMode && lanIp.value ? `http://${lanIp.value}:${port}/v1` : null,
    lanMode,
    lanUnknown: lanMode && !lanIp.value,
    pending,
    pendingPort: cfgPort,
    pendingLan: cfgHost === '0.0.0.0'
  };
});

const roleOptions: { value: RolePreset; label: string }[] = [
  { value: 'daily_assistant', label: '日常助手' },
  { value: 'writing', label: '文案写作' },
  { value: 'coding', label: '代码助手' }
];

watch(
  () => props.service?.model_id,
  async (modelId) => {
    if (!modelId) return;
    await refreshSessions();
    if (!currentSession.value) await loadOrCreateSession(modelId);
  },
  { immediate: true }
);

async function refreshSessions() {
  try {
    sessions.value = await getChatSessions();
  } catch (err) {
    error.value = String(err);
  }
}

async function loadOrCreateSession(modelId: string) {
  loadingHistory.value = true;
  error.value = null;
  try {
    sessions.value = await getChatSessions();
    currentSession.value = sessions.value.find((session) => session.model_id === modelId) ?? null;
    if (!currentSession.value) {
      currentSession.value = await createChatSession(modelId);
      sessions.value = [currentSession.value, ...sessions.value];
    }
    messages.value = await getChatMessages(currentSession.value.id);
  } catch (err) {
    error.value = String(err);
  } finally {
    loadingHistory.value = false;
  }
}

async function selectSession(session: ChatSession) {
  if (currentSession.value?.id === session.id) return;
  currentSession.value = session;
  loadingHistory.value = true;
  error.value = null;
  try {
    messages.value = await getChatMessages(session.id);
  } catch (err) {
    error.value = String(err);
  } finally {
    loadingHistory.value = false;
  }
}

async function newSession() {
  if (!props.service?.model_id) {
    error.value = 'AI服务还没有启动，无法新建对话。';
    return;
  }
  try {
    const session = await createChatSession(props.service.model_id, currentRole.value);
    sessions.value = [session, ...sessions.value];
    currentSession.value = session;
    messages.value = [];
  } catch (err) {
    error.value = String(err);
  }
}

async function renameCurrent(session: ChatSession) {
  const name = window.prompt('给这个对话起个名字', session.name);
  if (!name || !name.trim()) return;
  try {
    await renameChatSession(session.id, name.trim());
    session.name = name.trim();
  } catch (err) {
    error.value = String(err);
  }
}

async function removeSession(session: ChatSession) {
  if (!(await confirm(`确定删除对话「${session.name}」及其聊天记录吗？`, { title: '删除对话', kind: 'warning' }))) return;
  try {
    await deleteChatSession(session.id);
    sessions.value = sessions.value.filter((item) => item.id !== session.id);
    if (currentSession.value?.id === session.id) {
      currentSession.value = sessions.value[0] ?? null;
      messages.value = currentSession.value ? await getChatMessages(currentSession.value.id) : [];
    }
  } catch (err) {
    error.value = String(err);
  }
}

const currentRole = computed<RolePreset>(() => currentSession.value?.role_preset ?? 'daily_assistant');

async function selectRole(role: RolePreset) {
  if (!currentSession.value || currentSession.value.role_preset === role) return;
  try {
    await updateSessionRole(currentSession.value.id, role);
    currentSession.value.role_preset = role;
  } catch (err) {
    error.value = String(err);
  }
}

async function clearContext() {
  if (!currentSession.value) return;
  if (!(await confirm('清空当前对话上下文？历史消息会保留，但AI会忘记之前的内容。', { title: '清空上下文', kind: 'warning' }))) return;
  try {
    await clearChatContext(currentSession.value.id);
    messages.value = [];
  } catch (err) {
    error.value = String(err);
  }
}

function stopGeneration() {
  abortController?.abort();
}

async function copyText(text: string) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    /* clipboard unavailable; ignore */
  }
}

async function send() {
  if (!input.value.trim() || sending.value) return;
  if (!props.service?.port || !props.service.model_id) {
    error.value = 'AI服务还没有启动，请先完成安装或在设置中配置本地运行引擎。';
    return;
  }
  if (!currentSession.value) {
    await loadOrCreateSession(props.service.model_id);
  }
  if (!currentSession.value) return;

  const session = currentSession.value;
  const userText = input.value.trim();
  messages.value.push({ role: 'user', content: userText });
  messages.value.push({ role: 'assistant', content: '' });
  // Mutate through the array element (a reactive proxy) so streamed tokens
  // trigger re-renders; keep the reference to prune an empty reply later.
  const assistantMessage = messages.value[messages.value.length - 1];
  input.value = '';
  sending.value = true;
  error.value = null;
  abortController = new AbortController();

  try {
    await saveChatMessage(session.id, 'user', userText);
    const priorTurns = messages.value
      .slice(0, -1)
      .filter((message) => message.content.trim().length > 0)
      .map((message) => ({ role: message.role, content: message.content }));
    const response = await fetch(`http://127.0.0.1:${props.service.port}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: abortController.signal,
      body: JSON.stringify({
        model: props.service.model_id ?? 'local-model',
        messages: [{ role: 'system', content: systemPromptFor(currentRole.value, aiIdentity.value) }, ...priorTurns],
        temperature: temperature.value,
        stream: true
      })
    });
    if (!response.ok || !response.body) throw new Error('request failed');

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newlineIndex: number;
      while ((newlineIndex = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, newlineIndex).trim();
        buffer = buffer.slice(newlineIndex + 1);
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trim();
        if (payload === '[DONE]') continue;
        try {
          const json = JSON.parse(payload);
          const delta = json?.choices?.[0]?.delta?.content;
          if (delta) assistantMessage.content += delta;
        } catch {
          /* ignore keep-alive / partial frames */
        }
      }
    }
  } catch (err) {
    if ((err as Error)?.name !== 'AbortError') {
      error.value = '暂时没连上本地AI服务，请确认模型正在运行。';
    }
  } finally {
    sending.value = false;
    abortController = null;
    const finalText = assistantMessage.content.trim();
    if (finalText.length > 0) {
      try {
        await saveChatMessage(session.id, 'assistant', finalText);
      } catch {
        /* persistence best-effort */
      }
    } else {
      messages.value = messages.value.filter((message) => message !== assistantMessage);
    }
  }
}
</script>

<template>
  <section class="card chat-card">
    <div class="section-title">
      <span class="step-badge">5</span>
      <div>
        <h2>聊天</h2>
        <p>服务启动后即可在这里和本地AI对话，聊天记录会保存在本机。</p>
      </div>
    </div>
    <div class="chat-status" :class="props.service?.status">
      {{ props.service?.message ?? 'AI服务尚未启动。' }}
    </div>
    <div v-if="apiEndpoint" class="api-endpoint">
      <div class="api-endpoint-title">其他程序调用本机AI的地址（兼容 OpenAI 接口）</div>
      <div class="api-endpoint-row">
        <span class="api-label">本机</span>
        <code>{{ apiEndpoint.localUrl }}</code>
        <button class="secondary-button" @click="copyText(apiEndpoint.localUrl)">复制地址</button>
      </div>
      <div v-if="apiEndpoint.lanUrl" class="api-endpoint-row">
        <span class="api-label">局域网</span>
        <code>{{ apiEndpoint.lanUrl }}</code>
        <button class="secondary-button" @click="copyText(apiEndpoint.lanUrl)">复制地址</button>
      </div>
      <p v-else-if="apiEndpoint.lanUnknown" class="api-hint">
        已开启局域网访问，但没能自动识别本机局域网IP。请在系统里查看本机IP，用「http://本机IP:{{ apiEndpoint.pendingPort }}/v1」调用。
      </p>
      <p v-else-if="!apiEndpoint.lanMode" class="api-hint">
        目前只有本机能调用。想让局域网其他设备访问，请到右侧「设置」把「谁可以调用本机AI」改为局域网，再重启AI服务。
      </p>
      <div v-if="apiEndpoint.pending" class="api-pending">
        <span>接口设置已修改（新端口 {{ apiEndpoint.pendingPort }}{{ apiEndpoint.pendingLan ? '，局域网可访问' : '，仅本机' }}），重启AI服务后生效。</span>
        <button class="primary-button" @click="emit('restart-service')">重启AI服务以生效</button>
      </div>
    </div>
    <div v-if="props.service?.status === 'running' && props.service?.safe_mode" class="safe-mode-bar">
      <span>当前在更稳妥的 CPU 模式运行，速度较慢。</span>
      <button class="secondary-button" :disabled="props.restoringGpu" @click="emit('restore-gpu')">
        {{ props.restoringGpu ? '正在尝试恢复显卡加速' : '恢复显卡加速' }}
      </button>
    </div>

    <div class="chat-layout">
      <aside class="session-list">
        <button class="secondary-button new-session" @click="newSession">+ 新对话</button>
        <ul>
          <li
            v-for="session in sessions"
            :key="session.id"
            :class="{ active: session.id === currentSession?.id }"
            @click="selectSession(session)"
          >
            <span class="session-name">{{ session.name }}</span>
            <span class="session-actions">
              <button title="重命名" @click.stop="renameCurrent(session)">✎</button>
              <button title="删除" @click.stop="removeSession(session)">🗑</button>
            </span>
          </li>
        </ul>
      </aside>

      <div class="chat-main">
        <div class="chat-toolbar">
          <div class="role-picker">
            <button
              v-for="role in roleOptions"
              :key="role.value"
              class="role-chip"
              :class="{ active: currentRole === role.value }"
              :disabled="!currentSession"
              @click="selectRole(role.value)"
            >
              {{ role.label }}
            </button>
          </div>
          <div class="temperature">
            <span>更严谨</span>
            <input v-model.number="temperature" type="range" min="0" max="1" step="0.05" />
            <span>更有创意</span>
          </div>
          <button class="link-button" :disabled="!currentSession || !messages.length" @click="clearContext">
            清空上下文
          </button>
        </div>

        <div class="messages">
          <div v-if="loadingHistory" class="empty-chat">正在读取本机聊天记录...</div>
          <div v-else-if="messages.length === 0" class="empty-chat">还没有消息，试着问一个简单问题。</div>
          <div
            v-for="(message, index) in messages"
            :key="message.id ?? index"
            class="message"
            :class="message.role"
          >
            <div class="message-head">
              <strong>{{ message.role === 'user' ? '你' : 'AI' }}</strong>
              <button
                v-if="message.role === 'assistant' && message.content"
                class="copy-reply"
                title="复制回复"
                @click="copyText(message.content)"
              >
                复制
              </button>
            </div>
            <p v-if="message.role === 'user'" class="user-text">{{ message.content }}</p>
            <div v-else class="assistant-body">
              <template v-for="(block, bi) in parseMessage(message.content)" :key="bi">
                <div v-if="block.type === 'text'" class="md" v-html="block.html"></div>
                <div v-else class="code-block">
                  <button class="copy-code" @click="copyText(block.code ?? '')">复制代码</button>
                  <pre><code>{{ block.code }}</code></pre>
                </div>
              </template>
              <span v-if="sending && index === messages.length - 1 && !message.content" class="typing">正在思考…</span>
            </div>
          </div>
        </div>

        <form class="chat-input" @submit.prevent="send">
          <input v-model="input" placeholder="输入你想问的问题" />
          <button v-if="sending" type="button" class="secondary-button" @click="stopGeneration">停止</button>
          <button class="primary-button" :disabled="sending">{{ sending ? '生成中' : '发送' }}</button>
        </form>
        <p v-if="error" class="notice warning">{{ error }}</p>
      </div>
    </div>
  </section>
</template>

<style scoped>
.api-endpoint {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-top: 8px;
  padding: 10px 12px;
  border-radius: 8px;
  background: rgba(16, 185, 129, 0.1);
  border: 1px solid rgba(16, 185, 129, 0.25);
  font-size: 13px;
  color: #065f46;
}
.api-endpoint-title {
  font-weight: 700;
}
.api-endpoint-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}
.api-label {
  flex-shrink: 0;
  min-width: 42px;
  font-weight: 600;
  opacity: 0.8;
}
.api-endpoint code {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  background: rgba(6, 95, 70, 0.12);
  padding: 2px 6px;
  border-radius: 4px;
  user-select: all;
}
.api-hint {
  margin: 0;
  color: #047857;
  opacity: 0.9;
}
.api-pending {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-top: 2px;
  padding-top: 8px;
  border-top: 1px dashed rgba(6, 95, 70, 0.25);
  color: #92400e;
}
.chat-layout {
  display: flex;
  gap: 12px;
  margin-top: 12px;
}
.session-list {
  width: 160px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.session-list ul {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
  max-height: 320px;
  overflow-y: auto;
}
.session-list li {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 8px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
}
.session-list li.active {
  background: rgba(99, 102, 241, 0.12);
}
.session-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.session-actions button {
  border: none;
  background: transparent;
  cursor: pointer;
  font-size: 12px;
  opacity: 0.6;
}
.session-actions button:hover {
  opacity: 1;
}
.chat-main {
  flex: 1;
  min-width: 0;
}
.chat-toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 8px;
}
.role-picker {
  display: flex;
  gap: 4px;
}
.role-chip {
  border: 1px solid rgba(0, 0, 0, 0.12);
  background: transparent;
  border-radius: 999px;
  padding: 3px 10px;
  font-size: 12px;
  cursor: pointer;
}
.role-chip.active {
  background: rgba(99, 102, 241, 0.15);
  border-color: rgba(99, 102, 241, 0.5);
}
.temperature {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  opacity: 0.75;
}
.message-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.copy-reply,
.copy-code {
  border: none;
  background: transparent;
  cursor: pointer;
  font-size: 12px;
  opacity: 0.6;
}
.copy-reply:hover,
.copy-code:hover {
  opacity: 1;
}
.user-text {
  white-space: pre-wrap;
}
.assistant-body .md :deep(p) {
  margin: 4px 0;
}
.code-block {
  position: relative;
  margin: 6px 0;
}
.code-block pre {
  background: rgba(0, 0, 0, 0.06);
  padding: 10px;
  border-radius: 6px;
  overflow-x: auto;
}
.copy-code {
  position: absolute;
  top: 4px;
  right: 6px;
}
.typing {
  opacity: 0.6;
  font-size: 13px;
}
</style>
