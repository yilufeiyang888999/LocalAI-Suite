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
import KbCitations from './KbCitations.vue';
import ChatbiTrace from './ChatbiTrace.vue';
import {
  askChatbi,
  chatbiMetaToPlainText,
  normalizeChatbiBaseUrl,
  probeChatbiService,
  toChatbiMessageMeta
} from '../utils/chatbi';
import type { ChatbiProbeResult } from '../utils/chatbi';
import {
  askKb,
  kbMetaToPlainText,
  normalizeKbBaseUrl,
  probeKbService,
  toKbMessageMeta
} from '../utils/kb';
import type { KbProbeResult } from '../utils/kb';

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

// ---------------------------------------------------------------------------
// 知识库模式（作品一 · 对接）
// ---------------------------------------------------------------------------
/**
 * 知识库会话在 `chat_sessions.model_id` 里用这个哨兵值。
 *
 * 它**不是模型**：`create_chat_session` 只校验非空、不查模型表（见 `chat.rs`），
 * 唯一用途是把知识库问答的历史与普通聊天**在同一个列表里分开显示**。
 * 用哨兵值而不是新表，是为了不引入版本化迁移 —— 项目当时也没有迁移机制。
 */
const KB_SESSION_KEY = '__kb__';
/**
 * 智能问数会话在 `chat_sessions.model_id` 里的哨兵值。与 `KB_SESSION_KEY` 同一个理由：
 * 它不是模型，只是把一个模式的历史与其它模式**在同一个列表里分开显示**，
 * 从而不必引入版本化迁移。
 */
const CHATBI_SESSION_KEY = '__chatbi__';
const mode = ref<'chat' | 'kb' | 'chatbi'>('chat');
const kbBaseUrl = computed(() => normalizeKbBaseUrl(props.settings?.kb_base_url));
const kbProbe = ref<KbProbeResult | null>(null);
const kbBusy = ref(false);

// ---------------------------------------------------------------------------
// 智能问数 / 动作执行模式（作品二 · 对接）
// ---------------------------------------------------------------------------
/**
 * 作品二对外只有**一个入口**（编排服务 8096），它背后是三条链路。
 * 所以 APP 里只需要**一个模式**：用户不必先知道自己问的是制度类、数据类还是动作类问题
 * —— 那是系统的责任（路由），不是用户的负担。
 */
const chatbiBaseUrl = computed(() => normalizeChatbiBaseUrl(props.settings?.chatbi_base_url));
const chatbiProbe = ref<ChatbiProbeResult | null>(null);
const chatbiBusy = ref(false);
/**
 * 发起动作的**角色** —— 决定「能不能做写动作」。
 *
 * 🔴 角色是**调用方自己填的标签**，不是认证结果。真实部署要接 SSO / 网关注入；
 *    这里做的是「有这道关口 + 有留痕」，以及让越权这件事在界面上一眼看得到。
 */
const chatbiRole = ref<string>('');

const 角色选项 = computed(() => chatbiProbe.value?.health?.权限?.角色 ?? []);

/** 探活后把角色对齐服务端的「默认角色」（最少权限），避免界面默认给出更大权限。 */
watch(角色选项, (列表) => {
  if (!列表.length) return;
  if (!chatbiRole.value || !列表.includes(chatbiRole.value)) {
    chatbiRole.value = chatbiProbe.value?.health?.权限?.默认角色 ?? 列表[0];
  }
});

async function 探活智能问数() {
  chatbiBusy.value = true;
  try {
    chatbiProbe.value = await probeChatbiService(chatbiBaseUrl.value);
  } finally {
    chatbiBusy.value = false;
  }
}

/** 输入框占位：每个模式举一个**真实可用**的例子（比「输入你想问的问题」有用得多）。 */
const 问句占位 = computed(() => {
  if (mode.value === 'kb') return '向知识库提问，例如：不合格品应该怎么处理？';
  if (mode.value === 'chatbi') {
    return '制度类「不合格品应该怎么处理？」／ 数据类「上月哪家供应商不良率最高？」／ 动作类「给东莞精诚出一份质量整改通知」';
  }
  return '输入你想问的问题';
});

/** 三条链路的下游是否都通。8096 活着 ≠ 三条链路都通（知识问答那条依赖 8092）。 */
const 链路断的 = computed(() => {
  const 链 = chatbiProbe.value?.health?.链路;
  if (!链) return [];
  return Object.entries(链)
    .filter(([, v]) => v.可达 === false)
    .map(([k]) => k);
});

/** 当前模式下"会话归属谁"：普通模式认模型，另两个模式各认自己的哨兵值。 */
const 会话模型键 = computed(() => {
  if (mode.value === 'kb') return KB_SESSION_KEY;
  if (mode.value === 'chatbi') return CHATBI_SESSION_KEY;
  return props.service?.model_id ?? null;
});

/**
 * 知识库服务**内部**用的生成端地址 vs APP 实际在跑的端口。
 *
 * 🔴 这是一条真实存在的静默失败路径：知识库服务默认写死 `http://127.0.0.1:8000`，
 * 而 APP 的 `service_port` 用户可配。不一致时服务照常 200 返回，只是每次答案都是
 * 「资料不足 / (生成失败)」——**不报错，只是没结果**。所以必须显式核对并提示。
 */
const 生成端不一致 = computed(() => {
  const 服务端 = kbProbe.value?.health?.deps?.generate;
  const 端口 = props.service?.port;
  if (!服务端 || !端口) return null;
  const 抓到 = 服务端.match(/:(\d+)\s*$/);
  if (!抓到) return null;
  return Number(抓到[1]) === 端口 ? null : 服务端;
});

async function 探活知识库() {
  kbBusy.value = true;
  try {
    kbProbe.value = await probeKbService(kbBaseUrl.value);
  } finally {
    kbBusy.value = false;
  }
}

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
  [() => props.service?.model_id, mode],
  async () => {
    await 装载当前模式();
  },
  { immediate: true }
);

/** 切模式 / 换模型时重新装载：会话列表、当前会话、消息，全部按当前模式隔离。 */
async function 装载当前模式() {
  const 键 = 会话模型键.value;
  currentSession.value = null;
  messages.value = [];
  if (!键) {
    sessions.value = [];
    return;
  }
  await refreshSessions();
  await loadOrCreateSession(键);
  if (mode.value === 'kb') void 探活知识库();
  if (mode.value === 'chatbi') void 探活智能问数();
}

async function refreshSessions() {
  try {
    const 全部 = await getChatSessions();
    // 三种模式的历史分开显示：各自只看到自己的会话，互不混入
    sessions.value = 全部.filter((会话) => {
      if (mode.value === 'kb') return 会话.model_id === KB_SESSION_KEY;
      if (mode.value === 'chatbi') return 会话.model_id === CHATBI_SESSION_KEY;
      return 会话.model_id !== KB_SESSION_KEY && 会话.model_id !== CHATBI_SESSION_KEY;
    });
  } catch (err) {
    error.value = String(err);
  }
}

async function loadOrCreateSession(modelId: string) {
  loadingHistory.value = true;
  error.value = null;
  try {
    await refreshSessions();
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
  const 键 = 会话模型键.value;
  if (!键) {
    error.value =
      mode.value === 'kb'
        ? '知识库服务还没连上，无法新建对话。'
        : mode.value === 'chatbi'
          ? '智能问数服务还没连上，无法新建对话。'
          : 'AI服务还没有启动，无法新建对话。';
    return;
  }
  try {
    const session = await createChatSession(键, currentRole.value);
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
  if (mode.value === 'kb') return sendKb();
  if (mode.value === 'chatbi') return sendChatbi();
  if (!input.value.trim() || sending.value) return;
  if (!props.service?.port || !props.service.model_id) {
    error.value = 'AI服务还没有启动，请先完成安装或在设置中运行本地AI服务。';
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

/**
 * 知识库问答。
 *
 * 用独立函数而不是在 `send()` 里塞 if：两条链路的**失败模式完全不同**
 * （一个失败于模型服务，一个失败于检索服务），混在一起后错误提示只能含糊地说"失败"，
 * 用户不知道该修哪一头。
 *
 * 这里是**非流式** —— 一次拿全 `answer + citations`。检索必须先跑完才有料给生成端，
 * 本来也不存在"边检索边出字"；代价是这几秒界面只能转圈。
 */
async function sendKb() {
  if (!input.value.trim() || sending.value) return;
  // 连不上就别发 —— 直接给引导，而不是让 fetch 抛一个语焉不详的网络错误
  if (!kbProbe.value?.ok) {
    error.value = `知识库服务没连上（${kbProbe.value?.reason ?? '尚未检测'}）。请按上方提示启动它，或到设置里改地址。`;
    return;
  }
  const 问句 = input.value.trim();
  if (!currentSession.value) {
    await loadOrCreateSession(KB_SESSION_KEY);
  }
  if (!currentSession.value) return;

  const session = currentSession.value;
  messages.value.push({ role: 'user', content: 问句 });
  messages.value.push({ role: 'assistant', content: '' });
  const 助手消息 = messages.value[messages.value.length - 1];
  input.value = '';
  sending.value = true;
  error.value = null;
  abortController = new AbortController();

  try {
    await saveChatMessage(session.id, 'user', 问句);
    const 答案 = await askKb(kbBaseUrl.value, 问句, abortController.signal);
    助手消息.content = (答案.answer ?? '').trim() || '（空）';
    助手消息.kb = toKbMessageMeta(答案);
  } catch (错) {
    if ((错 as Error)?.name !== 'AbortError') {
      error.value = `知识库问答失败：${String(错)}`;
    }
  } finally {
    sending.value = false;
    abortController = null;
    if (助手消息.content.trim() && 助手消息.kb) {
      // 落库只存纯文本（表里没有放引用的列）—— 失败不该让界面上的答案消失
      try {
        await saveChatMessage(session.id, 'assistant', kbMetaToPlainText(助手消息.content, 助手消息.kb));
      } catch {
        /* 落库尽力而为 */
      }
    } else if (!助手消息.content.trim()) {
      messages.value = messages.value.filter((消息) => 消息 !== 助手消息);
    }
  }
}

/**
 * 智能问数 / 动作执行。
 *
 * 与 `sendKb` 分开成独立函数，理由同作品一那条：**三条链路的失败模式完全不同**
 * （检索不到 / 口径错 / 单据依据不切题），混在一个函数里错误提示只能含糊说「失败」。
 *
 * 🔴 **越权（HTTP 403）在这里是正常返回，不是异常。** 服务端把 403 和 200 都当
 *    「带完整应答的业务结论」，前端照常渲染 —— 越权单独一块红框（见 ChatbiTrace），
 *    与「依据不切题」同理：这类问题**表面上没有任何异常**，混在正文里就没人会注意到。
 */
async function sendChatbi() {
  if (!input.value.trim() || sending.value) return;
  // 连不上就别发 —— 直接给引导，而不是让 fetch 抛一个语焉不详的网络错误
  if (!chatbiProbe.value?.ok) {
    error.value = `智能问数服务没连上（${chatbiProbe.value?.reason ?? '尚未检测'}）。请按上方提示启动它，或到设置里改地址。`;
    return;
  }
  const 问句 = input.value.trim();
  if (!currentSession.value) {
    await loadOrCreateSession(CHATBI_SESSION_KEY);
  }
  if (!currentSession.value) return;

  const session = currentSession.value;
  messages.value.push({ role: 'user', content: 问句 });
  messages.value.push({ role: 'assistant', content: '' });
  const 助手消息 = messages.value[messages.value.length - 1];
  input.value = '';
  sending.value = true;
  error.value = null;
  abortController = new AbortController();

  try {
    await saveChatMessage(session.id, 'user', 问句);
    const 应答 = await askChatbi(
      chatbiBaseUrl.value,
      问句,
      chatbiRole.value || chatbiProbe.value?.health?.权限?.默认角色 || '操作员',
      abortController.signal
    );
    助手消息.content = (应答.回答 ?? '').trim() || '（空）';
    助手消息.chatbi = toChatbiMessageMeta(应答);
  } catch (错) {
    if ((错 as Error)?.name !== 'AbortError') {
      error.value = `智能问数失败：${String(错)}`;
    }
  } finally {
    sending.value = false;
    abortController = null;
    if (助手消息.content.trim() && 助手消息.chatbi) {
      // 落库只存纯文本（表里没有放轨迹的列）—— 失败不该让界面上的答案消失
      try {
        await saveChatMessage(
          session.id,
          'assistant',
          chatbiMetaToPlainText(助手消息.content, 助手消息.chatbi)
        );
      } catch {
        /* 落库尽力而为 */
      }
    } else if (!助手消息.content.trim()) {
      messages.value = messages.value.filter((消息) => 消息 !== 助手消息);
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
        <p>
          直接对话，或让本机助手查制度、查数据、起草单据（后两者会给出依据与执行轨迹）。
          聊天记录保存在本机。
        </p>
      </div>
    </div>

    <div class="mode-switch">
      <button class="mode-chip" :class="{ active: mode === 'chat' }" @click="mode = 'chat'">
        通用对话
      </button>
      <button class="mode-chip" :class="{ active: mode === 'kb' }" @click="mode = 'kb'">
        知识库问答
      </button>
      <button class="mode-chip" :class="{ active: mode === 'chatbi' }" @click="mode = 'chatbi'">
        智能问数
      </button>
    </div>

    <!-- 知识库服务状态。未连接时必须给出「怎么启动」—— 只说失败等于没说。 -->
    <div v-if="mode === 'kb'" class="kb-status" :class="kbProbe?.ok ? 'ok' : 'down'">
      <template v-if="kbProbe?.ok">
        <div class="kb-status-head">
          <span>
            已连接 <code>{{ kbBaseUrl }}</code> · {{ kbProbe.health?.docs }} 份文档 /
            {{ kbProbe.health?.blocks }} 块 · {{ kbProbe.health?.chain }}
          </span>
          <button class="secondary-button" :disabled="kbBusy" @click="探活知识库">
            {{ kbBusy ? '检测中' : '重新检测' }}
          </button>
        </div>
      </template>
      <template v-else>
        <div class="kb-status-head">
          <strong>知识库服务未连接</strong>
          <button class="secondary-button" :disabled="kbBusy" @click="探活知识库">
            {{ kbBusy ? '检测中' : '重新检测' }}
          </button>
        </div>
        <p v-if="kbProbe?.reason" class="kb-status-reason">原因：{{ kbProbe.reason }}</p>
        <p class="kb-status-help">
          它是一个独立的本地服务（不随本 APP 启动），需要在 RAG-Embedding 目录下单独运行：
        </p>
        <code class="kb-cmd">python scripts/知识库服务.py</code>
        <p class="kb-status-help">
          地址当前为 <code>{{ kbBaseUrl }}</code>，可在右侧「设置」里修改。
        </p>
      </template>
      <div v-if="生成端不一致" class="kb-mismatch">
        它现在用的生成端是 <code>{{ 生成端不一致 }}</code>，而本机 AI 服务在
        <code>http://127.0.0.1:{{ props.service?.port }}</code>。
        不一致时提问会一直得到「资料不足 / 生成失败」（**不会报错**）。按实际端口重启它：
        <code>--生成端 http://127.0.0.1:{{ props.service?.port }}</code>
      </div>
    </div>

    <!-- 智能问数服务状态。与知识库同一条纪律：未连接时必须给出「怎么启动」，只说失败等于没说。 -->
    <div v-if="mode === 'chatbi'" class="kb-status" :class="chatbiProbe?.ok ? 'ok' : 'down'">
      <template v-if="chatbiProbe?.ok">
        <div class="kb-status-head">
          <span>
            已连接 <code>{{ chatbiBaseUrl }}</code> · 编排服务 v{{ chatbiProbe.health?.版本 }} · 链路
            <template v-if="链路断的.length">部分可达（缺 {{ 链路断的.join('、') }}）</template>
            <template v-else>三条全通</template>
          </span>
          <button class="secondary-button" :disabled="chatbiBusy" @click="探活智能问数">
            {{ chatbiBusy ? '检测中' : '重新检测' }}
          </button>
        </div>
        <!--
          🔴 「服务在跑」不等于「三条链路都通」：知识问答那条是 HTTP 转发给 8092（作品一）的，
             8092 没起时 8096 照样 200。所以必须逐条显示，不能只看服务在不在。
        -->
        <p v-if="链路断的.length" class="cb-partial">
          下游不可达：<code v-for="名 in 链路断的" :key="名">{{ 名 }}</code>
          —— 「知识问答」依赖 8092（作品一，去 RAG-Embedding 目录用它的启动器起）。
          缺它只是这一条链路不可用，另外两条照常。
        </p>
        <div class="cb-role-row">
          <span>发起角色</span>
          <select v-model="chatbiRole" class="cb-role">
            <option v-for="名 in 角色选项" :key="名" :value="名">{{ 名 }}</option>
          </select>
          <span class="cb-role-hint">
            决定能不能提交申请 / 对外发文。角色不够时服务端回 403 且**不生成任何单据**。
          </span>
        </div>
      </template>
      <template v-else>
        <div class="kb-status-head">
          <strong>智能问数服务未连接</strong>
          <button class="secondary-button" :disabled="chatbiBusy" @click="探活智能问数">
            {{ chatbiBusy ? '检测中' : '重新检测' }}
          </button>
        </div>
        <p v-if="chatbiProbe?.reason" class="kb-status-reason">原因：{{ chatbiProbe.reason }}</p>
        <p class="kb-status-help">
          它是一个独立的本地服务（不随本 APP 启动），需要在 ChatBI 目录下单独运行：
        </p>
        <code class="kb-cmd">python scripts/启动-服务.py</code>
        <p class="kb-status-help">
          地址当前为 <code>{{ chatbiBaseUrl }}</code>，可在右侧「设置」里修改。
        </p>
      </template>
    </div>

    <div v-if="mode === 'chat'" class="chat-status" :class="props.service?.status">
      {{ props.service?.message ?? 'AI服务尚未启动。' }}
    </div>
    <div v-if="mode === 'chat' && apiEndpoint" class="api-endpoint">
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
    <!-- 启动成功但性能打折（0.1.7-1）：不能写进上面那条绿色状态条 ——
         绿底写着「只有一半层上了卡」，等于用配色宣布一切正常。 -->
    <div v-if="props.service?.status === 'running' && props.service?.warning" class="degraded-bar">
      <span class="degraded-icon">!</span>
      <span>{{ props.service.warning }}</span>
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
          <div v-if="mode === 'chat'" class="role-picker">
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
          <div v-if="mode === 'chat'" class="temperature">
            <span>更严谨</span>
            <input v-model.number="temperature" type="range" min="0" max="1" step="0.05" />
            <span>更有创意</span>
          </div>
          <span v-if="mode === 'kb'" class="kb-hint">问答不带上下文，每次都是独立检索</span>
          <span v-if="mode === 'chatbi'" class="kb-hint">
            一个入口三种问题：制度类 / 数据类 / 动作类。不带上下文，每次独立判断。
          </span>
          <button class="link-button" :disabled="!currentSession || !messages.length" @click="clearContext">
            {{ mode === 'chat' ? '清空上下文' : '清空问答记录' }}
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
              <KbCitations v-if="message.kb" :meta="message.kb" />
              <ChatbiTrace v-if="message.chatbi" :meta="message.chatbi" />
              <span v-if="sending && index === messages.length - 1 && !message.content" class="typing">
                {{ mode === 'kb' ? '正在检索并生成…' : mode === 'chatbi' ? '正在路由并取数…' : '正在思考…' }}
              </span>
            </div>
          </div>
        </div>

        <form class="chat-input" @submit.prevent="send">
          <input
            v-model="input"
            :placeholder="问句占位"
          />
          <button v-if="sending" type="button" class="secondary-button" @click="stopGeneration">停止</button>
          <button class="primary-button" :disabled="sending">{{ sending ? '生成中' : '发送' }}</button>
        </form>
        <p v-if="error" class="notice warning">{{ error }}</p>
      </div>
    </div>
  </section>
</template>

<style scoped>
.mode-switch {
  display: flex;
  gap: 6px;
  margin-top: 10px;
}
.mode-chip {
  border: 1px solid rgba(0, 0, 0, 0.12);
  background: transparent;
  border-radius: 999px;
  padding: 5px 14px;
  font-size: 13px;
  cursor: pointer;
}
.mode-chip.active {
  background: rgba(99, 102, 241, 0.15);
  border-color: rgba(99, 102, 241, 0.5);
  font-weight: 600;
}
.kb-status {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-top: 10px;
  padding: 10px 12px;
  border-radius: 8px;
  font-size: 13px;
  line-height: 1.6;
}
.kb-status.ok {
  background: rgba(16, 185, 129, 0.1);
  border: 1px solid rgba(16, 185, 129, 0.25);
  color: #065f46;
}
.kb-status.down {
  background: rgba(251, 191, 36, 0.12);
  border: 1px solid rgba(217, 119, 6, 0.3);
  color: #92400e;
}
.kb-status-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  flex-wrap: wrap;
}
.kb-status code,
.kb-cmd {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  background: rgba(0, 0, 0, 0.07);
  padding: 1px 6px;
  border-radius: 4px;
  user-select: all;
}
.kb-cmd {
  display: inline-block;
  padding: 6px 10px;
}
.kb-status-reason,
.kb-status-help {
  margin: 0;
  opacity: 0.9;
}
.kb-mismatch {
  margin-top: 4px;
  padding-top: 8px;
  border-top: 1px dashed rgba(146, 64, 14, 0.35);
}
.kb-hint {
  font-size: 12px;
  opacity: 0.7;
}
.cb-partial {
  margin: 0;
  padding-top: 6px;
  border-top: 1px dashed rgba(146, 64, 14, 0.35);
}
.cb-partial code,
.cb-role {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}
.cb-partial code {
  background: rgba(0, 0, 0, 0.07);
  padding: 1px 6px;
  border-radius: 4px;
  margin: 0 2px;
}
.cb-role-row {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  padding-top: 6px;
  border-top: 1px dashed rgba(6, 95, 70, 0.25);
}
.cb-role {
  border: 1px solid rgba(0, 0, 0, 0.15);
  border-radius: 6px;
  padding: 3px 8px;
  background: rgba(255, 255, 255, 0.6);
  font-size: 13px;
}
.cb-role-hint {
  font-size: 12px;
  opacity: 0.75;
}
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
