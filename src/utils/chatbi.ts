import type { ChatbiAnswer, ChatbiHealth, ChatbiMessageMeta } from '../types';

/**
 * 智能问数 / 动作执行服务的默认地址（作品二 · ChatBI 的**编排服务**）。
 *
 * 🔴 三处必须一致，改一处就得改另外两处：
 *   1. 这里
 *   2. `src-tauri/src/settings.rs::default_chatbi_base_url()`
 *   3. `ChatBI/scripts/编排服务.py` 的 `默认端口`
 *
 * 前端保留一份是兜底：设置项若被清空，不至于拼出一个空 URL 去 fetch
 * （那样只会得到一个语焉不详的网络错误）。
 */
export const DEFAULT_CHATBI_BASE_URL = 'http://127.0.0.1:8096';

/** 去掉末尾斜杠、补协议头，容忍用户填 `127.0.0.1:8096` 这种简写。 */
export function normalizeChatbiBaseUrl(raw: string | null | undefined): string {
  const 文本 = (raw ?? '').trim();
  if (!文本) return DEFAULT_CHATBI_BASE_URL;
  const 补协议 = /^https?:\/\//i.test(文本) ? 文本 : `http://${文本}`;
  return 补协议.replace(/\/+$/, '');
}

/** 三条链路的中文名 → 界面标签（服务返回的 `类型` 就是这几个值）。 */
export function 链路标签(类型: string): string {
  switch (类型) {
    case '知识问答':
      return '📚 知识问答';
    case '数据问数':
      return '📊 数据问数';
    case '动作执行':
      return '🖊️ 动作执行';
    default:
      return 类型 || '—';
  }
}

/** `路由.依据` → 说明。**兜底路由必须让用户看见**，否则「模型猜的」被当成确定结论。 */
export function 路由依据标签(依据: string): string {
  switch (依据) {
    case '确定性':
      return '确定性规则';
    case '模型':
      return '模型兜底判断';
    case '默认':
      return '默认链路';
    case '强制':
      return '手动指定';
    default:
      return 依据 || '—';
  }
}

export interface ChatbiProbeResult {
  ok: boolean;
  health?: ChatbiHealth;
  /** 面向用户的失败原因。**不允许只说「失败」** —— 用户需要知道下一步做什么。 */
  reason?: string;
}

/**
 * 探活。
 *
 * 超时**必须短**：这是给界面判断「要不要显示启动引导」用的，不是业务请求。
 * 一个未监听的端口在 Windows 上会立刻 RST（拒绝连接），但防火墙丢包或服务
 * 假死时会一直挂着 —— 不设超时就会让按钮转很久才失败。
 */
export async function probeChatbiService(baseUrl: string, timeoutMs = 3000): Promise<ChatbiProbeResult> {
  const 控制 = new AbortController();
  const 定时 = setTimeout(() => 控制.abort(), timeoutMs);
  try {
    const 响应 = await fetch(`${baseUrl}/health`, { signal: 控制.signal });
    if (!响应.ok) return { ok: false, reason: `服务返回 HTTP ${响应.status}` };
    const health = (await 响应.json()) as ChatbiHealth;
    if (health.status !== 'ok') return { ok: false, reason: `服务状态异常：${health.status}`, health };
    return { ok: true, health };
  } catch (错) {
    const 名 = (错 as Error)?.name;
    return {
      ok: false,
      reason: 名 === 'AbortError' ? `${timeoutMs} 毫秒内没有响应` : String(错)
    };
  } finally {
    clearTimeout(定时);
  }
}

/**
 * 提问（一个入口，三条链路 —— 不用先选问的是哪类问题）。
 *
 * 🔴 **HTTP 403 不是「请求失败」，是「越权」这一条业务结论。**
 *    它带着完整的应答体回来（路由 / 轨迹 / 拒绝理由），必须**照常返回**给调用方渲染。
 *    把它当异常抛出去，界面上就只剩一句「失败」—— 用户永远看不到「需要哪个角色」，
 *    也就不知道该换谁来做。**「读 body 看有没有被拒」这种判据，漏写一处就静默当成功。**
 *
 * 非流式：三条链路的共同点是「先有依据才能答」（检索 / 查数 / 取数），
 * 本来也不存在「边算边出字」。代价是这几秒界面只能转圈。
 */
export async function askChatbi(
  baseUrl: string,
  question: string,
  角色: string,
  signal?: AbortSignal
): Promise<ChatbiAnswer> {
  const 响应 = await fetch(`${baseUrl}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal,
    body: JSON.stringify({ 问题: question, 角色 })
  });
  // 200 = 正常；403 = 越权（同样是**业务结论**，带完整 body）；其余才是真失败。
  if (响应.ok || 响应.status === 403) {
    return (await 响应.json()) as ChatbiAnswer;
  }
  let 细节 = '';
  try {
    细节 = JSON.stringify(await 响应.json());
  } catch {
    /* 响应体不是 JSON（例如反向代理回了 HTML 错误页）—— 忽略，只用状态码 */
  }
  throw new Error(`智能问数服务返回 HTTP ${响应.status}${细节 ? `：${细节}` : ''}`);
}

/** 抽成落库用的元信息（内存态 `chatbi` 字段）。 */
export function toChatbiMessageMeta(答案: ChatbiAnswer): ChatbiMessageMeta {
  return {
    类型: 答案.类型 ?? '',
    路由依据: 答案.路由?.依据 ?? '',
    路由理由: 答案.路由?.理由 ?? '',
    步骤: 答案.步骤 ?? [],
    被拒: 答案.被拒 ?? false,
    拒绝理由: 答案.权限?.拒绝理由 ?? null,
    角色: 答案.权限?.角色 ?? null,
    // 服务端字段名是 `依据不切题`；这里**逐字对应**，不做二次命名
    依据不切题: 答案.依据不切题 ?? false,
    产物路径: 答案.产物?.路径 ?? null,
    耗时秒: 答案.总耗时秒 ?? 0
  };
}

/**
 * 把一条应答摊平成**可落库的纯文本**。
 *
 * 🔴 为什么要有这一步：`chat_messages` 表只有 `(id, session_id, role, content, created_at)`，
 * 没有列能放轨迹与权限结论。落库时把它们摊平到正文里，刷新后才不至于只剩一句答案、
 * 看不出「这句话是怎么来的（哪条链路 / 是不是模型兜底的）」。
 * 这是**有意选择的降级**，不是遗漏 —— 与作品一的引用卡片同一个取舍，见 `types.ts`。
 */
export function chatbiMetaToPlainText(正文: string, meta: ChatbiMessageMeta): string {
  const 行: string[] = [正文.trim() || '(空)'];

  if (meta.被拒) {
    行.push('', `> 🚫 权限不足，未生成任何单据。${meta.拒绝理由 ?? ''}`);
    行.push(`> 当前角色：${meta.角色 ?? '—'}`);
  }
  const 头: string[] = [];
  if (meta.类型) 头.push(`链路 ${meta.类型}`);
  if (meta.路由依据) 头.push(`路由 ${路由依据标签(meta.路由依据)}`);
  if (meta.耗时秒) 头.push(`耗时 ${meta.耗时秒.toFixed(2)} s`);
  if (头.length) 行.push('', `（${头.join(' ｜ ')}）`);

  if (meta.依据不切题) {
    行.push('', '> ⚠️ 本单据的依据可能不切题，请核对后再使用。');
  }
  if (meta.步骤.length) {
    行.push('', `执行轨迹（${meta.步骤.length} 步）：`);
    for (const 步 of meta.步骤) {
      行.push(`${步.步}. ${步.动作}${步.被拒 ? '（被拒）' : ''}`);
    }
  }
  if (meta.产物路径) {
    行.push('', `产物：${meta.产物路径}`);
  }
  return 行.join('\n');
}
