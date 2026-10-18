import type { KbAnswer, KbHealth, KbMessageMeta } from '../types';

/**
 * 知识库问答服务的默认地址。
 *
 * 🔴 三处必须一致，改一处就得改另外两处：
 *   1. 这里
 *   2. `src-tauri/src/settings.rs::default_kb_base_url()`
 *   3. `RAG-Embedding/scripts/知识库服务.py` 的 `监听端口`
 *
 * 前端保留一份是兜底：设置项若被清空，不至于拼出一个空 URL 去 fetch
 * （那样只会得到一个语焉不详的网络错误）。
 */
export const DEFAULT_KB_BASE_URL = 'http://127.0.0.1:8092';

/** 去掉末尾斜杠、补协议头，容忍用户填 `127.0.0.1:8092` 这种简写。 */
export function normalizeKbBaseUrl(raw: string | null | undefined): string {
  const 文本 = (raw ?? '').trim();
  if (!文本) return DEFAULT_KB_BASE_URL;
  const 补协议 = /^https?:\/\//i.test(文本) ? 文本 : `http://${文本}`;
  return 补协议.replace(/\/+$/, '');
}

export interface KbProbeResult {
  ok: boolean;
  health?: KbHealth;
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
export async function probeKbService(baseUrl: string, timeoutMs = 2500): Promise<KbProbeResult> {
  const 控制 = new AbortController();
  const 定时 = setTimeout(() => 控制.abort(), timeoutMs);
  try {
    const 响应 = await fetch(`${baseUrl}/health`, { signal: 控制.signal });
    if (!响应.ok) return { ok: false, reason: `服务返回 HTTP ${响应.status}` };
    const health = (await 响应.json()) as KbHealth;
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
 * 提问。
 *
 * **非流式**：检索（BM25 + 重排）本来就必须先跑完才有料给生成端，
 * 一次把 `answer + citations` 拿全反而更好渲染引用卡片。
 * 代价是这几秒里界面只能转圈 —— 所以调用方要给「正在检索」的阶段提示。
 */
export async function askKb(baseUrl: string, question: string, signal?: AbortSignal): Promise<KbAnswer> {
  const 响应 = await fetch(`${baseUrl}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal,
    body: JSON.stringify({ question, top_k: 3 })
  });
  if (!响应.ok) {
    let 细节 = '';
    try {
      细节 = JSON.stringify(await 响应.json());
    } catch {
      /* 响应体不是 JSON（例如反向代理回了 HTML 错误页）—— 忽略，只用状态码 */
    }
    throw new Error(`知识库服务返回 HTTP ${响应.status}${细节 ? `：${细节}` : ''}`);
  }
  return (await 响应.json()) as KbAnswer;
}

/** 抽成落库用的元信息（内存态 `kb` 字段）。 */
export function toKbMessageMeta(答案: KbAnswer): KbMessageMeta {
  return {
    citations: 答案.citations ?? [],
    refused: 答案.refused,
    refusalType: 答案.refusal_type ?? '',
    refusalReason: 答案.refusal_reason ?? '',
    offLibraryRefs: 答案.off_library_refs ?? [],
    topScore: 答案.top_score,
    timings: 答案.timings
  };
}

/** 拒答类型 → 中文说明（界面展示用）。 */
export function refusalLabel(类型: string): string {
  switch (类型) {
    case 'score_threshold':
      return '资料不足（库里没有相关内容）';
    case 'broken_citation_chain':
      return '资料不足（线索指向库外文档）';
    case 'model_refused':
      return '资料不足（依据不充分，AI 未作答）';
    default:
      return '资料不足';
  }
}

/**
 * 把一条知识库回答摊平成**可落库的纯文本**。
 *
 * 🔴 为什么要有这一步：`chat_messages` 表只有 `(id, session_id, role, content, created_at)`，
 * 没有列能放引用。落库时把卡片摊平到正文里，刷新后才不至于只剩一句答案、
 * 看不出依据是哪几份文件。这是**有意选择的降级**，不是遗漏 —— 见 `types.ts` 里 `kb` 字段的注释。
 */
export function kbMetaToPlainText(正文: string, meta: KbMessageMeta): string {
  const 行: string[] = [正文.trim() || '(空)'];

  if (meta.refused) {
    行.push('', `> ${refusalLabel(meta.refusalType)}` + (meta.refusalReason ? `：${meta.refusalReason}` : ''));
  }
  if (meta.offLibraryRefs.length) {
    行.push('', `线索指向未入库的文档：${meta.offLibraryRefs.join('、')}`);
  }
  const 引用 = meta.citations ?? [];
  if (引用.length) {
    行.push('', `依据（${引用.length} 条）：`);
    for (const 条 of 引用) {
      行.push(`${条.rank}. 【${条.doc_no}】${条.section}`);
    }
  }
  return 行.join('\n');
}
