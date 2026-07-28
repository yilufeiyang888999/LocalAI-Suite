// Minimal, dependency-free Markdown rendering for chat replies. Everything is
// HTML-escaped first, so the produced HTML is safe to bind with v-html even
// though the text comes from a local model. Only a small, well-known subset is
// supported (headings, bold, inline code, lists, code blocks).

export interface MessageBlock {
  type: 'code' | 'text';
  /** Raw code text, present for code blocks (used by the copy button). */
  code?: string;
  /** Language hint after the opening fence, if any. */
  lang?: string;
  /** Pre-rendered, escaped HTML, present for text blocks. */
  html?: string;
}

function escapeHtml(input: string): string {
  return input
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// Inline formatting applied to already-escaped text: inline code first so its
// contents aren't further transformed, then bold.
function renderInline(escaped: string): string {
  return escaped
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
}

// Render a non-code chunk: headings, unordered/ordered lists, and paragraphs.
function renderTextBlock(markdown: string): string {
  const lines = markdown.replace(/\r\n/g, '\n').split('\n');
  const out: string[] = [];
  let listType: 'ul' | 'ol' | null = null;

  const closeList = () => {
    if (listType) {
      out.push(`</${listType}>`);
      listType = null;
    }
  };

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();
    if (!line.trim()) {
      closeList();
      continue;
    }
    const heading = /^(#{1,3})\s+(.*)$/.exec(line);
    if (heading) {
      closeList();
      const level = heading[1].length;
      out.push(`<h${level}>${renderInline(escapeHtml(heading[2]))}</h${level}>`);
      continue;
    }
    const unordered = /^[-*]\s+(.*)$/.exec(line);
    if (unordered) {
      if (listType !== 'ul') {
        closeList();
        out.push('<ul>');
        listType = 'ul';
      }
      out.push(`<li>${renderInline(escapeHtml(unordered[1]))}</li>`);
      continue;
    }
    const ordered = /^\d+\.\s+(.*)$/.exec(line);
    if (ordered) {
      if (listType !== 'ol') {
        closeList();
        out.push('<ol>');
        listType = 'ol';
      }
      out.push(`<li>${renderInline(escapeHtml(ordered[1]))}</li>`);
      continue;
    }
    closeList();
    out.push(`<p>${renderInline(escapeHtml(line))}</p>`);
  }
  closeList();
  return out.join('');
}

// Split a message on ``` fenced code blocks and render each part.
export function parseMessage(content: string): MessageBlock[] {
  const blocks: MessageBlock[] = [];
  const fence = /```([^\n]*)\n([\s\S]*?)```/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = fence.exec(content)) !== null) {
    if (match.index > lastIndex) {
      const text = content.slice(lastIndex, match.index).trim();
      if (text) blocks.push({ type: 'text', html: renderTextBlock(text) });
    }
    blocks.push({ type: 'code', lang: match[1].trim(), code: match[2].replace(/\n$/, '') });
    lastIndex = fence.lastIndex;
  }

  const tail = content.slice(lastIndex).trim();
  if (tail) blocks.push({ type: 'text', html: renderTextBlock(tail) });
  if (blocks.length === 0) blocks.push({ type: 'text', html: renderTextBlock(content) });
  return blocks;
}

/**
 * Turn a model id into a human-friendly display name, e.g.
 * `ornith-1.0-9b` → `Ornith-1.0-9B`. Used only as the placeholder hint for the
 * "AI display name" setting, so the user can see what the name would look like.
 * The identity clause itself is opt-in and never auto-derives from this.
 */
export function displayNameFromModelId(modelId: string | null | undefined): string {
  if (!modelId) return 'AI助手';
  const cleaned = modelId.trim();
  if (!cleaned) return 'AI助手';
  return cleaned
    .split('-')
    .map((segment) => {
      if (!segment) return segment;
      // Pure size/quant tokens (9b, 8x7b, q4, q8, k, m …) read best fully upper-cased.
      if (/^[0-9]*[a-z]$/i.test(segment) || /^q[0-9]/i.test(segment)) {
        return segment.toUpperCase();
      }
      // Otherwise just capitalise a leading letter (keeps version dots intact).
      return /^[a-z]/.test(segment) ? segment[0].toUpperCase() + segment.slice(1) : segment;
    })
    .join('-');
}

/**
 * System prompt injected per role preset (§3.6.3). If the user has set a custom
 * name for the AI, an identity clause is prepended so it introduces itself by
 * that name. This is opt-in: with no custom name, `identity` is falsy and the
 * clause is skipped entirely — the model answers "who are you" however its
 * training decides (which for a Qwen-derived model is legitimately "Qwen").
 * The wording is neutral and does not force the model to deny its lineage.
 */
export function systemPromptFor(role: string, identity?: string | null): string {
  let base: string;
  switch (role) {
    case 'writing':
      base = '你是一名中文写作助手，擅长撰写文案、报告和邮件，语言流畅、得体、条理清晰。';
      break;
    case 'coding':
      base = '你是一名编程助手，擅长解释、生成和调试代码。回答时给出清晰、可运行的代码示例，并用中文说明。';
      break;
    default:
      base = '你是一个友好的AI助手，用通俗易懂的中文回答用户问题。';
      break;
  }

  const name = identity?.trim();
  if (!name) return base;
  return `你的名字是${name}，一个在本地运行的AI助手。当被问到你叫什么、你是谁时，可以称自己为${name}。\n${base}`;
}
