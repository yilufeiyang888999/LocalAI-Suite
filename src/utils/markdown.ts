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

/** System prompt injected per role preset (§3.6.3). */
export function systemPromptFor(role: string): string {
  switch (role) {
    case 'writing':
      return '你是一名中文写作助手，擅长撰写文案、报告和邮件，语言流畅、得体、条理清晰。';
    case 'coding':
      return '你是一名编程助手，擅长解释、生成和调试代码。回答时给出清晰、可运行的代码示例，并用中文说明。';
    default:
      return '你是一个友好的AI助手，用通俗易懂的中文回答用户问题。';
  }
}
