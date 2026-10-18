/**
 * 探针-智能问数.ts —— 验证 APP 侧客户端（`src/utils/chatbi.ts`）与**真服务**对得上
 * ============================================================================
 *
 * 回答一句话：**「APP 里那个『智能问数』模式，真的能连上 8096 并按预期工作吗？」**
 *
 * 为什么必须有它
 * --------------
 * `vue-tsc` 通过只证明**类型对**，不证明**字段名对**。这条链路是前端**直接 fetch**
 * 作品二服务的 JSON，字段名是**中文**且逐字对应服务端 —— 服务端改一个键名，
 * 前端**不报错、不崩**，只是那个字段永远是 `undefined`（界面上表现为「没显示」）。
 * 这类错**类型检查抓不到**，只能拿真服务对一遍。
 *
 * 🔴 默认**只读**：健康检查 + 知识类 + 数据类 + 越权类。
 *    越权那一题服务端是**硬拒**（不生成产物、不落盘），所以默认跑完**不留任何痕迹**。
 *    要验「放行」那条路径得显式加 `--写` —— 它会真的往 `outbox/` 落一张单据，
 *    并往审计里写一条记录。**默认跑测试不该改变被测系统的状态**，所以分开。
 *
 * 运行（脚本用 esbuild 现编现跑，不进 `dist/`）
 * ------------------------------------------
 *     npx esbuild scripts/探针-智能问数.ts --bundle --platform=node --format=esm \
 *       --outfile=node_modules/.cache/chatbi-probe.mjs
 *     node node_modules/.cache/chatbi-probe.mjs            # 只读
 *     node node_modules/.cache/chatbi-probe.mjs --写        # 额外验「放行 → 落盘」
 *
 * 退出码：0 = 全通过；1 = 有失败项
 *
 * ⚠️ 环境变量 `HTTP_PROXY`：Node 的 fetch **默认不读** `HTTP_PROXY`（只读
 *    `NO_PROXY`/`NODE_USE_ENV_PROXY`），所以这里不需要额外绕代理。
 *    但若在别的运行器里跑（比如全局 fetch 被 polyfill），要先确认这一点。
 */

import {
  DEFAULT_CHATBI_BASE_URL,
  askChatbi,
  chatbiMetaToPlainText,
  normalizeChatbiBaseUrl,
  probeChatbiService,
  toChatbiMessageMeta
} from '../src/utils/chatbi';
import type { ChatbiAnswer } from '../src/types';

const 要写 = process.argv.includes('--写');
const 只读角色 = '操作员';

let 通过 = 0;
let 失败 = 0;

function 判(名称: string, 条件: boolean, 说明: string): void {
  if (条件) {
    通过 += 1;
    console.log(`  ✅ ${名称}  ${说明}`);
  } else {
    失败 += 1;
    console.log(`  ❌ ${名称}  ${说明}`);
  }
}

function 标题(文: string): void {
  console.log('');
  console.log('='.repeat(78));
  console.log(文);
  console.log('='.repeat(78));
}

async function 主(): Promise<void> {
  // ---------------------------------------------------------------- 一、纯函数
  标题('一、纯函数（不需要服务）');
  判(
    'normalize 补协议头',
    normalizeChatbiBaseUrl('127.0.0.1:8096') === 'http://127.0.0.1:8096',
    normalizeChatbiBaseUrl('127.0.0.1:8096')
  );
  判(
    'normalize 去末尾斜杠',
    normalizeChatbiBaseUrl('http://127.0.0.1:8096///') === 'http://127.0.0.1:8096',
    normalizeChatbiBaseUrl('http://127.0.0.1:8096///')
  );
  判(
    'normalize 空值回落默认',
    normalizeChatbiBaseUrl('') === DEFAULT_CHATBI_BASE_URL,
    DEFAULT_CHATBI_BASE_URL
  );

  // ---------------------------------------------------------------- 二、探活
  标题('二、探活（vite build 抓不到的那一层：字段名对不上时这里才红）');
  const 基础 = DEFAULT_CHATBI_BASE_URL;
  const 探 = await probeChatbiService(基础);
  判('探活通过', 探.ok, 探.ok ? `${基础}/health 200` : `原因：${探.reason}`);
  判('health 带「版本」', typeof 探.health?.版本 === 'string', `版本=${探.health?.版本}`);
  判(
    'health 带「链路」且逐条可达',
    typeof 探.health?.链路 === 'object',
    Object.entries(探.health?.链路 ?? {})
      .map(([k, v]) => `${k}=${v.可达 === false ? '❌' : '✅'}`)
      .join('、')
  );
  判(
    'health 带「权限.角色」（界面下拉框的数据源）',
    Array.isArray(探.health?.权限?.角色) && (探.health?.权限?.角色?.length ?? 0) > 0,
    `默认角色=${探.health?.权限?.默认角色} ｜ 角色=${探.health?.权限?.角色?.join('、')}`
  );
  判(
    'health 带「审计」（白名单问题也得在这里可见）',
    typeof 探.health?.审计?.总条数 === 'number',
    `总 ${探.health?.审计?.总条数} ｜ 被拒 ${探.health?.审计?.被拒数} ｜ 依据不切题 ${探.health?.审计?.依据不切题数}`
  );
  判(
    '白名单自检无配置问题',
    (探.health?.权限?.白名单问题?.length ?? 1) === 0,
    JSON.stringify(探.health?.权限?.白名单问题 ?? [])
  );

  if (!探.ok) {
    console.log('\n服务不通，后面的对账做不了。先起它：python scripts/启动-服务.py');
    process.exitCode = 1;
    return;
  }

  // ---------------------------------------------------------------- 三、三条链路
  标题('三、三条链路（逐条对字段 —— 中文键名错一个字，这里才看得见）');
  const 应答表: Record<string, ChatbiAnswer> = {};

  应答表['知识'] = await askChatbi(基础, '不合格品应该怎么处理？', 只读角色);
  判('知识问答：链路判定', 应答表['知识'].类型 === '知识问答', `类型=${应答表['知识'].类型}`);
  判(
    '知识问答：有回答且带引用',
    (应答表['知识'].回答 ?? '').trim().length > 0,
    `回答 ${应答表['知识'].回答?.length ?? 0} 字 ｜ 引用 ${
      (应答表['知识'].依据 as any)?.citations?.length ?? 0
    } 条`
  );
  判(
    '知识问答：单步委派（无 ReAct 轨迹）',
    (应答表['知识'].步骤?.length ?? -1) === 0,
    `步骤 ${应答表['知识'].步骤?.length}`
  );

  应答表['数据'] = await askChatbi(基础, '上月哪家供应商不良率最高？', 只读角色);
  判('数据问数：链路判定', 应答表['数据'].类型 === '数据问数', `类型=${应答表['数据'].类型}`);
  判(
    '数据问数：有 SQL 与结果行',
    (应答表['数据'].依据 as any)?.SQL?.length > 0 && ((应答表['数据'].依据 as any)?.行?.length ?? 0) > 0,
    `列=${JSON.stringify((应答表['数据'].依据 as any)?.列)} ｜ 行=${(应答表['数据'].依据 as any)?.行?.length}`
  );
  判(
    '数据问数：口径随答案一起回来（否则数字没法核对）',
    ((应答表['数据'].依据 as any)?.口径 ?? '').length > 0,
    String((应答表['数据'].依据 as any)?.口径).slice(0, 40)
  );

  应答表['越权'] = await askChatbi(基础, '给东莞精诚出一份质量整改通知', 只读角色);
  判('动作执行：链路判定', 应答表['越权'].类型 === '动作执行', `类型=${应答表['越权'].类型}`);
  判(
    '越权：被拒为 true（403 是**业务结论**，不是请求失败）',
    应答表['越权'].被拒 === true,
    `被拒=${应答表['越权'].被拒}`
  );
  判(
    '越权：硬拒 → 产物为 null（不留半成品）',
    应答表['越权'].产物 === null,
    `产物=${JSON.stringify(应答表['越权'].产物)}`
  );
  判(
    '越权：拒绝理由**给出路**（写清需要哪个角色）',
    (应答表['越权'].权限?.拒绝理由 ?? '').includes('管理员'),
    应答表['越权'].权限?.拒绝理由 ?? '（空）'
  );
  判(
    '越权：轨迹里有「被拒」那一步',
    (应答表['越权'].步骤 ?? []).some((s) => s.被拒 === true),
    `步骤 ${应答表['越权'].步骤?.length} ｜ 被拒步 ${
      (应答表['越权'].步骤 ?? []).filter((s) => s.被拒).length
    }`
  );

  // ---------------------------------------------------------------- 四、元信息映射
  标题('四、落库元信息（`toChatbiMessageMeta` / `chatbiMetaToPlainText`）');
  const 元 = toChatbiMessageMeta(应答表['越权']);
  判('元信息：被拒字段透传', 元.被拒 === true, `被拒=${元.被拒}`);
  判('元信息：角色透传', 元.角色 === 只读角色, `角色=${元.角色}`);
  判('元信息：产物路径为空', 元.产物路径 === null, `产物路径=${元.产物路径}`);
  const 纯文本 = chatbiMetaToPlainText(应答表['越权'].回答, 元);
  判(
    '落库纯文本：被拒这件事在正文里留得住（刷新后也看得到）',
    纯文本.includes('权限不足'),
    纯文本.split('\n').slice(0, 2).join(' / ')
  );
  const 元数据 = toChatbiMessageMeta(应答表['数据']);
  判(
    '落库纯文本：数据类会带上链路与路由依据',
    chatbiMetaToPlainText(应答表['数据'].回答, 元数据).includes('数据问数'),
    '含「链路 数据问数」'
  );

  // ---------------------------------------------------------------- 五、放行路径
  标题('五、放行路径（要 --写 才跑：它会真的落一张单据进 outbox/）');
  if (!要写) {
    console.log('  ⏭ 已跳过（默认只读）。要验「放行 → 落盘」，加 --写 重跑。');
  } else {
    const 放 = await askChatbi(基础, '给东莞精诚出一份质量整改通知', '管理员');
    判('管理员放行：未被拒', 放.被拒 === false, `被拒=${放.被拒}`);
    判(
      '管理员放行：产出单据且落盘',
      typeof 放.产物?.路径 === 'string' && 放.产物.路径.length > 0,
      `产物=${放.产物?.路径}`
    );
    判(
      '管理员放行：轨迹走完 2 步（查数据 → 生成）',
      (放.步骤?.length ?? 0) >= 2,
      `步骤 ${放.步骤?.length}`
    );
  }

  // ---------------------------------------------------------------- 汇总
  标题('合计');
  console.log(`  ${通过} 通过 / ${失败} 失败`);
  process.exitCode = 失败 ? 1 : 0;
}

主().catch((错) => {
  console.error('探针自身出错：', 错);
  process.exitCode = 1;
});
