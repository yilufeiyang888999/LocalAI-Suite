# -*- coding: utf-8 -*-
"""诊断-生成端可复现：把「输出翻转」隔离到**生成端**还是**服务端语义缓存**。

🔴 结论（2026-10-15 深夜，已定案；此前的诊断是错的）
--------------------------------------------------------------------------------
D16 起观察到「同一句问句、同一提示词，输出会翻转」，当时归因于「**换一次会话**」——
**方向错了**。真机制是 **ChatBI 服务端的语义缓存**（`ChatBI/logs/意图缓存.json`，见
`意图缓存.py` / `问数服务.py` 的 0.x 前置判定）：**基线轮模型写下的意图被后续各轮原样读走**，
与「换会话」「换请求序列」都无关。
物证：40 题里 **31 题** `模型用时=0`（非拒答/元问题）= 缓存命中，全跑只约 8 题真问过模型；
`意图缓存.json` 里 T25/T26/T30/T29 首写于 10-14 18:44~18:50（用时 10.57/17.41/3.51/13.46 秒，
是基线模型**真算**的），而后续四轮这几题一律 `模型用时=0.0` = 100% 命中。
→ 「四轮逐字相同」被误读成「微调没效果」。污染处置与三道闸门见 `对账-微调前后.py` 边界声明。

本脚本职责因此收窄为**只测生成端**（自建 HF 壳，`do_sample=False` 贪婪）：
  · ① 同一提示词连发 3 次            → 测「同输入在同序列内是否同输出」
  · ② 中间插别的提示词后再发同一提示词 → 测「前序请求是否影响后序输出」
  · ③ 显式带 `cache_prompt=false/true` → 测「前缀复用是否就是原因」
       ⚠️ **壳根本没读这个字段**（`推理-服务.py` 里搜不到 `cache_prompt`）→ ③ 的差异
       不能拿来怪前缀复用；它只说明「发什么字段都一个样」。
  · ④ **整个序列连跑两遍并逐题对拍** → 把「请求序列」本身做成可复现性的自证：
       先打**序列摘要**（同一摘要 = 同一序列，两份结果才可比），再逐题比两次的原始文本。

为什么要有 ④：`temperature=0` 只保证「**同一请求序列**内可复现」。两次跑的输入提示词相同，
但**它前面排了什么**不同 → 输出就可能不同。所以「序列」必须和「提示词」一样被记账。
同序列两跑仍不一致 = **生成端本身不可复现**，那是引擎问题，必须单独查（退出码 1）。
ChatBI 侧的缓存判定**不归本脚本管** —— 那是 `意图缓存.py` 的活，别在这里查。

🔴 **2026-10-15 傍晚追加（#14 实验的产物）：可比性判据是「开关集」，不是「快慢」**
`lab/实验-只切SDPA.py` 实测：同一模型、同一提示词、**只切 SDPA 开关集** ——
`全开` 同壳两遍 **8/8 一致**，而 `cudnn`-only（`LAB_SDPA=训练`）**1/8**（7 题两遍就变）。
也就是说：**「生成端可复现」这句话挂在某一套开关集上，不挂在机器上。**
→ 所以本脚本现在会**先回读 `/health` 的 `SDPA开关集` 并打印**；不是「全开」时**明确告警**
（在那种路上「两跑不一致」是**预期**，不是异常 —— 别当成引擎坏了）。

🔴 **2026-10-15 傍晚（同一轮验证）追加：超时只能有一份定义**
拿一台**已知健康**的壳（`LAB_SDPA=全开 标签=diag-verify 端口=8012`）跑本脚本做**正向对照**时，
它**在第 4 次生成上超时退出**（`TimeoutError`，退出码 1）：

    SDPA 开关集：全开          ← 回读**通过**（这次要验的就是这一行）
    壳日志：生成 111.07s / 106.03s / 103.78s（prompt 2867 tok → out 259 tok，**约 2.5 tok/s**）

根因**不是**引擎坏了，是**超时定义有两份**：本脚本的 `发()` 把超时**硬编码成 180**，
而生产口径在 `意图解析._取超时()`（环境变量 `CHATBI_GEN_TIMEOUT`）。
设环境变量对**本脚本无效** → 一条 700 token 的生成在慢态下要 **283 s > 180 s** → 客户端先撤 →
壳侧 `CLOSE_WAIT`、slot 卡住（「掐客户端 = 亲手制造 slot 卡死」，**第 5 次复现**）。
→ 已改成默认取 `意图解析.超时秒`（**与生产同一份**），并把「客户端超时 / 慢态下 700 token 约需多久」
   打进抬头。**跑之前用 `CHATBI_GEN_TIMEOUT` 把它抬到最坏耗时之上。**

用法：
    python 诊断-生成端可复现.py                 # 默认端口 8000
    python 诊断-生成端可复现.py --端口 8010     # 自建壳的对照端口
退出码：0 = 同序列两跑逐题一致 ｜ 1 = 同序列两跑不一致（生成端不可复现，必须查）｜ 2 = 端口不通
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

本目录 = os.path.dirname(os.path.abspath(__file__))
项目目录 = os.path.dirname(本目录)
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)

import 语义层 as 语义                                        # noqa: E402
import 意图解析                                            # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:                                      # noqa: BLE001
        pass

代理绕过 = urllib.request.build_opener(urllib.request.ProxyHandler({}))
生成服务 = "http://127.0.0.1:8000/v1/chat/completions"      # main() 里按 --端口 覆盖

定义 = 语义.载入()


def 发(提示文本, 缓存=None, 温度=0, 超时=None):
    """发一次生成请求。

    🔴 **超时只能有一份定义**（2026-10-15 傍晚修的，踩过才改的）：
       本函数原来把超时**硬编码成 180**，而生产口径在 `意图解析._取超时()`
       （环境变量 `CHATBI_GEN_TIMEOUT`，默认 180）—— **两处定义**。
       后果实测：本机自建壳在慢态下长生成要 **105~283 s**，
       设了 `CHATBI_GEN_TIMEOUT` 对**本脚本无效**（它读的是那个常量），
       于是**只要模型写满就必然超时** → 客户端先撤 → 壳侧 `CLOSE_WAIT`、
       slot 卡住（「掐客户端 = 亲手制造 slot 卡死」，这已是第 5 次复现）。
       → 现在默认取 `意图解析.超时秒`，**与生产同一份**；要调就调那一个环境变量。
       （同族纪律：「凡『读配置/抽字段』的代码，先问『还有第二份吗』」。）
    """
    超时 = 意图解析.超时秒 if 超时 is None else 超时
    体 = {
        "messages": [{"role": "user", "content": 提示文本}],
        "max_tokens": 700,
        "temperature": 温度,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if 缓存 is not None:
        体["cache_prompt"] = 缓存
    req = urllib.request.Request(生成服务, data=json.dumps(体).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    应 = json.loads(代理绕过.open(req, timeout=超时).read().decode("utf-8"))
    return (应["choices"][0]["message"].get("content") or "").strip()


def 分类(文本):
    起 = 文本.strip().lstrip("`").replace("json\n", "", 1)
    起 = 起.strip().strip("`").strip()
    try:
        对 = json.loads(起[起.find("{"):起.rfind("}") + 1])
    except Exception:                                      # noqa: BLE001
        return "非JSON", 起[:70].replace("\n", " ")
    if 对.get("澄清"):
        return "澄清", str(对["澄清"])[:60].replace("\n", " ")
    if 对.get("指标"):
        return "指标", json.dumps(对, ensure_ascii=False)[:70]
    if 对.get("基础查询"):
        return "基础查询", json.dumps(对, ensure_ascii=False)[:70]
    return "其他", json.dumps(对, ensure_ascii=False)[:70]


def 序列摘要(步):
    """把「请求序列」压成一个短摘要。

    🔴 用途：两份结果要能**自证**是不是同一序列跑的 —— 光看「提示词相同」不够，
    因为温度=0 只保证「同序列内」可复现。摘要只哈希「名字 + 提示词内容 + cache_prompt」，
    不哈希输出（那是被测对象）。
    """
    原 = "\x00".join(
        "%s|%s|%s" % (名, hashlib.sha1((提示 or "").encode("utf-8")).hexdigest()[:8], 缓存)
        for 名, 提示, 缓存, 温 in 步)
    return hashlib.sha1(原.encode("utf-8")).hexdigest()[:12]


def 造步(P, R):
    return [
        ("① 同一提示词 · 第 1 次", P, None, 0),
        ("① 同一提示词 · 第 2 次", P, None, 0),
        ("① 同一提示词 · 第 3 次", P, None, 0),
        ("② 先发别的提示词", R, None, 0),
        ("② 紧接同提示词 · 第 4 次", P, None, 0),
        ("② 再发别的提示词", R, None, 0),
        ("② 紧接同提示词 · 第 5 次", P, None, 0),
        ("③ cache_prompt=false · 第 6 次", P, False, 0),
        ("③ cache_prompt=false · 第 7 次", P, False, 0),
        ("③ cache_prompt=true  · 第 8 次", P, True, 0),
    ]


def 跑一遍(步):
    """原样跑完整序列，返回 [(名, 类, 摘, 耗时, 原始文本)]。"""
    出 = []
    for 名, 提示, 缓存, 温 in 步:
        起 = time.time()
        文 = 发(提示, 缓存=缓存, 温度=温)
        类, 摘 = 分类(文)
        出.append((名, 类, 摘, time.time() - 起, 文))
    return 出


def main():
    参 = argparse.ArgumentParser(description="生成端可复现性诊断（含请求序列自证）")
    参.add_argument("--端口", type=int, default=8000)
    参.add_argument("--略过对拍", action="store_true",
                    help="只跑一遍看现象（省一半时间）；⚠️ 跳过 ④ 就不能报「可复现」")
    也 = 参.parse_args()

    global 生成服务
    生成服务 = "http://127.0.0.1:%d/v1/chat/completions" % 也.端口

    try:
        探 = urllib.request.Request("http://127.0.0.1:%d/health" % 也.端口)
        健康 = json.loads(代理绕过.open(探, timeout=8).read().decode("utf-8"))
    except Exception as 错:                                 # noqa: BLE001
        print("🔴 端口 %d 上没有生成端（/health 不通）：%s" % (也.端口, 错))
        print("   先起壳：python 推理-服务.py --端口 %d" % 也.端口)
        return 2

    P = 意图解析.构造提示(定义, "这个月怎么样？")
    R = 意图解析.构造提示(定义, "一共有多少家供应商？")
    步 = 造步(P, R)

    开 = 健康.get("SDPA开关集")
    print("生成端：%s" % 生成服务)
    # 🔴 开关集是**可比性判据**（2026-10-15 傍晚实测，#14）：`全开` 同壳两遍 8/8 一致；
    #    `训练`(cudnn-only) 只有 1/8 —— 那条路**本身不可复现**。所以「速度过门槛」不够，
    #    得先看这一项。自建壳会报；其它生成端（llama.cpp 8000）不报，显示「未自报」。
    print("SDPA 开关集：%s%s"
          % (开 or "（该生成端未自报）",
             "" if (开 in (None, "全开")) else
             "   🔴 **不是「全开」** —— 实测这条路同壳两遍只有 1/8 一致，"
             "「两跑不一致」在这里是**预期**，先换回默认再谈可复现性。"))
    print("提示词长度：%d 字符 ｜ 序列长度：%d 步" % (len(P), len(步)))
    # 🔴 超时必须**打印出来**：本机自建壳在慢态下，一条 700 token 的生成要 105~283 s，
    #    超时设小了就会「客户端先撤 → 壳侧 CLOSE_WAIT → slot 卡死」，
    #    而那看上去像「引擎坏了」。先把这一行摆出来，免得又去查引擎。
    print("客户端超时：%d s ｜ max_tokens=700 ｜ 慢态下 700 token 约需 %d s"
          % (意图解析.超时秒, int(700 / 2.5)))
    print("请求序列摘要：%s   ← 两份结果要**摘要相同**才可比（温度=0 只保证同序列内可复现）"
          % 序列摘要(步))
    print("=" * 100)

    甲 = 跑一遍(步)
    for 名, 类, 摘, 耗, _ in 甲:
        print("%-30s → %-6s ｜ %5.2fs ｜ %s" % (名, 类, 耗, 摘))

    if 也.略过对拍:
        print()
        print("⚠️ 已略过 ④ 两跑对拍 —— **本轮不能报「可复现」**（跳过必须与通过可区分）。")
        return 3

    print("-" * 100)
    print("④ 同序列两跑对拍（这是「可复现性」的判据本身）")
    乙 = 跑一遍(步)
    不一致 = []
    for (名, 类, 摘, 耗, 文), (_, 类2, _, _, 文2) in zip(甲, 乙):
        好 = 文 == 文2
        if not 好:
            不一致.append(名)
        print("%-30s %s ｜ 一跑 %-6s 二跑 %-6s%s"
              % (名, "✅" if 好 else "🔴", 类, 类2, "" if 好 else "  ← 两次原始文本不同"))
    print()

    单指标类 = {"澄清", "指标", "基础查询"}
    同词一致 = len({类 for 名, 类, _, _, _ in 甲}) == 1 and len({类 for 名, 类, _, _, _ in 乙}) == 1
    if 不一致:
        print("🔴 同序列两跑有 %d/%d 步不一致（%s）—— **生成端本身不可复现**。"
              % (len(不一致), len(步), "、".join(不一致)))
        if 开 and 开 != "全开":
            print("   🔴 **先看开关集：`SDPA开关集=%s`** —— 实测这条路同壳两遍只有 1/8 一致，"
                  "**它本身就是不可复现的**（`全开` 才有 8/8）。**换回 `全开` 重测**，别急着查引擎。"
                  % 开)
        print("   这不是 ChatBI 缓存的事（缓存不归本脚本管）：先查壳的解码开关、"
              "是否被别的实例占用、显存是否吃紧。")
        return 1

    if 同词一致:
        print("✅ 同序列两跑逐题一致 → 生成端在**同一请求序列内**可复现。")
    else:
        print("✅ 同序列两跑逐题一致 → 生成端可复现。")
        print("   注意：① vs ② 的类别若不同，说明**输出依赖请求序列**（前序请求会影响后序）——")
        print("   这正是「序列必须与提示词一起记账」的理由，也是 ④ 存在的意义。")
    print("   ③ 的 `cache_prompt=false/true` 若与①同类 → 壳不读该字段（前缀复用不是原因）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
