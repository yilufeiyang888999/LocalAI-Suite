# -*- coding: utf-8 -*-
"""ChatBI · 并发可复现性对照 —— 串行 vs 并发，逐题比输出

## 为什么另起一个（不靠 评测-盲测集.py）

评测脚本是**串行**发请求的：一题发出去、等回来、再发下一题。
在这种时序下 `--parallel 2` 只会用到 1 个 slot（LRU 选空闲槽，串行永远选 0 号），
于是「parallel=2 与 parallel=1 结果相同」是个**平凡结论** —— 根本没并发过。

`--parallel N` 的真实风险在**多槽同时推理**：llama.cpp 会把不同序列拼进同一个
batch 前向，batch 形状变了、归约顺序变了 ⇒ 浮点结果可能不同 ⇒ 贪心解码选中
不同 token。这个风险只有**并发发请求**才测得出来。

## 判据

对同一批题、同一提示词：
    · 串行跑一遍（基线）
    · 并发跑一遍（N 个线程同时发，确保真的同时在飞）
    · 逐题比**输出文本**（不是比分数 —— 分数相同但文本不同也算不可复现）

比分数会骗人：两轮都是 26.1%，可能刚好错在别的题上。

## 用法

    python scripts/对照-并发可复现.py --端口 18000 --题数 20
    python scripts/对照-并发可复现.py --端口 18000 --题数 20 --并发 2
"""

import argparse
import concurrent.futures
import json
import os
import sys
import urllib.request

脚本目录 = os.path.dirname(os.path.abspath(__file__))
项目目录 = os.path.dirname(脚本目录)
sys.path.insert(0, 脚本目录)

import 意图解析  # noqa: E402  —— 复用**同一份**提示词构造（不在本脚本抄第二份）
import 语义层 as 语义  # noqa: E402  —— 语义层定义同样只此一份

绕代理 = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def 说(文本=""):
    print(文本)
    sys.stdout.flush()


def 调(端口, 提示文本):
    """直接打生成端（不经问数服务）—— 被测对象就是引擎本身"""
    体 = {
        "messages": [{"role": "user", "content": 提示文本}],
        "max_tokens": 700,
        "temperature": 0,
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    请求 = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % 端口,
        data=json.dumps(体).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with 绕代理.open(请求, timeout=300) as 应:
        d = json.loads(应.read().decode("utf-8", "replace"))
    return (d["choices"][0]["message"].get("content") or "").strip()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    解 = argparse.ArgumentParser()
    解.add_argument("--端口", type=int, default=18000)
    解.add_argument("--题数", type=int, default=20)
    解.add_argument("--并发", type=int, default=0, help="0 = 取服务端 total_slots")
    解.add_argument("--题集", default="问数盲测集-40题.json")
    参数 = 解.parse_args()

    # 语义层定义：与问数服务/评测脚本同一入口（语义.载入），不在本脚本抄第二份
    定义 = 语义.载入()

    with open(os.path.join(项目目录, "eval", 参数.题集), encoding="utf-8") as 句柄:
        原 = json.load(句柄)
    题目 = (原.get("题目") if isinstance(原, dict) else 原)[:参数.题数]

    # 服务端报的槽数 —— 并发度默认跟随它，不自己猜
    props = json.loads(绕代理.open("http://127.0.0.1:%d/props" % 参数.端口, timeout=8)
                       .read().decode("utf-8", "replace"))
    槽 = props.get("total_slots")
    并发 = 参数.并发 or (槽 if isinstance(槽, int) and 槽 > 0 else 1)
    说("端口 %d ｜ total_slots=%s ｜ 本轮并发度 %d ｜ 题数 %d"
       % (参数.端口, 槽, 并发, len(题目)))
    if 并发 < 2:
        说("🔴 并发度 < 2 —— 这轮测不出并行风险。请用 --parallel 2 起的服务，或 --并发 2")
        return 3

    提示 = [(t["编号"], 意图解析.构造提示(定义, t["问题"])) for t in 题目]

    # ---- ① 串行基线 ----
    说("\n[1/3] 串行基线……")
    串行 = {}
    for 编号, 文 in 提示:
        串行[编号] = 调(参数.端口, 文)
    说("      完成 %d 题" % len(串行))

    # ---- ② 串行第二遍（🔴 最小对照：不先测它，就无法把「并发的锅」和
    #        「引擎本身就不确定」分开 —— §9.5 下结论前先做最小对照）----
    说("[2/3] 串行第二遍（最小对照）……")
    串行2 = {}
    for 编号, 文 in 提示:
        串行2[编号] = 调(参数.端口, 文)
    说("      完成 %d 题" % len(串行2))

    # ---- ③ 并发 ----
    说("[3/3] 并发 %d 路……" % 并发)
    并发果 = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=并发) as 池:
        任务 = {池.submit(调, 参数.端口, 文): 编号 for 编号, 文 in 提示}
        for 未 in concurrent.futures.as_completed(任务):
            编号 = 任务[未]
            并发果[编号] = 未.result()
    说("      完成 %d 题" % len(并发果))

    # ---- ④ 两组比对：串行vs串行（噪声下限）、串行vs并发（并发影响）----
    串行差异 = [k for k, v in 串行.items() if 串行2.get(k, "") != v]
    并发差异 = [k for k, v in 串行.items() if 并发果.get(k, "") != v]

    说("\n" + "=" * 66)
    说("对照一 · 串行第 1 遍 vs 串行第 2 遍（**噪声下限**，与并发无关）")
    if not 串行差异:
        说("  ✅ %d 题逐字一致 —— 该引擎在串行下是确定性的" % len(串行))
    else:
        说("  🔴 %d / %d 题不一致 —— **引擎本身就不确定**，于是下面「并发差异」不能全记在并发头上"
           % (len(串行差异), len(串行)))
        说("     不一致题号：%s" % ", ".join(串行差异))

    说("\n对照二 · 串行 vs 并发 %d 路（并发影响）" % 并发)
    if not 并发差异:
        说("  ✅ %d 题逐字一致 —— 并发 %d 路不改变输出" % (len(串行), 并发))
    else:
        说("  🔴 %d / %d 题不一致" % (len(并发差异), len(串行)))
        # 只有当串行两遍一致时，才可以说「是并发造成的」
        if not 串行差异:
            说("     且对照一已证明串行确定性 ⇒ **差异可归因于并发**")
        else:
            新增 = [k for k in 并发差异 if k not in 串行差异]
            说("     对照一已有 %d 题自发不一致；并发只**新增** %d 题（%s）"
               % (len(串行差异), len(新增), ", ".join(新增) or "无"))

        首 = 并发差异[0]
        说("\n  第一处差异（题号 %s）：" % 首)
        说("    串行：%s" % 串行[首][:260].replace("\n", " ⏎ "))
        说("    并发：%s" % 并发果[首][:260].replace("\n", " ⏎ "))

    # ---- ⑤ 最关键的一层：**文本不同 ≠ 业务不同** ----
    #    §9.5：判据要有分辨力。逐字比文本会把「修辞差异」判成失败，
    #    而真正影响用户的是抽出来的意图 JSON（进而 SQL、进而结果集）。
    def 意图键(文):
        try:
            j = 意图解析.抽JSON(文) if hasattr(意图解析, "抽JSON") else None
        except Exception:                                      # noqa: BLE001
            return ""
        return json.dumps(j, ensure_ascii=False, sort_keys=True) if j else ""

    说("\n对照三 · 抽出意图 JSON 层（**业务口径**，忽略修辞）")
    意图差异 = [k for k, v in 串行.items()
                if 意图键(并发果.get(k, "")) != 意图键(v)]
    if not 意图差异:
        说("  ✅ %d 题意图一致 —— 文本有差异，但**业务意图完全相同**" % len(串行))
        说("     推论：逐字比文本（对照二）过严；并发影响停留在措辞层")
    else:
        说("  🔴 %d / %d 题意图不一致 —— **并发改变了业务判断**，这层才是要处置的"
           % (len(意图差异), len(串行)))
        首 = 意图差异[0]
        说("     首例 %s：串行 %s ｜ 并发 %s"
           % (首, 意图键(串行[首])[:150] or "(抽不出)", 意图键(并发果[首])[:150] or "(抽不出)"))

    产物 = os.path.join(项目目录, "eval", "并发对照-题数%d-并发%d.json" % (len(串行), 并发))
    with open(产物, "w", encoding="utf-8") as 句柄:
        json.dump({"端口": 参数.端口, "total_slots": 槽, "并发": 并发,
                   "题数": len(串行),
                   "串行两遍不一致题号": 串行差异,
                   "串行与并发不一致题号": 并发差异,
                   "串行与并发意图不一致题号": 意图差异,
                   "串行1": 串行, "串行2": 串行2, "并发结果": 并发果},
                  句柄, ensure_ascii=False, indent=2)
    说("\n产物：%s" % 产物)
    return 0 if not 并发差异 else 1


if __name__ == "__main__":
    raise SystemExit(main())
