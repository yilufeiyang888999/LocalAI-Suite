# -*- coding: utf-8 -*-
"""
验证 P01 的结论：把「两跳证据」拼起来，生成端答得出来吗？（作品一 · D2）
=========================================================================

背景
----
P01 在五个检索通道下全部失败（22/41/33/31/30），但诊断显示
**原因证据块在 Top-3、结果证据块在 Top-1** —— 检索够用，
是「单块含全部金标准」这个判据在要求「一个块答完一道两跳题」。

本脚本做**可判定的验证**：
    把检索已经拿到的两块（原因侧 + 结果侧）拼成资料，喂给生成端，
    看它能不能答出「钝化体系变更 → 膜层致密度下降 → 不良率上升」这条因果链。

判据（能区分「通过」和「没通过」）
----------------------------------
    ✅ 通过：答案里出现「钝化」（说明用上了原因块），且出现因果连接词。
    ❌ 不通过：答案只复述不良率数据、没有原因 —— 那说明证据块选错了，不是判据问题。

零第三方依赖。
"""

import importlib.util
import json
import os
import re
import sys
import urllib.request

脚本目录 = os.path.dirname(os.path.abspath(__file__))
根目录 = os.path.dirname(脚本目录)
输出目录 = os.path.join(根目录, "eval")

嵌入服务 = "http://127.0.0.1:8081"
生成服务 = "http://127.0.0.1:8000"   # Local-AI 引擎（作品三）

原因侧词 = ["钝化"]
结果侧词 = ["不良率", "不良原因", "不良总数"]
因果词 = ["导致", "因为", "由于", "原因", "使得", "引起", "造成"]

提示词 = """请根据下面提供的资料回答问题。只依据资料作答，不要添加资料以外的信息。

【资料】
{资料}

【问题】
东莞精诚为什么 7 月起不良率突然升高？

【回答】"""


def 载入模块():
    路 = os.path.join(脚本目录, "实验-分块对照.py")
    规范 = importlib.util.spec_from_file_location("分块实验", 路)
    模块 = importlib.util.module_from_spec(规范)
    规范.loader.exec_module(模块)
    return 模块


def 主流程():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    模 = 载入模块()

    文档列表 = 模.读语料()
    块列表 = []
    for 文档 in 文档列表:
        for 前缀, 正文 in 模.切标题结构(文档, 1200, False, 软表格=True):
            if len(正文.strip()) >= 8:
                块列表.append((文档["文号"], 前缀, 正文))

    print("算块向量 …")
    块向量表 = 模.取向量(嵌入服务, [(前 + "\n" + 文) for _, 前, 文 in 块列表], 标签="  嵌入")
    块模表 = [模.求模长(向) for 向 in 块向量表]
    规范块文 = [模.规范化(文) for _, _, 文 in 块列表]

    问句 = "东莞精诚为什么 7 月起不良率突然升高"
    查询向量 = 模.取向量(嵌入服务, [问句])[0]
    查询模 = 模.求模长(查询向量)
    分 = [模.点积(查询向量, 块向量) / (查询模 * 块模 + 1e-12)
          for 块向量, 块模 in zip(块向量表, 块模表)]
    序 = sorted(range(len(分)), key=lambda i: -分[i])

    原因块 = next((s for s in 序[:10] if any(w in 规范块文[s] for w in 原因侧词)), None)
    结果块 = next((s for s in 序[:10] if any(w in 规范块文[s] for w in 结果侧词)), None)

    print("\n" + "=" * 92)
    print(" 检索拿到的两块（都在 Top-10 内）")
    print("=" * 92)
    for 名, 序_ in (("原因侧", 原因块), ("结果侧", 结果块)):
        if 序_ is None:
            print("  ✗ %s：Top-10 内未找到" % 名)
            continue
        排 = 序.index(序_) + 1
        print("\n  【%s】第 %d 名  向量分 %.4f  文号 %s" % (名, 排, 分[序_], 块列表[序_][0]))
        print("  前缀：%s" % 块列表[序_][1])
        print("  正文：%s" % 模.截断(规范块文[序_], 300))

    if 原因块 is None or 结果块 is None:
        print("\n✗ 证据不全，无法验证")
        return 1

    资料 = "【资料一｜%s】\n%s\n\n【资料二｜%s】\n%s" % (
        块列表[原因块][0], 块列表[原因块][2][:1400],
        块列表[结果块][0], 块列表[结果块][2][:1400])

    载荷 = json.dumps({
        "messages": [{"role": "user", "content": 提示词.format(资料=资料)}],
        "temperature": 0.1,
        "max_tokens": 500,
        "chat_template_kwargs": {"enable_thinking": False},
    }, ensure_ascii=False).encode("utf-8")
    请求 = urllib.request.Request(生成服务 + "/v1/chat/completions", data=载荷,
                                headers={"Content-Type": "application/json"})
    print("\n" + "=" * 92)
    print(" 生成端作答（资料 = 上面两块，无其他上下文）")
    print("=" * 92)
    with urllib.request.urlopen(请求, timeout=300) as 响应:
        数据 = json.loads(响应.read().decode("utf-8"))
    答案 = (数据["choices"][0]["message"].get("content") or "").strip()
    print("\n" + 答案)

    命中原因 = [w for w in ("钝化", "六价铬", "三价铬") if w in 答案]
    命中因果 = [w for w in 因果词 if w in 答案]

    print("\n" + "=" * 92)
    print(" 判定")
    print("=" * 92)
    print("  用上原因块（出现「钝化 / 六价铬 / 三价铬」）：%s" % ("、".join(命中原因) or "无"))
    print("  出现因果连接词：%s" % ("、".join(命中因果) or "无"))
    通过 = bool(命中原因) and bool(命中因果)
    print("  → %s" % ("✅ 通过：检索到的两块足以答出因果链 —— 印证「P01 不是检索问题」"
                      if 通过 else
                      "❌ 未通过：答案没接上原因 —— 需复查证据块选取"))
    return 0 if 通过 else 1


if __name__ == "__main__":
    sys.exit(主流程())
