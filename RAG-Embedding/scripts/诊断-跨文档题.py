# -*- coding: utf-8 -*-
"""
跨文档题诊断：单跳判据 vs 证据覆盖（作品一 · D2）
==================================================

为什么要做这个诊断
------------------
P01「东莞精诚为什么 7 月起不良率突然升高」在 **5 个检索通道下全部失败**
（A 向量 22 / B BM25 41 / C RRF 33 / D HyDE拼接 31 / E PRF二次 30）。
而另外三道跨文档题 P02/P03/P04 都在第 1~6 名。

差别在哪：**P02/P03/P04 的提问里，含有答案块里出现的词**
（「变更通知单」「复测」「补测通报」「不达标」）；
**P01 的提问与答案块零词重叠** ——
提问说的是「不良率升高」（结果），答案块说的是「钝化体系变更」（原因）。

所以要先回答一个问题：
    **是检索没找到，还是判据在要求「一个块答完一道两跳题」？**

两种判据
--------
    严格判据（分块实验沿用）  同一块内同时含全部金标准片段 → 衡量「单块可答性」
    证据覆盖                Top-k 内分别出现「原因块」与「结果块」 → 衡量「多跳可答性」

对两跳题，严格判据是不适用的 —— 但这不代表可以放宽判据，
而是说**这道题本来就该按「证据集合」考核**。本脚本把两者的差距量化出来。

零第三方依赖。
"""

import importlib.util
import json
import math
import operator
import os
import re
import sys
import urllib.request

脚本目录 = os.path.dirname(os.path.abspath(__file__))
根目录 = os.path.dirname(脚本目录)
输出目录 = os.path.join(根目录, "eval")
默认服务 = "http://127.0.0.1:8081"

# 原因/结果两侧的证据词 —— 都比金标准宽松，只看「有没有摸到那一侧」
原因证据词 = ["钝化"]
结果证据词 = ["不良率", "不良原因", "不良总数"]


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
    嵌入文本 = [(前 + "\n" + 文) for _, 前, 文 in 块列表]

    print("=" * 96)
    print(" 跨文档题诊断    基座 S13 软表保-1200    %d 块" % len(块列表))
    print("=" * 96)
    print("\n算块向量 …")
    块向量表 = 模.取向量(默认服务, 嵌入文本, 标签="  嵌入")
    块模表 = [模.求模长(向) for 向 in 块向量表]
    规范块文 = [模.规范化(文) for _, _, 文 in 块列表]

    def 含任何(序, 词表):
        return any(词 in 规范块文[序] for 词 in 词表)

    print()
    print("=" * 96)
    print(" 判据说明")
    print("   严格判据（分块实验沿用）：同一块内同时含全部金标准片段 → 衡量「单块可答性」")
    print("   证据覆盖              ：Top-k 内出现「原因块」与「结果块」 → 衡量「多跳可答性」")
    print("=" * 96)

    报告 = {}
    for 编号, 问句, 目标, 金标准, 类别 in 模.查询集:
        if 类别 != "跨文档":
            continue
        查询向量 = 模.取向量(默认服务, [问句])[0]
        查询模 = 模.求模长(查询向量)
        分 = [模.点积(查询向量, 块向量) / (查询模 * 块模 + 1e-12)
              for 块向量, 块模 in zip(块向量表, 块模表)]
        序 = sorted(range(len(分)), key=lambda i: -分[i])

        # 严格判据：同一块含全部金标准
        片段 = [模.规范化(x) for x in 金标准]
        严格 = [i for i in 序 if all(p in 规范块文[i] for p in 片段)]
        严格排 = (序.index(严格[0]) + 1) if 严格 else None

        # 证据覆盖：两侧分别在 Top-k 的最靠前排名
        原因排 = next((i + 1 for i, s in enumerate(序[:30]) if 含任何(s, 原因证据词)), None)
        结果排 = next((i + 1 for i, s in enumerate(序[:30]) if 含任何(s, 结果证据词)), None)

        print()
        print("-" * 96)
        print(" %s「%s」" % (编号, 问句))
        print("-" * 96)
        print("   严格判据：命中块 %d 个，最靠前排名 %s"
              % (len(严格), 严格排 if 严格排 else "未进前 30"))
        print("   原因证据（%s）：最靠前排名 %s" % ("/".join(原因证据词), 原因排 or "未进前 30"))
        print("   结果证据（%s）：最靠前排名 %s" % ("/".join(结果证据词), 结果排 or "未进前 30"))

        判定 = ("两跳证据在 Top-5 内齐备 → 检索够用，缺的是生成端拼装"
                if (原因排 or 99) <= 5 and (结果排 or 99) <= 5 else
                "证据未齐备 → 需要多跳/引用扩展检索")
        print("   → %s" % 判定)

        print()
        print("   Top-10：")
        print("     %-5s%-9s%-9s%-16s%s" % ("排名", "向量分", "侧", "文号", "片段"))
        展示 = []
        for i, s in enumerate(序[:10]):
            侧 = []
            if 含任何(s, 原因证据词):
                侧.append("原因")
            if 含任何(s, 结果证据词):
                侧.append("结果")
            标记 = "+".join(侧) if 侧 else "-"
            片段 = 模.截断(规范块文[s], 62)
            print("     %-5d%-9.4f%-9s%-16s%s" % (i + 1, 分[s], 标记, 块列表[s][0], 片段))
            展示.append({"排名": i + 1, "向量分": round(分[s], 4), "侧": 标记,
                         "文号": 块列表[s][0], "片段": 片段})

        报告[编号] = {"问句": 问句, "严格判据排名": 严格排,
                      "原因证据排名": 原因排, "结果证据排名": 结果排,
                      "Top10": 展示, "判定": 判定}

    with open(os.path.join(输出目录, "跨文档题-诊断.json"), "w", encoding="utf-8") as 文件:
        json.dump(报告, 文件, ensure_ascii=False, indent=2)

    print()
    print("=" * 96)
    print(" 汇总")
    print("=" * 96)
    print(" %-6s%-14s%-14s%-14s%s" % ("题号", "严格判据", "原因证据", "结果证据", "结论"))
    print(" " + "-" * 88)
    for 编号, 值 in 报告.items():
        print(" %-6s%-14s%-14s%-14s%s"
              % (编号,
                 str(值["严格判据排名"]) if 值["严格判据排名"] else "未进前30",
                 str(值["原因证据排名"]) if 值["原因证据排名"] else "未进前30",
                 str(值["结果证据排名"]) if 值["结果证据排名"] else "未进前30",
                 值["判定"][:30]))
    print()
    print(" 已落盘 %s" % os.path.join(输出目录, "跨文档题-诊断.json"))
    return 0


if __name__ == "__main__":
    sys.exit(主流程())
