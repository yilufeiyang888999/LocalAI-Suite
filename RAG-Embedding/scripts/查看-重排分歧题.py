# -*- coding: utf-8 -*-
"""
查重排量化分歧题（作品一 · D3 补充）
====================================

`实验-重排量化对照.py` 的结论是「Q8_0 与 FP16 指标完全一致、逐题正确项排名 0 处变化」，
但表 2 同时报告了 **Top-3 集合 64/66、Top-5 集合 57/66**。

按判据铁律：**「差几题」不能习惯性当噪声放过 —— 放宽容差只是让判据闭嘴，
查清机制才让判据有意义。** 本脚本把这几个分歧题摊开看：
换掉的是哪一块？正确块有没有被换出去？首名有没有变？

用法：
    python scripts/查看-重排分歧题.py              默认 T02、O01（Top-3 分歧）
    python scripts/查看-重排分歧题.py --题 T02 O01 K09
"""

import importlib.util
import json
import os
import sys

脚本目录 = os.path.dirname(os.path.abspath(__file__))
根目录 = os.path.dirname(脚本目录)
评测集文件 = os.path.join(根目录, "eval", "题目.json")

服务甲 = ("Q8_0", "http://127.0.0.1:8091")
服务乙 = ("FP16", "http://127.0.0.1:8090")
向量池, BM25池 = 20, 50


def 加载模块(文件名, 模块名):
    路 = os.path.join(脚本目录, 文件名)
    规范 = importlib.util.spec_from_file_location(模块名, 路)
    模块 = importlib.util.module_from_spec(规范)
    规范.loader.exec_module(模块)
    return 模块


def 截断(文本, 宽):
    文本 = 文本.replace("\n", " ").strip()
    return 文本 if len(文本) <= 宽 else 文本[:宽] + "…"


def 主流程():
    检 = 加载模块("检索-混合.py", "混合检索")
    模 = 检.载入分块模块()
    检.模 = 模
    排 = 加载模块("实验-重排对照.py", "重排对照")
    模.设终端编码()

    看题 = None
    if "--题" in sys.argv:
        i = sys.argv.index("--题")
        看题 = sys.argv[i + 1:i + 1 + 10]
    看题 = 看题 or ["T02", "O01"]

    文档列表 = 模.读语料()
    块列表 = []
    for 文档 in 文档列表:
        for 前缀, 正文 in 模.切标题结构(文档, 1200, False, 软表格=True):
            if len(正文.strip()) >= 8:
                块列表.append((文档["文号"], 前缀, 正文))
    嵌入文本 = [(前 + "\n" + 文) for _, 前, 文 in 块列表]

    with open(评测集文件, encoding="utf-8") as 文件:
        题集 = {x["编号"]: x for x in json.load(文件)["题目"]}

    print("算块向量（%s）…" % 检.默认服务, flush=True)
    块向量表 = 模.取向量(检.默认服务, 嵌入文本, 标签="  嵌入")
    块模表 = [模.求模长(向) for 向 in 块向量表]
    BM25索引 = 检.建BM25索引(嵌入文本)
    正确块表 = 检.找正确块(块列表, [(x["编号"], x["问"], x.get("目标", []),
                                  x.get("金标准", []), x["类别"])
                                 for x in 题集.values()])
    向量缓存 = {}

    for 编号 in 看题:
        题 = 题集.get(编号)
        if not 题:
            print("⚠ 评测集里没有 %s" % 编号)
            continue
        问句 = 题["问"]
        序A, _ = 检.向量排序(问句, 块向量表, 块模表, 向量缓存)
        序B, _ = 检.BM25排序(问句, BM25索引)
        池 = sorted(set(序A[:向量池]) | set(序B[:BM25池]))

        print()
        print("=" * 100)
        print(" %s「%s」  类别 %s" % (编号, 问句, 题["类别"]))
        print("   金标准：%s" % "、".join(题.get("金标准", [])) or "（库外题，无金标准）")
        print("   正确块（含完整金标准的块）：%s"
              % ("无" if not 正确块表[编号] else
                 "、".join("第%d块[%s]" % (i, 块列表[i][0]) for i in sorted(正确块表[编号]))))
        print("=" * 100)

        前五表 = {}
        for 名, 服务 in (服务甲, 服务乙):
            分 = 排.重排打分(服务, 问句, [嵌入文本[i] for i in 池], 标签=编号)
            分表 = {块: 值 for 块, 值 in zip(池, 分)}
            序 = 排.派生顺序(序B, {i: 分表[i] for i in 序B[:BM25池]}, BM25池)
            前五表[名] = 序[:5]
            print(" 【%s】Top-5（RB50 口径）" % 名)
            for 位, i in enumerate(序[:5], 1):
                标 = "  ← 正确块" if i in 正确块表[编号] else ""
                print("   %d. 分 %8.4f  [%s] %s%s"
                      % (位, 分表[i], 块列表[i][0], 截断(块列表[i][1] or 块列表[i][2], 46), 标))
            名次 = 检.求排名(序, 正确块表[编号]) if 题["类别"] != "库外" else None
            print("   正确块排名：%s" % ("—（库外题）" if 名次 is None else 名次))

        甲集, 乙集 = set(前五表[服务甲[0]][:3]), set(前五表[服务乙[0]][:3])
        if 甲集 != 乙集:
            print("   ▶ Top-3 差异：Q8_0 独有 %s；FP16 独有 %s"
                  % ([块列表[i][0] for i in 甲集 - 乙集] or "无",
                     [块列表[i][0] for i in 乙集 - 甲集] or "无"))
        else:
            print("   ▶ Top-3 集合相同")
    return 0


if __name__ == "__main__":
    sys.exit(主流程())
