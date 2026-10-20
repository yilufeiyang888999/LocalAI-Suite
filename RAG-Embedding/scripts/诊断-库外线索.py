# -*- coding: utf-8 -*-
"""
库外线索诊断（作品一 · D4 前置）
================================

为什么先做这个
--------------
10-12 的知识库服务要做**两类拒答**：

    类① 完全没线索  → 分数阈值可拦（报告 §三.4 已量化）
    类② 有线索但指向库外 → 阈值拦不住（O03 = 2.65 落在库内分数区间内部）

但**类②的判定规则不能凭空设计**。必须先看清 O03 / O06 的检索现场长什么样：
  · 它们的 Top-3 块里到底有什么？
  · 块里有没有「指向库外文档」的交叉引用（如 QW-DD-03 提到 QW-DD-05）？
  · 库内题与库外题在「块内编号的库内/库外比例」上有没有可分的差异？

判据
----
    · 每题取**全库 290 块**的最高重排分（与 `重排对照-报告.md` §三.4 的阈值表同口径）
    · 同时验证「池内 Top-50 的最高分 ≡ 全库最高分」—— 这是 RB50 分池策略成立的前提
    · 打印 Top-3 块的**文号 + 章节路径 + 文本**，以及块内抽到的**文档编号**及其库内/库外归属
    · 库内题作对照（不只看库外，否则「差异」可能只是这 6 题的偶然）

用法：
    python scripts/诊断-库外线索.py            # 6 道库外 + 4 道库内对照
    python scripts/诊断-库外线索.py --全库否     # 只算池内 Top-50（快）
"""

import importlib.util
import json
import os
import re
import sys
import time

脚本目录 = os.path.dirname(os.path.abspath(__file__))
根目录 = os.path.dirname(脚本目录)
输出目录 = os.path.join(根目录, "eval")
题目文件 = os.path.join(输出目录, "题目.json")

向量服务 = "http://127.0.0.1:8082"      # 单槽，跨进程可复现（见 §五.3）
重排服务 = "http://127.0.0.1:8085"      # Q8_0，--parallel 1

池深 = 50

# 文档编号模式：QW-DD-03 / JC-GY-2026-006 / GB-T-19001 这类
# 故意放宽（2~6 段字母 + 1~2 段数字），**宁可多抽再人工判**，不要漏掉交叉引用
编号模式 = re.compile(r"[A-Z]{2,6}(?:-[A-Z]{2,6}){1,2}-\d{2,6}")


def 载入(文件名, 别名):
    路 = os.path.join(脚本目录, 文件名)
    规范 = importlib.util.spec_from_file_location(别名, 路)
    模块 = importlib.util.module_from_spec(规范)
    规范.loader.exec_module(模块)
    return 模块


def 截断(文本, 宽):
    文本 = re.sub(r"\s+", " ", 文本).strip()
    return 文本 if len(文本) <= 宽 else 文本[:宽] + "…"


def 主流程():
    sys.stdout.reconfigure(errors="replace")
    全库否 = ("--全库否" in sys.argv)

    检 = 载入("检索-混合.py", "检索")
    模 = 检.载入分块模块()
    检.模 = 模
    重 = 载入("实验-重排对照.py", "重排")

    # ---------- 建基座（与 检索-混合 / 重排对照 同一口径：S13 软表保-1200） ----------
    文档列表 = 模.读语料()
    块列表 = []
    for 文档 in 文档列表:
        for 前缀, 正文 in 模.切标题结构(文档, 1200, False, 软表格=True):
            if len(正文.strip()) >= 8:
                块列表.append((文档["文号"], 前缀, 正文))
    嵌入文本 = [(前 + "\n" + 文) for _, 前, 文 in 块列表]
    库内文号 = sorted({文档["文号"] for 文档 in 文档列表})

    print("=" * 100)
    print(" 库外线索诊断    基座 S13 软表保-1200    %d 块    库内文号 %d 份"
          % (len(块列表), len(库内文号)))
    print("=" * 100)
    print(" 库内文号表：" + "、".join(库内文号))
    print()

    print("算块向量 / BM25 索引 …")
    块向量表 = 模.取向量(向量服务, 嵌入文本, 标签="  嵌入")
    块模表 = [模.求模长(v) for v in 块向量表]
    BM25索引 = 检.建BM25索引(嵌入文本)
    print("      就绪（%d 块 × %d 维）\n" % (len(块列表), len(块向量表[0])))

    # ---------- 选题：库外全要 + 库内抽对照 ----------
    题集 = json.load(open(题目文件, encoding="utf-8"))["题目"]
    库外题 = [t for t in 题集 if t.get("类别") == "库外"]
    库内题 = [t for t in 题集 if t.get("类别") != "库外"]
    # 库内对照：取 4 道不同类型的
    对照编号 = ["T01", "K01", "L01", "P01"]
    对照题 = [t for t in 库内题 if t["编号"] in 对照编号] or 库内题[:4]
    待测 = 库外题 + 对照题

    print(" 待测：%s" % "、".join("%s(%s)" % (t["编号"], t["类别"]) for t in 待测))
    print()

    汇总 = []
    起 = time.time()
    for 题 in 待测:
        编号, 问句 = 题["编号"], 题["问"]
        序B, _ = 检.BM25排序(问句, BM25索引)
        池 = 序B[:池深]

        # 打分对象：全库 或 仅池内
        打分序 = list(range(len(块列表))) if not 全库否 else 池
        文本 = [(块列表[i][1] + "\n" + 块列表[i][2]) for i in 打分序]
        分 = 重.重排打分(重排服务, 问句, 文本, 标签=编号)
        分表 = {i: s for i, s in zip(打分序, 分)}
        全库最高 = max(分表.values())
        池内最高 = max((分表[i] for i in 池), default=None)

        # Top-3 = 全库按重排分降序
        Top = sorted(分表, key=lambda i: -分表[i])[:3]

        # 块内编号抽取（Top-3 与 Top-50 两个口径都看）
        洞 = {"top3": set(), "top50": set()}
        for 范围, 键 in ((Top, "top3"), (池[:50], "top50")):
            for i in 范围:
                洞[键] |= set(编号模式.findall(嵌入文本[i]))

        print("-" * 100)
        print(" %s  「%s」  类别 %s" % (编号, 问句, 题["类别"]))
        print("     全库最高重排分 %.2f   池内(Top-50)最高 %.2f   %s"
              % (全库最高, 池内最高,
                 "✅ 池内 ≡ 全库" if abs(全库最高 - 池内最高) < 1e-6 else "🔴 池内 < 全库（池深不够）"))
        for 位, i in enumerate(Top, start=1):
            文号, 前缀, 正文 = 块列表[i]
            章节 = 前缀.split(">")[-1].strip() if ">" in 前缀 else 前缀
            print("   [%d] %.2f  %-12s %s" % (位, 分表[i], 文号, 截断(章节, 46)))
            print("        %s" % 截断(正文, 92))
        for 键, 名 in (("top3", "Top-3"), ("top50", "Top-50")):
            库外引 = sorted(洞[键] - set(库内文号))
            print("     %s 内文档编号：库内 %s ｜ 库外 %s"
                  % (名, "、".join(sorted(洞[键] & set(库内文号))) or "—",
                     "、".join(库外引) or "—"))

        汇总.append({
            "编号": 编号, "类别": 题["类别"], "问": 问句,
            "全库最高": 全库最高, "池内最高": 池内最高,
            "top3文号": [块列表[i][0] for i in Top],
            "top3编号": sorted(洞["top3"]),
            "top3库外编号": sorted(洞["top3"] - set(库内文号)),
            "top50库外编号": sorted(洞["top50"] - set(库内文号)),
        })

    print()
    print("=" * 100)
    print(" 汇总（%.1f s，%s）" % (time.time() - 起, "全库打分" if not 全库否 else "仅池内"))
    print("=" * 100)
    print(" " + "题".ljust(6) + "类别".ljust(10) + "全库最高".ljust(10)
          + "Top-3 文号（去重）".ljust(40) + "Top-3 里的库外编号")
    print(" " + "-" * 96)
    for x in 汇总:
        去重 = []
        for 文 in x["top3文号"]:
            if 文 not in 去重:
                去重.append(文)
        print(" " + x["编号"].ljust(6) + x["类别"].ljust(10)
              + ("%.2f" % x["全库最高"]).ljust(10)
              + "、".join(去重).ljust(40)
              + ("、".join(x["top3库外编号"]) or "—"))

    # ---------- 阈值分离度（复算报告 §三.4，用本轮同一口径） ----------
    print()
    print("=" * 100)
    print(" 分离度：库内最低 vs 库外各题")
    print("=" * 100)
    库内分 = sorted(x["全库最高"] for x in 汇总 if x["类别"] != "库外")
    库外分 = sorted(((x["全库最高"], x["编号"]) for x in 汇总 if x["类别"] == "库外"),
                    reverse=True)
    if 库内分:
        print("  库内（对照 %d 题）最高分区间：%.2f … %.2f" % (len(库内分), 库内分[-1], 库内分[0]))
    print("  库外：" + "  ".join("%s=%.2f" % (n, s) for s, n in 库外分))

    with open(os.path.join(输出目录, "库外线索-诊断.json"), "w", encoding="utf-8") as 文件:
        json.dump({"基座": "S13 软表保-1200", "池深": 池深,
                   "库内文号": 库内文号, "明细": 汇总}, 文件,
                  ensure_ascii=False, indent=2)
    print("\n 已落盘 %s" % os.path.join(输出目录, "库外线索-诊断.json"))
    return 0


if __name__ == "__main__":
    sys.exit(主流程())
