# -*- coding: utf-8 -*-
"""
端到端验收：知识库服务（作品一 · D4）
======================================

为什么要有这一步
----------------
`实验-拒答规则.py` 只测了**前置拒答规则本身**（不看生成端）。
服务里有三层拒答（类1 分数阈值 / 类2 引用链 / 类3 生成端），
三层叠加后的**真实表现**只能端到端测。

判据（全部可自动判定；每条都能区分「通过」和「没检查」）
--------------------------------------------------------
    ① 拒答矩阵 —— 库外 6 题拒答几个、库内 60 题误拒几个，**分母固定**
                   并**按拒答类型拆分**（类1 / 类2 / 类3）——
                   否则说不清是哪一层在起作用
    ② 数值忠实度 —— 答案里的百分比/编号必须**逐字**出现在送进去的 context 里；
                   **分母是数值个数**，不是题数（否则「一个数值都没有」会被算成通过）
    ③ 引用可核对性 —— citations 的 doc_no 必须都在库内文号表、
                   section 非空、rank 从 1 连续。**引用溯源不能只是个姿势**
    ④ 耗时 —— 检索 / 生成 / 总，报中位数与 p90（上线要对得住用户体验）

直接 import 服务模块（不走 HTTP）—— 与 HTTP 路径共用同一份 `回答()`，不重复实现。

用法：
    python scripts/验收-知识库服务.py
    python scripts/验收-知识库服务.py --前N 5        # 冒烟
    python scripts/验收-知识库服务.py --不生成       # 只测类1/类2（快）
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
结果文件 = os.path.join(输出目录, "知识库服务-验收.json")

百分比模式 = re.compile(r"\d+(?:\.\d+)?\s*%")
编号模式 = re.compile(r"[A-Z]{2,}(?:-[A-Z0-9]+)+")


def 载入(文件名, 别名):
    路 = os.path.join(脚本目录, 文件名)
    规范 = importlib.util.spec_from_file_location(别名, 路)
    模块 = importlib.util.module_from_spec(规范)
    规范.loader.exec_module(模块)
    return 模块


def 去空格(文本):
    return re.sub(r"\s+", "", 文本)


def 抽数值(文本):
    出 = []
    for m in 百分比模式.finditer(文本):
        出.append(去空格(m.group(0)))
    for m in 编号模式.finditer(文本):
        s = m.group(0)
        if any(c.isdigit() for c in s):
            出.append(s)
    return 出


def 分位(表, 比):
    if not 表:
        return None
    表 = sorted(表)
    i = min(len(表) - 1, int(round(比 * (len(表) - 1))))
    return 表[i]


def 主流程():
    sys.stdout.reconfigure(errors="replace")
    前N = int(sys.argv[sys.argv.index("--前N") + 1]) if "--前N" in sys.argv else None
    要生成 = "--不生成" not in sys.argv

    服务 = 载入("知识库服务.py", "知识库服务")
    if not 要生成:
        服务.不生成 = True
    # 拒答模式对照（默认 = 生产配置：类2 软警示）
    模式后缀 = ""
    if "--宽松" in sys.argv:
        服务.类2启用 = False
        模式后缀 = "-宽松"
    elif "--类2硬" in sys.argv:
        服务.类2软 = False
        模式后缀 = "-类2硬"

    print("=" * 100)
    print(" 知识库服务 · 端到端验收    生成端 %s    拒答模式 %s"
          % ("开" if 要生成 else "关", 模式后缀 or "默认(类2软警示)"))
    print("=" * 100)
    print("建基座 …")
    服务.建基座()
    库内文号 = 服务.基座["库内文号"]
    print("  %d 份文档 / %d 块 / 库内文号 %d 个" % (
        len(服务.基座["文档列表"]), len(服务.基座["块列表"]), len(库内文号)))
    print("  拒答：类1 阈值 %+.2f ｜ 类2 %s (Top-%d) ｜ 类3 %s"
          % (服务.分数阈值,
             ("软" if 服务.类2软 else "硬") if 服务.类2启用 else "关",
             服务.类2范围, "关" if 服务.不生成 else "开"))
    print()

    题集 = json.load(open(题目文件, encoding="utf-8"))["题目"]
    if 前N:
        题集 = 题集[:前N]

    明细 = []
    print("[跑] %d 题 …\n" % len(题集))
    总起 = time.time()
    for 序号, 题 in enumerate(题集, start=1):
        编号, 问句, 类别 = 题["编号"], 题["问"], 题["类别"]
        起 = time.time()
        try:
            r = 服务.回答(问句)
            错 = None
        except Exception as e:                                     # noqa: BLE001
            r, 错 = None, str(e)
        墙钟 = time.time() - 起

        if 错:
            print("  [%2d/%d] %-5s %-8s ✗ %s" % (序号, len(题集), 编号, 类别, 错))
            明细.append({"编号": 编号, "类别": 类别, "问": 问句, "错误": 错})
            continue

        # 判据② 数值忠实度
        规范 = 去空格(r["context"])
        数值 = 抽数值(r["answer"])
        编造 = [v for v in 数值 if v not in 规范]

        # 判据③ 引用可核对性
        引用问题 = []
        for k, c in enumerate(r["citations"], start=1):
            if c["rank"] != k:
                引用问题.append("rank 不连续")
            if c["doc_no"] not in 库内文号:
                引用问题.append("文号 %s 不在库内" % c["doc_no"])
            if not c["section"]:
                引用问题.append("第 %d 条 section 为空" % k)

        前置拒 = r["refusal_type"] in ("score_threshold", "broken_citation_chain")
        生成拒 = r["refusal_type"] == "model_refused"

        明细.append({
            "编号": 编号, "类别": 类别, "问": 问句,
            "拒答": r["refused"], "拒答类型": r["refusal_type"],
            "前置拒答": 前置拒, "生成端拒答": 生成拒,
            "最高分": r["top_score"],
            "引用文号": [c["doc_no"] for c in r["citations"]],
            "引用问题": 引用问题,
            "数值": 数值, "编造": 编造,
            "答案": r["answer"], "耗时": r["timings"], "墙钟": round(墙钟, 2),
        })
        标 = "拒" if r["refused"] else "答"
        print("  [%2d/%d] %-5s %-8s %s %-22s 最高分 %6.2f  数值 %d 编造 %d  引用%s %5.2fs"
              % (序号, len(题集), 编号, 类别, 标, r["refusal_type"] or "—",
                 r["top_score"], len(数值), len(编造),
                 "✅" if not 引用问题 else "✗", 墙钟))

    print("\n      总用时 %.1f s" % (time.time() - 总起))

    有效 = [x for x in 明细 if "错误" not in x]
    库内 = [x for x in 有效 if x["类别"] != "库外"]
    库外 = [x for x in 有效 if x["类别"] == "库外"]

    # ---------------- 判据① 拒答矩阵 ----------------
    print()
    print("=" * 100)
    print(" 判据① 拒答矩阵（分母固定；按**层**拆分）")
    print("=" * 100)
    print(" " + "集合".ljust(16) + "题数".ljust(8) + "拒答".ljust(12)
          + "其中类1".ljust(10) + "其中类2".ljust(10) + "其中类3(生成端)")
    print(" " + "-" * 88)
    for 名, 组 in (("库内", 库内), ("库外", 库外)):
        if not 组:
            continue
        拒 = [x for x in 组 if x["拒答"]]
        print(" " + 名.ljust(16) + str(len(组)).ljust(8)
              + ("%d/%d" % (len(拒), len(组))).ljust(12)
              + str(sum(1 for x in 拒 if x["拒答类型"] == "score_threshold")).ljust(10)
              + str(sum(1 for x in 拒 if x["拒答类型"] == "broken_citation_chain")).ljust(10)
              + str(sum(1 for x in 拒 if x["拒答类型"] == "model_refused")))
    print()
    print("  库内**误拒**（用户体验损失）：%s"
          % ("、".join(x["编号"] for x in 库内 if x["拒答"]) or "无 ✅"))
    print("  库外**漏网**（会硬答库外问题）：%s"
          % ("、".join(x["编号"] for x in 库外 if not x["拒答"]) or "无 ✅"))

    # ---------------- 判据② 数值忠实度 ----------------
    print()
    print("=" * 100)
    print(" 判据② 数值忠实度（库内题；分母 = 数值**个数**，不是题数）")
    print("=" * 100)
    总数值 = sum(len(x["数值"]) for x in 库内)
    总编造 = sum(len(x["编造"]) for x in 库内)
    有数值 = sum(1 for x in 库内 if x["数值"])
    print("  答案里百分比/编号总数：%d（其中 %d 题一个数值都没有）" % (总数值, len(库内) - 有数值))
    if 总数值:
        print("  **编造 %d / %d = %.1f%%**" % (总编造, 总数值, 100.0 * 总编造 / 总数值))
    else:
        print("  ⚠ 一个数值都没有 —— 判据不适用，不能报「0% 编造」")
    if 总编造:
        for x in 库内:
            for v in x["编造"]:
                print("    ✗ %s 编造 %s" % (x["编号"], v))

    # ---------------- 判据③ 引用可核对性 ----------------
    print()
    print("=" * 100)
    print(" 判据③ 引用可核对性（每条 citation 的文号/章节/rank）")
    print("=" * 100)
    坏 = [x for x in 有效 if x["引用问题"]]
    无引用 = [x for x in 有效 if not x["引用文号"]]
    print("  引用有问题的题：%d / %d %s" % (len(坏), len(有效), "✅" if not 坏 else ""))
    for x in 坏:
        print("    ✗ %s：%s" % (x["编号"], "；".join(x["引用问题"])))
    print("  未返回引用的题：%d（拒答题不返回引用是正常的；非拒答则不正常）"
          % len(无引用))
    异常 = [x for x in 无引用 if not x["拒答"]]
    if 异常:
        print("    ✗ 非拒答却无引用：%s" % "、".join(x["编号"] for x in 异常))

    # ---------------- 判据④ 耗时 ----------------
    print()
    print("=" * 100)
    print(" 判据④ 耗时（秒）")
    print("=" * 100)
    答的 = [x for x in 有效 if not x["拒答"]]
    拒的 = [x for x in 有效 if x["拒答"]]
    for 名, 组 in (("正常答", 答的), ("拒答", 拒的)):
        if not 组:
            continue
        总 = [x["耗时"].get("total", x["墙钟"]) for x in 组]
        检 = [x["耗时"].get("retrieve") for x in 组 if x["耗时"].get("retrieve")]
        生 = [x["耗时"].get("generate") for x in 组 if x["耗时"].get("generate")]
        print("  %s（%d 题）：总 中位 %.2f / p90 %.2f ｜ 检索 %.2f/%.2f ｜ 生成 %s"
              % (名, len(组), 分位(总, 0.5), 分位(总, 0.9),
                 分位(检, 0.5) or 0, 分位(检, 0.9) or 0,
                 ("%.2f/%.2f" % (分位(生, 0.5), 分位(生, 0.9))) if 生 else "—"))

    # ---------------- 逐题明细 ----------------
    print()
    print("=" * 100)
    print(" 逐题明细")
    print("=" * 100)
    print(" " + "题".ljust(6) + "类别".ljust(10) + "判".ljust(4) + "拒绝类型".ljust(24)
          + "最高分".ljust(9) + "引用文号（去重）".ljust(34) + "答案（截断）")
    print(" " + "-" * 98)
    for x in 明细:
        if "错误" in x:
            print(" " + x["编号"].ljust(6) + x["类别"].ljust(10) + "✗ " + x["错误"])
            continue
        去重 = []
        for 文 in x["引用文号"]:
            if 文 not in 去重:
                去重.append(文)
        print(" " + x["编号"].ljust(6) + x["类别"].ljust(10)
              + ("拒  " if x["拒答"] else "答  ")
              + (x["拒答类型"] or "—").ljust(24)
              + ("%.2f" % x["最高分"]).ljust(9)
              + "、".join(去重).ljust(34)
              + re.sub(r"\s+", " ", x["答案"])[:44])

    with open(os.path.join(输出目录, "知识库服务-验收%s.json" % 模式后缀), "w",
              encoding="utf-8") as 文件:
        json.dump({
            "服务参数": {"池深": 服务.池深, "取块数": 服务.默认取块数,
                        "分数阈值": 服务.分数阈值, "类2启用": 服务.类2启用,
                        "类2软": 服务.类2软, "类2范围": 服务.类2范围,
                        "生成": 要生成},
            "汇总": {
                "库内题数": len(库内), "库外题数": len(库外),
                "库内拒答": [x["编号"] for x in 库内 if x["拒答"]],
                "库外漏网": [x["编号"] for x in 库外 if not x["拒答"]],
                "数值总数": 总数值, "编造数": 总编造,
                "引用问题题": [x["编号"] for x in 坏],
            },
            "明细": 明细,
        }, 文件, ensure_ascii=False, indent=2)
    print("\n 已落盘 %s" % os.path.join(输出目录, "知识库服务-验收%s.json" % 模式后缀))
    return 0


if __name__ == "__main__":
    sys.exit(主流程())
