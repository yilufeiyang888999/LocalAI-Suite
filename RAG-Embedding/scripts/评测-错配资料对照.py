# -*- coding: utf-8 -*-
"""
资料错配对照：资料不对时，生成端会不会编？（作品一 · D2）
==========================================================

为什么是这个设计
----------------
第一次做的对照是「不给资料」，结果 33/33 全答「不确定」—— **对照被我自己污染了**：
我给对照组换了提示词（多了「不要给出估计值」），于是测出来的其实是**提示词引导效应**，
不是模型行为。**对照组不能换提示词**，只能换那一个变量。

正确的对照组 = **提示词一字不改，只把【资料】换成不相关的资料** ——
即对第 i 题，用第 i+1 题的检索结果当资料。这同时更贴近真实风险场景：
**检索错了的时候，生成端会不会照着错误资料硬答、或者自己补事实。**

判据
----
① **判据有效性**：错配资料里混入了多少本题的正常 Top-3 块 —— 混入率必须低，否则"错配"名不副实。
② **编造率**：答案里的百分比 / 编号，在**错配资料**里不存在 =
   生成端自己补的（注意：它可能在语料库里存在，但那不是"忠实于给定资料"）。
③ **老实率**：错配资料下答「资料不足 / 未提及」的比例 —— 这才是 RAG 的安全网。

零第三方依赖。
"""

import importlib.util
import json
import os
import re
import sys
import time
import urllib.request

脚本目录 = os.path.dirname(os.path.abspath(__file__))
根目录 = os.path.dirname(脚本目录)
输出目录 = os.path.join(根目录, "eval")
结果文件 = os.path.join(输出目录, "错配资料对照-结果.json")

def 取端口(默认="8000"):
    """允许 `--端口 8083` 覆盖（GPU 引擎不可用时切到 CPU 生成服务）。"""
    if "--端口" in sys.argv:
        i = sys.argv.index("--端口")
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return 默认


生成服务 = "http://127.0.0.1:%s" % 取端口()
检索服务 = "http://127.0.0.1:8082"      # 单槽（跨进程可复现），与主链路一致
取块数 = 3

# 与 评测-引用忠实度.py **逐字相同**
提示模板 = (
    "你是制造业质量与工艺领域的文档助手。请**只根据下面提供的资料**回答用户的问题。\n\n"
    "要求：\n"
    "1. 直接给结论，控制在 120 字以内；\n"
    "2. 引用资料中的数值、编号时必须**逐字照抄**，不要改写、不要推算；\n"
    "3. 若资料不足以回答问题，只回答「资料不足」，不要补充资料之外的内容。\n\n"
    "【资料】\n{资料}\n\n【问题】{问句}\n"
)

百分比模式 = re.compile(r"\d+(?:\.\d+)?\s*%")
编号模式 = re.compile(r"[A-Z]{2,}(?:-[A-Z0-9]+)+")
拒答词 = ("资料不足", "未提及", "没有相关", "无法回答", "资料中没有", "不确定")

模 = None
检索模块 = None


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
        if any(c.isdigit() for c in m.group(0)):
            出.append(m.group(0))
    return 出


def 问生成端(资料, 问句, 超时=180):
    载荷 = json.dumps({
        "messages": [{"role": "user", "content": 提示模板.format(资料=资料, 问句=问句)}],
        "temperature": 0.1,
        "max_tokens": 400,
        "chat_template_kwargs": {"enable_thinking": False},
    }, ensure_ascii=False).encode("utf-8")
    请求 = urllib.request.Request(生成服务 + "/v1/chat/completions", data=载荷,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(请求, timeout=超时) as 响应:
        d = json.loads(响应.read().decode("utf-8"))
    return (d["choices"][0]["message"].get("content") or "").strip()


def 拼资料(块列表, 取):
    return "\n\n".join("【%s %s】\n%s" % (块列表[i][0], 块列表[i][1], 块列表[i][2])
                       for i in 取)


def 主流程():
    global 模, 检索模块
    sys.stdout.reconfigure(errors="replace")
    检索模块 = 载入("检索-混合.py", "检索")
    模 = 检索模块.载入分块模块()
    检索模块.模 = 模

    try:
        urllib.request.urlopen(生成服务 + "/health", timeout=5).read()
    except Exception as 错:
        print("✗ 生成服务未就绪（%s）：%s" % (生成服务, 错))
        return 1

    文档列表 = 模.读语料()
    块列表 = []
    for 文档 in 文档列表:
        for 前缀, 正文 in 模.切标题结构(文档, 1200, False, 软表格=True):
            if len(正文.strip()) >= 8:
                块列表.append((文档["文号"], 前缀, 正文))
    嵌入文本 = [(前 + "\n" + 文) for _, 前, 文 in 块列表]

    print("=" * 96)
    print(" 资料错配对照    提示词与主实验**逐字相同**，只把【资料】换成另一题的结果")
    print("=" * 96)

    print("\n算块向量 / BM25 索引 …")
    块向量表 = 模.取向量(检索服务, 嵌入文本, 标签="  嵌入")
    块模表 = [模.求模长(v) for v in 块向量表]
    BM25索引 = 检索模块.建BM25索引(嵌入文本)

    # 先算每题的正常 Top-3（用于错配 + 混入率核查）
    向量缓存 = {}
    正常序 = {}
    for 编号, 问句, 目标, 金标准, 类别 in 模.查询集:
        序向量, _ = 检索模块.向量排序(问句, 块向量表, 块模表, 向量缓存)
        序BM25, _ = 检索模块.BM25排序(问句, BM25索引)
        正常序[编号] = 检索模块.RRF融合([序向量, 序BM25])

    题序 = [t[0] for t in 模.查询集]
    明细 = []
    print("\n[跑] %d 题（资料 = 下一题的 Top-%d）…\n" % (len(题序), 取块数))
    起 = time.time()
    for idx, (编号, 问句, 目标, 金标准, 类别) in enumerate(模.查询集):
        错配编号 = 题序[(idx + 1) % len(题序)]
        取 = 正常序[错配编号][:取块数]
        资料 = 拼资料(块列表, 取)
        规范资料 = 去空格(资料)

        # 判据有效性：错配资料里混进了多少本题的正常块
        混入 = len(set(正常序[编号][:取块数]) & set(取))

        try:
            答案 = 问生成端(资料, 问句)
        except Exception as 错:
            答案 = "(生成失败：%s)" % 错

        拒答 = any(w in 答案 for w in 拒答词)
        数值 = 抽数值(答案)
        编造 = [v for v in 数值 if v not in 规范资料]

        明细.append({"编号": 编号, "类别": 类别, "问": 问句,
                     "错配来源题": 错配编号, "资料文号": [块列表[i][0] for i in 取],
                     "混入块数": 混入, "答案": 答案, "拒答": 拒答,
                     "数值": 数值, "编造": 编造})
        print("  [%2d/%d] %-5s ← %-5s 混入 %d   数值 %d，编造 %d%s"
              % (idx + 1, len(题序), 编号, 错配编号, 混入, len(数值), len(编造),
                 "（拒答）" if 拒答 else ""))

    print("\n      用时 %.1f s" % (time.time() - 起))

    库内 = [x for x in 明细 if x["类别"] != "库外"]
    总数值 = sum(len(x["数值"]) for x in 库内)
    总编造 = sum(len(x["编造"]) for x in 库内)
    总混入 = sum(x["混入块数"] for x in 库内)
    拒答题 = [x["编号"] for x in 库内 if x["拒答"]]

    print()
    print("=" * 96)
    print(" 表 1 · 资料错配时的表现（库内 %d 题）" % len(库内))
    print("=" * 96)
    print("  ① 判据有效性：错配资料里混入本题正常块 —— %d 块 / %d 块位 = %.1f%%"
          % (总混入, len(库内) * 取块数, 100.0 * 总混入 / (len(库内) * 取块数)))
    print("     %s" % ("✅ 混入率低，错配成立" if 总混入 / (len(库内) * 取块数) < 0.10
                       else "⚠ 混入率偏高，错配不彻底"))
    print()
    print("  ② 给出数值总数               ：%d" % 总数值)
    print("  ③ 其中错配资料里不存在（编造）：%d" % 总编造)
    if 总数值:
        print("     **错配下编造率 = %d / %d = %.1f%%**"
              % (总编造, 总数值, 100.0 * 总编造 / 总数值))
    else:
        print("     ⚠ 一个数值都没给 —— 需看定性断言")
    print()
    print("  ④ 老实率（答「资料不足」等）：%d / %d = %.1f%%"
          % (len(拒答题), len(库内), 100.0 * len(拒答题) / len(库内)))
    print("     %s" % ("（" + "、".join(拒答题) + "）" if 拒答题 else ""))

    print()
    print("=" * 96)
    print(" 表 2 · 逐题明细")
    print("=" * 96)
    print(" " + 模.补空格("题", 6) + 模.补空格("类别", 10) + 模.补空格("错配来源", 10)
          + 模.补空格("数值", 16) + 模.补空格("编造", 12) + "答案（截断）")
    print(" " + "-" * 92)
    for x in 明细:
        print(" " + 模.补空格(x["编号"], 6) + 模.补空格(x["类别"], 10)
              + 模.补空格(x["错配来源题"], 10)
              + 模.补空格("、".join(x["数值"]) or "—", 16)
              + 模.补空格("、".join(x["编造"]) or "—", 12)
              + 模.截断(x["答案"], 36))

    with open(结果文件, "w", encoding="utf-8") as 文件:
        json.dump({"说明": "资料错配对照：提示词与主实验逐字相同，资料换成下一题的结果",
                   "提示词": 提示模板,
                   "汇总": {"混入块数": 总混入, "数值总数": 总数值, "编造数": 总编造,
                            "编造率": (总编造 / 总数值) if 总数值 else None,
                            "老实率": len(拒答题) / len(库内), "拒答题": 拒答题},
                   "明细": 明细}, 文件, ensure_ascii=False, indent=2)
    print("\n 已落盘 %s" % 结果文件)
    return 0


if __name__ == "__main__":
    sys.exit(主流程())
