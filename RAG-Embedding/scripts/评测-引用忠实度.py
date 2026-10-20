# -*- coding: utf-8 -*-
"""
引用忠实度评测：生成端会不会编造资料里没有的东西？（作品一 · D2）
==================================================================

为什么做
--------
昨天只验了 P01 一例（数字逐字对得上）。**一例不能说明系统行为** ——
10-12 要做「引用溯源」，它的前置问题就是：

    **生成端多频繁地编造资料里没有的数值 / 编号？**

做法
----
对 33 道库内题：用 C RRF 检索 Top-3 块 → 拼成【资料】 → 让生成端**只依据资料**回答。
对 4 道库外题：同一流程，正确行为是**拒答**（不能硬答）。

判据（全部可自动判定）
----------------------
① **数值忠实度**：答案里出现的百分比 / 编号，必须**逐字**出现在给定资料里。
   **分母 = 答案里数值的总个数**，不是题数 ——
   这样「答案里一个数值都没有」的题不会被算成「通过」（否则判据分不清通过和没检查）。
② **拒答率**：库外题答「资料不足」的比例。
③ **过度拒答**：库内题也答「资料不足」= 该题资料不够 → 记下来，但不计作幻觉。

🔴 不做「答案正确性」判定（那要人工）。只做**可自动判定**的引用忠实度。
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
结果文件 = os.path.join(输出目录, "引用忠实度-结果.json")

def 取端口(默认="8000"):
    """允许 `--端口 8083` 覆盖（用于 GPU 引擎不可用时切到 CPU 生成服务）。"""
    if "--端口" in sys.argv:
        i = sys.argv.index("--端口")
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return 默认


生成服务 = "http://127.0.0.1:%s" % 取端口()   # 默认 = Local-AI 引擎（作品三）
检索服务 = "http://127.0.0.1:8082"      # bge-m3 embedding（单槽，跨进程可复现）
取块数 = 3

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
拒答词 = ("资料不足", "未提及", "没有相关", "无法回答", "资料中没有")

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
    """抽两类**必须先知道事实才能写出**的东西：百分比、文档编号。"""
    出 = []
    for m in 百分比模式.finditer(文本):
        出.append(("百分比", 去空格(m.group(0))))
    for m in 编号模式.finditer(文本):
        s = m.group(0)
        # 排除纯英文缩写（如 SCAR、SDS），必须含数字才是编号
        if any(c.isdigit() for c in s):
            出.append(("编号", s))
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
    return (d["choices"][0]["message"].get("content") or "").strip(), d


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
    print(" 引用忠实度评测    基座 S13 软表保-1200    %d 块    检索 C RRF Top-%d"
          % (len(块列表), 取块数))
    print("=" * 96)

    print("\n算块向量 / BM25 索引 …")
    块向量表 = 模.取向量(检索服务, 嵌入文本, 标签="  嵌入")
    块模表 = [模.求模长(v) for v in 块向量表]
    BM25索引 = 检索模块.建BM25索引(嵌入文本)

    向量缓存 = {}
    明细 = []
    print("[跑] %d 题 …\n" % len(模.查询集))
    起 = time.time()
    for 序号, (编号, 问句, 目标, 金标准, 类别) in enumerate(模.查询集, start=1):
        序向量, _ = 检索模块.向量排序(问句, 块向量表, 块模表, 向量缓存)
        序BM25, _ = 检索模块.BM25排序(问句, BM25索引)
        序 = 检索模块.RRF融合([序向量, 序BM25])
        取 = 序[:取块数]
        资料 = "\n\n".join("【%s %s】\n%s" % (块列表[i][0], 块列表[i][1], 块列表[i][2])
                          for i in 取)
        规范资料 = 去空格(资料)

        try:
            答案, 应答 = 问生成端(资料, 问句)
        except Exception as 错:
            答案, 应答 = "(生成失败：%s)" % 错, {}

        拒答 = any(w in 答案 for w in 拒答词)
        数值 = 抽数值(答案)
        编造 = [(类, 值) for 类, 值 in 数值 if 值 not in 规范资料]

        明细.append({"编号": 编号, "类别": 类别, "问": 问句,
                     "资料文号": [块列表[i][0] for i in 取],
                     "答案": 答案, "拒答": 拒答,
                     "数值": [值 for _, 值 in 数值],
                     "编造": 编造})
        print("  [%2d/%d] %-5s %-8s 数值 %d 个，编造 %d 个%s"
              % (序号, len(模.查询集), 编号, 类别, len(数值), len(编造),
                 "（拒答）" if 拒答 else ""))

    print("\n      用时 %.1f s" % (time.time() - 起))

    # ---------- 汇总 ----------
    库内 = [x for x in 明细 if x["类别"] != "库外"]
    库外 = [x for x in 明细 if x["类别"] == "库外"]
    总数值 = sum(len(x["数值"]) for x in 库内)
    总编造 = sum(len(x["编造"]) for x in 库内)
    有数值的题 = sum(1 for x in 库内 if x["数值"])

    print()
    print("=" * 96)
    print(" 表 1 · 数值忠实度（库内 %d 题；分母是**数值个数**，不是题数）" % len(库内))
    print("=" * 96)
    print("  答案里出现的百分比 / 编号总数 ：%d" % 总数值)
    print("  其中在给定资料中逐字存在     ：%d" % (总数值 - 总编造))
    print("  编造（资料中不存在）         ：%d" % 总编造)
    print("  只含不含数值的答案           ：%d 题" % (len(库内) - 有数值的题))
    print()
    if 总数值:
        print("  **编造率 = %d / %d = %.1f%%**" % (总编造, 总数值, 100.0 * 总编造 / 总数值))
    else:
        print("  ⚠ 答案里一个数值都没有 —— 判据不适用，不能报「0% 编造」")

    print()
    print("=" * 96)
    print(" 表 2 · 逐题明细")
    print("=" * 96)
    print(" " + 模.补空格("题", 6) + 模.补空格("类别", 10) + 模.补空格("数值", 24)
          + 模.补空格("编造", 16) + "答案（截断）")
    print(" " + "-" * 92)
    for x in 明细:
        print(" " + 模.补空格(x["编号"], 6) + 模.补空格(x["类别"], 10)
              + 模.补空格("、".join(x["数值"]) or "—", 24)
              + 模.补空格("、".join(v for _, v in x["编造"]) or "—", 16)
              + 模.截断(x["答案"], 40))

    print()
    print("=" * 96)
    print(" 表 3 · 库外题（正确行为 = 拒答）与库内过度拒答")
    print("=" * 96)
    库外拒答 = sum(1 for x in 库外 if x["拒答"])
    print("  库外 %d 题拒答：%d / %d" % (len(库外), 库外拒答, len(库外)))
    for x in 库外:
        print("    %s %s  「%s」" % (模.补空格(x["编号"], 6),
                                   "✅ 拒答" if x["拒答"] else "❌ 硬答",
                                   模.截断(x["答案"], 46)))
    过度 = [x for x in 库内 if x["拒答"]]
    print()
    print("  库内过度拒答：%d / %d%s" % (len(过度), len(库内),
                                       "：" + "、".join(x["编号"] for x in 过度) if 过度 else ""))

    # ---------- 落盘 ----------
    with open(结果文件, "w", encoding="utf-8") as 文件:
        json.dump({"基座": "S13 软表保-1200", "检索": "C RRF Top-%d" % 取块数,
                   "提示词": 提示模板,
                   "汇总": {"库内题数": len(库内), "库外题数": len(库外),
                            "数值总数": 总数值, "编造数": 总编造,
                            "编造率": (总编造 / 总数值) if 总数值 else None,
                            "库外拒答": 库外拒答,
                            "库内过度拒答": [x["编号"] for x in 过度]},
                   "明细": 明细}, 文件, ensure_ascii=False, indent=2)
    print("\n 已落盘 %s" % 结果文件)
    return 0


if __name__ == "__main__":
    sys.exit(主流程())
