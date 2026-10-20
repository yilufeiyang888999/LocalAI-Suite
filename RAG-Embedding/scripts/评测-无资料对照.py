# -*- coding: utf-8 -*-
"""
无资料对照：不给资料时，生成端会不会编？（作品一 · D2）
========================================================

为什么必须有这个对照
--------------------
`评测-引用忠实度.py` 测出「有资料 → 编造率 0%」。但**没有对照就说明不了问题**：
这个 0% 可能是提示词压制出来的（"数值必须逐字照抄"），也可能是模型本来就不编。

本脚本做对照：**不给任何资料**，只让模型「根据你的知识回答」。

判据
----
答案里的百分比 / 编号，**在整个语料库里都不存在** = 纯编造。
本语料是虚构的企业私有文档 —— 通用模型不可能"知道"其中数值，说出来就是编的。

区分三种情况（别把"回避"当"通过"）：
    ① 给出了语料中不存在的数值        → **编造**
    ② 明确说"不确定" / 拒绝给数值      → 回避（不是编造，也不是"答对"）
    ③ 给出的数值恰好在语料库里        → 少见（本语料不该出现，出现要单独查）

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
结果文件 = os.path.join(输出目录, "无资料对照-结果.json")

def 取端口(默认="8000"):
    """允许 `--端口 8083` 覆盖（GPU 引擎不可用时切到 CPU 生成服务）。"""
    if "--端口" in sys.argv:
        i = sys.argv.index("--端口")
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return 默认


生成服务 = "http://127.0.0.1:%s" % 取端口()
提问模板 = ("请直接根据你的知识回答下面的问题。\n"
            "若你并不掌握该事实，请直接说「不确定」，**不要给出估计值或示例值**。\n\n"
            "问题：{问句}")

百分比模式 = re.compile(r"\d+(?:\.\d+)?\s*%")
编号模式 = re.compile(r"[A-Z]{2,}(?:-[A-Z0-9]+)+")
不确定词 = ("不确定", "无法确定", "不知道", "不掌握", "无从得知", "无法回答")


def 载入():
    路 = os.path.join(脚本目录, "实验-分块对照.py")
    规范 = importlib.util.spec_from_file_location("分块实验", 路)
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


def 问生成端(问句, 超时=120):
    载荷 = json.dumps({
        "messages": [{"role": "user", "content": 提问模板.format(问句=问句)}],
        "temperature": 0.1,
        "max_tokens": 200,
        "chat_template_kwargs": {"enable_thinking": False},
    }, ensure_ascii=False).encode("utf-8")
    请求 = urllib.request.Request(生成服务 + "/v1/chat/completions", data=载荷,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(请求, timeout=超时) as 响应:
        d = json.loads(响应.read().decode("utf-8"))
    return (d["choices"][0]["message"].get("content") or "").strip()


def 主流程():
    sys.stdout.reconfigure(errors="replace")
    模 = 载入()

    try:
        urllib.request.urlopen(生成服务 + "/health", timeout=5).read()
    except Exception as 错:
        print("✗ 生成服务未就绪（%s）：%s" % (生成服务, 错))
        return 1

    语料全文 = 去空格("".join(d["正文"] for d in 模.读语料()))

    print("=" * 96)
    print(" 无资料对照    不给任何资料，只问「根据你的知识回答」    %d 题" % len(模.查询集))
    print("=" * 96)
    print(" 判据：答案里的百分比/编号若**整个语料库里都不存在** = 编造")
    print()

    明细 = []
    起 = time.time()
    for 序号, (编号, 问句, 目标, 金标准, 类别) in enumerate(模.查询集, start=1):
        try:
            答案 = 问生成端(问句)
        except Exception as 错:
            答案 = "(生成失败：%s)" % 错
        数值 = 抽数值(答案)
        编造 = [v for v in 数值 if v not in 语料全文]
        回避 = any(w in 答案 for w in 不确定词) and not 数值
        明细.append({"编号": 编号, "类别": 类别, "问": 问句, "答案": 答案,
                     "数值": 数值, "编造": 编造, "回避": 回避})
        print("  [%2d/%d] %-5s %-8s 数值 %d，编造 %d%s"
              % (序号, len(模.查询集), 编号, 类别, len(数值), len(编造),
                 "（回避）" if 回避 else ""))

    print("\n      用时 %.1f s" % (time.time() - 起))

    库内 = [x for x in 明细 if x["类别"] != "库外"]
    总数值 = sum(len(x["数值"]) for x in 库内)
    总编造 = sum(len(x["编造"]) for x in 库内)
    编造题 = [x["编号"] for x in 库内 if x["编造"]]
    回避题 = [x["编号"] for x in 库内 if x["回避"]]

    print()
    print("=" * 96)
    print(" 表 1 · 无资料时的表现（库内 %d 题）" % len(库内))
    print("=" * 96)
    print("  给出数值总数               ：%d" % 总数值)
    print("  其中语料库中不存在（编造） ：%d" % 总编造)
    print("  编造题数                   ：%d / %d  %s"
          % (len(编造题), len(库内), "（" + "、".join(编造题) + "）" if 编造题 else ""))
    print("  明确回避（说'不确定'且无数值）：%d / %d  %s"
          % (len(回避题), len(库内), "（" + "、".join(回避题) + "）" if 回避题 else ""))
    print()
    if 总数值:
        print("  **无资料编造率 = %d / %d = %.1f%%**" % (总编造, 总数值, 100.0 * 总编造 / 总数值))
    else:
        print("  ⚠ 无资料时也没有给出任何数值 —— 对照不成立，需换判据（例如看定性断言）")

    print()
    print("=" * 96)
    print(" 表 2 · 逐题明细")
    print("=" * 96)
    print(" " + 模.补空格("题", 6) + 模.补空格("类别", 10) + 模.补空格("数值", 20)
          + 模.补空格("编造", 16) + "答案（截断）")
    print(" " + "-" * 92)
    for x in 明细:
        print(" " + 模.补空格(x["编号"], 6) + 模.补空格(x["类别"], 10)
              + 模.补空格("、".join(x["数值"]) or "—", 20)
              + 模.补空格("、".join(x["编造"]) or "—", 16)
              + 模.截断(x["答案"], 40))

    with open(结果文件, "w", encoding="utf-8") as 文件:
        json.dump({"说明": "无资料对照：不给资料，只让模型凭知识回答",
                   "汇总": {"数值总数": 总数值, "编造数": 总编造,
                            "编造率": (总编造 / 总数值) if 总数值 else None,
                            "编造题": 编造题, "回避题": 回避题},
                   "明细": 明细}, 文件, ensure_ascii=False, indent=2)
    print("\n 已落盘 %s" % 结果文件)
    return 0


if __name__ == "__main__":
    sys.exit(主流程())
