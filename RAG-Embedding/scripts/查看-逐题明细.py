# -*- coding: utf-8 -*-
"""
查看分块对照实验的逐题明细
==========================

`实验-分块对照.py` 只输出汇总表。要回答「这条题为什么漏」，得看它每一条的排名。

用法
----
    python 查看-逐题明细.py                 # 打印全部方案的全部题
    python 查看-逐题明细.py S07 标题-1200    # 只看某个方案
    python 查看-逐题明细.py --失败             # 只看排名 > 3 或不可召回的题
    python 查看-逐题明细.py --题 P03           # 追问单条题在各方案下的排名

零第三方依赖（只用标准库）。
"""

import json
import os
import sys

根目录 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
结果路径 = os.path.join(根目录, "eval", "分块对照-结果.json")


def 设终端编码():
    for 流 in (sys.stdout, sys.stderr):
        if hasattr(流, "reconfigure"):
            流.reconfigure(errors="replace")


def 显示宽(文本):
    宽 = 0
    for 字符 in str(文本):
        宽 += 2 if ord(字符) > 0x2E80 else 1
    return 宽


def 补空格(文本, 目标宽):
    文本 = str(文本)
    return 文本 + " " * max(0, 目标宽 - 显示宽(文本))


def 主流程():
    设终端编码()
    if not os.path.exists(结果路径):
        print("[X] 找不到 " + 结果路径 + "，请先跑 实验-分块对照.py")
        return 2
    with open(结果路径, encoding="utf-8") as 文件:
        数据 = json.load(文件)

    参数 = sys.argv[1:]
    只看失败 = "--失败" in 参数
    指定题 = None
    if "--题" in 参数:
        指定题 = 参数[参数.index("--题") + 1]
    关键字 = [x for x in 参数 if not x.startswith("--")
              and (指定题 is None or x != 指定题)]

    方案们 = 数据["方案"]
    if 关键字:
        方案们 = [s for s in 方案们 if any(k in s["方案"] for k in 关键字)]
    if not 方案们:
        print("[X] 没有匹配的方案")
        return 2

    # 收集全部题号（保序）
    题号们 = []
    for s in 方案们:
        for 编, 类, 排, 分 in s["明细"]:
            if 编 not in 题号们:
                题号们.append(编)

    if 指定题:
        print("=" * 78)
        print(" 单题跨方案排名追踪：" + 指定题)
        print("=" * 78)
        print(" " + 补空格("方案", 24) + 补空格("类别", 10)
              + 补空格("排名", 10) + "正确项相似度")
        print(" " + "-" * 74)
        for s in 方案们:
            行 = [x for x in s["明细"] if x[0] == 指定题]
            if not 行:
                continue
            _, 类, 排, 分 = 行[0]
            排文 = "不可召回" if 排 is None else str(排)
            分文 = "—" if 分 is None else "%.4f" % 分
            print(" " + 补空格(s["方案"], 24) + 补空格(类, 10)
                  + 补空格(排文, 10) + 分文)
        return 0

    print("=" * 78)
    print(" 逐题明细（排名 1 = 首选；不可召回 = 没有任何一块含完整金标准）")
    print("=" * 78)
    for 题 in 题号们:
        行集 = []
        for s in 方案们:
            行 = [x for x in s["明细"] if x[0] == 题]
            if not 行:
                continue
            _, 类, 排, 分 = 行[0]
            if 只看失败 and 排 is not None and 排 <= 3:
                continue
            行集.append((s["方案"], 类, 排, 分))
        if not 行集:
            continue
        类 = 行集[0][1]
        print()
        print(" " + 题 + "  [" + 类 + "]")
        print(" " + " " * 3 + 补空格("方案", 24) + 补空格("排名", 8) + "相似度")
        for 名, _, 排, 分 in 行集:
            排文 = "不可召回" if 排 is None else str(排)
            分文 = "—" if 分 is None else "%.4f" % 分
            print(" " + " " * 3 + 补空格(名, 24) + 补空格(排文, 8) + 分文)
    return 0


if __name__ == "__main__":
    sys.exit(主流程())
