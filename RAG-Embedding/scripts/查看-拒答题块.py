# -*- coding: utf-8 -*-
"""
查看「触发类2 的题」的块级证据（作品一 · D4）
=============================================

`实验-拒答规则.py` 测出类2（Top-3 出现库外文档号）**误伤 4/60 库内题**：

    L03  记录编号能反查出归属的程序文件吗        → QP-04
    L05  提供给客户的非受控文件更新后要更换吗     → QW-WM-03
    P05  宁波恒力为什么被要求提交交付改善计划     → QW-WM-03
    P06  向欧盟出口的表面处理件为什么必须确认钝化体系是三价铬 → QW-WM-03

而真正该被拦的 O03（电镀废液具体怎么处理）也是同一形态（QW-DD-05 等）。

**形态一样，但一个该答、一个该拒** —— 所以要看块级证据：
库外号出现在 Top-1（答案的直接依据）还是 Top-3（顺带提及）？
这决定「类2」能不能靠**收窄到 Top-1** 救回来。

用法：
    python scripts/查看-拒答题块.py                       # 默认看 4 误伤 + O03
    python scripts/查看-拒答题块.py --题 L03 O03 T01
"""

import importlib.util
import json
import os
import re
import sys

脚本目录 = os.path.dirname(os.path.abspath(__file__))
根目录 = os.path.dirname(脚本目录)
输出目录 = os.path.join(根目录, "eval")
题目文件 = os.path.join(输出目录, "题目.json")

向量服务 = "http://127.0.0.1:8082"
重排服务 = "http://127.0.0.1:8085"
池深 = 50


def 载入(文件名, 别名):
    路 = os.path.join(脚本目录, 文件名)
    规范 = importlib.util.spec_from_file_location(别名, 路)
    模块 = importlib.util.module_from_spec(规范)
    规范.loader.exec_module(模块)
    return 模块


def 去空白(文本):
    return re.sub(r"\s+", " ", 文本).strip()


def 主流程():
    sys.stdout.reconfigure(errors="replace")
    看题 = ["L03", "L05", "P05", "P06", "O03", "T01"]
    if "--题" in sys.argv:
        i = sys.argv.index("--题")
        看题 = [x for x in sys.argv[i + 1:] if not x.startswith("--")]

    检 = 载入("检索-混合.py", "检索")
    模 = 检.载入分块模块()
    检.模 = 模
    重 = 载入("实验-重排对照.py", "重排")

    文档列表 = 模.读语料()
    块列表 = []
    for 文档 in 文档列表:
        for 前缀, 正文 in 模.切标题结构(文档, 1200, False, 软表格=True):
            if len(正文.strip()) >= 8:
                块列表.append((文档["文号"], 前缀, 正文))
    嵌入文本 = [(前 + "\n" + 文) for _, 前, 文 in 块列表]
    库内文号 = {文档["文号"] for 文档 in 文档列表}
    BM25索引 = 检.建BM25索引(嵌入文本)

    题集 = {t["编号"]: t for t in json.load(open(题目文件, encoding="utf-8"))["题目"]}

    print("=" * 100)
    print(" 类2 触发题的块级证据    （库内文号 %d 份）" % len(库内文号))
    print("=" * 100)

    for 编号 in 看题:
        题 = 题集.get(编号)
        if not 题:
            print("  ✗ 无此题 %s" % 编号)
            continue
        问句 = 题["问"]
        序B, _ = 检.BM25排序(问句, BM25索引)
        池 = 序B[:池深]
        分 = 重.重排打分(重排服务, 问句,
                        [(块列表[i][1] + "\n" + 块列表[i][2]) for i in 池], 标签=编号)
        分表 = {i: s for i, s in zip(池, 分)}
        序 = sorted(池, key=lambda i: -分表[i])

        print()
        print("-" * 100)
        print(" %s  [%s]  「%s」   最高分 %.2f" % (编号, 题["类别"], 问句, 分表[序[0]]))
        for 位, i in enumerate(序[:5], start=1):
            文号, 前缀, 正文 = 块列表[i]
            段 = 嵌入文本[i]
            引号 = sorted(检.抽文档号(段) - 库内文号)
            标 = ("🔴 " + "、".join(引号)) if 引号 else ""
            章节 = 前缀.split(">")[-1].strip() if ">" in 前缀 else 前缀
            print("   Top-%d  %.2f  %-12s %-34s %s"
                  % (位, 分表[i], 文号, 去空白(章节)[:34], 标))
            if 位 == 1:
                print("         正文：%s" % 去空白(正文)[:150])
    return 0


if __name__ == "__main__":
    sys.exit(主流程())
