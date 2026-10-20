# -*- coding: utf-8 -*-
"""提示词 × 评测题 污染检查 —— 保护 A-4 的「外部性」

## 为什么必须有这个脚本

A-4 这一项的全部价值，在于**评测题没进过提示词迭代**。
但改提示词时很容易顺手把失败题的原句当例子抄进去 —— 那就成了
「照着答案出题」，题目转绿不算本事，数字虚高、不可交付。

**本轮（包 9）真的发生了**：为了修「问『最高的哪家』丢维度」，
新加的规则 12 举了三个例子，其中

    · 「上个月不合格率最高的三家厂商是谁？」   ≈ 盲测 T01
    · 「八月份 IQC 不良率最高的是哪家？」       = 盲测 T02（一字不差）
    · 「上月不良率头名是谁？」                  = 盲测 T11（一字不差）

→ v3 跑到 87.0%（20/23），看着像大涨，其实含练题成分。
发现后把例子改成模板式表述，重跑 v4 才拿到可交付的数字。

**结论：改提示词之后、跑评测之前，先跑这个脚本。**

## 判据（阈值故意保守：宁可误报，不许漏报）

    ① 题面整句（去标点后 ≥ 6 字）原样出现在提示词里        → 🔴 题面级污染
    ② 题面去掉「的」之后整串出现                            → 🔴 同上
       （中文里多一个「的」不构成改写，等价于原句）
    ③ 题面与提示词的最长公共子串 ≥ 8 字                     → ⚠️ 片段级可疑

第 ③ 条要人工确认：它可能是**合法词表**（「厂商 / 供货商」这类
「X 是 Y 的口语说法」本来就必须写进提示词，属于词表知识、
能泛化到别的指标），也可能是抄了大半句。脚本不能替你判断，
但能保证**你不会漏看**。

## 退出码

    0 = 干净 ｜ 3 = 有污染（⚠️ 与「通过」可区分 —— 项目铁律：
    「降级/跳过必须与通过可区分」；退出码 1 留给脚本自身出错）
"""

import io
import json
import os
import sys

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

根 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(根, "scripts"))

import 语义层 as 语义          # noqa: E402
import 意图解析               # noqa: E402

题集 = os.path.join(根, "eval", "问数盲测集-40题.json")

# 评测脚本用的样例问题（保持与 评测-盲测集.py 一致，否则拼出来的
# 提示词与真实评测时不是同一份）
样例问题 = "样例问题：上月不良率最高的三家供应商"


def 去标点(s):
    return s.strip().strip("？?。.，,！!；;：: ")


def 读题():
    d = json.load(io.open(题集, encoding="utf-8"))
    题 = d["题目"] if isinstance(d, dict) and "题目" in d else d
    return [(t.get("编号"), t.get("类"), 去标点(t.get("问题") or "")) for t in 题]


def 最长公共子串(a, b):
    """返回 (长度, 内容)。中文短句用 O(n*m) DP 足够。"""
    if not a or not b:
        return 0, ""
    m = len(b)
    上 = [0] * (m + 1)
    最长, 末尾 = 0, 0
    for i in range(1, len(a) + 1):
        cur = [0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                cur[j] = 上[j - 1] + 1
                if cur[j] > 最长:
                    最长, 末尾 = cur[j], i
        上 = cur
    return 最长, a[末尾 - 最长:末尾]


def main(argv):
    提示文件 = argv[1] if len(argv) >= 2 and argv[0] == "--提示词文件" else None
    if 提示文件:
        提示 = io.open(提示文件, encoding="utf-8").read()
        来源 = "（从文件读：%s —— 用于回归验证本脚本真能抓到已知污染）" % 提示文件
    else:
        定义 = 语义.载入()
        提示 = 意图解析.构造提示(定义, 样例问题, None)
        来源 = "（由 scripts/意图解析.py 现场构造，与评测时同口径）"
    提示平 = 提示.replace("的", "")

    print("=" * 74)
    print("提示词 × 评测题 污染检查（A-4 外部性护栏）")
    print("=" * 74)
    print("题集：%s（%d 题）" % (os.path.relpath(题集, 根), len(读题())))
    print("提示词：%d 字符 %s" % (len(提示), 来源))
    print()

    硬命中, 软命中 = [], []
    for 编号, 类, 问 in 读题():
        if len(问) < 6:
            continue
        # ① 整句
        if 问 in 提示:
            硬命中.append((编号, 类, 问, "整句原样"))
            continue
        # ② 去「的」
        if len(问.replace("的", "")) >= 6 and 问.replace("的", "") in 提示平:
            硬命中.append((编号, 类, 问, "去「的」后原样"))
            continue
        # ③ 最长公共子串
        n, s = 最长公共子串(问, 提示)
        if n >= 8:
            软命中.append((编号, 类, 问, n, s))

    if 硬命中:
        print("🔴 题面级污染（%d 条）—— 等于照着答案出题，必须先改提示词再评测" % len(硬命中))
        for 编号, 类, 问, 方式 in 硬命中:
            print("   %s [%s] %s" % (编号, 类, 问))
            print("        命中方式：%s" % 方式)
        print()

    if 软命中:
        print("⚠️  片段级可疑（%d 条）—— 人工确认是「合法词表」还是「抄了大半句」" % len(软命中))
        for 编号, 类, 问, n, s in sorted(软命中, key=lambda x: -x[3]):
            print("   %s [%s] %s" % (编号, 类, 问))
            print("        与提示词共有 %d 字：%s" % (n, s))
        print("   说明：「厂商 / 供货商 / 厂家」这类**词表**是有意写进提示词的（属于词汇知识、"
              "能泛化到别的指标）；只有**整句或大半个句子**才算污染。")
        print()

    if not 硬命中 and not 软命中:
        print("✅ 干净：40 道题没有一道的题面（或 ≥8 字的片段）出现在提示词里。")
        print()

    print("-" * 74)
    if 硬命中:
        print("结论：🔴 有题面级污染 —— 评测结果不可当「外部性证据」。")
        print("      改法：把例子换成模板式表述（如「……最高的那家是谁 → 维度+降序+限制 1」），")
        print("      只教规则、不抄句子。改完重跑本脚本，再跑评测。")
        return 3
    if 软命中:
        print("结论：⚠️ 无题面级污染，但有 %d 条片段级可疑 —— 请人工过一遍上面几条。" % len(软命中))
        return 0
    print("结论：✅ 无污染，可以放心跑评测。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
