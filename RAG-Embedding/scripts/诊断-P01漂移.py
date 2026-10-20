# -*- coding: utf-8 -*-
"""一次性诊断：查清 A 通道 P01 排名 21 / 22 漂移的真实来源。

三个待验假设：
    ① 服务端槽位复用（前置请求不同 → 向量末位比特不同）
    ② 请求累计条数（脚本里 P01 是第 ~46 个单条请求）
    ③ 批量 vs 单条（块向量一次性 290 条分批 vs 逐条）
"""

import importlib.util
import os
import sys

sys.stdout.reconfigure(errors="replace")
脚本目录 = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def 载入(文件名, 模块名):
    路 = os.path.join(脚本目录, 文件名)
    规范 = importlib.util.spec_from_file_location(模块名, 路)
    模块 = importlib.util.module_from_spec(规范)
    规范.loader.exec_module(模块)
    return 模块


检 = 载入("检索-混合.py", "混合检索")
模 = 检.载入分块模块()
检.模 = 模

文档列表 = 模.读语料()
块列表 = []
for 文档 in 文档列表:
    for 前缀, 正文 in 模.切标题结构(文档, 1200, False, 软表格=True):
        if len(正文.strip()) >= 8:
            块列表.append((文档["文号"], 前缀, 正文))
嵌入文本 = [前 + "\n" + 文 for _, 前, 文 in 块列表]

块向量 = 模.取向量(检.默认服务, 嵌入文本)
块模 = [模.求模长(v) for v in 块向量]

题目 = 模.查询集
问集 = [t[1] for t in 题目]
编号表 = [t[0] for t in 题目]
P01位置 = 编号表.index("P01")
P01问句 = 问集[P01位置]
片段 = [模.规范化(x) for x in 题目[P01位置][3]]
规范块 = [模.规范化(文) for _, _, 文 in 块列表]
正确块 = set(i for i in range(len(块列表))
             if 片段 and all(p in 规范块[i] for p in 片段))
print("P01 金标准 %s → 正确块序号 %s" % (题目[P01位置][3], sorted(正确块)))
print("P01 在查询集里的位置（0 基）= %d / %d" % (P01位置, len(问集)))
print()


def 算排名(查询文本):
    v = 模.取向量(检.默认服务, [查询文本])[0]
    m = 模.求模长(v)
    分 = [模.点积(v, bv) / (m * 模.求模长(bv)) for bv in 块向量]
    序 = sorted(range(len(分)), key=lambda i: -分[i])
    # 同时报告边界分差：正确块与它前一名差多少
    位 = min(序.index(b) + 1 for b in 正确块)
    前分 = 分[序[位 - 2]] if 位 >= 2 else float("nan")
    本分 = 分[序[位 - 1]]
    return 位, 本分 - 前分


位, 差 = 算排名(P01问句)
print("A. 直接算（P01 是本进程第一个单条请求）      → 名次 %2d   边界分差 %+.6f" % (位, 差))

for 数 in (10, 30, 45, 46, 47, 60, 66):
    for q in 问集[:数]:
        模.取向量(检.默认服务, [q])
    位, 差 = 算排名(P01问句)
    print("B. 先打 %2d 个单条请求，再算 P01              → 名次 %2d   边界分差 %+.6f" % (数, 位, 差))

print()
缓存 = {}
位序 = []
for q in 问集:
    序 = 检.向量排序(q, 块向量, 块模, 缓存)
    if q == P01问句:
        位序.append(检.求排名(序, 正确块))
print("C. 完全走 `检索-混合.py` 的流程（含缓存）      → 名次 %s" % 位序)
