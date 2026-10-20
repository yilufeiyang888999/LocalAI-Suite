# -*- coding: utf-8 -*-
"""诊断：向量服务（8081）跨进程是否可复现。

判据：同一批文本、同样的分批方式，两个独立进程算出的向量指纹必须一致。
      不一致 → 服务侧存在状态相关性（多槽并行 / LCP 槽位复用），
      **任何在近似并列处的 R@1 都会在会话之间漂移**。
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


模 = 载入("实验-分块对照.py", "分块实验")
服务 = "http://127.0.0.1:%s" % (sys.argv[1] if len(sys.argv) > 1 else "8081")
print("服务 = %s" % 服务)

文本 = [
    "QR-QC-07-01 §4.2 S001 东莞精诚 · 专项分析\n8 月不良原因分布…钝化体系由六价铬改为三价铬",
    "JC-GY-2026-006 工艺变更通知单\n拟生效日期 2026-07-01，一号线钝化体系变更",
    "QP-07 检验作业指导书\n镀层厚度检测方法",
    "QW-DD-03 废水处理\n电镀废液收集",
    "QP-06 抽样方案\nAQL 取值",
    "JC-GY-2026-011 来料数据对照\n不良率 7.66%",
    "QR-SB-01-05 供应商审核记录\n3B 记法",
    "QP-08 测量系统分析\nGRR",
]

一次 = 模.取向量(服务, 文本)
再算 = 模.取向量(服务, 文本)

最大差 = max(abs(a - b) for 甲, 乙 in zip(一次, 再算) for a, b in zip(甲, 乙))
print("同进程内、同一批两次调用：最大分量差 = %.3e" % 最大差)

指纹 = ["%.12f" % sum(v) for v in 一次]
print("本次进程指纹（每条向量分量和）：")
for i, f in enumerate(指纹):
    print("  [%d] %s" % (i, f))

单条指纹 = []
for t in 文本:
    v = 模.取向量(服务, [t])[0]
    单条指纹.append("%.12f" % sum(v))
print("逐条单独请求的指纹：")
for i, f in enumerate(单条指纹):
    print("  [%d] %s   与批量一致: %s" % (i, f, "是" if f == 指纹[i] else "否 ← 分批影响结果"))
