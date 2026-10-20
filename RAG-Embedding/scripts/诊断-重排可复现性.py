# -*- coding: utf-8 -*-
"""
诊断：重排服务的三个前提（作品一 · D3）
====================================

在拿 Q8_0 与 FP16 做量化对照之前，必须先确认**重排侧没有未受控变量**。
向量服务刚在「多槽 + LCP 槽位复用」上栽过（见 `诊断-向量可复现性.py`），
而重排服务启动日志同样写着 `n_slots = 4` —— **同一个风险面**。

本脚本回答三问：

  问 1  同一批 (问句, 块) 连打两次，分数是否逐位一致？        【可复现性】
  问 2  16 块整批打 vs 拆成 8+8 打，分数是否一致？            【批量无关性】
  问 3  Q8_0 与 FP16 在同一批上的分数差多大、名次是否同序？    【量化差异量级】

为什么问 2 是硬要求
-------------------
`实验-重排对照.py` 的核心设计建立在

    先给全库打分再取 Top-N 重排  ≡  只把 Top-N 送去打分

之上。这个等价性**只依赖于「逐对独立」**。若批量大小会改变分数，
那么「一次全库重排 + 离线派生」（批 16）与「真实上线时只打 Top-50」
（批不同）就会算出**不同的排序** —— 报告的方案表就是在拿一个上线时复现不了的
数字做推荐。所以必须实测，不能靠"理论上应该独立"。

用法：
    python scripts/诊断-重排可复现性.py                默认 8085 / 8086 各测一遍
    python scripts/诊断-重排可复现性.py --端口 8086     只测一个服务
"""

import importlib.util
import json
import os
import sys
import urllib.request

脚本目录 = os.path.dirname(os.path.abspath(__file__))
根目录 = os.path.dirname(脚本目录)
评测集文件 = os.path.join(根目录, "eval", "题目.json")

待测服务 = [("Q8_0", "http://127.0.0.1:8085"), ("FP16", "http://127.0.0.1:8086")]
探测题数 = 3      # 取前 3 题，每题全库打分（够用，且快）
块数上限 = 60     # 每题只取前 60 块，避免拖时间


def 取参数(名, 默认=None):
    if 名 in sys.argv:
        i = sys.argv.index(名)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return 默认


def 加载模块(文件名, 模块名):
    路 = os.path.join(脚本目录, 文件名)
    规范 = importlib.util.spec_from_file_location(模块名, 路)
    模块 = importlib.util.module_from_spec(规范)
    规范.loader.exec_module(模块)
    return 模块


def 打分(服务, 问句, 文档段):
    """返回与 文档段 等长同序的分数表（按返回的 index 回填，不靠顺序）。"""
    体 = json.dumps({"query": 问句, "documents": 文档段},
                    ensure_ascii=False).encode("utf-8")
    请求 = urllib.request.Request(服务 + "/rerank", data=体,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(请求, timeout=600) as 响应:
        数据 = json.loads(响应.read())
    表 = [None] * len(文档段)
    for 项 in 数据["results"]:
        表[int(项["index"])] = float(项["relevance_score"])
    缺 = [i for i, x in enumerate(表) if x is None]
    if 缺:
        raise RuntimeError("有 %d 块未拿到分数（如第 %d 块）" % (len(缺), 缺[0]))
    return 表


def 比分数(甲, 乙):
    """返回 (最大绝对差, 第几个不同, 相同个数, 总数)。"""
    最大 = 0.0
    首个 = None
    同 = 0
    for i, (a, b) in enumerate(zip(甲, 乙)):
        d = abs(a - b)
        if d > 最大:
            最大 = d
        if a == b:
            同 += 1
        elif 首个 is None:
            首个 = i
    return 最大, 首个, 同, len(甲)


def 比名次(甲, 乙):
    """两个分数表的名次是否同序 → (Top3 集合是否相同, 全序是否相同)。"""
    序甲 = sorted(range(len(甲)), key=lambda i: -甲[i])
    序乙 = sorted(range(len(乙)), key=lambda i: -乙[i])
    return set(序甲[:3]) == set(序乙[:3]), 序甲 == 序乙


def 跨服务比较(库内题, 用的块, 甲端口, 乙端口):
    """两个服务在同一批 (问句, 块) 上的分数对照。

    用途有二：
      · 同模型不同槽配置（如 8085 vs 8088）→ 判定**槽数是否影响该模型**；
      · 同槽配置不同量化（如 8085-Q8 vs 8086-FP16）→ 量化的质量代价。
    两种用法的关键都在于「同一批对、同一次调用序列」，所以放在同一段代码里。
    """
    甲服务 = "http://127.0.0.1:%s" % 甲端口
    乙服务 = "http://127.0.0.1:%s" % 乙端口
    print()
    print("=" * 96)
    print(" 跨服务比较：%s  vs  %s（同一批对）" % (甲端口, 乙端口))
    print("=" * 96)
    最大差 = 0.0
    top3同数 = 0
    全序同数 = 0
    for 题 in 库内题:
        编号, 问句 = 题.get("编号"), 题.get("问")
        甲 = 打分(甲服务, 问句, 用的块)
        乙 = 打分(乙服务, 问句, 用的块)
        差, 首, 同, 总 = 比分数(甲, 乙)
        最大差 = max(最大差, 差)
        top3同, 全序同 = 比名次(甲, 乙)
        top3同数 += 1 if top3同 else 0
        全序同数 += 1 if 全序同 else 0
        print("   %-5s  最大分量差 %.4f  逐位相同 %d/%d  Top3 同 %s  全序 %s"
              % (编号, 差, 同, 总, "✅" if top3同 else "✗",
                 "完全相同" if 全序同 else "**有挪动**"))
    print("   —— 小结：最大分量差 **%.4f**；Top-3 集合全同 %d/%d 题；全序全同 %d/%d 题"
          % (最大差, top3同数, len(库内题), 全序同数, len(库内题)))
    print()
    return 0


def 主流程():
    模 = 加载模块("实验-分块对照.py", "分块")
    模.设终端编码()

    文档列表 = 模.读语料()
    问题 = 模.校验金标准(文档列表)
    if 问题:
        print("✗ 金标准校验失败，拒跑：")
        for 行 in 问题:
            print(行)
        return 1

    with open(评测集文件, encoding="utf-8") as 文件:
        题目 = json.load(文件)
    题目列表 = 题目["题目"] if isinstance(题目, dict) else 题目
    库内题 = [x for x in 题目列表 if x.get("类别") != "库外"][:探测题数]

    # 块列表与检索基座完全一致（S13 软表保-1200）
    块列表 = []
    for 文档 in 文档列表:
        for 前缀, 正文 in 模.切标题结构(文档, 1200, False, 软表格=True):
            if len(正文.strip()) >= 8:
                块列表.append(前缀 + "\n" + 正文)
    用的块 = 块列表[:块数上限]

    print("=" * 96)
    print(" 重排服务诊断    块 %d（探测用前 %d）    题 %d    批 16"
          % (len(块列表), len(用的块), len(库内题)))
    print("=" * 96)
    print(" 探测题：" + "、".join("%s「%s」" % (x.get("编号"), x.get("问"))
                                  for x in 库内题))

    指定端口 = 取参数("--端口")
    比甲 = 取参数("--比甲")
    比乙 = 取参数("--比乙")
    if 比甲 and 比乙:
        return 跨服务比较(库内题, 用的块, 比甲, 比乙)

    服务表 = 待测服务
    if 指定端口:
        服务表 = [(("端口%s" % 指定端口),
                  "http://127.0.0.1:%s" % 指定端口)]

    # 缓存每个服务每题的第一遍分数，供跨服务比较
    分数缓存 = {}

    for 名, 服务 in 服务表:
        print()
        print("-" * 96)
        print(" 【%s】%s" % (名, 服务))
        print("-" * 96)
        最大同服务差 = 0.0
        批量不一致数 = 0
        名次不一致数 = 0
        总对数 = 0

        for 题 in 库内题:
            编号, 问句 = 题.get("编号"), 题.get("问")

            # --- 问 1：连打两次，比可复现性 ---
            甲 = 打分(服务, 问句, 用的块)
            乙 = 打分(服务, 问句, 用的块)
            差, 首, 同, 总 = 比分数(甲, 乙)
            最大同服务差 = max(最大同服务差, 差)

            # --- 问 2：整批 vs 拆半批，比批量无关性 ---
            半 = len(用的块) // 2
            拆 = 打分(服务, 问句, 用的块[:半]) + 打分(服务, 问句, 用的块[半:])
            差2, 首2, 同2, _ = 比分数(甲, 拆)
            批一致 = (差2 == 0.0)
            if not 批一致:
                批量不一致数 += 1

            # --- 名次（Top-3 / 全序）---
            top3同, 全序同 = 比名次(甲, 拆)
            if not 全序同:
                名次不一致数 += 1

            分数缓存[编号] = 甲
            总对数 += len(用的块)

            print("   %-5s  连打两次最大差 %.3e %s | 整批 vs 拆半最大差 %.3e %s | Top3同 %s 全序同 %s"
                  % (编号, 差, "✅" if 差 == 0.0 else "⚠",
                     差2, "✅" if 批一致 else "⚠",
                     "✅" if top3同 else "✗", "✅" if 全序同 else "✗"))

        print("   —— 小结：同一批连打两次最大差 **%.3e**；批量不一致 %d/%d 题；全序不一致 %d/%d 题"
              % (最大同服务差, 批量不一致数, len(库内题), 名次不一致数, len(库内题)))

    # --- 问 3：跨模型比较 ---
    if len(服务表) == 2:
        print()
        print("=" * 96)
        print(" 问 3 · Q8_0 vs FP16 的分数差异（同一批对）")
        print("=" * 96)
        最大跨模型差 = 0.0
        top3全同题 = 0
        全序同题 = 0
        for 题 in 库内题:
            编号 = 题.get("编号")
            甲 = 分数缓存.get(编号)
            if 甲 is None:
                continue
            乙 = 打分("http://127.0.0.1:8086", 题.get("问"), 用的块)
            差, 首, 同, 总 = 比分数(甲, 乙)
            最大跨模型差 = max(最大跨模型差, 差)
            top3同, 全序同 = 比名次(甲, 乙)
            top3全同题 += 1 if top3同 else 0
            全序同题 += 1 if 全序同 else 0
            print("   %-5s  最大分量差 %.4f  逐位相同 %d/%d  Top3 同 %s  全序 %s"
                  % (编号, 差, 同, 总, "✅" if top3同 else "✗",
                     "完全相同" if 全序同 else "**有挪动**"))
        print("   —— 小结：跨模型最大分量差 **%.4f**；Top-3 集合全同 %d/%d 题；全序全同 %d/%d 题"
              % (最大跨模型差, top3全同题, len(库内题), 全序同题, len(库内题)))

    print()
    return 0


if __name__ == "__main__":
    sys.exit(主流程())
