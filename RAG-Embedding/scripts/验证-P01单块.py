# -*- coding: utf-8 -*-
"""
P01 单块验证：这一块自己就能答完吗？（作品一 · D2）
====================================================

背景
----
昨天用「两块拼装」验证过 P01（喂原因块 + 结果块，生成端答出因果链）。
今天做证据覆盖率时发现 **A 通道第 3 名那一块**（`QR-QC-07-01` 的结论段）
本身含完整因果链：

    原因「一号线钝化体系由六价铬改为三价铬，7 月 1 日生效」
    结果「不良率 7 月跳升至 5.46%、8 月达 7.66%」
    因果「时间线吻合 → 根因明确为供应商的工艺变更」

那就不需要"两跳"了。本脚本把它**单独**喂给生成端：

    答对 → P01 是**单块可答**题；严格判据（连续串 + 同块）把第 3 名排除了 → 判据错
    答错 → P01 确实需要多跳拼装 → 检索层就得上多跳

判据（可判定，不靠"看起来对"）
------------------------------
    ① 答案含「钝化」                     —— 原因侧实体
    ② 答案含「六价铬」或「三价铬」        —— 变更细节
    ③ 答案含 5.46% / 7.66% 中的至少一个   —— 结果侧数值（随后逐字核对它在给定资料里）
    ④ 给定资料里没有的实体不得出现        —— 人工复核（脚本打印资料供对照）

零第三方依赖。
"""

import importlib.util
import json
import os
import re
import sys
import urllib.request

脚本目录 = os.path.dirname(os.path.abspath(__file__))
根目录 = os.path.dirname(脚本目录)
输出目录 = os.path.join(根目录, "eval")
生成服务 = "http://127.0.0.1:8000"      # Local-AI 引擎（作品三）
默认服务 = "http://127.0.0.1:8082"      # bge-m3 embedding（单槽，跨进程可复现）

问题 = "东莞精诚为什么 7 月起不良率突然升高"
证据条件 = [("六价铬", "三价铬"), ("不良率",)]   # 与题目.json 的 P01 证据组一致


def 载入模块():
    路 = os.path.join(脚本目录, "实验-分块对照.py")
    规范 = importlib.util.spec_from_file_location("分块实验", 路)
    模块 = importlib.util.module_from_spec(规范)
    规范.loader.exec_module(模块)
    return 模块


def 建块(模):
    块列表 = []
    for 文档 in 模.读语料():
        for 前缀, 正文 in 模.切标题结构(文档, 1200, False, 软表格=True):
            if len(正文.strip()) >= 8:
                块列表.append((文档["文号"], 前缀, 正文))
    return 块列表


def 找单块齐备(模, 块列表):
    """返回 (排名, 块序号) —— 最靠前的「一块同时满足全部证据条件」的块。"""
    嵌入文本 = [(前 + "\n" + 文) for _, 前, 文 in 块列表]
    块向量表 = 模.取向量(默认服务, 嵌入文本, 标签="  嵌入")
    块模表 = [模.求模长(v) for v in 块向量表]
    查询向量 = 模.取向量(默认服务, [问题])[0]
    查询模 = 模.求模长(查询向量)
    分 = [模.点积(查询向量, v) / (查询模 * w + 1e-12) for v, w in zip(块向量表, 块模表)]
    序 = sorted(range(len(分)), key=lambda i: -分[i])
    for 位次, i in enumerate(序, start=1):
        规范 = 模.规范化(块列表[i][2])
        if all(all(p in 规范 for p in 组) for 组 in 证据条件):
            return 位次, i, 分[i], 块向量表
    return None, None, None, 块向量表


def 问生成端(资料文本, 超时=180):
    提示 = (
        "你是制造业质量与工艺领域的文档助手。请**只根据下面提供的资料**回答用户的问题。\n\n"
        "要求：\n"
        "1. 回答控制在 180 字以内，直接给结论与因果链，不要复述资料；\n"
        "2. 引用资料中的具体数值时必须逐字照抄，不要改写、不要推算；\n"
        "3. 若资料不足以回答，直接说「资料不足」，不要补充资料之外的内容。\n\n"
        "【资料】\n" + 资料文本 + "\n\n"
        "【问题】" + 问题 + "\n"
    )
    载荷 = json.dumps({
        "messages": [{"role": "user", "content": 提示}],
        "temperature": 0.1,
        "max_tokens": 500,
        "chat_template_kwargs": {"enable_thinking": False},
    }, ensure_ascii=False).encode("utf-8")
    请求 = urllib.request.Request(生成服务 + "/v1/chat/completions", data=载荷,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(请求, timeout=超时) as 响应:
        return json.loads(响应.read().decode("utf-8"))


def 主流程():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    模 = 载入模块()

    print("=" * 96)
    print(" P01 单块验证：只喂「单块齐备」的那一块，看生成端能否答完因果链")
    print("=" * 96)

    try:
        urllib.request.urlopen(生成服务 + "/health", timeout=5).read()
    except Exception as 错:
        print("✗ 生成服务未就绪（%s）：%s" % (生成服务, 错))
        return 1

    块列表 = 建块(模)
    排名, 序, 相似度, _ = 找单块齐备(模, 块列表)
    if 排名 is None:
        print("✗ 语料中没有「一块同时满足全部证据条件」的块 —— 判据不成立，别下结论")
        return 1

    文号, 前缀, 正文 = 块列表[序]
    print("\n单块齐备块：**第 %d 名**（向量相似度 %.4f）" % (排名, 相似度))
    print("  文号  ：%s" % 文号)
    print("  章节  ：%s" % 前缀)
    print("  长度  ：%d 字" % len(正文))
    print("\n- - - 该块全文 - - -")
    print(正文)
    print("- - - 全文结束 - - -\n")

    资料 = "【%s %s】\n%s" % (文号, 前缀, 正文)
    print("→ 只喂这一块给生成端（%s）…" % 生成服务)
    应答 = 问生成端(资料)
    答案 = (应答["choices"][0]["message"].get("content") or "").strip()
    用时 = 应答.get("timings", {}).get("predicted_per_second", 0)
    print("   生成 %.1f t/s\n" % 用时)

    print("=" * 96)
    print(" 生成端回答（仅凭上述单块）")
    print("=" * 96)
    print(答案)

    # ---------- 判据 ----------
    规范答案 = 模.规范化(答案)
    规范资料 = 模.规范化(资料)
    检查 = [
        ("① 含「钝化」", "钝化" in 规范答案),
        ("② 含「六价铬」或「三价铬」", ("六价铬" in 规范答案) or ("三价铬" in 规范答案)),
        ("③ 含不良率数值 5.46% / 7.66% 之一",
         ("5.46" in 规范答案) or ("7.66" in 规范答案)),
    ]
    # 数值必须**逐字来自给定资料**（防幻觉）
    数值 = re.findall(r"\d+(?:\.\d+)?%", 规范答案)
    编造 = [x for x in 数值 if x.replace(" ", "") not in 规范资料.replace(" ", "")]

    print()
    print("=" * 96)
    print(" 判据")
    print("=" * 96)
    for 名, 通过 in 检查:
        print("  %s %s" % ("✅" if 通过 else "❌", 名))
    print("  %s 答案引用的百分比数值 [%s] 全部逐字来自给定资料"
          % ("✅" if not 编造 else "❌", "、".join(数值) or "无"))
    if 编造:
        print("     ⚠ 疑似编造：%s" % "、".join(编造))

    结论 = ("P01 是**单块可答**题 —— 严格判据（连续串 + 同块）把第 %d 名的正确答案排除了，"
            "问题在判据而不在检索" % 排名) if all(v for _, v in 检查) and not 编造 else \
           "单块不足以回答 —— P01 确实需要多跳证据拼装"
    print("\n 结论：%s" % 结论)

    with open(os.path.join(输出目录, "P01单块验证-结果.json"), "w", encoding="utf-8") as 文件:
        json.dump({"问题": 问题, "单块齐备排名": 排名, "文号": 文号, "章节": 前缀,
                   "向量相似度": round(相似度, 4), "块全文": 正文, "提示词": 资料,
                   "答案": 答案, "判据": {名: 值 for 名, 值 in 检查},
                   "引用数值": 数值, "疑似编造": 编造, "结论": 结论},
                  文件, ensure_ascii=False, indent=2)
    print("\n 已落盘 %s" % os.path.join(输出目录, "P01单块验证-结果.json"))
    return 0


if __name__ == "__main__":
    sys.exit(主流程())
