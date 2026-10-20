# -*- coding: utf-8 -*-
"""
查询改写（HyDE）—— 为混合检索的 D 通道生成「假设答案」（作品一 · D2）
=======================================================================

原理
----
P01「东莞精诚为什么 7 月起不良率突然升高」在 16 个分块方案下正确块排名 8~86、
相似度仅 0.475~0.522 —— 落在**库外区间**（库外均分 0.40，某些库外题最高到 0.71）。
根因：query 说的是「结果」（不良率升高），正确块说的是「原因」（钝化体系变更），
**两者字面零重叠**。

HyDE（Hypothetical Document Embeddings）：
    先让 LLM 按企业文档口吻**编一段可能含答案的片段**，再用它去检索 ——
    用词一下就对齐了（「不良率升高」→「钝化体系变更导致膜层致密性下降」）。
    假设答案**只用于检索，不进入任何判定**。

🔴 一条自制硬约束：禁止模型编造具体数值与编号
----------------------------------------------
否则会人为污染实验：假设答案若恰好写出语料里的真数字（如 7.66%），
那一块就会被**巧合地**召回、分数虚高，而归因却记在 HyDE 头上。
本实验要验证的是**概念对齐**，不是数值搬运。
脚本末尾另跑一遍「编造检测」：把假设答案里的数字与语料全文对表，列出可疑项。

用法
----
    python scripts/改写-假设答案.py            # 用默认 8000 端口的生成服务（Local-AI 引擎）
    python scripts/改写-假设答案.py --端口 8000
"""

import argparse
import importlib.util
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

脚本目录 = os.path.dirname(os.path.abspath(__file__))
根目录 = os.path.dirname(脚本目录)
输出目录 = os.path.join(根目录, "eval")
输出文件 = os.path.join(输出目录, "改写-假设答案.json")

默认端口 = 8000  # Local-AI 引擎端口（作品三常驻服务，复用可省一份 4B 权重显存）

提示词模板 = """你是制造业质量与工艺领域的文档助手。请针对下面的问题，写一段 150 字左右、可能出现在企业正式文件里的技术内容。

要求：
1. **必须写出具体的技术内容** —— 点出最可能涉及的工序名、工艺名、检验项目或管理环节
   （例如：钝化、镀锌层厚度、盐雾试验、附着力、来料检验、变更评审、安全库存）；
2. 只写实质内容。**禁止**写「需进一步核实」「建议与相关部门沟通确认」这类没有信息量的话；
3. 不要复述问题，不要说明你在推测；
4. 不要编造具体数值、编号、日期；需要量化时用「偏高」「不达标」「超出上限」这类表述；
5. 只写一段，不要分点、不要小标题、不要 Markdown 标记。

问题：{问句}"""


def 载入模块():
    路 = os.path.join(脚本目录, "实验-分块对照.py")
    规范 = importlib.util.spec_from_file_location("分块实验", 路)
    模块 = importlib.util.module_from_spec(规范)
    规范.loader.exec_module(模块)
    return 模块


def 调生成(问句, 端口, 超时=180):
    """调 llama-server 的 OpenAI 兼容 chat 端点。

    ⚠ 思考模式必须显式关掉：Spark-X2.5 的 chat template 里
    `enable_thinking = enable_thinking | default(true)` —— **默认开**，
    正文会全部落进 `reasoning_content`，`content` 是空串。
    `chat_template_kwargs` 是最干净的做法（不污染 prompt 文本）。
    首次请求另有预热开销（实测 prefill 0.7 t/s → 第二次 716 t/s），别拿第一条计时。
    """
    载荷 = json.dumps({
        "messages": [{"role": "user", "content": 提示词模板.format(问句=问句)}],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 400,
        "chat_template_kwargs": {"enable_thinking": False},
    }, ensure_ascii=False).encode("utf-8")
    请求 = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % 端口,
        data=载荷, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(请求, timeout=超时) as 响应:
        数据 = json.loads(响应.read().decode("utf-8"))
    正文 = 数据["choices"][0]["message"]["content"]
    return 清洗(正文)


def 清洗(正文):
    """去掉思考残留、Markdown 标记与多余空白，压成一段。"""
    正文 = re.sub(r"<think[^>]*>.*?</think[^>]*>", "", 正文, flags=re.S)
    正文 = re.sub(r"</?think[^>]*>", "", 正文)
    正文 = 正文.replace("**", "").replace("```", "")
    正文 = re.sub(r"^\s*[-*#>\d.]+\s*", "", 正文, flags=re.M)
    正文 = re.sub(r"\s+", " ", 正文)
    return 正文.strip()


def 编造检测(假设答案, 语料全文):
    """把假设答案里的数字/百分比与语料全文对表，列出语料中不存在的项。

    这是护栏，不是判据 —— 它是为了证明「HyDE 的收益不来自数字搬运」。
    """
    数字 = set(re.findall(r"\d+(?:\.\d+)?\s*%?", 假设答案))
    可疑 = sorted(d for d in 数字 if d.replace(" ", "") not in 语料全文.replace(" ", ""))
    return 可疑


def 主流程():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    解析 = argparse.ArgumentParser()
    解析.add_argument("--端口", type=int, default=默认端口)
    参数 = 解析.parse_args()

    模 = 载入模块()
    文档列表 = 模.读语料()
    语料全文 = "\n".join(d["正文"] for d in 文档列表)

    # 先探活，避免跑到一半才发现服务没起
    模型名 = "未知"
    try:
        urllib.request.urlopen("http://127.0.0.1:%d/health" % 参数.端口, timeout=5).read()
        元数据 = json.load(urllib.request.urlopen(
            "http://127.0.0.1:%d/props" % 参数.端口, timeout=10))
        模型名 = os.path.basename(元数据.get("model_path", "")) or "未知"
    except Exception as 错:
        print("✗ 生成服务未就绪（127.0.0.1:%d）：%s" % (参数.端口, 错))
        print("  先起服务：llama-server --model <XX.gguf> --port %d -c 4096 -ngl 99" % 参数.端口)
        return 1

    查询集 = 模.查询集
    print("=" * 92)
    print(" HyDE 查询改写    %d 条问题    生成端 127.0.0.1:%d" % (len(查询集), 参数.端口))
    print("=" * 92)

    改写表, 编造表 = {}, {}
    起 = time.time()
    for 序, (编号, 问句, 目标, 金标准, 类别) in enumerate(查询集, start=1):
        尝试 = 0
        while True:
            try:
                结果 = 调生成(问句, 参数.端口)
                break
            except Exception as 错:
                尝试 += 1
                if 尝试 >= 3:
                    print("  ✗ %s 生成失败：%s" % (编号, 错))
                    结果 = ""
                    break
                time.sleep(2)
        改写表[编号] = 结果
        可疑 = 编造检测(结果, 语料全文) if 结果 else []
        编造表[编号] = 可疑
        标记 = "  ⚠ 可疑数字 %s" % "、".join(可疑) if 可疑 else ""
        占位 = "%s %s" % (模.补空格(编号, 5), 模.截断(结果, 62))
        print("  [%2d/%d] %s  %5.1fs%s" % (序, len(查询集), 占位, time.time() - 起, 标记))
        起 = time.time()

    with open(输出文件, "w", encoding="utf-8") as 文件:
        json.dump({
            "模型": 模型名,
            "端口": 参数.端口,
            "说明": "HyDE 假设答案：只用于检索，不进入任何判定。",
            "提示词": 提示词模板,
            "改写": 改写表,
            "编造检查": 编造表,
        }, 文件, ensure_ascii=False, indent=2)

    print()
    print(" 已写入 %s" % 输出文件)
    可疑总数 = sum(len(v) for v in 编造表.values())
    if 可疑总数:
        print(" ⚠ 共 %d 处可疑数字（语料中不存在）—— 需人工确认是否影响结论：" % 可疑总数)
        for 编号, 项 in 编造表.items():
            if 项:
                print("    %s: %s" % (编号, "、".join(项)))
    else:
        print(" ✓ 编造检测：假设答案里的数字全部能在语料中找到（无凭空编造）")
    return 0


if __name__ == "__main__":
    sys.exit(主流程())
