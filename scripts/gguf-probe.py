# -*- coding: utf-8 -*-
"""
GGUF 元数据探针
================
只解析 GGUF 文件头的「元数据段」与「张量信息段」，不加载张量数据，因此对
8GB 的文件也是秒级完成。

用途：
1. 为 Local-AI 的 GGUF 元数据解析模块提供真实字段清单（避免凭猜测写代码）
2. 为 KV 缓存公式提供真实层数 / KV 头数 / 全注意力层数
3. 统计实际量化分布，验证文件与标称量化档是否一致

用法：
    python gguf-probe.py                      # 扫默认模型目录
    python gguf-probe.py D:\\某目录            # 扫指定目录
    python gguf-probe.py 某文件.gguf           # 只查单个文件

输出：
    控制台摘要 + 同目录 UTF-8 报告文件 gguf探针结果.md
"""

import os
import struct
import sys
import traceback

魔数 = b"GGUF"

# 元数据类型枚举（GGUF 规范）
类型名称 = {
    0: "UINT8", 1: "INT8", 2: "UINT16", 3: "INT16", 4: "UINT32",
    5: "INT32", 6: "FLOAT32", 7: "BOOL", 8: "STRING", 9: "ARRAY",
    10: "UINT64", 11: "INT64", 12: "FLOAT64",
}
定长类型 = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
    10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}

# general.file_type 枚举（llama.cpp ggml_ftype）
文件类型名 = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 4: "Q4_1_SOME_F16",
    7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
    14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
    19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS",
    24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
    29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
    36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE",
}

# 单个张量的 ggml_type 枚举
张量类型名 = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 4: "Q4_2", 5: "Q4_3",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
    12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
    17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64",
    28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0",
    39: "MXFP4",
}


def 读串(文件):
    """GGUF 字符串：u64 长度前缀 + UTF-8 字节，无结尾 \\0。"""
    长度 = struct.unpack("<Q", 文件.read(8))[0]
    if 长度 > 64 * 1024 * 1024:
        raise ValueError(f"字符串长度异常：{长度}")
    return 文件.read(长度).decode("utf-8", errors="replace")


def 读值(文件, 类型码):
    """按类型读一个值。数组只保留前若干个元素，避免词表撑爆内存。"""
    if 类型码 == 8:  # STRING
        return 读串(文件)
    if 类型码 == 9:  # ARRAY
        元素类型 = struct.unpack("<I", 文件.read(4))[0]
        个数 = struct.unpack("<Q", 文件.read(8))[0]
        上限 = 64  # 只留前 64 个元素供审查，其余读掉不存
        if 元素类型 == 8:  # 字符串数组（词表、对话模板等）
            保留 = []
            for 序号 in range(个数):
                项 = 读串(文件)
                if 序号 < 上限:
                    保留.append(项)
            return {"_数组": True, "元素类型": "STRING", "元素个数": 个数,
                    "前若干个": 保留, "已截断": 个数 > 上限}
        if 元素类型 in 定长类型:
            格式, 字节数 = 定长类型[元素类型]
            保留 = []
            for 序号 in range(个数):
                项 = struct.unpack(格式, 文件.read(字节数))[0]
                if 序号 < 上限:
                    保留.append(项)
            return {"_数组": True, "元素类型": 类型名称.get(元素类型, str(元素类型)),
                    "元素个数": 个数, "前若干个": 保留, "已截断": 个数 > 上限}
        raise ValueError(f"不支持的数组元素类型：{元素类型}")
    if 类型码 in 定长类型:
        格式, 字节数 = 定长类型[类型码]
        return struct.unpack(格式, 文件.read(字节数))[0]
    raise ValueError(f"未知的元数据类型码：{类型码}")


def 解析(路径):
    """解析一个 GGUF 文件，返回字典。任何异常都收进「错误」字段，不上抛。"""
    结果 = {
        "文件": 路径,
        "文件名": os.path.basename(路径),
        "大小GB": round(os.path.getsize(路径) / (1024 ** 3), 3),
        "错误": None,
        "元数据": {},
        "张量统计": {},
        "张量数": 0,
    }
    try:
        with open(路径, "rb") as 文件:
            if 文件.read(4) != 魔数:
                结果["错误"] = "不是 GGUF 文件（magic 不匹配）"
                return 结果
            结果["版本"] = struct.unpack("<I", 文件.read(4))[0]
            结果["张量数"] = struct.unpack("<Q", 文件.read(8))[0]
            元数据个数 = struct.unpack("<Q", 文件.read(8))[0]

            for 序号 in range(元数据个数):
                起点 = 文件.tell()
                try:
                    键 = 读串(文件)
                    类型码 = struct.unpack("<I", 文件.read(4))[0]
                    结果["元数据"][键] = 读值(文件, 类型码)
                except Exception as 异常:
                    结果["错误"] = f"第 {序号} 项元数据（文件位置 {起点}）解析失败：{异常}"
                    return 结果

            # 张量信息段：统计各量化类型的张量个数
            统计 = {}
            总元素 = 0
            for 张量序号 in range(结果["张量数"]):
                起点 = 文件.tell()
                try:
                    读串(文件)  # 张量名，此处不需要
                    维数 = struct.unpack("<I", 文件.read(4))[0]
                    # 注意：GGUF 规范的张量维度是 uint64（ggml 的 ne[] 数组），
                    # 不是 uint32。按 u32 读会导致每个张量少读 4*n_dims 字节，
                    # 偏移逐项累积后必然错位。
                    维度 = struct.unpack(f"<{维数}Q", 文件.read(8 * 维数)) if 维数 else ()
                    类型码 = struct.unpack("<I", 文件.read(4))[0]
                    struct.unpack("<Q", 文件.read(8))[0]  # 数据偏移，不需要
                except Exception as 异常:
                    结果["错误"] = (f"第 {张量序号}/{结果['张量数']} 个张量信息"
                                 f"（文件位置 {起点}）解析失败：{异常}")
                    return 结果
                名称 = 张量类型名.get(类型码, f"未知({类型码})")
                统计[名称] = 统计.get(名称, 0) + 1
                if 维数:
                    乘积 = 1
                    for 值 in 维度:
                        乘积 *= 值
                    总元素 += 乘积
            结果["张量统计"] = 统计
            结果["总元素数"] = 总元素
    except Exception as 异常:
        结果["错误"] = f"{type(异常).__name__}: {异常}"
        结果["堆栈"] = traceback.format_exc()
    return 结果


def 取(元数据, 后缀, 前缀候选):
    """按候选架构前缀找形如 {架构}.{后缀} 的键。"""
    for 前缀 in 前缀候选:
        键 = f"{前缀}.{后缀}"
        if 键 in 元数据:
            return 键, 元数据[键]
    # 兜底：任意以该后缀结尾的键
    for 键, 值 in 元数据.items():
        if 键.endswith("." + 后缀):
            return 键, 值
    return None, None


def 算KV真值(结果, ctx, 元素字节=2):
    """按 KV 缓存的结构定义计算真实占用（GiB），并返回所用机制说明。

    KV 只与层结构有关，与参数量无关：
        per_token = n_kv_heads × (key_length + value_length) × bytes_per_element
    混合注意力模型的非全注意力层不随上下文增长：
        - 状态空间层（如 Qwen3.5 的 SSM）：KV 恒定，无 ctx 项
        - 滑动窗口层：上限为 min(ctx, window)
    """
    元数据 = 结果["元数据"]
    架构 = 元数据.get("general.architecture", "")

    层数 = 取(元数据, "block_count", [架构])[1] or 0
    kv头 = 取(元数据, "attention.head_count_kv", [架构])[1] or 0
    key长 = 取(元数据, "attention.key_length", [架构])[1] or 128
    val长 = 取(元数据, "attention.value_length", [架构])[1] or 128

    if isinstance(kv头, dict):  # 每层不同的数组形式：取最大值，避免低估
        候选 = [项 for 项 in kv头.get("前若干个", []) if isinstance(项, int)]
        kv头 = max(候选) if 候选 else 0

    滑窗模式 = 取(元数据, "attention.sliding_window_pattern", [架构])[1]
    滑窗大小 = 取(元数据, "attention.sliding_window", [架构])[1]
    全注意力间隔 = 取(元数据, "full_attention_interval", [架构])[1]
    # MTP（多 token 预测）附加层。它们计在 block_count 里，但 llama.cpp 不加载、
    # 不参与推理、也不产生 KV —— 用 block_count 推层结构时必须先扣掉。
    mtp层数 = 取(元数据, "nextn_predict_layers", [架构])[1]
    if not isinstance(mtp层数, int) or isinstance(mtp层数, bool) or mtp层数 < 0:
        mtp层数 = 0
    主干层数 = max(层数 - mtp层数, 0) or 层数  # 异常（MTP ≥ 总层数）时退回原口径

    if isinstance(滑窗模式, dict) and 滑窗模式.get("_数组"):
        项目 = 滑窗模式.get("前若干个", [])
        全注意力层 = sum(1 for 项 in 项目 if not 项)
        滑窗层数 = sum(1 for 项 in 项目 if 项)
        机制 = f"滑动窗口混合（{滑窗层数} 层窗口 {滑窗大小} + {全注意力层} 层全注意力）"
    elif isinstance(全注意力间隔, int) and 全注意力间隔 > 1:
        全注意力层 = -(-主干层数 // 全注意力间隔)  # 向上取整，宁可高估不可低估
        滑窗层数 = 0
        机制 = (f"状态空间混合（每 {全注意力间隔} 层含 1 层全注意力，"
                f"主干 {主干层数} 层里共 {全注意力层} 层）")
    else:
        全注意力层 = 层数
        滑窗层数 = 0
        机制 = "纯全注意力"

    每层每token字节 = kv头 * (key长 + val长) * 元素字节
    KV字节 = 全注意力层 * 每层每token字节 * ctx
    if 滑窗层数 and 滑窗大小:
        KV字节 += 滑窗层数 * 每层每token字节 * min(ctx, 滑窗大小)
    return {
        # 一律 GiB（1024³）：显存侧的每个量最终都要和 nvidia-smi 报的 MiB 相减，
        # 用 10⁹ 会引入约 7% 的系统偏差。与 service.rs / gguf.rs 同口径。
        "KV_GiB": KV字节 / (1024 ** 3),
        "机制": 机制,
        "全注意力层": 全注意力层,
        "滑窗层数": 滑窗层数,
        "每token_KB": 全注意力层 * 每层每token字节 / 1024,
    }


# 旧经验公式的系数（2026-09-24 起已废弃）：
# service.rs 现按 GGUF 真实层结构正算 KV，此处只用于生成历史偏差对照表。
APP_KV系数 = 0.000008


def 摘要(结果):
    """从元数据里抽出关键架构参数。"""
    元数据 = 结果["元数据"]
    架构 = 元数据.get("general.architecture", "?")
    前缀 = [架构, "llama", "qwen2", "qwen3", "qwen3next", "general"]

    字段 = {}
    for 中文名, 后缀 in [
        ("名称", "name"),
        ("层数", "block_count"),
        ("上下文长度", "context_length"),
        ("嵌入维度", "embedding_length"),
        ("注意力头数", "attention.head_count"),
        ("KV 头数", "attention.head_count_kv"),
        ("Key 维度", "attention.key_length"),
        ("Value 维度", "attention.value_length"),
        ("RoPE 基准", "rope.freq_base"),
        ("专家总数", "expert_count"),
        ("每 token 激活专家", "expert_used_count"),
        ("全注意力间隔", "full_attention_interval"),
        ("滑动窗口", "attention.sliding_window"),
        ("滑动窗口模式", "attention.sliding_window_pattern"),
        ("层类型数组", "attention.layer_types"),
        ("SSM 卷积核", "ssm.conv_kernel"),
        ("SSM 状态维度", "ssm.state_size"),
    ]:
        键, 值 = 取(元数据, 后缀, 前缀)
        if 键 is not None:
            字段[中文名] = 值

    文件类型码 = 元数据.get("general.file_type")
    if isinstance(文件类型码, int):
        文件类型文本 = 文件类型名.get(文件类型码, f"未知({文件类型码})")
    else:
        文件类型文本 = "未提供"

    结果["摘要"] = {
        "架构": 架构,
        "名称": 元数据.get("general.name", "?"),
        "参数量": 元数据.get("general.parameter_count"),
        "文件类型": 文件类型文本,
        "量化版本": 元数据.get("general.quantization_version"),
        **字段,
    }
    return 结果


def 生成报告(结果集, 输出路径):
    行 = []
    行.append("# GGUF 元数据探针结果\n")
    行.append("本文件由 `scripts/gguf-probe.py` 自动生成，用于校准 Local-AI 的模型元数据解析模块。\n")
    行.append("> 说明：只读取文件头的元数据段与张量信息段，不加载张量数据。\n")

    # ---- 总览表 ----
    行.append("\n## 一、总览\n")
    行.append("| 文件 | 大小GB | 架构 | 参数量 | 实测bpp | 文件类型 | 层数 | 上下文 | KV头数 | 张量数 |")
    行.append("|---|---|---|---|---|---|---|---|---|---|")
    for 结果 in 结果集:
        if 结果["错误"]:
            行.append(f"| {结果['文件名']} | {结果['大小GB']} | 解析失败 | | | | | | | |")
            continue
        摘 = 结果["摘要"]
        参数总数 = 结果.get("总元素数") or 0
        if 参数总数:
            参数文本 = f"{参数总数/1e9:.3f}B"
            bpp文本 = f"{结果['大小GB'] * (1024 ** 3) / 参数总数:.3f}"
        else:
            参数文本 = "未提供"
            bpp文本 = "-"
        行.append(
            f"| {结果['文件名']} | {结果['大小GB']} | {摘['架构']} | {参数文本} | {bpp文本} | "
            f"{摘['文件类型']} | {摘.get('层数','-')} | {摘.get('上下文长度','-')} | "
            f"{摘.get('KV 头数','-')} | {结果['张量数']} |"
        )

    # ---- KV 真值对照 ----
    行.append("\n## 二、KV 缓存真值 vs 旧经验公式（历史对照）\n")
    行.append("> 2026-09-24 起 `service.rs` 已改为**按 GGUF 真实层结构正算** KV，")
    行.append("> 下表保留旧公式的偏差记录，用于说明当初为什么必须改。\n")
    行.append("旧公式（已废弃，仅作对照）：\n")
    行.append("```text")
    行.append("KV = 0.000008 × active_params(B) × ctx × kv_factor")
    行.append("```\n")
    行.append("它把 KV 与**参数量**绑定，但 KV 的真实结构只与层结构有关：\n")
    行.append("```text")
    行.append("per_token = n_kv_heads × (key_length + value_length) × bytes_per_element")
    行.append("```\n")
    行.append("下表按 fp16 KV（2 字节/元素）计算，上下文取 32768，体积单位 GiB：\n")
    行.append("| 模型 | 注意力机制 | 全注意力层 | 真实 KB/token | 旧公式 KB/token | 偏差 | 32K 真实 KV | 32K 旧公式估值 |")
    行.append("|---|---|---|---|---|---|---|---|")
    for 结果 in 结果集:
        if 结果["错误"]:
            continue
        参数量 = (结果.get("总元素数") or 0) / 1e9
        if 参数量 <= 0:
            参数量 = 结果["大小GB"] * (1024 ** 3) / 4e9  # 兜底：按 4 字节/参数粗估
        真值 = 算KV真值(结果, 32768)
        # 旧公式给的是 GiB/token，换算成 KB 要乘 1024×1024
        公式每token_KB = APP_KV系数 * 参数量 * 1024 * 1024
        比值 = 真值["每token_KB"] / 公式每token_KB if 公式每token_KB else 0
        行.append(
            f"| {结果['文件名']} | {真值['机制']} | {真值['全注意力层']} | "
            f"{真值['每token_KB']:.1f} | {公式每token_KB:.1f} | "
            f"**{比值:.2f}×** | {真值['KV_GiB']:.3f} GiB | {APP_KV系数 * 参数量 * 32768:.3f} GiB |"
        )
    行.append("\n> 偏差 > 1 表示旧公式**低估** KV —— 有显存溢出风险（表现为"
              "「启动成功但极慢」，即 llama.cpp 把 KV 溢出到共享内存）；")
    行.append("> 偏差 < 1 表示**高估** —— 参数偏保守，上下文开不大、显存被浪费。\n")

    # ---- 逐文件详情 ----
    行.append("\n## 三、逐文件详情\n")
    for 结果 in 结果集:
        行.append(f"\n### {结果['文件名']}\n")
        if 结果["错误"]:
            行.append(f"**解析失败**：{结果['错误']}\n")
            continue
        行.append(f"- 路径：`{结果['文件']}`")
        行.append(f"- 大小：{结果['大小GB']} GB　GGUF 版本：v{结果.get('版本')}　张量数：{结果['张量数']}")

        摘 = 结果["摘要"]
        行.append(f"- 架构：`{摘['架构']}`　名称：{摘['名称']}　文件类型：{摘['文件类型']}")

        参数量 = 结果.get("总元素数") or 0
        if 参数量:
            每参数字节 = 结果["大小GB"] * (1024 ** 3) / 参数量
            行.append(f"- 参数量：{参数量:,}（{参数量/1e9:.3f}B）　"
                      f"实测 {每参数字节:.3f} 字节/参数")
        else:
            行.append("- 参数量：无法从张量维度累加")

        # KV 结构：这是混合注意力的关键判据
        kv头 = 摘.get("KV 头数")
        if isinstance(kv头, dict) and kv头.get("_数组"):
            行.append(f"- **KV 头数为数组**（共 {kv头['元素个数']} 项）："
                      f"前若干个 = {kv头['前若干个']}")
            零值 = sum(1 for x in kv头["前若干个"] if x == 0)
            行.append(f"  - 其中为 0 的有 {零值} 项 → 这些层不存 KV，即线性注意力层")
        else:
            行.append(f"- KV 头数：{kv头}（标量，即所有层相同）")

        层类型 = 摘.get("层类型数组")
        if isinstance(层类型, dict) and 层类型.get("_数组"):
            项目 = 层类型["前若干个"]
            行.append(f"- **层类型数组**（共 {层类型['元素个数']} 项）：{项目[:20]}")
            全注意力 = sum(1 for x in 项目 if "full" in str(x).lower())
            行.append(f"  - 前 {len(项目)} 项中全注意力层 {全注意力} 项")

        for 键 in ["全注意力间隔", "滑动窗口", "滑动窗口模式", "SSM 卷积核", "SSM 状态维度"]:
            if 键 in 摘:
                行.append(f"- {键}：{摘[键]}")

        行.append(f"- 张量类型分布：{结果['张量统计']}")

        真值 = 算KV真值(结果, 32768)
        行.append(f"- **KV 真值（32K / fp16）**：{真值['KV_GiB']:.3f} GiB"
                  f"（{真值['每token_KB']:.1f} KB/token）　机制：{真值['机制']}")

    # ---- 待人工审查的键 ----
    行.append("\n## 四、全部元数据键（供审查混合注意力结构）\n")
    行.append("以下列出所有与层结构 / 注意力 / 专家 / 状态空间相关的键，"
              "用于确认是否还有未处理的混合注意力字段。\n")
    关键词 = ("attention", "layer", "block", "expert", "ssm", "conv", "rope", "sliding", "recurrent")
    for 结果 in 结果集:
        if 结果["错误"]:
            continue
        行.append(f"\n### {结果['文件名']}\n")
        行.append("```")
        for 键 in sorted(结果["元数据"]):
            if any(词 in 键.lower() for 词 in 关键词):
                值 = 结果["元数据"][键]
                if isinstance(值, dict) and 值.get("_数组"):
                    值 = f"<数组 {值['元素个数']} 项，前几个={值['前若干个'][:8]}>"
                行.append(f"{键} = {值}")
        行.append("```")

    # ---- 全部键名清单 ----
    行.append("\n## 五、元数据键名全清单\n")
    for 结果 in 结果集:
        if 结果["错误"]:
            continue
        行.append(f"\n**{结果['文件名']}**（共 {len(结果['元数据'])} 个键）\n")
        行.append("```")
        for 键 in sorted(结果["元数据"]):
            行.append(键)
        行.append("```")

    with open(输出路径, "w", encoding="utf-8") as 文件:
        文件.write("\n".join(行) + "\n")


def 主函数():
    参数 = sys.argv[1:]
    默认目录 = r"D:\workspace\10-LocalAI-Installer\models"

    目标文件 = []
    if 参数:
        for 项 in 参数:
            if os.path.isfile(项) and 项.lower().endswith(".gguf"):
                目标文件.append(项)
            elif os.path.isdir(项):
                for 根, _, 文件名集 in os.walk(项):
                    for 名 in 文件名集:
                        if 名.lower().endswith(".gguf"):
                            目标文件.append(os.path.join(根, 名))
            else:
                print(f"跳过不存在的路径：{项}")
    else:
        for 根, _, 文件名集 in os.walk(默认目录):
            for 名 in 文件名集:
                if 名.lower().endswith(".gguf"):
                    目标文件.append(os.path.join(根, 名))

    目标文件.sort()
    if not 目标文件:
        print("没有找到 .gguf 文件。用法：python gguf-probe.py <目录或文件>")
        return 1

    结果集 = []
    for 路径 in 目标文件:
        print(f"解析 {os.path.basename(路径)} ...")
        结果集.append(摘要(解析(路径)))

    输出路径 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gguf探针结果.md")
    生成报告(结果集, 输出路径)
    print(f"\n报告已生成：{输出路径}")

    # 控制台只打极简摘要
    for 结果 in 结果集:
        if 结果["错误"]:
            print(f"  [失败] {结果['文件名']}：{结果['错误']}")
        else:
            摘 = 结果["摘要"]
            print(f"  [成功] {结果['文件名']} | {摘['架构']} | {摘['文件类型']} | "
                  f"层数={摘.get('层数','-')} | 上下文={摘.get('上下文长度','-')}")
    return 0


if __name__ == "__main__":
    sys.exit(主函数())
