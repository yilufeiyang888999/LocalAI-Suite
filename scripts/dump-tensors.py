# -*- coding: utf-8 -*-
"""
dump-tensors.py —— 临时诊断脚本：把 GGUF 的张量名清单打出来

用途：llama.cpp 报 "missing tensor 'blk.N.xxx'" 时，用它确认
      文件里到底有没有这个张量、以及同族模型之间的命名差异。

复用 gguf-probe.py 的 读串 / 读值，不另起一套解析实现
（项目约定：同一格式有多个实现时必须交叉验证）。
"""
import importlib.util
import os
import struct
import sys

探针路径 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gguf-probe.py")
_spec = importlib.util.spec_from_file_location("gguf探针", 探针路径)
探针 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(探针)


def 读张量表(路径):
    """返回 [(名称, 类型码, 维度元组), ...]"""
    with open(路径, "rb") as 文件:
        if 文件.read(4) != 探针.魔数:
            raise ValueError("不是 GGUF 文件")
        struct.unpack("<I", 文件.read(4))[0]
        张量数 = struct.unpack("<Q", 文件.read(8))[0]
        元数据个数 = struct.unpack("<Q", 文件.read(8))[0]
        for _ in range(元数据个数):
            探针.读串(文件)
            类型码 = struct.unpack("<I", 文件.read(4))[0]
            探针.读值(文件, 类型码)
        表 = []
        for _ in range(张量数):
            名称 = 探针.读串(文件)
            维数 = struct.unpack("<I", 文件.read(4))[0]
            维度 = struct.unpack(f"<{维数}Q", 文件.read(8 * 维数)) if 维数 else ()
            类型码 = struct.unpack("<I", 文件.read(4))[0]
            struct.unpack("<Q", 文件.read(8))[0]
            表.append((名称, 类型码, 维度))
    return 表


def 取层号(名称):
    if not 名称.startswith("blk."):
        return None
    段 = 名称.split(".")
    try:
        return int(段[1])
    except (IndexError, ValueError):
        return None


def 报告(路径):
    表 = 读张量表(路径)
    出 = []
    出.append(f"### {os.path.basename(路径)}")
    出.append(f"张量总数：{len(表)}")
    层号集 = sorted({n for n in (取层号(x[0]) for x in 表) if n is not None})
    出.append(f"层号范围：{层号集[0]} ~ {层号集[-1]}（共 {len(层号集)} 层）")

    # 含 ssm 的张量与它们所在的层
    ssm = [(取层号(n), n) for n, _, _ in 表 if "ssm" in n]
    出.append(f"含 'ssm' 的张量：{len(ssm)} 个，分布在层 "
             f"{sorted({l for l, _ in ssm if l is not None})}")
    # 含 conv1d 的（不限 ssm 前缀）
    c1 = [(取层号(n), n) for n, _, _ in 表 if "conv1d" in n or "conv_kernel" in n]
    出.append(f"含 'conv1d' 的张量：{len(c1)} 个")
    for l, n in c1[:6]:
        出.append(f"    {n}")
    # 每层张量名模板（取三种代表性层：首个全注意力层、首个 SSM 层、最后一层）
    出.append("")
    出.append("按层名后缀归类的模板（取最后 4 层逐一列出）：")
    组 = {}
    for 名, 码, 维 in 表:
        l = 取层号(名)
        if l is None:
            continue
        组.setdefault(l, []).append((名, 码, 维))
    for l in sorted(组)[-4:]:
        出.append(f"  blk.{l}（{len(组[l])} 个张量）：")
        for 名, 码, 维 in 组[l]:
            后缀 = 名.split(".", 2)[2] if 名.count(".") >= 2 else 名
            出.append(f"     -{后缀:32} {探针.张量类型名.get(码, 码):8} {维}")
    # 最后一层之后还有没有别的（nextn）
    出.append("")
    尾部 = [n for n, _, _ in 表 if 取层号(n) is None]
    出.append(f"非 blk.N.* 命名的张量：{尾部}")
    return "\n".join(出)


def 主函数():
    路径表 = sys.argv[1:]
    if not 路径表:
        print("用法：python dump-tensors.py 文件1.gguf [文件2.gguf ...]")
        return 1
    块 = []
    for 路径 in 路径表:
        try:
            块.append(报告(路径))
        except Exception as 异常:
            块.append(f"### {os.path.basename(路径)}\n解析失败：{异常!r}")
    文本 = "\n\n".join(块)
    输出 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "张量清单.md")
    with open(输出, "w", encoding="utf-8") as f:
        f.write(文本 + "\n")
    print(f"已写 {输出}（{len(文本)} 字符）")
    return 0


if __name__ == "__main__":
    sys.exit(主函数())
