#!/usr/bin/env python3
# sm_60 fp16 精度验证 —— 第三轮（最终版）
#
# 前两轮的教训：
#   第一轮：加了非标准字段 return_tokens=True → 两引擎对未知字段处理不同，
#           导致返回文本被截断的位置不同（Vulkan 414 / CUDA 436 字符），
#           **这是我引入的假阳性，不是精度问题**。
#   第二轮：/completion 不给 tokens 字段（需 return_tokens），比对空转。
#
# 本轮原则：**只用标准字段**，把变量降到最少。
#   · /completion 端点，raw prompt，不走 chat 模板
#   · temperature=0 + top_k=1 + seed 固定 = 纯贪心
#   · n_predict=512，只取 content，逐**字节**比对
#   · 三组：Vulkan / CUDA / CUDA+MMQ
#   · 额外加一组 CUDA 复跑，确认后端自洽
#
# 判据（这次是对的判据）：
#   · 若 CUDA 与 Vulkan 的文本**在相同长度下前 N 字节全同** → 无精度差异
#   · 若出现**中段差异**（两边长度相当但中间某处不同） → 确证精度问题
#   · 只比长度没意义（长度差可能来自 stop token 判定），
#     关键看**最长公共前缀**占全文比例

import base64
import json
import os
import subprocess
import time
import urllib.request

模型 = "/opt/llama.cpp/models/Qwen3.8-27B-UD-Q3_K_XL.gguf"
VULKAN = "/opt/llama-vulkan/llama-server"
CUDA = "/opt/llama-src-b10853/build/bin/llama-server"
端口 = 8000
上下文 = 4096
输出长度 = 512

# 定长 seed 文本：给模型一个确定性的续写起点
提示词 = "\n".join(str(i) for i in range(1, 201)) + "\n"


def 跑(命令, 超时=120):
    r = subprocess.run(["bash", "-lc", 命令], capture_output=True, timeout=超时)
    return r.stdout.decode("utf-8", errors="replace")


def 启动(引擎路径, 日志, 额外环境=""):
    跑("pkill -f llama-server ; sleep 3")
    目录 = 引擎路径.rsplit("/", 1)[0]
    脚本 = (
        "#!/bin/bash\n"
        f"cd {目录}\n"
        f"export LD_LIBRARY_PATH={目录}\n"
        f"{额外环境}"
        f"exec {引擎路径} --model {模型} --ctx-size {上下文} --batch-size 256 "
        "--ubatch-size 256 --port 8000 --host 0.0.0.0 --n-gpu-layers 99 "
        "--cache-type-k q8_0 --cache-type-v q8_0 "
        "--no-mmap --parallel 1 "
        f"> {日志} 2>&1\n"
    )
    open("/tmp/_eng_f.sh", "wb").write(base64.b64decode(base64.b64encode(脚本.encode())))
    os.chmod("/tmp/_eng_f.sh", 0o755)
    subprocess.Popen(["bash", "-lc", "setsid /tmp/_eng_f.sh < /dev/null > /dev/null 2>&1 &"])
    for _ in range(60):
        time.sleep(5)
        尾部 = 跑(f"tail -60 {日志} 2>/dev/null")
        if "listening" in 尾部:
            return True, 尾部
        真错误 = [L for L in 尾部.splitlines()
                 if "error" in L.lower() and "unused tensor" not in L.lower()
                 and "missing tensor" not in L.lower()]
        if 真错误:
            return False, 尾部
    return False, 尾部


def 生成():
    """纯标准字段，返回 (文本, 服务端生成 token 数)"""
    载荷 = {"prompt": 提示词,
           "n_predict": 输出长度,
           "temperature": 0,
           "top_k": 1,
           "seed": 12345,
           "cache_prompt": False}
    数据 = json.dumps(载荷).encode()
    请 = urllib.request.Request(f"http://127.0.0.1:{端口}/completion", data=数据,
                              headers={"Content-Type": "application/json"})
    响应 = json.loads(urllib.request.urlopen(请, timeout=1200).read())
    停因 = 响应.get("stop_type", "?")
    return 响应.get("content", ""), 响应.get("tokens_predicted", -1), 停因


def 一组(名称, 引擎路径, 日志, 额外环境="", 重复=1):
    print()
    print("=" * 72)
    print(f"  {名称}")
    print("=" * 72)
    成功, 尾部 = 启动(引擎路径, 日志, 额外环境)
    if not 成功:
        print("  ❌ 启动失败：")
        print("  " + 尾部[-800:].replace("\n", "\n  "))
        return []
    升 = [L.strip()[:100] for L in 尾部.splitlines()
         if any(k in L.lower() for k in ["cuda : archs", "vulkan0", "offloaded"])]
    for L in 升[:3]:
        print(f"  [启动] {L}")
    time.sleep(3)
    出 = []
    for i in range(重复):
        文本, 个数, 停因 = 生成()
        print(f"  ── 第 {i+1} 次：n_predict={个数}  文本={len(文本)} 字符  stop={停因}")
        print(f"     尾部: {文本[-60:]!r}")
        出.append(文本)
    跑("pkill -f llama-server ; sleep 3")
    return 出


print("=" * 72)
print("  第三轮（最终）：标准字段 / 纯贪心 / 逐字节比对")
print("  " + time.strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 72)
print(f"  prompt 长度: {len(提示词)} 字符 | n_predict={输出长度}")

锅 = {}
锅["vulkan"] = 一组("① Vulkan（fp32 基准）", VULKAN, "/tmp/f_vulkan.log")
锅["cuda"] = 一组("② CUDA 默认", CUDA, "/tmp/f_cuda.log")
锅["cuda2"] = 一组("③ CUDA 默认 · 复跑（自洽性）", CUDA, "/tmp/f_cuda2.log")
锅["mmq"] = 一组("④ CUDA + FORCE_MMQ=1", CUDA, "/tmp/f_mmq.log",
              "export GGML_CUDA_FORCE_MMQ=1\n")


def 比(甲名, 甲, 乙名, 乙):
    print()
    print(f"  ── {甲名}  ↔  {乙名}")
    if not 甲 or not 乙:
        print("     有一方失败，跳过")
        return
    a, b = 甲[0], 乙[0]
    print(f"     长度：{甲名}={len(a)}  {乙名}={len(b)} 字符")
    if a == b:
        print("     ✅ 逐字节完全相同")
        return
    最短 = min(len(a), len(b))
    前缀 = next((i for i in range(最短) if a[i] != b[i]), 最短)
    占比 = 前缀 / max(len(a), len(b))
    print(f"     最长公共前缀 = {前缀} 字符（占较长者的 {占比:.1%}）")
    if 前缀 == 最短:
        print("     ⚠️ 一方是另一方的前缀 —— **只是长度不同，内容无分歧**")
        print(f"        短的一方尾部: {a[-50:]!r}")
        print(f"        长的一方续写: {b[最短:最短+50]!r}")
    else:
        print(f"     ❌ 中段分歧！")
        print(f"        {甲名}: …{a[max(0,前缀-50):前缀+50]!r}")
        print(f"        {乙名}: …{b[max(0,前缀-50):前缀+50]!r}")


print()
print("=" * 72)
print("  比对")
print("=" * 72)
比("Vulkan", 锅.get("vulkan"), "CUDA", 锅.get("cuda"))
比("Vulkan", 锅.get("vulkan"), "CUDA+MMQ", 锅.get("mmq"))
比("CUDA", 锅.get("cuda"), "CUDA+MMQ", 锅.get("mmq"))
比("CUDA", 锅.get("cuda"), "CUDA复跑", 锅.get("cuda2"))

print()
print("=" * 72)
print("  判读")
print("=" * 72)
v, c, m = 锅.get("vulkan", []), 锅.get("cuda", []), 锅.get("mmq", [])
if v and c and m:
    if v[0] == c[0] == m[0]:
        print("  ✅ 三者逐字节完全相同")
        print("  → sm_60 上 CUDA 的 fp16 快速路径与 Vulkan fp32 **输出一致**")
        print("  → 本模型（Qwen3.8-27B Q3_K_XL）+ 本提示词下，**未重现 fp16 精度缺陷**")
    elif c[0] == m[0]:
        print("  ⚠️ CUDA 内部自洽（fp16 路径 == MMQ fp32 路径），但与 Vulkan 不同")
        print("  → 分歧不是 fp16 造成的（若是，CUDA 两条路径必然不同）")
        print("  → 可能是后端『停止条件判定』或『不同 kernel 的浮点归约顺序』")
    else:
        print("  ❌ CUDA 的 fp16 路径与 MMQ 路径都不同 → **fp16 精度缺陷确证**")
        print("  → 建议：Local-AI 在 sm_60 上对 CUDA 后端自动加 GGML_CUDA_FORCE_MMQ=1")
