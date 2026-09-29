#!/usr/bin/env python3
# sm_60 fp16 精度验证 —— 在 P100 本机运行（不经 ssh）
#
# 背景：P100 = Pascal sm_60，无原生 fp16 加速。上游已知 fp16 路径在 sm_60 上
#      有精度缺陷（约每 29 个 token 错一个，且**不报错**，静默跑偏）。
#      Vulkan 后端在 Pascal 上只能 fp32 计算，所以天然不受影响；
#      CUDA 后端会走 fp16 快速路径 —— 这正是要验的。
#
# 设计要点：
#   1. **贪心解码全程锁死**：temperature=0 → 无采样随机性；
#      固定 prompt + 固定 max_tokens + cache_prompt=False → 三次运行输入的
#      token 序列逐字节相同。**否则比对无意义**（这是本实验最大的失效点）。
#   2. **三步对照，而不是两两比对**：
#        Vulkan（Pascal 上恒 fp32）  → 作为「基准」
#        CUDA  默认（走 fp16）        → 待测
#        CUDA  GGML_CUDA_FORCE_MMQ=1  → 强制走 MMQ（fp32 计算）
#      判据：若「CUDA 默认」偏离基准、「CUDA+MMQ」贴回基准
#            → fp16 缺陷确证 + 附带绕开手段。
#   3. **定位首次分歧**：给出第一个不同的字节位置，并打印分歧点上下文。
#      若首次分歧出现在约 29 token 的倍数附近 → 与上游「每 29 个 token」吻合。
#   4. **同进程内多次生成**：除三组对照外，对 CUDA 默认组再重复一次，
#      确认「同一后端重复运行是自洽的」（排除偶发），这是判据的基线噪声。
#
# 用法：python3 p100-fp16-precision.py
# 预计耗时：5 组 × （启动约 10s + 生成约 30s）≈ 4 分钟

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
上下文 = 4096          # 精度实验不需要长上下文，短上下文启动快、省显存
输出长度 = 512         # 要跑够长才能暴露「每 29 token 一错」的累积效应

# 提示词要求它做长文本续写 —— 我们要的是**确定性的长 token 序列**，
# 不是有意义的答案。重复计数是最理想的选择：模型几乎不可能中途"换思路"，
# 贪心解码下必然是一条极长的确定序列。
提示词 = "Count from 1 to 400, one number per line. Output only the numbers."


def 跑(命令, 超时=120):
    结果 = subprocess.run(["bash", "-lc", 命令], capture_output=True, timeout=超时)
    return 结果.stdout.decode("utf-8", errors="replace")


def 显存():
    return 跑("nvidia-smi --query-gpu=memory.used --format=csv,noheader").strip()


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
        "--flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 "
        "--no-mmap --parallel 1 --metrics "
        f"> {日志} 2>&1\n"
    )
    编码 = base64.b64encode(脚本.encode()).decode()
    open("/tmp/_eng_prec.sh", "wb").write(base64.b64decode(编码))
    os.chmod("/tmp/_eng_prec.sh", 0o755)
    subprocess.Popen(["bash", "-lc",
                      "setsid /tmp/_eng_prec.sh < /dev/null > /dev/null 2>&1 &"])

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
    """一次确定性生成，返回 (文本, 服务端返回的 token 列表, 用量)"""
    载荷 = {"model": "m",
           "messages": [{"role": "user", "content": 提示词}],
           "max_tokens": 输出长度,
           "temperature": 0,
           "top_k": 1,              # 贪心：双保险，杜绝采样随机
           "seed": 12345,           # 固定种子，进一步锁定
           "cache_prompt": False,
           "return_tokens": True}   # 关键：拿 token id 序列，而不是字符串
    数据 = json.dumps(载荷).encode()
    请 = urllib.request.Request(
        f"http://127.0.0.1:{端口}/v1/chat/completions", data=数据,
        headers={"Content-Type": "application/json"})
    响应 = json.loads(urllib.request.urlopen(请, timeout=1200).read())
    文本 = 响应["choices"][0]["message"]["content"]
    # llama-server 在 return_tokens 下可能把 token 放在不同字段，逐处兜底
    令牌 = (响应.get("tokens")
            or 响应["choices"][0].get("tokens")
            or 响应.get("choices", [{}])[0].get("tokens")
            or [])
    return 文本, 令牌, 响应["usage"]


def 比对(基准名, 基准文本, 基准令牌, 待测名, 待测文本, 待测令牌):
    print(f"\n  ── 比对：{待测名}  ↔  {基准名}")
    同文 = 基准文本 == 待测文本
    print(f"     字符数：{基准名}={len(基准文本)}  {待测名}={len(待测文本)}")
    if 同文:
        print("     ✅ 文本完全相同")
    else:
        # 定位首个分歧字符
        位置 = next((i for i, (a, b) in enumerate(zip(基准文本, 待测文本)) if a != b),
                   min(len(基准文本), len(待测文本)))
        print(f"     ❌ 首次分歧在**字符**位置 {位置}")
        print(f"        基准：…{基准文本[max(0,位置-40):位置+40]!r}")
        print(f"        待测：…{待测文本[max(0,位置-40):位置+40]!r}")

    令牌比对 = "（未取到 token 序列，退化为字符比对）"
    if 基准令牌 and 待测令牌 and len(基准令牌) == len(待测令牌):
        分歧 = [i for i, (a, b) in enumerate(zip(基准令牌, 待测令牌)) if a != b]
        令牌比对 = f"共 {len(基准令牌)} 个 token，分歧 {len(分歧)} 个"
        if 分歧:
            令牌比对 += f"，**首个分歧在第 {分歧[0] + 1} 个 token**"
            # 分歧位置间隔序列 —— 这是「每 29 token 一错」的直接指纹
            前几 = 分歧[:12]
            print(f"     [token] {令牌比对}")
            print(f"     [token] 前 12 个分歧位置：{前几}")
            if len(分歧) >= 3:
                间隔 = [dis - 分歧[i - 1] for i, dis in enumerate(分歧[1:], 1)]
                常用 = max(set(间隔), key=间隔.count)
                占比 = 间隔.count(常用) / len(间隔)
                print(f"     [token] 分歧间隔中位数≈{sorted(间隔)[len(间隔)//2]}"
                      f"，最频繁间隔={常用}（占 {占比:.0%}）")
                if 29 in 间隔 or (常用 in (28, 29, 30)):
                    print("     ⚠️ 间隔聚集在 29 附近 —— **与上游『每 29 个 token 一错』"
                          "的指纹高度吻合**")
        else:
            print(f"     [token] {令牌比对} ✅ token 序列完全一致")
    else:
        print(f"     [token] {令牌比对}")

    return {"文本相同": 同文, "文本": 待测文本, "令牌": 待测令牌}


def 一组(名称, 引擎路径, 日志, 额外环境="", 重复=1):
    print()
    print("=" * 72)
    print(f"  {名称}")
    print("=" * 72)
    print(f"  轮前显存: {显存()}  环境: {额外环境 or '(无)'}")
    成功, 尾部 = 启动(引擎路径, 日志, 额外环境)
    if not 成功:
        print("  ❌ 启动失败：")
        print("  " + 尾部[-900:].replace("\n", "\n  "))
        return None

    for 关键词 in ["CUDA : ARCHS", "ggml_vulkan", "flash_attn", "offloaded",
                   "type_k", "type_v", "MMQ"]:
        命中 = [L for L in 尾部.splitlines() if 关键词.lower() in L.lower()]
        for L in 命中[:1]:
            print(f"  [启动] {L.strip()[:110]}")
    time.sleep(3)
    print(f"  载入后显存: {显存()}")

    结果 = []
    for i in range(重复):
        起 = time.time()
        文本, 令牌, 用量 = 生成()
        耗时 = time.time() - 起
        print(f"  ── 第 {i+1} 次生成：{用量['completion_tokens']} token / {耗时:.1f}s "
              f"({用量['completion_tokens']/耗时:.2f} tok/s)")
        结果.append({"文本": 文本, "令牌": 令牌})

    if 重复 > 1:
        同 = 结果[0]["文本"] == 结果[1]["文本"]
        print(f"  [自洽性] 同后端重复两次 → {'✅ 完全一致' if 同 else '❌ 竟然不同（有非确定性！）'}")

    跑("pkill -f llama-server ; sleep 3")
    return 结果[0]


print("=" * 72)
print("  P100 sm_60 fp16 精度验证（三步对照）")
print("  " + time.strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 72)
print(f"  模型: Qwen3.8-27B-UD-Q3_K_XL | ctx={上下文} | KV=q8_0 | -ngl 99 | 贪心解码")
print(f"  prompt: {提示词}")
print(f"  输出: {输出长度} token × 5 组 | 轮前总显存: {显存()}")

基准 = 一组("① Vulkan（Pascal 上恒 fp32）—— 基准", VULKAN, "/tmp/prec_vulkan.log")
cuda默认 = 一组("② CUDA 默认（预期走 fp16 快速路径）", CUDA, "/tmp/prec_cuda.log")
cuda复查 = 一组("③ CUDA 默认 · 复跑（自洽性对照）", CUDA, "/tmp/prec_cuda2.log")
cuda_mmq = 一组("④ CUDA + GGML_CUDA_FORCE_MMQ=1（强制 fp32 计算路径）",
              CUDA, "/tmp/prec_mmq.log", "export GGML_CUDA_FORCE_MMQ=1\n")

print()
print("=" * 72)
print("  比对结果")
print("=" * 72)
if 基准:
    比对("Vulkan基准", 基准["文本"], 基准["令牌"],
        "CUDA默认", (cuda默认 or {}).get("文本", ""), (cuda默认 or {}).get("令牌", []))
    if cuda_mmq:
        比对("Vulkan基准", 基准["文本"], 基准["令牌"],
            "CUDA+MMQ", cuda_mmq["文本"], cuda_mmq["令牌"])
    if cuda默认 and cuda_mmq:
        比对("CUDA默认", cuda默认["文本"], cuda默认["令牌"],
            "CUDA+MMQ", cuda_mmq["文本"], cuda_mmq["令牌"])

print()
print("=" * 72)
print("  判读规则")
print("=" * 72)
print("  · CUDA默认 ≠ 基准，且 CUDA+MMQ == 基准  → fp16 缺陷确证，MMQ 可绕开")
print("  · 三者全等                              → sm_60 上 fp16 路径无此缺陷")
print("  · 三者全不同                            → 非 fp16 因素，另找变量")
if cuda默认 and cuda复查:
    print(f"  · CUDA 复跑自洽性: "
          f"{'✅ 一致（上面的分歧可归因于后端）' if cuda默认['文本'] == cuda复查['文本'] else '❌ 不一致（有随机性，结论不可信）'}")
print()
print("  完整输出已留存：")
for f in ["/tmp/prec_vulkan.log", "/tmp/prec_cuda.log", "/tmp/prec_mmq.log"]:
    print(f"    {f}")
