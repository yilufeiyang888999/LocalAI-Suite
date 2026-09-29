#!/usr/bin/env python3
# Vulkan vs CUDA 横评 —— 在 P100 本机运行（不经 ssh）
#
# 用法：python3 p100-backend-ab.py
# 设计要点（吸取 2026-09-28 的教训）：
#   1. 每轮开始前确认 GPU 独占（显存基线 + 杀残留进程）
#   2. 每轮记录服务端 metrics 基线，结束后做差值对账（防第三方流量污染）
#   3. 测速用 400 token 输出（短样本会被每请求固定开销摊薄）
#   4. 同时采温度/时钟，确认没降频
#   5. 两轮之间重启服务，避免 KV/预热效应跨轮干扰

import base64
import json
import os
import subprocess
import threading
import time
import urllib.request

模型 = "/opt/llama.cpp/models/Qwen3.8-27B-UD-Q3_K_XL.gguf"
VULKAN = "/opt/llama-vulkan/llama-server"
CUDA = "/opt/llama-src-b10853/build/bin/llama-server"
端口 = 8000


def 跑(命令, 超时=60):
    结果 = subprocess.run(["bash", "-lc", 命令], capture_output=True, timeout=超时)
    return 结果.stdout.decode("utf-8", errors="replace")


def 显存():
    return 跑("nvidia-smi --query-gpu=memory.used --format=csv,noheader").strip()


def 取指标():
    原始 = 跑(f"curl -s -m 5 http://127.0.0.1:{端口}/metrics 2>/dev/null")
    结果 = {}
    for 行 in 原始.splitlines():
        if 行.startswith("llamacpp:"):
            段 = 行.split(" ", 1)
            if len(段) == 2:
                try:
                    结果[段[0].replace("llamacpp:", "")] = float(段[1])
                except ValueError:
                    pass
    return 结果


def 启动(引擎路径, 日志):
    跑("pkill -f llama-server ; sleep 3")
    目录 = 引擎路径.rsplit("/", 1)[0]
    脚本 = (
        "#!/bin/bash\n"
        f"cd {目录}\n"
        f"export LD_LIBRARY_PATH={目录}\n"
        f"exec {引擎路径} --model {模型} --ctx-size 49152 --batch-size 256 "
        "--ubatch-size 256 --port 8000 --host 0.0.0.0 --n-gpu-layers 99 "
        f"--cache-type-k q8_0 --cache-type-v q8_0 --no-mmap --parallel 1 --metrics "
        f"> {日志} 2>&1\n"
    )
    编码 = base64.b64encode(脚本.encode()).decode()
    open("/tmp/_eng.sh", "wb").write(base64.b64decode(编码))
    os.chmod("/tmp/_eng.sh", 0o755)
    subprocess.Popen(["bash", "-lc", f"setsid /tmp/_eng.sh < /dev/null > /dev/null 2>&1 &"])

    for _ in range(60):
        time.sleep(5)
        尾部 = 跑(f"tail -40 {日志} 2>/dev/null")
        if "listening" in 尾部:
            return True, 尾部
        真错误 = [L for L in 尾部.splitlines()
                 if "error" in L.lower() and "unused tensor" not in L.lower()
                 and "missing tensor" not in L.lower()]
        if 真错误:
            return False, 尾部
    return False, 尾部


def 请求(输出长度):
    载荷 = {"model": "m",
           "messages": [{"role": "user",
                         "content": "Repeat the word hello 200 times separated by spaces."}],
           "max_tokens": 输出长度, "temperature": 0, "cache_prompt": False}
    数据 = json.dumps(载荷).encode()
    请 = urllib.request.Request(
        f"http://127.0.0.1:{端口}/v1/chat/completions", data=数据,
        headers={"Content-Type": "application/json"})
    起 = time.time()
    响应 = json.loads(urllib.request.urlopen(请, timeout=900).read())
    return 响应["usage"], time.time() - 起


def 一轮(名称, 路径, 日志):
    print()
    print("=" * 70)
    print(f"  {名称}")
    print("=" * 70)
    print(f"  轮前显存: {显存()}")
    成功, 尾部 = 启动(路径, 日志)
    if not 成功:
        print("  ❌ 启动失败：")
        print("  " + 尾部[-800:].replace("\n", "\n  "))
        return None
    for 关键词 in ["CUDA : ARCHS", "ggml_vulkan", "offloaded", "n_ctx_slot", "type"]:
        命中 = [L for L in 尾部.splitlines() if 关键词.lower() in L.lower()]
        for L in 命中[:2]:
            print(f"  [启动] {L.strip()[:108]}")
    time.sleep(5)
    基线 = 取指标()
    print(f"  载入后显存: {显存()}")

    采样, 停止 = [], [False]

    def 采():
        while not 停止[0]:
            采样.append(跑("nvidia-smi --query-gpu=temperature.gpu,clocks.sm,"
                         "clocks_throttle_reasons.active --format=csv,noheader").strip())
            time.sleep(3)

    线 = threading.Thread(target=采)
    线.start()
    用量, 墙钟 = 请求(400)
    停止[0] = True
    线.join()
    完成 = 取指标()

    pt, ct = 用量["prompt_tokens"], 用量["completion_tokens"]
    print(f"  prompt={pt}  out={ct}  墙钟={墙钟:.1f}s")
    print(f"  ▶ decode = {ct / 墙钟:.2f} tok/s")

    对账生成 = None
    if 基线 and 完成:
        差_p = 完成.get("prompt_tokens_total", 0) - 基线.get("prompt_tokens_total", 0)
        对账生成 = 完成.get("tokens_predicted_total", 0) - 基线.get("tokens_predicted_total", 0)
        差_s = 完成.get("tokens_predicted_seconds_total", 0) - 基线.get("tokens_predicted_seconds_total", 0)
        print(f"  [对账] 服务端差值 prompt={差_p:.0f} 生成={对账生成:.0f}（我的 prompt={pt} 生成={ct}）")
        if 对账生成 > ct * 1.15:
            print("  ⚠️ 服务端生成量多于本客户端 —— 疑有第三方流量")
        else:
            print("  ✅ 对账通过，无第三方流量")
        if 差_s > 0:
            print(f"  [服务端口径] decode = {对账生成 / 差_s:.2f} tok/s")
    if 采样:
        print(f"  [温度/时钟] 首={采样[0]} | 中={采样[len(采样) // 2]} | 末={采样[-1]}")
    跑("pkill -f llama-server ; sleep 3")
    return {"名称": 名称, "decode": ct / 墙钟, "墙钟": 墙钟, "out": ct}


print("=" * 70)
print("  P100 Vulkan vs CUDA 横评")
print("  " + time.strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 70)
print("  模型: Qwen3.8-27B-UD-Q3_K_XL | ctx=49152 | KV=q8_0 | -ngl 99 | parallel=1")
print(f"  轮前总显存: {显存()}")

结果 = [一轮("Vulkan b10853（预编译包）", VULKAN, "/tmp/run_vulkan.log"),
      一轮("CUDA   b10853（自编 12.2）", CUDA, "/tmp/run_cuda.log")]

print()
print("=" * 70)
print("  横评结论")
print("=" * 70)
有效 = [r for r in 结果 if r]
for r in 有效:
    print(f"  {r['名称']:30s} decode = {r['decode']:6.2f} tok/s  (墙钟 {r['墙钟']:.1f}s)")
if len(有效) == 2:
    倍率 = 有效[1]["decode"] / 有效[0]["decode"]
    print()
    print(f"  ⮕ CUDA / Vulkan = {倍率:.2f}×")
    print("  ⮕ 判读：" + ("CUDA 明显更快，A 路线价值确认"
                        if 倍率 > 1.5 else "两者接近 —— 瓶颈可能不在后端，而在带宽或算子"))
