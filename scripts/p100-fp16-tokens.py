#!/usr/bin/env python3
# sm_60 fp16 精度验证 —— 第二轮：token 级比对（绕开 chat 模板与文本拼接）
#
# 第一轮的问题：三组服务端日志完全一致（n_tokens=583, truncated=0），
# 但返回文本长度不同（Vulkan 414 vs CUDA 436 字符）。
# → 差异不在**模型输出**，而在「token 序列 → 文本」这一步。
#
# 本轮改用 /completion 端点：
#   · 不走 chat 模板（少一层变量）
#   · 直接返回 token id 序列（tokens 字段）
#   · 直接设置 logit_bias / n_probs 都不需要，只要贪心
# 这样比对的就是**模型真正吐出的 id**，不受 detokenize 实现差异影响。

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

# 用 /completion 的 raw prompt（不加 chat 模板），内容仍是确定性计数
提示词 = "1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n11\n12\n13\n14\n15\n16\n17\n18\n19\n20\n21\n22\n23\n24\n25\n26\n27\n28\n29\n30\n31\n32\n33\n34\n35\n36\n37\n38\n39\n40\n41\n42\n43\n44\n45\n46\n47\n48\n49\n50\n51\n52\n53\n54\n55\n56\n57\n58\n59\n60\n61\n62\n63\n64\n65\n66\n67\n68\n69\n70\n71\n72\n73\n74\n75\n76\n77\n78\n79\n80\n81\n82\n83\n84\n85\n86\n87\n88\n89\n90\n91\n92\n93\n94\n95\n96\n97\n98\n99\n100\n101\n102\n103\n104\n105\n106\n107\n108\n109\n110\n111\n112\n113\n114\n115\n116\n117\n118\n119\n120\n121\n122\n123\n124\n125\n126\n127\n128\n129\n130\n131\n132\n133\n134\n135\n136\n137\n138\n139\n140\n141\n142\n143\n144\n145\n146\n147\n148\n149\n150\n"


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
    open("/tmp/_eng_tok.sh", "wb").write(base64.b64decode(base64.b64encode(脚本.encode())))
    os.chmod("/tmp/_eng_tok.sh", 0o755)
    subprocess.Popen(["bash", "-lc", "setsid /tmp/_eng_tok.sh < /dev/null > /dev/null 2>&1 &"])
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
    """用 /completion 端点，返回 (token id 列表, 文本)"""
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
    return 响应.get("tokens", []), 响应.get("content", "")


def 一组(名称, 引擎路径, 日志, 额外环境=""):
    print()
    print("=" * 72)
    print(f"  {名称}")
    print("=" * 72)
    成功, 尾部 = 启动(引擎路径, 日志, 额外环境)
    if not 成功:
        print("  ❌ 启动失败：")
        print("  " + 尾部[-800:].replace("\n", "\n  "))
        return None
    time.sleep(3)
    令牌, 文本 = 生成()
    print(f"  服务端返回 token 数 = {len(令牌)}，文本 {len(文本)} 字符")
    print(f"  前 20 个 token: {令牌[:20]}")
    print(f"  末 20 个 token: {令牌[-20:]}")
    跑("pkill -f llama-server ; sleep 3")
    return {"令牌": 令牌, "文本": 文本}


print("=" * 72)
print("  第二轮：token 级精度比对（/completion 端点，绕开 chat 模板）")
print("  " + time.strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 72)

基准 = 一组("① Vulkan（fp32 基准）", VULKAN, "/tmp/tok_vulkan.log")
cuda = 一组("② CUDA 默认", CUDA, "/tmp/tok_cuda.log")
mmq = 一组("③ CUDA + FORCE_MMQ=1", CUDA, "/tmp/tok_mmq.log",
          "export GGML_CUDA_FORCE_MMQ=1\n")

print()
print("=" * 72)
print("  token 级比对")
print("=" * 72)


def 比(甲名, 甲, 乙名, 乙):
    if not 甲 or not 乙:
        print(f"  {甲名} ↔ {乙名}：有一方失败，跳过")
        return
    A, B = 甲["令牌"], 乙["令牌"]
    print(f"\n  {甲名} ↔ {乙名}")
    print(f"    token 数：{甲名}={len(A)}  {乙名}={len(B)}")
    if A == B:
        print("    ✅ token 序列完全一致")
        return
    最短 = min(len(A), len(B))
    分歧 = [i for i in range(最短) if A[i] != B[i]]
    print(f"    ❌ 分歧 {len(分歧)} 个，首个在第 {分歧[0] + 1} 个 token"
          if 分歧 else "    ⚠️ 前段全同，仅长度不同（提前停止）")
    if 分歧:
        print(f"       位置: {分歧[:15]}")
        for i in 分歧[:3]:
            print(f"       第 {i+1}: {甲名}={A[i]}  {乙名}={B[i]}"
                  f"  （前一个 token: {甲名}={A[i-1]} {乙名}={B[i-1]}）")
        if len(分歧) >= 2:
            间隔 = [分歧[j] - 分歧[j - 1] for j in range(1, len(分歧))]
            中位 = sorted(间隔)[len(间隔) // 2]
            print(f"       分歧间隔中位数 = {中位}，前几间隔 = {间隔[:12]}")
            if any(28 <= x <= 30 for x in 间隔):
                print("       ⚠️ 间隔命中 29 附近 —— 与上游指纹吻合")


比("Vulkan", 基准, "CUDA", cuda)
比("Vulkan", 基准, "CUDA+MMQ", mmq)
比("CUDA", cuda, "CUDA+MMQ", mmq)

print()
print("  判读：")
print("   · token 序列一致 → **模型输出无差异**，上一轮的字符差异是 detokenize 造成")
print("   · token 序列有规律分歧 → 确证 fp16 精度问题")
