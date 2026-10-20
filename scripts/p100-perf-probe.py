#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P100 性能探针（2026-09-24）

把「10 t/s 到底正常吗」拆成四个可以独立回答的问题：

  1. 现在跑的到底是哪个引擎？哪些参数？有没有算子悄悄掉到 CPU？
  2. prefill（prompt processing）速度：短提示 vs 长提示
  3. 首次调用 vs 同一进程内重复调用
     —— 用来区分「首次触发 Vulkan 管线编译/预热」和「稳态就是这么慢」
  4. decode（逐 token 生成）速度 + 期间 gpu 利用率
     —— 利用率很低，说明瓶颈不在 GPU 算力，而在别处（掉 CPU / 串行化 / 数据搬运）

只发 HTTP 请求 + 只读 /proc 与 nvidia-smi，不做任何持久化改动。
日志里的中文一律走「打印到终端」，不写文件，避免跨机编码问题。
"""

import glob
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

# 保留终端原生编码（Windows 中文终端默认 GBK），只兜住无法表示的字符。
# 不要强制改成 utf-8 —— 那反而会乱码。
sys.stdout.reconfigure(errors="replace")

API = "http://127.0.0.1:8000"


def 打印(*args):
    print(*args)
    sys.stdout.flush()


def 分隔(标题=""):
    打印("")
    打印("=" * 72)
    if 标题:
        打印(标题)
        打印("=" * 72)


def 找运行中的引擎():
    """从 /proc 读 llama-server 的完整命令行 —— 一眼看清是哪个二进制、带了哪些参数。"""
    结果 = []
    for 路径 in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            with open(路径, "rb") as f:
                内容 = f.read().decode("utf-8", "replace")
        except Exception:
            continue
        if "llama-server" not in 内容 and "llama_server" not in 内容:
            continue
        参数 = [x for x in 内容.split("\0") if x]
        if 参数 and 参数 not in 结果:
            结果.append(参数)
    return 结果


def 取最新服务器日志():
    """llm-switch.sh 会把启动输出 tee 到 mktemp 文件，取最新的那个。"""
    文件 = glob.glob("/tmp/llm-switch-*.log")
    if not 文件:
        return None
    文件.sort(key=lambda p: os.path.getmtime(p))
    return 文件[-1]


def 关键行(路径, 关键词, 上限=16):
    命中 = []
    try:
        with open(路径, "r", errors="replace") as f:
            for 行 in f:
                小写 = 行.lower()
                if any(k.lower() in 小写 for k in 关键词):
                    命中.append(行.rstrip())
    except Exception as 异常:
        return ["（读取失败：%s）" % 异常]
    return 命中[-上限:] if len(命中) > 上限 else 命中


class GPU采样(threading.Thread):
    """请求期间每秒采一次 gpu 利用率与显存 —— 用来判断瓶颈在不在 GPU。"""

    def __init__(self):
        super().__init__(daemon=True)
        self.数据 = []
        self.停 = False

    def run(self):
        while not self.停:
            try:
                输出 = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip()
                if 输出:
                    self.数据.append(输出.splitlines()[0])
            except Exception:
                pass
            time.sleep(1)


def 发请求(提示词, 生成数=128, 超时=1800):
    载荷 = json.dumps({
        "prompt": 提示词,
        "n_predict": 生成数,
        "temperature": 0,
        # 关键：默认 cache_prompt=true，第二次发同样的提示会整段命中缓存，
        # prefill 显示成 0 token，就把「预热效应」这件事盖掉了。这里必须关掉。
        "cache_prompt": False,
    }).encode("utf-8")

    请求 = urllib.request.Request(API + "/completion", data=载荷,
                                  headers={"Content-Type": "application/json"})
    采样 = GPU采样()
    采样.start()
    try:
        t0 = time.time()
        with urllib.request.urlopen(请求, timeout=超时) as 响应:
            数据 = json.loads(响应.read().decode("utf-8", "replace"))
        墙钟 = time.time() - t0
    finally:
        采样.停 = True
        time.sleep(1.2)
    return 数据, 墙钟, 采样.数据


def 报告(标签, 数据, 墙钟, 采样数据):
    计时 = 数据.get("timings", {}) or {}
    行 = ["【%s】墙钟 %.1f s" % (标签, 墙钟)]
    if 计时:
        prefill数 = 计时.get("prompt_n") or 0
        decode数 = 计时.get("predicted_n") or 0
        行.append("  prefill: %5s token  %8.2f tok/s  （每 token %.0f ms）" % (
            prefill数, 计时.get("prompt_per_second", 0) or 0,
            (计时.get("prompt_ms", 0) or 0) / max(prefill数, 1)))
        行.append("  decode : %5s token  %8.2f tok/s  （每 token %.0f ms）" % (
            decode数, 计时.get("predicted_per_second", 0) or 0,
            (计时.get("predicted_ms", 0) or 0) / max(decode数, 1)))

    if 采样数据:
        利用率, 显存 = [], []
        for 项 in 采样数据:
            部分 = [x.strip() for x in 项.split(",")]
            if len(部分) >= 2:
                try:
                    利用率.append(float(部分[0]))
                    显存.append(int(部分[1]))
                except ValueError:
                    pass
        if 利用率:
            行.append("  GPU    : 利用率 平均%.0f%% 峰值%.0f%%   显存 平均%d MiB 峰值%d MiB" % (
                sum(利用率) / len(利用率), max(利用率),
                sum(显存) / len(显存), max(显存)))

    # 生成的文字样例（前 60 字），用于确认服务真的在出结果、不是空转
    文字 = (数据.get("content") or "").replace("\n", " ").strip()
    if 文字:
        行.append("  样例   : %s" % 文字[:60])

    打印("\n".join(行))
    return 计时


def main():
    分隔("P100 性能探针  %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    打印("目标：%s" % API)

    分隔("一、现在跑的是哪个引擎、带了哪些参数")
    引擎们 = 找运行中的引擎()
    if not 引擎们:
        打印("❌ 没找到正在运行的 llama-server ——")
        打印("   请先在另一个终端把服务起起来（./llm-switch.sh），再跑本探针。")
        return 1
    for 参数 in 引擎们:
        打印(" 二进制：%s" % 参数[0])
        打印(" 参数  ：")
        for 项 in 参数[1:]:
            打印("          %s" % 项)
        打印("")

    分隔("二、启动日志里的关键行")
    日志 = 取最新服务器日志()
    if 日志:
        打印("（日志文件：%s）" % 日志)
        for 行 in 关键行(日志, [
            "offloaded", "Vulkan", "CUDA", "ARCHS", "compute buffer",
            "KV self size", "n_ctx", "unsupported", "assigned to cpu",
            "missing tensor", "error", "abort",
        ]):
            打印("   %s" % 行)
    else:
        打印("（没找到 /tmp/llm-switch-*.log，本步跳过）")

    # prefill 需要一个长度接近真实场景的提示：作者那次真实对话是 371 token，
    # 这里用同一量级（约 400 token），数字可直接对比。
    短提示 = "Reply with one word: hello."
    长提示 = "A local language model runs on a single GPU. " * 40

    分隔("三、实测三发请求")
    打印("三个测试都显式关闭 prompt 缓存，否则第二次的 prefill 会显示成 0 token。")
    打印("")

    打印("测试 1 / 3：短提示 —— 看 decode 稳态速度")
    数据, 墙钟, 采样 = 发请求(短提示, 生成数=128)
    短计时 = 报告("测试 1 短提示", 数据, 墙钟, 采样)

    打印("")
    打印("测试 2 / 3：长提示（约 400 token）—— 冷 prefill，看有没有首次编译/预热开销")
    数据, 墙钟, 采样 = 发请求(长提示, 生成数=128)
    冷计时 = 报告("测试 2 长提示（本进程内首次）", 数据, 墙钟, 采样)

    打印("")
    打印("测试 3 / 3：同一长提示再来一次 —— 只可能更快，不会更慢")
    数据, 墙钟, 采样 = 发请求(长提示, 生成数=128)
    热计时 = 报告("测试 3 长提示（重复）", 数据, 墙钟, 采样)

    分隔("四、判读参照值")
    打印("  P100 HBM2 带宽 732 GB/s，本模型实际参与推理的权重 ≈ 11.92 GiB")
    打印("  decode 理论上限 ≈ 732 ÷ 12.8 ≈ 57 tok/s；CUDA 上通常能跑到 30~35")
    打印("  prefill 是算力受限，本该比 decode 快一个数量级")
    打印("     —— 371 token 量级在 P100 上应在 1~3 秒内跑完")
    打印("")

    冷 = (冷计时 or {}).get("prompt_per_second", 0) or 0
    热 = (热计时 or {}).get("prompt_per_second", 0) or 0
    if 冷 and 热:
        倍数 = 热 / 冷
        打印("  本次实测 prefill：首次 %.2f tok/s → 重复 %.2f tok/s（%.1f 倍）" % (冷, 热, 倍数))
        if 倍数 >= 3:
            打印("  → 提升明显：首次调用里含有一次性开销（Vulkan 管线编译 / 预热）")
        elif 倍数 >= 1.3:
            打印("  → 有提升但不彻底：一次性开销存在，同时稳态也偏慢")
        else:
            打印("  → 几乎没变：不是预热问题，是稳态就慢（要查算子掉 CPU / 串行化）")

    分隔("完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
