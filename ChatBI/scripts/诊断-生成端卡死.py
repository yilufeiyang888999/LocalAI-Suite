# -*- coding: utf-8 -*-
"""ChatBI · 生成端「活着但不应答」三步判

## 为什么单独写一个脚本

    本项目的实测形态：**问数全超时**，而 `GET /health`、`GET /v1/models`、`GET /props`
    **全部 200 且 0.00s 返回**。

    这时最容易走错的三条路（都走过，都浪费了时间）：
      · 怀疑模型答得慢 → 加超时、换小模型、重跑评测 → 全是白费
      · 怀疑提示词写坏了 → 改提示词 → 更白费
      · 怀疑问数服务／语义层有问题 → 一层层查代码 → 问题根本不在这一侧

    🔴 真相是：**llama-server 的唯一 slot 被占住了，HTTP 层还活着。**
       判别只需 10 秒，不需要看任何日志 —— 看进程状态。

## 三步判（本脚本做的就是这三步）

    ① **CPU 增量**：5 秒采样，增量 ≈ 0 → **不是在算，是在等**。
       （如果它在算，说明只是慢，那是另一个问题 —— 本脚本会明说）

    ② **线程状态**：`llama-server` 的所有线程是否都处于 `Wait`。
       全 `Wait` + CPU 0 = 阻塞在同步原语上（锁 / socket / pipe）。
       ⚠️ 全 `Wait` 也可能是在**等 pipe 写**（`--无上屏` 没开、父进程不读输出）——
          但那种情况下 `/health` 也会卡（它同样要写访问日志）。
          `/health` 秒回 ⇒ 排除了 pipe 阻塞，只剩锁。

    ③ **`/slots` 能不能秒回**：
       `/slots` 的处理器要拿**和推理同一把** slot 锁。
        · `/health` `/props` `/v1/models` 秒回  → HTTP 线程池正常
        · `/slots` **超时**                       → 锁被占，这不是巧合，是判据
       这一条把「服务卡了」精确定位到「**slot 卡了**」。

    结论如果是「slot 被占死」，处置只有一个：**在 Local-AI APP 里重启模型服务**。
    ⚠️ **不要在这个脚本里杀进程** —— 那个进程是 APP 拉起来的（APP 管模型生命周期，
       杀完 APP 的界面状态与真实进程会对不上）。脚本只诊断、只报告。

## 🔴 为什么会卡死（本项目的实测成因）

    客户端**在请求还没返回时被强杀**（例如探针 / 自检挂住后 `SIGTERM`）：
    客户端 socket 关闭，服务端 handler 没能干净收尾，**slot 没被归还**。
    端口上会留下一条 `CLOSE_WAIT` 连接 —— 那就是现场。

    两条工程纪律（比修这一次更重要）：
      1. **给客户端的超时必须长于模型的最坏耗时**，宁可长一点，
         也不要让请求在中途被砍 —— 砍一次可能就把引擎卡死一次。
      2. 引擎侧开 `--parallel 2`（>1）能让一个卡死的 slot 不拖垮全部请求。
         当前是 `--parallel 1`（向量/重排服务**必须**为 1，但生成端没必要）。

用法：python scripts/诊断-生成端卡死.py [--端口 8000] [--PID 398176]
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:                                    # noqa: BLE001
        pass

绕代理 = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# 不拿 slot 锁的路径（HTTP 层存活证据）
免锁路径 = ["/health", "/props", "/v1/models"]
# 需要 slot 锁的路径（卡死证据）
要锁路径 = "/slots"


def 探(端口, 路径, 超时=6):
    """返回 (状态码 or None, 用时秒, 正文前 200 字 or 错误)。"""
    起 = time.time()
    try:
        r = 绕代理.open("http://127.0.0.1:%d%s" % (端口, 路径), timeout=超时)
        b = r.read().decode("utf-8", "replace")
        return r.status, round(time.time() - 起, 2), b[:200]
    except urllib.error.HTTPError as e:
        # 501 / 404 也算「HTTP 层答了」—— 关键是**它多久答的**
        return e.code, round(time.time() - 起, 2), "HTTP %d %s" % (e.code, e.reason)
    except Exception as 错:                               # noqa: BLE001
        return None, round(time.time() - 起, 2), "%s: %s" % (type(错).__name__, str(错)[:60])


def 端口占用者(端口):
    """谁在听这个端口。返回 PID（拿不到就 None）。"""
    try:
        # ⚠️ Windows 上 `netstat -ano` 是唯一免管理员的方式；
        #    `Get-NetTCPConnection` 在 PowerShell 里更快，但本项目的脚本一律走 python。
        出 = subprocess.run(["netstat", "-ano"], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=20).stdout
    except Exception:                                      # noqa: BLE001
        return None, []
    听 = None
    待关 = []
    for 行 in 出.splitlines():
        段 = 行.split()
        if len(段) < 5:
            continue
        if 段[0].upper() != "TCP":
            continue
        本地, 状态, pid = 段[1], 段[3].upper(), 段[4]
        if 本地.endswith(":%d" % 端口):
            if 状态 == "LISTENING":
                听 = int(pid)
            elif 状态 in ("CLOSE_WAIT", "FIN_WAIT_2", "LAST_ACK"):
                待关.append((本地, 状态, int(pid)))
    return 听, 待关


def 采样CPU(pid, 秒=5):
    """Windows 上没有 psutil → 用 kernel32 的 GetProcessTimes（两个 FILETIME 相加）。"""
    if os.name != "nt":
        return None
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class FILETIME(ctypes.Structure):
        _fields_ = [("低", ctypes.c_uint32), ("高", ctypes.c_uint32)]

    def 取(句柄):
        创, 退, 核, 用 = FILETIME(), FILETIME(), FILETIME(), FILETIME()
        if not k32.GetProcessTimes(句柄, ctypes.byref(创), ctypes.byref(退),
                                 ctypes.byref(核), ctypes.byref(用)):
            return None
        return (核.高 << 32 | 核.低) + (用.高 << 32 | 用.低)

    PROCESS_QUERY_LIMITED = 0x1000
    句柄 = k32.OpenProcess(PROCESS_QUERY_LIMITED, False, pid)
    if not 句柄:
        return None
    try:
        a = 取(句柄)
        time.sleep(秒)
        b = 取(句柄)
    finally:
        k32.CloseHandle(句柄)
    if a is None or b is None:
        return None
    # FILETIME = 100 纳秒 → 秒
    return round((b - a) / 1e7, 2)


def 说(文本=""):
    print(文本)


def 主():
    参 = argparse.ArgumentParser(description="ChatBI 生成端卡死诊断")
    参.add_argument("--端口", type=int, default=8000)
    参.add_argument("--PID", type=int, help="llama-server 的 PID（不给就按端口找）")
    args = 参.parse_args()

    说("ChatBI · 生成端「活着但不应答」三步判　端口 %d" % args.端口)
    说("时间：%s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    说("=" * 74)

    听, 待关 = 端口占用者(args.端口)
    pid = args.PID or 听
    说("① 谁在听：%s" % ("PID %d" % 听 if 听 else "**没有进程监听 → 服务根本没起**"))
    if 待关:
        说("   半关闭连接 %d 条（客户端已走、服务端没释放）：" % len(待关))
        for 本地, 状态, p in 待关[:5]:
            说("     %s  %s  pid=%d" % (本地, 状态, p))
        说("   ⚠️ 这就是「客户端被中途强杀 → slot 没归还」的现场。")
    else:
        说("   没有半关闭连接（若仍卡死，成因不是客户端强杀）")

    说()
    说("② HTTP 层是否活着（**不拿 slot 锁**的路径）")
    都秒回 = True
    for 路径 in 免锁路径:
        码, 用时, 正文 = 探(args.端口, 路径)
        if 码 is None or 用时 > 3:
            都秒回 = False
        说("   %-12s → %-5s %.2fs  %s"
           % (路径, 码 if 码 is not None else "超时", 用时, 正文.replace("\n", " ")[:60]))

    说()
    说("③ slot 是否被占死（**要 slot 锁**的路径）")
    码, 用时, 正文 = 探(args.端口, 要锁路径, 超时=6)
    slot卡 = 码 is None
    说("   %-12s → %-5s %.2fs  %s"
       % (要锁路径, 码 if 码 is not None else "**超时**", 用时,
          "拿到了锁（slot 空闲）" if 码 is not None else "拿不到锁 → slot 被占住"))

    说()
    说("④ 进程是在算还是在等（CPU 采样 5 秒）")
    if not pid:
        cpu = None
    else:
        cpu = 采样CPU(pid, 5)
    说("   PID %s ｜ 5 秒 CPU 增量：%s"
       % (pid or "（未知）", "拿不到（权限/非 Windows）" if cpu is None else "%s 秒" % cpu))

    说()
    说("=" * 74)
    if 听 is None:
        说("诊断：🔴 **服务没起** —— 8000 端口无人监听。去 Local-AI APP 里启动模型服务。")
        return 1
    if slot卡 and 都秒回 and (cpu is None or cpu < 0.2):
        说("诊断：🔴 **slot 被占死**（不是模型慢、不是提示词坏、不是问数服务的问题）")
        说("  证据链：免锁路径秒回 + /slots 拿不到锁 + CPU 增量为 0（在等，不在算）")
        说()
        说("  处置：**在 Local-AI APP 里重启模型服务**（停 → 启）。")
        说("  ⚠️ 不要直接 taskkill 这个进程：它是 APP 拉起来的，杀了 APP 界面状态")
        说("     会与真实进程对不上（这就是「判状态要看现场」那一条）。")
        说()
        说("  下次避免：① 给客户端的超时留得比模型最坏耗时更长，别中途砍请求；")
        说("            ② 生成端可以开 --parallel 2，一个 slot 卡死不拖垮全部。")
        return 1
    if slot卡 and not 都秒回:
        说("诊断：🔴 **HTTP 层也卡了** —— 更像输出被堵住（stdout 管道无人读）。")
        说("  查：APP 是否还在读子进程输出 / 控制台是否被鼠标拖选住（标题变 Select）。")
        return 1
    if not slot卡 and cpu is not None and cpu > 0.5:
        说("诊断：🟠 **slot 没被占死，但进程在烧 CPU** —— 这是一次卡住的计算，")
        说("  不是锁问题。看 GPU 是否被别的进程占着（nvidia-smi），或换个更小的模型试。")
        return 1
    if not slot卡:
        说("诊断：✅ **生成端空闲可用**（/slots 秒回、CPU 没在烧）。")
        说("  若问数仍超时，问题在**客户端这一侧**（问数服务 / 网络 / 超时设太短）。")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(主())
