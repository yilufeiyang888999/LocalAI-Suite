# -*- coding: utf-8 -*-
"""
启动-服务.py —— 一键起「作品一」的本地服务（零第三方依赖，只用标准库）
=====================================================================

回答一句话：**「这套服务该怎么起、起对了没有？」**

它管谁、不管谁
--------------
管（本脚本负责起）：
  · **重排**   8085  llama-server --reranking     ← 检索链的必需依赖，必须 `--parallel 1`
  · **知识库** 8092  scripts/知识库服务.py        ← 作品一的服务本体
不管（只检查，不代起）：
  · **生成端** 8000  **由 Local-AI APP 负责**。APP 管模型生命周期与显存调优
    （换模型 / 调上下文 / 卸载显存都在 APP 里）。脚本代起会和 APP 抢同一份资源，
    也会让"谁在管这个进程"变得不清楚。
不需要：
  · **向量**   8082  知识库服务起手走 BM25，**压根不碰向量服务**（建基座 0.05 s）

启动顺序（有依赖关系，不是随便排的）
------------------------------------
    ① 重排 8085   ──→ ② 知识库 8092   ──→ ③ 用的时候需要生成端 8000（APP）
      等它 /health 200 才继续（模型加载要几秒）

设计要点（每条都是实际踩过的坑换来的）
--------------------------------------
1. 🔴 **起之前先数一遍端口上有几个 PID**。Python/llama-server 都能多进程绑同一端口
   （`SO_REUSEADDR`），**不报错，请求随机分流** —— 症状是「我明明重启了，
   改动就是不生效」。所以本脚本把「实例数」当成第一位的判据。
2. 🔴 **发现问题不静默重起**。默认只报告并要求确认；`--强制` 才跳过询问。
3. 🔴 **起完必须自证**，而且要用**能区分版本的判据**：
   知识库服务看 `/health` 里有没有 `class3` 字段（旧代码没有）。
   **启动 banner 不能当版本判据** —— 新旧版的 banner 可能一字不差。
4. **只读的东西单独一档**（`--状态` / `--演练`）：不猜、不动手，先看清现场。
5. **不吞错误**：每个服务各开一个自己的控制台窗口，日志原样留在那里。

用法
----
    python scripts\\启动-服务.py                 # 体检 → 起缺的 → 自证（各开一个新窗口）
    python scripts\\启动-服务.py --状态           # 只看现状，什么都不动
    python scripts\\启动-服务.py --演练           # 只打印将要执行的命令（不执行）
    python scripts\\启动-服务.py --重起           # 先停旧实例（要确认）再起
    python scripts\\启动-服务.py --重起 --强制     # 不停问，直接停旧的
    python scripts\\启动-服务.py --前台           # 不开新窗口，输出汇总到当前终端
    python scripts\\启动-服务.py --停             # 停掉本脚本管的两个服务（要确认）

退出码：0 = 该起的都在跑且自证通过；1 = 有失败项
"""

import ctypes
import ctypes.wintypes as wintypes
import datetime
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request

# ======================= 配置（要改就改这一段） =======================

脚本目录 = os.path.dirname(os.path.abspath(__file__))
仓库根 = os.path.dirname(脚本目录)

# 引擎（llama.cpp b10853）。后端由**同目录的 ggml-*.dll** 决定，不看 exe 名 ——
# 该目录当前只有 ggml-cuda.dll；但重排命令刻意不带 -ngl，
# llama.cpp 是惰性初始化，**不碰 CUDA 就不建 context、不占显存**（实测：720 MiB 的
# 重排服务不在 `nvidia-smi` 的计算进程列表里）。留着 0.6~1 GiB 显存给 4B 生成端。
引擎目录 = r"D:\workspace\10-LocalAI-Installer\engine\backends"
引擎程序 = "llama-server.exe"
重排模型 = os.path.join(仓库根, "models", "bge-reranker-v2-m3-Q8_0.gguf")


def 引擎程序路径():
    """引擎 exe 的**绝对路径**。`重排命令()` 与 `检查前置条件()` 都只能从这里取。

    🔴 这里踩过一个把判据整个废掉的坑（2026-10-10 作者实跑暴露）：
    早先 `重排命令()` 第 0 段写的是**裸名字** `"llama-server.exe"`，靠 `cwd=引擎目录`
    指望它"顺便"被找到。**但 Windows 的 `CreateProcess` 不用 `lpCurrentDirectory`
    （子进程的 cwd）去找 exe —— 它用父进程的 cwd + PATH。** 结果是：

      · `检查前置条件()` 查的 `引擎目录/llama-server.exe` **在** → 判据 ✅ 全绿；
      · 真正 `Popen(["llama-server.exe", ...], cwd=引擎目录)` → **`FileNotFoundError: [WinError 2]`**。

    **判据检查的路径 ≠ 实际执行的路径**，判据就成了摆设 —— 与「同一份数据有两处定义」
    同一个毛病（改一处、跑另一处，不报错，只出错）。所以这里**只留一处定义**，
    前置条件也改成直接查**命令里真正会执行的那一段**（`重排命令()[0]`）。

    实测（2026-10-10，同一台机、同一个 exe）：
      裸名字 + cwd       → `FileNotFoundError: [WinError 2]` ✅ 稳定复现
      绝对路径 + cwd     → 正常，`version: 0.4.0-dev (build 10853)`，0.37 s
    """
    return os.path.join(引擎目录, 引擎程序)

# 端口（与 README / 知识库服务.py 一致）
重排端口 = 8085
知识库端口 = 8092
生成端端口 = 8000
向量端口 = 8082          # 本脚本不检查、不启动，仅列出来避免误会

# 重排线程数：本机 32 核，留 8 核给系统。重排走 CPU + 单槽，线程是实打实用得上的。
重排线程 = min(24, os.cpu_count() or 8)

# 起服务后等 /health 的上限（重排要加载 636 MB 模型，宽一点）
等重排秒 = 90
等知识库秒 = 60

# 知识库服务的"版本判据"字段：旧代码的 /health 里**没有**这个键
版本判据键 = "class3"

# ======================= 小工具 =======================

监听模式 = re.compile(rb"^TCP\S*\s+\S+:(\d+)\s+\S+\s+LISTENING\s+(\d+)\s*$")


def 显示宽(文本):
    """中文按 2 宽的显示宽度。f-string 的 `<12` 对中文会错位，表格对齐只能自己算。"""
    宽 = 0
    for 字 in 文本:
        宽 += 2 if unicodedata.east_asian_width(字) in ("W", "F") else 1
    return 宽


def 补位(文本, 宽, 右对齐=False):
    空 = max(0, 宽 - 显示宽(文本))
    return (" " * 空 + 文本) if 右对齐 else (文本 + " " * 空)


def 端口占用者(端口):
    """返回监听该端口的 PID 列表（去重、保序）。

    🔴 用二进制正则解析 `netstat -ano`，不解码 —— 中文 Windows 的 netstat 输出是
    GBK，而我们需要的字段全是 ASCII。少一次解码就少一类编码坑。
    """
    结果 = []
    try:
        出 = subprocess.run(["netstat", "-ano"], capture_output=True, timeout=25).stdout
    except Exception:
        return 结果
    for 行 in 出.splitlines():
        匹配 = 监听模式.match(行.strip())
        if 匹配 and int(匹配.group(1)) == 端口:
            pid = int(匹配.group(2))
            if pid not in 结果:
                结果.append(pid)
    return 结果


def 窗口标题(pid):
    """取进程的控制台窗口标题（失败返回 None）。

    🔴 为什么要它：`Select ` 前缀是 **conhost 进入「标记/拖选」模式**的标志。
       有它 = 窗口正被鼠标选中 = 该进程的输出已经被冻住。这是把「服务不应答」
       从一句现象**定位到具体原因**的那条判据（2026-10-10 实测踩到，见 `诊断不应答`）。
    """
    try:
        出 = subprocess.run(["tasklist", "/V", "/FI", "PID eq %d" % pid,
                             "/FO", "CSV", "/NH"],
                            capture_output=True, timeout=15).stdout
    except Exception:                                          # noqa: BLE001
        return None
    # 中文 Windows 的 tasklist 输出是 GBK；标题里的中文丢了无所谓 —— 要判的 `Select ` 是 ASCII。
    文 = 出.decode("utf-8", "replace").strip()
    if not 文 or "No tasks" in 文 or "没有运行" in 文:
        return None
    # CSV 最后一个字段就是窗口标题，而标题本身可能含逗号 → 从**最后一个 `","` 边界**倒着切。
    尾 = 文.rstrip('"')
    位 = 尾.rfind('","')
    return 尾[位 + 3:] if 位 >= 0 else None


def 诊断不应答(端口):
    r"""`/health` 不通时，把「**没起来**」和「**起来了但被卡住**」分开 —— 处置完全不同。

    🔴 后者最坑：端口在 LISTENING、进程活着（CPU 接近 0），但**所有请求超时**，
       连 `/health` 都挂 —— 看着像"服务崩了"，其实进程好好的。
       2026-10-10 实测到的成因是 **Windows 传统控制台被鼠标拖选**：
       conhost 一进「标记」模式，进程的**下一次控制台输出就永久阻塞**，
       而服务每个请求都要往控制台打一行 → 服务就冻死在那一行。
       铁证：同一 PID、没重启，窗口标题从 `Select C:\...\python.exe` 变回
       `C:\...\python.exe` 的瞬间，`/health` 从 6 s 超时变成 0.03 s 返回。
    """
    pids = 端口占用者(端口)
    if not pids:
        return ["  · 端口 %d 上没有监听进程 —— 服务**没起来**（去看它的启动窗口报错）。" % 端口]
    行 = ["  · 端口 %d 上有进程（PID %s）却在 LISTENING 状态下不应答 —— "
          "**进程没死，是被卡住了**。" % (端口, "、".join(str(p) for p in pids))]
    确认 = [p for p in pids if (窗口标题(p) or "").lstrip().startswith("Select ")]
    if 确认:
        行.append("  · 🔴 **已确认**：PID %s 的控制台窗口标题以 `Select ` 开头 —— 窗口正被鼠标拖选。"
                  % "、".join(str(p) for p in 确认))
    else:
        行.append("  · 瞄一眼那个窗口的标题：以 `Select ` 开头 = 窗口正被选中。")
    行 += [
        "    成因：Windows 传统控制台一进「标记/拖选」模式，进程的下一次输出就**永久阻塞**；",
        "    本服务每个请求都要往控制台打一行 → 冻在那一行（端口还在听、CPU 0%、全超时）。",
        "    恢复：到那个窗口**按一下 Esc**（或点一下窗口）—— 立刻恢复，**不用重启**。",
        "    （新代码已把请求日志改成「先落盘、再交给后台线程上屏」，窗口拖选不再影响服务。）",
    ]
    return 行


_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.OpenProcess.restype = wintypes.HANDLE
_kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_kernel32.QueryFullProcessImageNameW.argtypes = [
    wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)
]

进程查询权限 = 0x1000                # PROCESS_QUERY_LIMITED_INFORMATION
进程终止权限 = 0x0001                # PROCESS_TERMINATE


# FILETIME 的起点（1601-01-01）到 Unix epoch（1970-01-01）的秒数
FILETIME_基点差 = 11644473600


def 进程启动时刻(pid):
    """进程创建时刻 —— 返回**本地时间**的 datetime（失败返回 None）。

    🔴 时区这一步最容易错，而且错了**看不出来**：
    `GetProcessTimes` 给的 FILETIME 是 **UTC**（1601-01-01 起的 100 纳秒数）。
    如果直接拿 `datetime(1601,1,1) + timedelta(...)` 当结果，得到的是 **UTC 时刻**，
    而源码 mtime / `datetime.now()` 都是**本地时间** —— 拿两者相比会差 8 小时，
    却依然能"讲通一个故事"（"进程启动早于改代码 → 跑的是旧代码"），
    这正是「判据看着能分辨、其实不能分辨」的典型形态。
    所以统一先减基点差换算成 Unix 秒，再 `fromtimestamp` 交给系统转本地。
    """
    句柄 = _kernel32.OpenProcess(进程查询权限, False, pid)
    if not 句柄:
        return None
    try:
        创建 = wintypes.FILETIME()
        退出 = wintypes.FILETIME()
        内核 = wintypes.FILETIME()
        用户 = wintypes.FILETIME()
        成功 = _kernel32.GetProcessTimes(
            句柄, ctypes.byref(创建), ctypes.byref(退出), ctypes.byref(内核), ctypes.byref(用户)
        )
    finally:
        _kernel32.CloseHandle(句柄)
    if not 成功:
        return None
    值 = (创建.dwHighDateTime << 32) | 创建.dwLowDateTime
    try:
        return datetime.datetime.fromtimestamp(值 / 10_000_000 - FILETIME_基点差)
    except (OSError, OverflowError, ValueError):
        return None


def 进程启动时间(pid):
    """`进程启动时刻` 的字符串形式，取不到时说清楚是「取不到」。"""
    时刻 = 进程启动时刻(pid)
    return 时刻.strftime("%Y-%m-%d %H:%M:%S") if 时刻 else "取不到"


def 源码新鲜度(源文件, 进程时刻):
    """进程是在源码最后一次修改**之后**启动的吗？

    🔴 这是「重启了 ≠ 生效了」唯一硬判据：比**时刻**，不比动作。
    返回 (源码改动时刻, 是否比源码新) 或 None（缺数据时绝不猜）。
    """
    if 进程时刻 is None or not os.path.isfile(源文件):
        return None
    改动 = datetime.datetime.fromtimestamp(os.path.getmtime(源文件))
    return 改动, 进程时刻 > 改动


def 进程程序路径(pid):
    句柄 = _kernel32.OpenProcess(进程查询权限, False, pid)
    if not 句柄:
        return "打不开(错误 %d)" % ctypes.get_last_error()
    try:
        缓冲 = ctypes.create_unicode_buffer(1024)
        长度 = wintypes.DWORD(1024)
        成功 = _kernel32.QueryFullProcessImageNameW(句柄, 0, 缓冲, ctypes.byref(长度))
    finally:
        _kernel32.CloseHandle(句柄)
    return 缓冲.value if 成功 else "取路径失败"


def 结束进程(pid):
    """终止单个进程。返回 (成功, 说明)。

    走 ctypes 而不是 `taskkill`，有两个理由：
      · Git Bash 会把 `taskkill /F` 的 `/F` 当路径转换（报 Invalid argument - 'F:/'）；
      · taskkill 的输出也是 GBK，容易糊在终端里。
    """
    句柄 = _kernel32.OpenProcess(进程终止权限, False, pid)
    if not 句柄:
        return False, "打不开进程（错误 %d，可能需要管理员权限）" % ctypes.get_last_error()
    try:
        成功 = _kernel32.TerminateProcess(句柄, 1)
    finally:
        _kernel32.CloseHandle(句柄)
    return bool(成功), "已发送终止" if 成功 else "终止失败（错误 %d）" % ctypes.get_last_error()


def 探活(端口, 超时=1.5):
    import socket
    套 = socket.socket()
    套.settimeout(超时)
    try:
        return 套.connect_ex(("127.0.0.1", 端口)) == 0
    finally:
        套.close()


def 等健康(端口, 上限秒, 说明):
    """轮询 /health 直到 200。返回 (是否通过, 响应体或 None, 耗时)。"""
    地址 = "http://127.0.0.1:%d/health" % 端口
    起 = time.time()
    while time.time() - 起 < 上限秒:
        try:
            with urllib.request.urlopen(地址, timeout=3) as 响应:
                if 响应.status == 200:
                    return True, json.loads(响应.read().decode("utf-8", "replace")), time.time() - 起
        except Exception:
            pass
        time.sleep(1.0)
    return False, None, time.time() - 起


def 询问(强制, 提示):
    if 强制:
        print("     （--强制：不询问，直接执行）")
        return True
    try:
        答 = input("     %s [y/N] " % 提示).strip().lower()
    except EOFError:
        print("     （非交互环境，视为放弃）")
        return False
    return 答 in ("y", "yes")


# ======================= 命令拼装（--演练 也走这里，保证预览=实际） =======================

def 重排命令():
    return [
        引擎程序路径(),              # 🔴 必须是绝对路径 —— 裸名字在 Windows 上找不到（见该函数注释）
        "--model", 重排模型,
        "--reranking",
        "--host", "127.0.0.1",
        "--port", str(重排端口),
        "-c", "8192",
        "-b", "2048", "-ub", "2048",
        "-t", str(重排线程),
        "--parallel", "1",          # 🔴 不能省：多槽会让浮点权重跨请求不可复现
        "--no-webui",               # 少一个会误报 -1 的 WebUI 面
    ]


def 知识库命令():
    # 用**当前解释器**：用户终端里的 python 与 APP 无关，不必锁版本（服务只用标准库）
    return [sys.executable, os.path.join(脚本目录, "知识库服务.py")]


def 命令可读(命令):
    """人类可读的一行命令（仅用于打印，不用于执行 —— 带空格的路径引号是 shell 的事）。"""
    部 = []
    for 段 in 命令:
        部.append('"%s"' % 段 if (" " in 段 and not 段.startswith('"')) else 段)
    return " ".join(部)


# ======================= 体检 =======================

角色表 = [
    ("重排", 重排端口, "本脚本起"),
    ("知识库", 知识库端口, "本脚本起"),
    ("生成端", 生成端端口, "APP 起"),
]

# 单文件 Python 服务可以做「跑的是不是最新代码」的硬判据（二进制服务比不了）
服务源文件 = {"知识库": os.path.join(脚本目录, "知识库服务.py")}


def 体检(打印=True):
    """数一遍每个端口上有几个实例 —— 这是本脚本第一位的判据。"""
    现场 = {}
    for 角色, 端口, 归属 in 角色表:
        占用 = 端口占用者(端口)
        现场[角色] = {"端口": 端口, "PID": 占用, "归属": 归属}

    if not 打印:
        return 现场

    print("=" * 96)
    print("现场体检：每个端口上有几个实例？（>1 就是「重启了却没生效」的根因）")
    print("=" * 96)
    print("  %s %s %s %s %s" % (
        补位("角色", 10), 补位("端口", 7), 补位("实例", 6),
        补位("PID / 启动时间", 42), "判定"))
    for 角色, 端口, 归属 in 角色表:
        信息 = 现场[角色]
        pids = 信息["PID"]
        if not pids:
            判定 = "未运行（%s）" % 归属
        elif len(pids) == 1:
            判定 = "✅ 单实例"
        else:
            判定 = "❌ 重复实例！请求会随机分流"
        明细 = "、".join("%d @ %s" % (p, 进程启动时间(p)) for p in pids) if pids else "—"
        print("  %s %s %s %s %s" % (
            补位(角色, 10), 补位(str(端口), 7), 补位(str(len(pids)), 6),
            补位(明细, 42), 判定))
        if len(pids) > 1:
            for p in pids:
                print("      └ PID %-7d %s" % (p, 进程程序路径(p)))
        # 「重启了 ≠ 生效了」的硬判据：比时刻，不比动作。重复实例时能直接看出哪个新旧。
        源 = 服务源文件.get(角色)
        if 源 and pids:
            for p in pids:
                结论 = 源码新鲜度(源, 进程启动时刻(p))
                if not 结论:
                    continue
                改动, 新鲜 = 结论
                print("      └ PID %-7d 源码最后改于 %s → %s" % (
                    p, 改动.strftime("%m-%d %H:%M:%S"),
                    "进程比它新 ✅（跑的是新代码）" if 新鲜
                    else "**进程比它旧 ❌ 跑的是旧代码，重启才生效**"))
    print()
    return 现场


def 检查前置条件():
    """起之前先确认"该有的东西在不在" —— 缺了就直接说清，别等它崩。

    🔴 这里查的必须是**命令里真正会执行的那一段**（`重排命令()[0]`），
    不能另写一份"我以为的路径"。2026-10-10 的 `WinError 2` 就是因为这两者不一致：
    前置条件查的是 `引擎目录/llama-server.exe`（存在 → 全绿），
    而命令第 0 段是裸名字 `llama-server.exe`（Windows 找不到）→ `Popen` 崩。
    **判据与执行各走各的，判据就等于没有。**
    """
    问题 = []
    argv0 = 重排命令()[0]
    if not os.path.isabs(argv0):
        问题.append("引擎程序不是绝对路径（Windows 不会用 cwd 去找 exe）：%s" % argv0)
    elif not os.path.isfile(argv0):
        问题.append("引擎不存在：%s" % argv0)
    if not os.path.isfile(重排模型):
        问题.append("重排模型不存在：%s" % 重排模型)
    if not os.path.isfile(os.path.join(脚本目录, "知识库服务.py")):
        问题.append("服务本体不存在：%s" % os.path.join(脚本目录, "知识库服务.py"))
    return 问题


def 验引擎可执行(不打印=False):
    """把引擎 exe **真的拉起来一次**（`--version`），证明 Windows 找得到它、DLL 也加载得上。

    🔴 为什么"文件存在"不算数：上面那个 `WinError 2` 正是「文件在、但命令第 0 段不是绝对路径」。
    `os.path.isfile()` 全绿，`Popen` 崩 —— **"看得到文件" ≠ "起得来"**，
    与「`/health` 通 ≠ 链路通」是同一类毛病。

    顺带解决另一个难查的现象：**新窗口一闪就没**。缺 DLL 时起进程会立刻退出，
    控制台窗口闪一下就没了，什么都看不到；这里把退出码与它的输出**接回来打印**。
    实测 `--version` 只打印版本就退出、耗时 0.37 s，是当前最便宜的真启动证据。

    返回 None = 通过，否则返回一行说明。
    """
    exe = 重排命令()[0]
    try:
        p = subprocess.Popen([exe, "--version"], cwd=引擎目录,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        出, _ = p.communicate(timeout=60)
    except FileNotFoundError:
        return "找不到 `%s`（命令第 0 段；Windows 不会用 cwd 去找 exe）" % exe
    except Exception as 错:                                        # noqa: BLE001
        return "起不来：%s：%s" % (type(错).__name__, 错)
    文本 = 出.decode("utf-8", "replace").strip()
    if p.returncode != 0:
        return "退出码 %d ｜ %s" % (p.returncode, 文本[:160] or "（无输出）")
    if not 不打印:
        首行 = 文本.splitlines()[0] if 文本 else "（无输出）"
        print("  ✅ 引擎可执行：%s" % 首行)
    return None


# ======================= 动作：状态 / 演练 =======================

def 动作_状态():
    体检()
    print("=" * 96)
    print("依赖与版本判据")
    print("=" * 96)
    for 角色, 端口, _归属 in 角色表:
        if not 探活(端口):
            print("  %s（%d） ⏭ 不可达 —— 没法验" % (补位(角色, 10), 端口))
            continue
        通过, 体, 耗 = 等健康(端口, 5, 角色)
        if not 通过:
            print("  %s（%d） ❌ 端口活着但 /health 不通（%s 有「健康但请求挂起」的先例）"
                  % (补位(角色, 10), 端口, 角色))
            continue
        if 角色 == "知识库":
            有判据 = 版本判据键 in 体
            指纹 = "有 %s（新代码）" % 版本判据键 if 有判据 else "**没有 %s —— 跑的是旧代码**" % 版本判据键
            print("  %s（%d） ✅ /health 通了 %.1fs ｜ 版本判据：%s"
                  % (补位(角色, 10), 端口, 耗, 指纹))
            if 有判据:
                print("      └ %s" % json.dumps(体.get(版本判据键), ensure_ascii=False))
            print("      └ deps = %s" % json.dumps(体.get("deps"), ensure_ascii=False))
            print("      └ 问答页 http://127.0.0.1:%d/   ← 浏览器直接打开" % 端口)
            # 🔴 页面是**独立文件**（web/index.html），所以 `/health` 通了不代表它在 ——
            #    必须单独探一次。同一条教训：健康 ≠ 链路通。
            try:
                # 🔴 必须**读完 body 并关闭**。早先这版只取响应头就把 response 丢下不管，
                #    socket 被 GC 回收时发 RST，而服务端此刻还在写这 14 KB 页面 ——
                #    于是**服务端**控制台打出一整页 `WinError 10053`，
                #    看起来像服务崩了，其实是我这个探针在捣乱。
                #    「客户端偷懒」会伪装成「服务端故障」，别再写这种一次性用法。
                with urllib.request.urlopen("http://127.0.0.1:%d/" % 端口, timeout=8) as 页:
                    类 = 页.headers.get("Content-Type") or ""
                    大小 = len(页.read())
                if 页.status == 200 and "text/html" in 类:
                    print("      └ 问答页探测：✅ 200 HTML ｜ %d 字节" % 大小)
                else:
                    print("      └ 问答页探测：❌ 状态 %d ｜ %s" % (页.status, 类))
            except urllib.error.HTTPError as 错:
                # 服务活着但这条路没接上 = 端口上多半跑着**没有 Web 页的旧实例**。
                # 与下面的「连不上」是两种问题，分开报，别让人去查错方向。
                print("      └ 问答页探测：❌ HTTP %d —— 路由没接上（端口上多半是旧实例，重启）"
                      % 错.code)
            except Exception as 错:                            # noqa: BLE001
                print("      └ 问答页探测：❌ %s —— 检查 web/index.html 是否存在"
                      % type(错).__name__)
            # 日志落点也一起报：窗口拖选时屏幕会丢行，但日志文件里一行没缺 —— 差别就在这。
            if 体.get("log_file"):
                print("      └ 请求日志 %s ｜ 屏幕丢行 %s"
                      % (体.get("log_file"), 体.get("console_dropped")))
            else:
                print("      └ 请求日志 ❌ /health 无 log_file —— 旧实例（请求日志只上屏，"
                      "窗口拖选会冻住服务）")
        else:
            print("  %s（%d） ✅ /health 通了 %.1fs" % (补位(角色, 10), 端口, 耗))
    print()
    print("  提示：`--状态` 只读，不会启动或停止任何东西。")


def 动作_演练():
    print("=" * 96)
    print("演练（只打印，不执行）—— 实际执行的命令与此完全相同")
    print("=" * 96)
    print()
    print("【① 重排】cwd = %s" % 引擎目录)
    print("    %s" % 命令可读(重排命令()))
    print("    · --parallel 1 不能省：多槽时浮点权重跨请求不可复现（Q8_0 免疫，但留着是保险）")
    print("    · 不带 -ngl → 惰性初始化 → 只跑 CPU、不占显存")
    print("    · 起之前会先真跑一次 `<exe> --version` 验可执行（**文件在 ≠ 起得来**）")
    print()
    print("【② 知识库】cwd = %s" % 仓库根)
    print("    %s" % 命令可读(知识库命令()))
    print("    · 启动时会自检依赖：重排 ✅／生成端 ✅；起手走 BM25，不碰向量服务")
    print()
    print("【③ 生成端】**本脚本不起** —— 请在 Local-AI APP 里点「启动模型」")
    print("    APP 管模型生命周期与显存调优；脚本代起会和 APP 抢资源。")
    print()
    print("顺序：① 重排 /health 200 → ② 知识库 → 用的时候 ③ 生成端（APP）")
    print()
    print("前置条件检查：")
    for 问题 in 检查前置条件():
        print("  ❌ %s" % 问题)
    if not 检查前置条件():
        print("  ✅ 引擎 / 重排模型 / 服务本体 都在")


# ======================= 动作：停 =======================

def 动作_停(端口列表, 强制=False):
    print("=" * 96)
    print("停止服务")
    print("=" * 96)
    for 端口 in 端口列表:
        pids = 端口占用者(端口)
        if not pids:
            print("  端口 %d：无监听，跳过" % 端口)
            continue
        print("  端口 %d 上有 %d 个进程：" % (端口, len(pids)))
        for p in pids:
            print("      PID %-7d 启动于 %s" % (p, 进程启动时间(p)))
            print("         %s" % 进程程序路径(p))
        if not 询问(强制, "停掉端口 %d 上的 %d 个进程？" % (端口, len(pids))):
            print("      → 已跳过端口 %d" % 端口)
            continue
        for p in pids:
            成功, 说明 = 结束进程(p)
            print("      PID %-7d %s" % (p, 说明))
        # 终止是异步的，等端口真的空出来再报结论（别把"发了信号"当"停了"）
        起 = time.time()
        while time.time() - 起 < 6 and 端口占用者(端口):
            time.sleep(0.3)
        剩 = 端口占用者(端口)
        print("      → %s" % ("✅ 已释放" if not 剩 else "❌ 仍有 %d 个进程占着：%s"
                              % (len(剩), 剩)))
    print()


# ======================= 动作：起 =======================

def 起一个(名称, 命令, 工作目录, 前台, 日志=None):
    """起一个服务。返回 Popen 或 None。

    默认 CREATE_NEW_CONSOLE：每个服务一个自己的窗口，日志原样留在那里，**关窗即停** ——
    跟"手工开两个终端"是同一个心智模型。
    """
    try:
        if 前台:
            print("  → 前台模式：输出汇总到当前终端（Ctrl+C 会一起停）")
            子 = subprocess.Popen(命令, cwd=工作目录)
        else:
            旗标 = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
            子 = subprocess.Popen(命令, cwd=工作目录, creationflags=旗标)
    except FileNotFoundError:
        # 别把 `WinError 2` 以裸 Traceback 甩给用户 —— 它只说明"命令第 0 段找不到"。
        print("  ❌ %s 起不来：找不到要执行的文件" % 名称)
        print("     argv[0] = %s" % 命令[0])
        print("     🔴 这段才是 Windows 实际要加载的 exe；它不是绝对路径时，")
        print("        Windows **不会**用 cwd 去找它（用的是父进程 cwd + PATH）。")
        return None
    except OSError as 错:                                          # 权限 / 格式不符等
        print("  ❌ %s 起不来：%s：%s" % (名称, type(错).__name__, 错))
        return None
    print("  → 已启动 %s，PID %d" % (名称, 子.pid))
    return 子


def 动作_起(前台=False, 强制=False, 重起=False):
    现场 = 体检()

    # --- 前置条件 ---
    问题 = 检查前置条件()
    for 一 in 问题:
        print("  ❌ %s" % 一)
    if 问题:
        print("\n  修好上面这些再跑。")
        return 1

    # --- 旧实例：默认不静默重起（本脚本最想防的就是"以为重启了"） ---
    要停 = []
    for 角色 in ("重排", "知识库"):
        信息 = 现场[角色]
        if 信息["PID"] and (重起 or len(信息["PID"]) > 1):
            要停.append(信息)
    if 要停:
        print("  发现已在运行的实例：")
        for 信息 in 要停:
            print("      端口 %d：PID %s" % (信息["端口"], 信息["PID"]))
        if not 重起:
            print("  重复实例会让请求随机分流（改了半天不生效就是这么来的）。")
        if 询问(强制, "停掉它们再起？"):
            动作_停([信息["端口"] for 信息 in 要停], 强制=True)
            现场 = 体检(打印=False)
        else:
            print("  → 保留现有实例，只补缺的。\n")

    子进程表 = []

    # --- ① 重排（知识库的依赖，必须先起） ---
    if 现场["重排"]["PID"]:
        print("  ① 重排（%d）已在运行 —— 跳过启动" % 重排端口)
    else:
        print("  ① 起重排（%d）…" % 重排端口)
        # 前置条件只证明"文件在"；**真拉起来一次**才证明"起得来"（见 `验引擎可执行()`）。
        引擎问题 = 验引擎可执行()
        if 引擎问题:
            print("  ❌ 引擎预检未过：%s" % 引擎问题)
            print("     不往下走了 —— 硬起只会得到一个看不懂的报错、或一闪就没的窗口。")
            return 1
        子 = 起一个("重排", 重排命令(), 引擎目录, 前台)
        if 子 is None:
            return 1
        子进程表.append(子)
        print("     等 /health（模型 636 MB，首次要几秒）…")
        通过, 体, 耗 = 等健康(重排端口, 等重排秒, "重排")
        if not 通过:
            print("     ❌ %g s 内 /health 没通。" % 等重排秒)
            for 一 in 诊断不应答(重排端口):
                print(一)
            return 1
        print("     ✅ 重排就绪（%.1f s）" % 耗)

    # --- ② 知识库 ---
    if 现场["知识库"]["PID"]:
        print("  ② 知识库（%d）已在运行 —— 跳过启动" % 知识库端口)
    else:
        print("  ② 起知识库（%d）…" % 知识库端口)
        子 = 起一个("知识库", 知识库命令(), 仓库根, 前台)
        if 子 is None:
            return 1
        子进程表.append(子)
        通过, 体, 耗 = 等健康(知识库端口, 等知识库秒, "知识库")
        if not 通过:
            print("     ❌ %g s 内 /health 没通。" % 等知识库秒)
            for 一 in 诊断不应答(知识库端口):
                print(一)
            return 1
        print("     ✅ 知识库就绪（%.1f s）" % 耗)

    # --- ③ 生成端：只检查（APP 负责） ---
    if 探活(生成端端口):
        print("  ③ 生成端（%d）✅ 已在跑" % 生成端端口)
    else:
        print("  ③ 生成端（%d）❌ 没在跑 —— 请在 Local-AI APP 里点「启动模型」" % 生成端端口)
        print("      （不影响现在起服务；只是这时候问问题会答不出东西）")

    # --- 自证：起完必须验，而且要用**能区分版本**的判据 ---
    print()
    print("=" * 96)
    print("自证（起完不算数，验过才算数）")
    print("=" * 96)
    坏 = 0
    通过, 体, 耗 = 等健康(知识库端口, 10, "知识库")
    if not 通过:
        print("  ❌ 知识库 /health 不通")
        for 一 in 诊断不应答(知识库端口):
            print(一)
        坏 += 1
    else:
        指纹 = 体.get(版本判据键)
        if 指纹 is None:
            print("  ❌ /health 里**没有 %s** —— 跑的是旧代码（banner 不能当版本判据！）" % 版本判据键)
            坏 += 1
        else:
            print("  ✅ 版本判据 %s = %s" % (版本判据键, json.dumps(指纹, ensure_ascii=False)))
        print("  ✅ 语料 %s 份 / %s 块 ｜ 检索链 %s"
              % (体.get("docs"), 体.get("blocks"), 体.get("chain")))
        print("  ✅ deps = %s" % json.dumps(体.get("deps"), ensure_ascii=False))
        # 日志落点：请求日志已改成「落盘不跟控制台共命运」→ 窗口被拖选不再冻住服务。
        # 🔴 这里必须区分三种情况，**不能只看值**：
        #    · 字段**缺失**  = 跑的是旧代码（旧版只上屏）→ 算失败，重启才生效；
        #    · 字段在但为 null = 新代码，但 logs/ 开不出来 → 同样会冻，也算失败；
        #    · 字段在且有路径 = 新代码且日志可用。
        if "log_file" not in 体:
            print("  ❌ /health 里**没有 log_file** —— 跑的是「请求日志只上屏」的旧实例：")
            print("     窗口一被鼠标拖选，服务就被冻住（端口还在听、CPU 0%、请求全超时）。**重启才生效。**")
            坏 += 1
        elif not 体.get("log_file"):
            print("  ❌ 有新代码，但日志文件没开成（logs/ 不可写？）—— 窗口拖选仍会冻住服务。")
            坏 += 1
        else:
            print("  ✅ 请求日志落盘 = %s（屏幕丢行 %s）"
                  % (体.get("log_file"), 体.get("console_dropped")))
    for 角色 in ("重排", "知识库"):
        剩余 = 端口占用者(dict((r, p) for r, p, _ in 角色表)[角色])
        if len(剩余) > 1:
            print("  ❌ %s 端口上有 %d 个实例 —— 请求会随机分流" % (角色, len(剩余)))
            坏 += 1

    print()
    if 坏:
        print("总判定：❌ 有 %d 项没通过" % 坏)
    else:
        print("总判定：✅ 服务就位。")
        print("  下一步（脚本代劳不了的那一步）：")
        print("    在 Local-AI APP 里切到「知识库问答」，问一句 —— 看答案 + 引用卡片。")
    if 前台 and 子进程表:
        print()
        print("前台模式：Ctrl+C 会一起停掉刚起的 %d 个服务。" % len(子进程表))
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n  收到中断，正在停子进程…")
            for 子 in 子进程表:
                try:
                    子.terminate()
                except Exception:
                    pass
    return 1 if 坏 else 0


# ======================= 入口 =======================

def 主函数(argv):
    参数 = set(argv[1:])

    if not argv[1:]:
        print("=" * 96)
        print("作品一 · 本地服务启动器（重排 8085 + 知识库 8092；生成端 8000 由 APP 负责）")
        print("=" * 96)
        print()
        return 动作_起(前台=False, 强制=False, 重起=False)

    if {"--状态", "--status", "状态"} & 参数:
        动作_状态()
        return 0
    if {"--演练", "--dry-run", "演练"} & 参数:
        动作_演练()
        return 0
    if {"--停", "--stop", "停"} & 参数:
        动作_停([重排端口, 知识库端口], 强制=bool({"--强制", "--force", "-f"} & 参数))
        return 0

    前台 = bool({"--前台", "--foreground"} & 参数)
    强制 = bool({"--强制", "--force", "-f"} & 参数)
    重起 = bool({"--重起", "--restart"} & 参数)
    if not (前台 or 强制 or 重起):
        print("未知参数：%s" % " ".join(sorted(参数)))
        print("可用：--状态 / --演练 / --停 / --重起 / --强制 / --前台")
        return 2

    print("=" * 96)
    print("作品一 · 本地服务启动器（重排 8085 + 知识库 8092；生成端 8000 由 APP 负责）")
    print("=" * 96)
    print()
    return 动作_起(前台=前台, 强制=强制, 重起=重起)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")   # 终端 cp936，不要 reconfigure(encoding=...)
    except Exception:
        pass
    sys.exit(主函数(sys.argv))
