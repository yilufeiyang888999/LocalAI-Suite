# -*- coding: utf-8 -*-
"""ChatBI · 重启生成端（8000）—— **只杀不代起**，把「起」交还给 Local-AI APP

## 为什么要有这个脚本

生成端会进入一种 **「活着但不应答」** 的卡死态：`/health`、`/props`、`/v1/models`
全部秒回 200，而任何生成请求**永远不返回**（判据与成因见同目录
`诊断-生成端卡死.py`；一句话：客户端在请求返回前被强杀 → slot 没被归还）。

处置只有一个：**重启模型服务**。以前必须手工到 APP 里点「停止」再「启动」，
本脚本把这个动作变成一条命令。

## 🔴 边界：脚本只做「杀」，不做「起」——「一个进程只能有一个主」

`启动-服务.py` 立过这条规矩：**生成端归 APP 管**（模型生命周期、显存调优都在 APP 里）。
本脚本**不破**这条规矩，办法是借用 APP **自带的看门狗**：

    APP 的 `service_watchdog_loop`（`src-tauri/src/service.rs`）每 5 秒
    `child.try_wait()` 一次；进程死了 → 清句柄 → **立即**调 `start_service_core()`
    重新拉起（首轮无 backoff）。它自己的注释写得很清楚，支持的场景就包括
    「**user killed it from Task Manager**」。

所以「杀进程」不是绕开 APP，而是**用 APP 自己认可的方式**告诉它「服务挂了，重启吧」。
进程的主自始至终是 APP，脚本只是触发者；界面状态也会跟着同步
（看门狗走 `set_running` → `emit("service_status_changed")`）。

> 如果脚本自己 `spawn` 一个 llama-server，就会出现「两个主」：APP 的界面状态、
> 下次 `recover_service()` 的 reattach 判断、显存账本全都对不上。
> **这条线一步都不能越。**

## 🔴 前提：只有「APP 直管」的服务，杀完才会自动回来

看门狗判死的那一步是这样写的：

    match inner.service.process.as_mut() {
        Some(child) => child.try_wait()...,   // APP 自己 spawn 的 → 有句柄，能盯着
        None => None,                         // reattach 来的   → **拿不到句柄，跳过**
    }

`None` 出现在 **APP 重启后 reattach 到旧服务** 的情形（`recover_service()`：
「A healthy port means the old process is still serving — reattach to it」）。
**此时看门狗是空转的，杀了没人拉。**

判据（本脚本第 ② 步就做）：**看引擎进程的父进程是谁**。

    · 父进程 = 当前活着的 `local-ai-installer.exe`  → **直管态** ✅ 可以杀
    · 父进程不存在 / 不是 APP                       → **接管态** ❌ 杀了回不来，脚本拒不动手

Windows 的 PPID 是**创建时写死**的、父进程死后不更新 —— 所以「PPID 指向一个已死的
PID」正好是 reattach 的指纹，这个判据成立。

### 顺带解释一个反直觉的现象：「重启 Local-AI APP 常常没用」

全仓库**没有任何**地方在退出时清理子进程（没有 `Drop for Child`、没有
`RunEvent::Exit` 钩子）。于是：

    关掉 APP  →  llama-server 变孤儿，**端口还活着**（卡死态尤其如此）
    重开 APP  →  `recover_service()` 一试 `health_check_once(端口)` → **200**（卡死不影响它）
              → 直接 reattach，**把卡死的引擎原样接管回来**

所以「重启 APP」不但重（dev 模式还要重新编译），而且在**卡死这个场景下方向就是错的**：
它优先复用那个坏掉的进程。**先杀引擎、再让它被重新拉起** 才是最短路径。

（推论：既然 APP 退出不清子进程，那么「重启 APP 生效了」的真实原因，
多半是旧引擎恰好已经死了/被手工清了，而不是「重启 APP」这个动作本身有效。）

## 🔴 第三条：不能只看「端口活」就算重启成功

本项目反复踩的坑 —— **「活着」≠「能干这件事」**（`MEMORY.md` §9.5）。
生成端尤其如此：**卡死态本身端口就是活的**。所以复验缺一不可：

    · **PID 必须变了**  → 旧进程没死透时端口照样活，只有 PID 变了才说明真的换了进程
    · **`/slots` 必须拿得到锁** → 新进程起来但 slot 又被占死，得继续查（本轮会判失败）

## 用法

    python scripts\\重启-生成端.py                    # 体检 → 询问 → 杀 → 等 APP 拉起 → 复验
    python scripts\\重启-生成端.py --演练              # 只打印将要做什么，一个进程都不动
    python scripts\\重启-生成端.py --强制              # 不问，直接干
    python scripts\\重启-生成端.py --端口 8000 --等待秒 180
    python scripts\\重启-生成端.py --接管态也杀        # 明知 APP 拉不回来，仍要杀掉（比如就是要腾显存）

退出码：0 = 重启成功且复验通过；1 = 失败（附现场与下一步）；3 = 跳过（演练 / 用户拒绝 / 本来就没起）
"""

import argparse
import ctypes
import ctypes.wintypes as wintypes
import datetime
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    try:
        # 终端是 cp936；不要 reconfigure(encoding="utf-8")，用 errors=replace 就够
        sys.stdout.reconfigure(errors="replace")
    except Exception:                                          # noqa: BLE001
        pass

脚本目录 = os.path.dirname(os.path.abspath(__file__))
当前进程 = os.getpid()


# ======================= 复用诊断模块（判据同源，不抄第二份） =======================
#
# 🔴 为什么用 importlib 而不是复制那几个函数：
#    `端口占用者()` / `探()` / `采样CPU()` 的判据必须与 `诊断-生成端卡死.py`
#    **逐字同源**。抄一份到自己文件里，就又多出一处「同一份事实两处定义」——
#    本项目已经因为这类问题吃过亏（改一处跑另一处，不报错、只给错报告）。
#
#    文件名带连字符（`诊断-生成端卡死`）不是合法标识符，不能直接 `import`，
#    所以走 importlib 的 spec_from_file_location。

def 载入诊断模块():
    路径 = os.path.join(脚本目录, "诊断-生成端卡死.py")
    if not os.path.isfile(路径):
        raise SystemExit("找不到同目录的诊断模块：%s" % 路径)
    规格 = importlib.util.spec_from_file_location("诊断生成端卡死", 路径)
    模块 = importlib.util.module_from_spec(规格)
    规格.loader.exec_module(模块)
    return 模块


诊断 = 载入诊断模块()
探 = 诊断.探
端口占用者 = 诊断.端口占用者
采样CPU = 诊断.采样CPU
免锁路径 = 诊断.免锁路径


# ======================= 进程表（拿 PPID，判「谁拉起谁」） =======================

APP进程名 = "local-ai-installer.exe"
默认端口 = 8000
默认等待秒 = 180
看门狗轮询秒 = 5

TH32CS_SNAPPROCESS = 0x00000002
进程查询权限 = 0x1000                 # PROCESS_QUERY_LIMITED_INFORMATION
进程终止权限 = 0x0001                 # PROCESS_TERMINATE

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)


class 进程条目(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", ctypes.c_wchar * 260),
    ]


_k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
_k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
_k32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(进程条目)]
_k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(进程条目)]
_k32.CloseHandle.argtypes = [wintypes.HANDLE]
_k32.OpenProcess.restype = wintypes.HANDLE
_k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_k32.QueryFullProcessImageNameW.argtypes = [
    wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)
]
_k32.GetProcessTimes.argtypes = [
    wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
]

FILETIME_基点差 = 11644473600           # 1601-01-01 → 1970-01-01 的秒数


def 进程表():
    """返回 {pid: (父pid, 映像名)}。

    🔴 为什么非要 PPID：它是本脚本**唯一**能区分「APP 直管」与「APP 接管」的外部判据。
       看门狗只在 `service.process` 是 `Some`（自己 spawn 的）时才盯得住进程；
       reattach 来的服务 `process` 是 `None`，看门狗空转 —— 杀了没人拉。
       而「父进程是不是当前活着的 APP」正好把这两态分开。
    """
    表 = {}
    句柄 = _k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if not 句柄 or 句柄 == wintypes.HANDLE(-1).value:
        return 表
    try:
        条目 = 进程条目()
        条目.dwSize = ctypes.sizeof(进程条目)
        好 = _k32.Process32FirstW(句柄, ctypes.byref(条目))
        while 好:
            表[int(条目.th32ProcessID)] = (int(条目.th32ParentProcessID), 条目.szExeFile)
            好 = _k32.Process32NextW(句柄, ctypes.byref(条目))
    finally:
        _k32.CloseHandle(句柄)
    return 表


def 进程程序路径(pid):
    句柄 = _k32.OpenProcess(进程查询权限, False, pid)
    if not 句柄:
        return None
    try:
        缓冲 = ctypes.create_unicode_buffer(1024)
        长度 = wintypes.DWORD(1024)
        成功 = _k32.QueryFullProcessImageNameW(句柄, 0, 缓冲, ctypes.byref(长度))
    finally:
        _k32.CloseHandle(句柄)
    return 缓冲.value if 成功 else None


def 进程启动时刻(pid):
    """进程创建时刻（本地时间）。时区这一步最容易错：FILETIME 是 UTC，先换算成 Unix 秒再交给系统转本地。"""
    句柄 = _k32.OpenProcess(进程查询权限, False, pid)
    if not 句柄:
        return None
    try:
        创建 = wintypes.FILETIME()
        退出 = wintypes.FILETIME()
        内核 = wintypes.FILETIME()
        用户 = wintypes.FILETIME()
        成功 = _k32.GetProcessTimes(句柄, ctypes.byref(创建), ctypes.byref(退出),
                                   ctypes.byref(内核), ctypes.byref(用户))
    finally:
        _k32.CloseHandle(句柄)
    if not 成功:
        return None
    值 = (创建.dwHighDateTime << 32) | 创建.dwLowDateTime
    try:
        return datetime.datetime.fromtimestamp(值 / 10_000_000 - FILETIME_基点差)
    except (OSError, OverflowError, ValueError):
        return None


def 结束进程(pid):
    """强杀单个进程。返回 (是否成功, 说明)。

    🔴 走 ctypes 而不是 `taskkill`：Git Bash 会把 `taskkill /F` 的 `/F` 当路径转换
       （报 Invalid argument - 'F:/'），而且 taskkill 输出是 GBK 容易糊在终端里。
       这与 `启动-服务.py` 的 `结束进程()` 是同一个做法。
    """
    句柄 = _k32.OpenProcess(进程终止权限, False, pid)
    if not 句柄:
        return False, "打不开进程（错误 %d，可能需要管理员权限）" % ctypes.get_last_error()
    try:
        成功 = _k32.TerminateProcess(句柄, 1)
    finally:
        _k32.CloseHandle(句柄)
    return bool(成功), "已发送终止" if 成功 else "终止失败（错误 %d）" % ctypes.get_last_error()


# ======================= 小工具 =======================

def 显存():
    """(已用 MiB, 总量 MiB) 或 (None, None)。拿不到就不报 —— 不阻断主流程。"""
    try:
        出 = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, timeout=20, encoding="utf-8", errors="replace").stdout.strip()
        行 = 出.splitlines()[0] if 出 else ""
        用, 总 = [x.strip() for x in 行.split(",")[:2]]
        return int(用), int(总)
    except Exception:                                          # noqa: BLE001
        return None, None


def 取JSON(端口, 路径, 超时=6):
    """绕代理取 JSON。本环境有 HTTP_PROXY，本机探测必须显式绕开，否则「活着却连不上」。"""
    开启器 = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with 开启器.open("http://127.0.0.1:%d%s" % (端口, 路径), timeout=超时) as 响应:
        return json.loads(响应.read().decode("utf-8", "replace"))


def 当前模型名(端口):
    try:
        体 = 取JSON(端口, "/v1/models")
        条目 = (体.get("models") or [{}])[0]
        名字 = 条目.get("name") or ""
        return os.path.basename(名字) if 名字 else None
    except Exception:                                          # noqa: BLE001
        return None


def 槽位数(端口):
    """当前 `--parallel` 生效值（llama-server 的 /props 里有 total_slots）。"""
    try:
        return 取JSON(端口, "/props").get("total_slots")
    except Exception:                                          # noqa: BLE001
        return None


def 说(文本=""):
    print(文本)


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


# ======================= ① 现场：谁是引擎、谁拉起它 =======================

def 摸现场(端口):
    """返回一个字典：引擎进程、APP 主进程、两者关系（直管 / 接管 / 游离）。"""
    听, 待关 = 端口占用者(端口)
    表 = 进程表()
    # 同名的 APP 可能有多个（比如 dev 期间遗留的旧实例）→ 全列出来，别只取第一个
    APP们 = sorted(p for p, (_, 名) in 表.items()
                  if 名 and 名.lower() == APP进程名.lower())

    引擎 = None
    if 听:
        引擎 = {
            "PID": 听,
            "映像": 表.get(听, (None, "?"))[1],
            "父PID": 表.get(听, (None, None))[0],
            "路径": 进程程序路径(听),
            "启动": 进程启动时刻(听),
        }

    关系 = "无引擎"
    父嫌疑 = None
    if 引擎:
        父PID = 引擎["父PID"]
        父名 = 表.get(父PID, (None, None))[1] if 父PID else None
        父活着 = 父PID in 表 and 父PID != 0
        if 父活着 and 父名 and 父名.lower() == APP进程名.lower():
            关系 = "直管态"                      # APP 自己 spawn 的 → 看门狗盯得住
            父嫌疑 = 父PID
        elif 父活着 and 父名 and 父名.lower() == 引擎["映像"].lower():
            关系 = "直管态"                      # 引擎被自己同类拉起（极少见，仍按直管报）
            父嫌疑 = 父PID
        elif not 父活着:
            关系 = "接管态"                      # PPID 指向已死的进程 = reattach 的指纹
        else:
            关系 = "游离态"                      # 父进程是别的东西（人为起的、或 PID 被复用）

    return {
        "端口": 端口, "听": 听, "待关": 待关, "引擎": 引擎, "APP们": APP们,
        "关系": 关系, "父嫌疑": 父嫌疑, "表": 表,
    }


def 报现场(现场):
    说("① 现场")
    if not 现场["听"]:
        说("   %d 端口上**没有进程监听** —— 生成端本来就没起。" % 现场["端口"])
        return
    e = 现场["引擎"]
    说("   引擎：PID %d ｜ %s ｜ 启动于 %s"
       % (e["PID"], e["映像"], e["启动"].strftime("%Y-%m-%d %H:%M:%S") if e["启动"] else "取不到"))
    说("         %s" % (e["路径"] or "（取不到路径）"))
    父 = e["父PID"]
    父名 = 现场["表"].get(父, (None, "（已不存在）"))[1] if 父 else "（无）"
    说("   父进程：PID %s ｜ %s" % (父, 父名))
    if 现场["APP们"]:
        for p in 现场["APP们"]:
            说("   APP 主进程：PID %d ｜ %s" % (p, 进程程序路径(p) or "（取不到路径）"))
    else:
        说("   APP 主进程：**一个都没在跑**")

    关系 = 现场["关系"]
    if 关系 == "直管态":
        说("   → 🟢 **直管态**：引擎由当前 APP 拉起，APP 的看门狗盯得住它。")
        说("        杀掉后 %d 秒内 APP 会自己 `start_service_core()` 把它拉回来。" % 看门狗轮询秒)
    elif 关系 == "接管态":
        说("   → 🔴 **接管态**：引擎的父进程已经不存在了 —— 说明 APP 重启时")
        说("        `recover_service()` 见到端口健康，**直接接管**了这个进程（没有自己 spawn）。")
        说("        这种情况看门狗**拿不到 Child 句柄，是空转的** —— 杀了不会自动回来。")
    elif 关系 == "游离态":
        说("   → 🟠 **游离态**：拉起它的不是本 APP（是别的程序、或旧 PID 被系统复用了）。")
        说("        看门狗同样管不到它 —— 杀了不会自动回来。")
    else:
        说("   → （无引擎，无需判定）")


# ======================= ② 诊断：它到底是不是卡死的 =======================

def 报诊断(端口, pid):
    """复用三步判的口径，但只报结论（详细现场让 `诊断-生成端卡死.py` 去打）。"""
    说("② 诊断（卡死判据与 `诊断-生成端卡死.py` 同源）")
    都秒回 = True
    for 路径 in 免锁路径:
        码, 用时, _ = 探(端口, 路径)
        if 码 is None or 用时 > 3:
            都秒回 = False
    码, 用时, _ = 探(端口, "/slots", 超时=6)
    slot卡 = 码 is None
    说("   免锁路径 %s：%s" % ("、".join(免锁路径), "✅ 全秒回" if 都秒回 else "⚠️ 有慢的/超时的"))
    说("   /slots（要 slot 锁）：%s"
       % ("🔴 **拿不到锁 → slot 被占死**" if slot卡
          else "✅ 拿到了锁（slot 空闲，用时 %.2fs）" % 用时))
    cpu = 采样CPU(pid, 5) if pid else None
    说("   PID %s ｜ 5 秒 CPU 增量：%s"
       % (pid, "拿不到" if cpu is None else "%s 秒" % cpu))

    if slot卡 and 都秒回:
        结论 = "卡死（活着但不应答）"
    elif slot卡 and not 都秒回:
        结论 = "HTTP 层也不通（更像输出被堵，查控制台是否被鼠标拖选）"
    elif not slot卡 and cpu is not None and cpu > 0.5:
        结论 = "没卡死，但进程在烧 CPU（可能在算，或显存被别的进程占着）"
    elif not slot卡:
        结论 = "空闲可用（本来就不卡）"
    else:
        结论 = "现象不典型"
    说("   → %s" % 结论)
    return {"卡死": slot卡, "结论": 结论}


# ======================= ③ 执行：杀 =======================

def 杀引擎(pid):
    说("③ 终止引擎进程")
    成功, 说明 = 结束进程(pid)
    说("   PID %d ｜ %s" % (pid, 说明))
    return 成功


# ======================= ④ 等 APP 把服务拉回来 =======================

def 等复苏(端口, 旧PID, 等待秒):
    """轮询直到：新 PID 监听 + /health 200 + /slots 拿得到锁。返回 (是否成功, 新PID, 耗时, 末态)。"""
    起 = time.time()
    上次 = None
    末态 = "（未开始）"
    while True:
        耗 = time.time() - 起
        if 耗 >= 等待秒:
            return False, None, 耗, 末态

        听, _ = 端口占用者(端口)
        if not 听:
            态 = "端口无监听 —— 等 APP 看门狗发现（轮询周期 %d 秒）" % 看门狗轮询秒
        elif 听 == 旧PID:
            态 = "端口仍挂在旧 PID %d 上（还没死透）" % 旧PID
        elif 听 == 当前进程:
            态 = "端口挂在本脚本进程上？（不该发生）"
        else:
            码, _, _ = 探(端口, "/health", 3)
            if 码 != 200:
                态 = "新 PID %d 已在监听，/health 还没通（模型正在加载/上显存）" % 听
            else:
                码2, 用时2, _ = 探(端口, "/slots", 超时=6)
                if 码2 is None:
                    态 = "新 PID %d /health 通了，但 /slots 拿不到锁（**还没转好**）" % 听
                else:
                    return True, 听, 耗, "新 PID %d ｜ /health 200 ｜ /slots 拿到锁（%.2fs）" % (听, 用时2)
        末态 = 态
        if 态 != 上次:
            说("   %6.1fs  %s" % (耗, 态))
            上次 = 态
        time.sleep(1.5)


# ======================= ⑤ 复验 =======================

def 复验(端口, 旧PID, 旧模型, 新PID):
    说("⑤ 复验（**不能只看端口活** —— 卡死态本身端口就是活的）")
    项 = []
    项.append(("PID 变更", 新PID is not None and 新PID != 旧PID,
              "旧 %s → 新 %s" % (旧PID, 新PID)))
    码, 用时, _ = 探(端口, "/slots", 超时=6)
    项.append(("/slots 拿得到锁", 码 is not None,
              "%.2fs 拿到" % 用时 if 码 is not None else "**超时：slot 仍被占死**"))
    新模型 = 当前模型名(端口)
    项.append(("模型一致", bool(新模型) and (旧模型 is None or 新模型 == 旧模型),
              "%s（重启前 %s）" % (新模型 or "取不到", 旧模型 or "取不到")))
    用, 总 = 显存()
    if 用 is not None:
        项.append(("显存占用", True, "%d / %d MiB（%.1f%%）" % (用, 总, 100.0 * 用 / 总)))
    槽 = 槽位数(端口)
    if 槽 is not None:
        项.append(("槽位数（--parallel）", True,
                  "%s%s" % (槽, "  ← 若为 1，卡死一次就全瘫；开 2 可缓解" if 槽 == 1 else "")))

    坏 = 0
    for 名, 好, 说明 in 项:
        说("   %s %-22s %s" % ("✅" if 好 else "❌", 名, 说明))
        if not 好:
            坏 += 1
    return 坏


# ======================= 主流程 =======================

def 主():
    参 = argparse.ArgumentParser(description="重启 ChatBI 生成端（借 Local-AI APP 的看门狗把手）")
    参.add_argument("--端口", type=int, default=默认端口)
    参.add_argument("--等待秒", type=int, default=默认等待秒,
                    help="等 APP 把服务拉回来的上限（模型要重新上显存，给足）")
    参.add_argument("--演练", "--dry-run", dest="演练", action="store_true",
                    help="只打印将要做什么，一个进程都不动")
    参.add_argument("--强制", "--force", dest="强制", action="store_true", help="不询问")
    参.add_argument("--接管态也杀", dest="接管也杀", action="store_true",
                    help="明知 APP 拉不回来（接管态/游离态）仍要杀掉该进程")
    args = 参.parse_args()

    说("ChatBI · 重启生成端　端口 %d" % args.端口)
    说("时间：%s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    说("=" * 74)

    现场 = 摸现场(args.端口)
    报现场(现场)

    if not 现场["听"]:
        说()
        说("=" * 74)
        说("结论：⏭ **跳过** —— 生成端本来就没起，没有「重启」可言。")
        说("  去 Local-AI APP 里点「启动模型服务」。")
        return 3

    引擎PID = 现场["引擎"]["PID"]
    旧模型 = 当前模型名(args.端口)
    旧槽 = 槽位数(args.端口)
    说()
    诊断结果 = 报诊断(args.端口, 引擎PID)

    # ---- 能不能杀：先看是不是 APP 直管的 ----
    说()
    说("④ 处置")
    if 现场["关系"] != "直管态" and not args.接管也杀:
        说("   ⏸ **不动手** —— 当前是「%s」，APP 的看门狗不会把服务拉回来。" % 现场["关系"])
        说("      杀了它 = 生成端直接没了（不是重启，是停用）。")
        说("      怎么把它变回直管态：**在 APP 里点一次「停止服务」→「启动服务」**")
        说("        （走的是 `start_service_core`，会重新 spawn 并把看门狗挂上）。")
        说("      若你就是要腾显存（不指望它回来），加 `--接管态也杀`。")
        return 3 if args.演练 else 1

    说("   将要执行：终止 PID %d（%s）" % (引擎PID, 现场["引擎"]["映像"]))
    说("   然后等 APP 的看门狗把它拉回来（最多 %d 秒）" % args.等待秒)
    说("   ⚠️ 这一步会**杀掉当前生成端进程**（正在跑的请求会失败）。")

    if args.演练:
        说()
        说("   （--演练：到此为止，没有动任何进程）")
        return 3
    if not 询问(args.强制, "确认重启生成端？"):
        说("   → 已放弃")
        return 3

    说()
    t0 = time.time()
    if not 杀引擎(引擎PID):
        return 1
    # 等端口真的松开（TerminateProcess 是异步的，「发了信号」不等于「停了」）
    起 = time.time()
    while time.time() - 起 < 8:
        听, _ = 端口占用者(args.端口)
        if 听 != 引擎PID:
            break
        time.sleep(0.3)

    说()
    说("④ 等 APP 把服务拉回来（上限 %d 秒；看门狗 %d 秒一轮 + 模型要重新上显存）"
       % (args.等待秒, 看门狗轮询秒))
    好, 新PID, 耗, 末态 = 等复苏(args.端口, 引擎PID, args.等待秒)
    说("   用时 %.1fs ｜ %s" % (耗, 末态))

    说()
    if not 好:
        说("=" * 74)
        说("结论：❌ **没等回来**")
        说("  可能原因（按概率）：")
        说("    · APP 那一刻不是 Running 态（看门狗只在 Running 时盯）→ 到 APP 里看状态")
        说("    · APP 自己也没了 / 被关掉了 → 重新打开 APP，它会自动启动上次的模型")
        说("    · 模型太大 / 显存不够，加载超时 → 加长 `--等待秒`，或看 APP 日志")
        说("    · 从没走到过 `start_service_core`（比如引擎是手工起的）")
        说("  看 APP 日志找 watchdogs 那两行：")
        说("    `watchdog: llama-server crashed, scheduling restart` / `watchdog: restart succeeded`")
        说("  日志目录：%%LOCALAPPDATA%%\\LocalAI-Installer\\data")
        说("  兜底：在 APP 里点「启动服务」（或重开 APP，它会自动起上次的模型）。")
        return 1

    说("=" * 74)
    坏 = 复验(args.端口, 引擎PID, 旧模型, 新PID)
    说()
    if 坏:
        说("结论：🟠 **进程换新了，但有 %d 项没通过** —— 按上面 ❌ 那几行继续查。" % 坏)
        return 1
    说("结论：✅ **重启成功且复验通过**")
    说("  引擎 %d → %d ｜ 用时 %.1fs ｜ 模型 %s ｜ 槽位 %s"
       % (引擎PID, 新PID, time.time() - t0, 旧模型 or "?", 旧槽 if 旧槽 is not None else "?"))
    说("  现在可以重跑评测了。")
    return 0


if __name__ == "__main__":
    raise SystemExit(主())
