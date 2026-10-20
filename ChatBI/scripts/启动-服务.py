# -*- coding: utf-8 -*-
"""
启动-服务.py —— 一键起「作品二 · ChatBI」的本地服务（零第三方依赖，只用标准库）
==============================================================================

回答一句话：**「这套服务该怎么起、起对了没有？」**

它管谁、不管谁（**这条边界是包 7 的核心结论，不是随手划的**）
----------------------------------------------------------
管（本脚本负责起）：
  · **问数**   8095  scripts/问数服务.py    ← 作品二的数据链路本体
  · **编排**   8096  scripts/编排服务.py    ← 作品二的对外唯一入口（一个入口三条链路）

不管（只检查，不代起）：
  · **生成端** 8000  **由 Local-AI APP 负责**。APP 管模型生命周期与显存调优
    （换模型 / 调上下文 / 卸载显存都在 APP 里）。脚本代起会和 APP 抢同一份资源，
    也会让「谁在管这个进程」变得不清楚。
  · **重排**   8085  **由作品一的启动器负责**。它是检索链的依赖，归 `RAG-Embedding`。
  · **知识库** 8092  **由作品一的启动器负责**。8096 的知识问答链路是**HTTP 转发**给它的
    （不 import、不共进程）—— 所以 8092 在不在，直接决定 8096 的三条链路里有一条通不通。

🔴 为什么**不**代起 8085 / 8092：**一个进程只能有一个主**。
   两个仓库各写一份「起 8092」的命令，就会出现「谁的版本对」「谁改的端口生效」这类
   查不清的问题。作品二只**探活 + 报告**，缺了就把话说清楚：「去作品一的启动器起它」。

启动顺序（有依赖关系，不是随便排的）
------------------------------------
    ① 问数 8095  ──► ② 编排 8096
      等它 /health 200 才继续（服务起手要读语义层定义、开 SQLite）

  8096 起手会**探一遍下游**（8092 / 8095）并把结果写进 `/health` 的 `链路` 字段 ——
  所以「8096 起来了」不等于「三条链路都通」，自证时必须看 `链路` 里每一项的 `可达`。

设计要点（每条都是实际踩过的坑换来的，与作品一的启动器同源）
------------------------------------------------------------
1. 🔴 **起之前先数一遍端口上有几个 PID**。Python 能多进程绑同一端口
   （`SO_REUSEADDR`），**不报错，请求随机分流** —— 症状是「我明明重启了，
   改动就是不生效」。所以本脚本把「实例数」当成第一位的判据。
2. 🔴 **发现问题不静默重起**。默认只报告并要求确认；`--强制` 才跳过询问。
3. 🔴 **起完必须自证，而且要用能区分版本的判据**：
   编排服务看 `/health` 里有没有 **`权限` 块**（包 6 新增的键，旧代码没有）。
   **启动 banner 不能当版本判据** —— 新旧版的 banner 可能一字不差。
4. **只读的东西单独一档**（`--状态` / `--演练`）：不猜、不动手，先看清现场。
5. **不吞错误**：每个服务各开一个自己的控制台窗口，日志原样留在那里。
6. 🔴 **`--演练` 与实际执行走同一个命令拼装函数** —— 否则「预览的命令」和
   「真正跑的命令」各写一份，又是「同一份事实两处定义」。

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

# 端口（与 README / 各服务脚本的 `默认端口` 一致）
问数端口 = 8095
编排端口 = 8096
知识库端口 = 8092            # 作品一，**不代起**，只检查
重排端口 = 8085              # 作品一，**不代起**，只检查
生成端端口 = 8000            # APP，**不代起**，只检查

# 起服务后等 /health 的上限（都是纯 Python，起手要读定义 + 开 SQLite）
等问数秒 = 45
等编排秒 = 45

# 🔴 版本判据：**能区分新旧代码的那个键**。
#    编排服务 1.1（包 6）的 `/health` 里新加了 `权限` 块；1.0 没有。
#    问数服务的 `log_file` 是「请求日志落盘」那一版的标志（旧版只上屏 → 窗口拖选会冻住服务）。
编排版本判据键 = "权限"
编排期望版本 = "1.1"

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
    except Exception:                                          # noqa: BLE001
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
       从一句现象**定位到具体原因**的那条判据。
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
       连 `/health` 都挂 —— 看着像「服务崩了」，其实进程好好的。
       成因是 **Windows 传统控制台被鼠标拖选**：conhost 一进「标记」模式，
       进程的**下一次控制台输出就永久阻塞**，而服务每个请求都要往控制台打一行
       → 服务就冻死在那一行。恢复只需在那个窗口**按一下 Esc**，不用重启。
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
        "    恢复：到那个窗口**按一下 Esc**（或点一下窗口）—— 立刻恢复，**不用重启**。",
        "    （本仓库的服务已把请求日志改成「先落盘、再交给后台线程上屏」，窗口拖选不再影响服务。）",
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

    🔴 时区这一步最容易错，而且错了**看不出来**：`GetProcessTimes` 给的 FILETIME 是
    **UTC**。直接 `datetime(1601,1,1) + timedelta(...)` 得到的是 UTC 时刻，而源码 mtime
    是本地时间 —— 两者相比会差 8 小时，却依然能「讲通一个故事」。
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
    时刻 = 进程启动时刻(pid)
    return 时刻.strftime("%Y-%m-%d %H:%M:%S") if 时刻 else "取不到"


def 源码新鲜度(源文件, 进程时刻):
    """进程是在源码最后一次修改**之后**启动的吗？返回 (改动时刻, 是否比源码新) 或 None。

    🔴 这是「重启了 ≠ 生效了」唯一硬判据：比**时刻**，不比动作。
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

    走 ctypes 而不是 `taskkill`：Git Bash 会把 `taskkill /F` 的 `/F` 当路径转换
    （报 Invalid argument - 'F:/'），而且 taskkill 的输出是 GBK，容易糊在终端里。
    """
    句柄 = _kernel32.OpenProcess(进程终止权限, False, pid)
    if not 句柄:
        return False, "打不开进程（错误 %d，可能需要管理员权限）" % ctypes.get_last_error()
    try:
        成功 = _kernel32.TerminateProcess(句柄, 1)
    finally:
        _kernel32.CloseHandle(句柄)
    return bool(成功), "已发送终止" if 成功 else "终止失败（错误 %d）" % ctypes.get_last_error()


# HTTP 与进程控制

def 绕代理开启器():
    """本机探测**必须显式绕代理** —— 本环境有 `HTTP_PROXY`，urllib 默认会走它，
    后果是「服务明明活着却连不上」（代理那边当然没有 127.0.0.1:8095）。

    🔴 用 `build_opener(ProxyHandler({}))` 而不是删环境变量：删环境变量会影响
    同一个进程里后面所有代码，而且**在这个进程里对子进程不生效**（子进程继承的是
    父进程启动时的环境）。局部绕代理是唯一干净的做法。
    """
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


_开启器 = 绕代理开启器()


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
            with _开启器.open(地址, timeout=3) as 响应:
                if 响应.status == 200:
                    return True, json.loads(响应.read().decode("utf-8", "replace")), time.time() - 起
        except Exception:                                      # noqa: BLE001
            pass
        time.sleep(1.0)
    return False, None, time.time() - 起


def 取健康(端口, 超时=5):
    """单次取 `/health`。返回 (通过, 体或 None, 耗时)。"""
    地址 = "http://127.0.0.1:%d/health" % 端口
    起 = time.time()
    try:
        with _开启器.open(地址, timeout=超时) as 响应:
            if 响应.status == 200:
                return True, json.loads(响应.read().decode("utf-8", "replace")), time.time() - 起
    except Exception:                                          # noqa: BLE001
        pass
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

def 问数命令():
    # 用**当前解释器**：用户终端里的 python 与 APP 无关，不必锁版本（服务只用标准库 + sqlite3）
    return [sys.executable, os.path.join(脚本目录, "问数服务.py"), "--端口", str(问数端口)]


def 编排命令():
    return [sys.executable, os.path.join(脚本目录, "编排服务.py"), "--端口", str(编排端口)]


def 命令可读(命令):
    """人类可读的一行命令（仅用于打印，不用于执行 —— 带空格的路径引号是 shell 的事）。"""
    部 = []
    for 段 in 命令:
        部.append('"%s"' % 段 if (" " in 段 and not 段.startswith('"')) else 段)
    return " ".join(部)


# ======================= 体检 =======================

# (角色, 端口, 归属)。归属只有三种：本脚本起 / 作品一起 / APP 起。
角色表 = [
    ("问数", 问数端口, "本脚本起"),
    ("编排", 编排端口, "本脚本起"),
    ("重排", 重排端口, "作品一起"),
    ("知识库", 知识库端口, "作品一起"),
    ("生成端", 生成端端口, "APP 起"),
]

# 单文件 Python 服务可以做「跑的是不是最新代码」的硬判据（二进制服务比不了）
服务源文件 = {
    "问数": os.path.join(脚本目录, "问数服务.py"),
    "编排": os.path.join(脚本目录, "编排服务.py"),
}


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
    """起之前先确认「该有的东西在不在」—— 缺了就直接说清，别等它崩。

    🔴 这里查的必须是**命令里真正会执行的那一段**（`问数命令()[0]` / 脚本本体），
    不能另写一份「我以为的路径」。判据与执行各走各的，判据就等于没有。
    """
    问题 = []
    for 名称, 命令 in (("问数服务本体", 问数命令()), ("编排服务本体", 编排命令())):
        本体 = 命令[1]
        if not os.path.isfile(本体):
            问题.append("%s 不存在：%s" % (名称, 本体))
    if not os.path.isabs(问数命令()[0]):
        问题.append("解释器 argv[0] 不是绝对路径（Windows 不会用 cwd 去找 exe）：%s" % 问数命令()[0])
    # 数据库是问数链路的硬依赖 —— 它不在，服务照样起得来，但每条问数都会失败（静默降级）。
    库 = os.path.join(os.path.dirname(仓库根), "AI大模型应用开发实战训练营",
                      "30天学习计划", "资产-制造业Demo库", "制造业Demo.db")
    if not os.path.isfile(库):
        问题.append("业务库不存在（问数链路会全失败）：%s" % 库)
    return 问题


# ======================= 包 8：生产就绪体检 =======================

# 包 8 新增的五个可观测块。**缺任何一个 = 端口上跑的是包 8 之前的旧实例。**
# 这正是「版本判据」（与包 6 用 `权限` 块判新旧同一个套路）：
# 光看「/health 通了」分不出新旧，而新旧实例的功能差着五项。
包8健康键 = ("反馈", "会话", "意图缓存", "身份", "行级")

包8环境变量 = [
    ("CHATBI_TOKEN_SECRET", "B-4 令牌签名密钥 —— **不设 = 谁都能签出合法令牌**"),
    ("CHATBI_REQUIRE_TOKEN", "B-4 强制令牌模式（裸角色一律 401）"),
    ("CHATBI_FEEDBACK_DIR", "A-3 反馈台账目录（默认 logs/）"),
    ("CHATBI_NO_INSIGHT", "A-2 关掉解读（评测 / 自检用）"),
    ("CHATBI_NO_SESSION", "A-1 关掉会话继承（**评测必须关**）"),
    ("CHATBI_NO_CACHE / CHATBI_CACHE_DIR", "B-2 关语义缓存 / 改缓存目录（**评测必须关**）"),
    ("CHATBI_NO_MASK / CHATBI_NO_ROWLEVEL", "B-3 / B-5 关掉脱敏 / 行级（仅排障）"),
]


def 包8就绪(端口):
    """包 8 五项能力「配得对不对」的体检。

    🔴 为什么要有这一段：这五项**都能在「没配好」的状态下正常工作** ——
       不设令牌密钥照样签得出令牌、不设强制模式照样接受裸角色、不关缓存照样答得对。
       不看一眼的话，**演示态与生产态在界面上完全一样**
       （本项目 §9.5 那一类「拿表现当状态」的错误，换了个形态又来一次）。
    """
    通过, 体, _ = 取健康(端口)
    if not 通过 or not 体:
        print("      └ 包 8 ❌ 取 /health 失败 —— 没法判新旧")
        return False
    缺 = [k for k in 包8健康键 if k not in 体]
    if 缺:
        print("      └ 包 8 ❌ /health 缺块：%s" % "、".join(缺))
        print("         端口上跑的是**包 8 之前的旧实例** —— 不重启就验不到这五项能力")
        return False

    反 = 体.get("反馈") or {}
    会 = 体.get("会话") or {}
    缓 = 体.get("意图缓存") or {}
    身 = 体.get("身份") or {}
    行 = 体.get("行级") or {}

    print("      └ 包 8 ✅ 五个观测块都在"
          " ｜ 反馈 %s 条（👍%s / 👎%s / 带纠正 %s）"
          % (反.get("总条数"), 反.get("好评数"), 反.get("差评数"), 反.get("带纠正数")))
    print("      └ 会话：%s 个 / %s 轮 ｜ 新建 %s 命中 %s 承接 %s ｜ 淘汰 %s 过期 %s%s"
          % (会.get("会话数"), 会.get("轮次"), 会.get("新建"), 会.get("命中"),
             会.get("继承"), 会.get("淘汰"), 会.get("过期"),
             " ｜ ⚠️ 已关闭" if 会.get("已关闭") else ""))
    print("      └ 缓存：%s 条 ｜ 命中 %s ｜ 指纹不符作废 %s ｜ 失败不缓存 %s%s"
          % (缓.get("条目"), 缓.get("累计命中次数"), 缓.get("指纹不符"),
             缓.get("失败不缓存"), " ｜ ⚠️ 已关闭" if 缓.get("已关闭") else ""))

    # 🔴 身份这一行是本段的重点：两个键的组合决定「演示态」还是「生产态」
    if 身.get("要求令牌"):
        密钥态 = ("✅ 有密钥" if not 身.get("用开发密钥")
                 else "🔴 **要求令牌 + 用开发密钥** = 谁都能签合法令牌（假收紧）")
    else:
        密钥态 = ("⚠️ 未强制令牌（裸角色可自称管理员）"
                 + ("" if not 身.get("用开发密钥") else " ｜ 密钥：内置开发密钥"))
    print("      └ 身份：验过 %s ｜ 失败 %s ｜ 缺令牌被拒 %s ｜ 自称 %s ｜ 默认角色 %s ｜ %s"
          % (身.get("验签通过"), 身.get("验签失败"), 身.get("缺令牌被拒"),
             身.get("自称身份"), 身.get("默认角色"), 密钥态))
    print("      └ 行级：已遮蔽 %s 次 ｜ 建视图 %s ｜ **失败 %s**（非 0 = 有查询被阻断过）%s"
          % (行.get("已遮蔽连接"), 行.get("建视图"), 行.get("建视图失败"),
             " ｜ ⚠️ 已关闭" if 行.get("已关闭") else ""))

    # 敏感列 / 行级 / 角色的清单与配置自查在 /catalog 与 /health?deep=1 里 ——
    # 单独取一次（deep 要连库核行级取值，比普通 /health 慢得多，所以不放进上面那次）。
    try:
        目 = _取JSON(端口, "/catalog", timeout=10)
        print("      └ 目录：角色 %s 个 ｜ 敏感列 %s 个 ｜ 行级规则 %s 个"
              % (len(目.get("角色") or []), len(目.get("敏感列") or []),
                 len(目.get("行级") or [])))
    except Exception as 错:                                      # noqa: BLE001
        print("      └ 目录：❌ %s: %s" % (type(错).__name__, str(错)[:50]))
    try:
        深 = _取JSON(端口, "/health?deep=1", timeout=30)
        问 = 深.get("配置问题")
        print("      └ 四项配置自查（权限 / 敏感列 / 身份 / 行级）：%s"
              % (问 if isinstance(问, str) else ("🔴 " + "；".join(问) if 问 else "（没返回）")))
    except Exception as 错:                                      # noqa: BLE001
        print("      └ 四项配置自查：❌ %s: %s（deep 会连库核行级取值，可能只是慢）"
              % (type(错).__name__, str(错)[:40]))

    print("      └ 包 8 环境变量（改了要**重启服务**才生效）：")
    for 键, 说明 in 包8环境变量:
        print("          %-40s %s" % (键, 说明))
    return True


def _取JSON(端口, 路径, timeout=10):
    """取一个 JSON 端点（只用标准库；必须**读完 body 再关**，理由见页面探测那段）。"""
    with _开启器.open("http://127.0.0.1:%d%s" % (端口, 路径), timeout=timeout) as 响应:
        return json.loads(响应.read().decode("utf-8", "replace"))


def 生成端卡死(端口, 超时=6):
    """生成端「活着但不应答」的判据。返回 True = slot 被占死。

    🔴 **`探活()` 与 `/health` 对生成端都是假绿灯**：实测过 —— llama-server 的唯一
       slot 被一个中途断开的客户端占死时，`/health`・`/props`・`/v1/models` 全部
       **秒回 200**，而任何生成请求**永远不返回**（客户端只看到超时）。
       所以这里必须补一枪 `/slots`：它的处理器要拿**和推理同一把**锁，
       拿不到就是 slot 卡死 —— 处置是**重启模型服务**，不是改代码。
       现场（进程全 `Wait`、CPU 增量 0、端口留 `CLOSE_WAIT`）与三步判见
       `scripts/诊断-生成端卡死.py`。
    """
    起 = time.time()
    try:
        with _开启器.open("http://127.0.0.1:%d/slots" % 端口, timeout=超时) as 响应:
            响应.read(64)
        return False                                     # 拿到了锁 → slot 空闲
    except Exception:                                    # noqa: BLE001
        # 只有**超时**算卡死；明确的 4xx（这个版本没开 /slots）不算。
        return (time.time() - 起) >= 超时 - 0.5


# ======================= 动作：状态 / 演练 =======================

def 动作_状态():
    体检()
    print("=" * 96)
    print("依赖与版本判据")
    print("=" * 96)

    # ---- 本脚本管的两件：验得最细 ----
    for 角色, 端口 in (("问数", 问数端口), ("编排", 编排端口)):
        if not 探活(端口):
            print("  %s（%d） ⏭ 端口不可达 —— 服务没起，没法验" % (补位(角色, 10), 端口))
            continue
        通过, 体, 耗 = 等健康(端口, 5, 角色)
        if not 通过:
            print("  %s（%d） ❌ 端口活着但 /health 不通" % (补位(角色, 10), 端口))
            for 一 in 诊断不应答(端口):
                print("      %s" % 一.lstrip())
            continue
        print("  %s（%d） ✅ /health 通了 %.1fs ｜ 版本 %s"
              % (补位(角色, 10), 端口, 耗, 体.get("版本")))
        # 版本判据（能区分新旧代码的那个键）
        if 角色 == "编排":
            有判据 = 编排版本判据键 in 体
            print("      └ 版本判据：%s"
                  % ("✅ 有 `%s` 块（包 6 新代码）" % 编排版本判据键 if 有判据
                     else "❌ **没有 `%s`** —— 跑的是 1.0 旧代码" % 编排版本判据键))
            if 体.get("版本") != 编排期望版本:
                print("      └ ⚠️ 版本 = %s，期望 %s（本脚本按后者写的自证判据）"
                      % (体.get("版本"), 编排期望版本))
            链 = 体.get("链路") or {}
            通 = sum(1 for v in 链.values() if v.get("可达"))
            print("      └ 下游链路 %d/%d 可达：%s" % (
                通, len(链),
                "、".join("%s=%s" % (k, "✅" if v.get("可达") else "❌")
                          for k, v in 链.items()) or "—"))
            if 链 and 通 < len(链):
                for k, v in 链.items():
                    if not v.get("可达"):
                        print("         · %s 不可达 → %s" % (k, v.get("地址")))
                print("         （知识问答 / 数据问数各自的依赖没起；见下面「依赖」一节）")
            审 = 体.get("审计") or {}
            if 审:
                print("      └ 审计：总 %s ｜ 被拒 %s ｜ 落盘 %s ｜ 依据不切题 %s（%s）"
                      % (审.get("总条数"), 审.get("被拒数"), 审.get("落盘数"),
                         审.get("依据不切题数"), 审.get("文件")))
            权 = 体.get("权限") or {}
            if 权:
                print("      └ 权限：默认角色=%s ｜ 角色=%s ｜ 白名单问题=%s"
                      % (权.get("默认角色"),
                         "、".join(权.get("角色") or []) or "—",
                         (权.get("白名单问题") or "无")))
        # 日志落点（三种情况必须分开报）
        if "log_file" not in 体:
            print("      └ 请求日志 ❌ /health 无 log_file —— 旧实例（请求日志只上屏，窗口拖选会冻住服务）")
        elif not 体.get("log_file"):
            print("      └ 请求日志 ❌ 字段在但为空 —— logs/ 不可写（窗口拖选仍会冻住服务）")
        else:
            print("      └ 请求日志 %s ｜ 屏幕丢行 %s"
                  % (体.get("log_file"), 体.get("console_dropped")))
        # 页面是**独立文件**（web/*.html），所以 /health 通了不代表它在 —— 必须单独探一次
        try:
            # 🔴 必须**读完 body 并关闭**：只取响应头就把 response 丢下不管，socket 被 GC
            #    回收时发 RST，而服务端此刻还在写页面 —— **服务端**会打出一整页 `WinError 10053`，
            #    看起来像服务崩了，其实是我这个探针在捣乱。「客户端偷懒」会伪装成「服务端故障」。
            with _开启器.open("http://127.0.0.1:%d/" % 端口, timeout=8) as 页:
                类 = 页.headers.get("Content-Type") or ""
                大小 = len(页.read())
            if 页.status == 200 and "text/html" in 类:
                print("      └ 页面探测：✅ 200 HTML ｜ %d 字节 ｜ http://127.0.0.1:%d/" % (大小, 端口))
            else:
                print("      └ 页面探测：❌ 状态 %d ｜ %s" % (页.status, 类))
        except urllib.error.HTTPError as 错:
            print("      └ 页面探测：❌ HTTP %d —— 路由没接上（端口上多半是旧实例，重启）" % 错.code)
        except Exception as 错:                                  # noqa: BLE001
            print("      └ 页面探测：❌ %s —— 检查 web/ 下对应 html 是否存在" % type(错).__name__)

        # ---- 包 8：生产就绪体检（只对问数服务；编排服务的 /health 是另一套键）----
        if 角色 == "问数":
            包8就绪(端口)

    # ---- 依赖三项：只报可达性，不验版本 ----
    print()
    print("  ---- 依赖（不代起，只管报告） ----")
    依赖 = [
        ("重排", 重排端口, "作品一起：RAG-Embedding/scripts/启动-服务.py"),
        ("知识库", 知识库端口, "作品一起：RAG-Embedding/scripts/启动-服务.py"),
        ("生成端", 生成端端口, "APP 起：Local-AI 里点「启动模型」"),
    ]
    for 角色, 端口, 归属 in 依赖:
        if not 探活(端口):
            print("  %s（%d） ❌ 不可达 —— %s" % (补位(角色, 10), 端口, 归属))
            continue
        通过, 体, 耗 = 等健康(端口, 5, 角色)
        if not 通过:
            print("  %s（%d） ⚠️ 端口活着但 /health 不通" % (补位(角色, 10), 端口))
            continue
        if 角色 == "知识库":
            print("  %s（%d） ✅ 语料 %s 份 / %s 块 ｜ 链路 %s"
                  % (补位(角色, 10), 端口, 体.get("docs"), 体.get("blocks"), 体.get("chain")))
        elif 角色 == "生成端":
            # 🔴 「可达」不等于「能生成」—— 这里必须补一枪 /slots（见 生成端卡死() 的注释）。
            #    不补的话，启动器会报一个绿灯，而所有问数都在超时。
            if 生成端卡死(端口):
                print("  %s（%d） 🔴 **活着但 slot 卡死**（/health 通、生成永不返回）"
                      % (补位(角色, 10), 端口))
                print("        处置：跑 `python scripts/诊断-生成端卡死.py` 确认；"
                      "然后在 Local-AI APP 里**重启模型服务**（别直接杀进程）")
                print("        成因多半是「客户端在请求返回前被强杀」→ slot 没归还。"
                      "生成端开 `--parallel 2` 可以让一个卡死的 slot 不拖垮全部。")
            else:
                print("  %s（%d） ✅ 可达且 /slots 拿得到锁（**能生成**）"
                      % (补位(角色, 10), 端口))
        else:
            print("  %s（%d） ✅ 可达（%.1fs）" % (补位(角色, 10), 端口, 耗))

    print()
    print("  提示：`--状态` 只读，不会启动或停止任何东西。")


def 动作_演练():
    print("=" * 96)
    print("演练（只打印，不执行）—— 实际执行的命令与此完全相同")
    print("=" * 96)
    print()
    print("【① 问数】cwd = %s" % 仓库根)
    print("    %s" % 命令可读(问数命令()))
    print("    · 只读护栏：SQLite 以 `mode=ro` 打开 + 语句白名单 + `query_only` + 超时")
    print()
    print("【② 编排】cwd = %s" % 仓库根)
    print("    %s" % 命令可读(编排命令()))
    print("    · 起手探一遍下游（8092 / 8095），结果写进 /health 的 `链路` 字段")
    print("    · 对外唯一入口：一个入口三条链路（知识问答 / 数据问数 / 动作执行）")
    print("    · 权限默认角色 = 操作员；越权回 HTTP 403，且这次调用会进审计")
    print()
    print("【③ 依赖】**本脚本不起** —— 缺了按归属去找对应的启动方：")
    print("    · 重排 8085 / 知识库 8092 → 作品一：RAG-Embedding/scripts/启动-服务.py")
    print("    · 生成端 8000           → Local-AI APP 里点「启动模型」")
    print()
    print("顺序：① 问数 /health 200 → ② 编排 → 用的时候 ③ 重排/知识库（作品一）+ 生成端（APP）")
    print()
    print("前置条件检查：")
    问题 = 检查前置条件()
    for 一 in 问题:
        print("  ❌ %s" % 一)
    if not 问题:
        print("  ✅ 服务本体 / 业务库 都在")


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
        # 终止是异步的，等端口真的空出来再报结论（别把「发了信号」当「停了」）
        起 = time.time()
        while time.time() - 起 < 6 and 端口占用者(端口):
            time.sleep(0.3)
        剩 = 端口占用者(端口)
        print("      → %s" % ("✅ 已释放" if not 剩 else "❌ 仍有 %d 个进程占着：%s"
                              % (len(剩), 剩)))
    print()


# ======================= 动作：起 =======================

def 起一个(名称, 命令, 工作目录, 前台):
    """起一个服务。返回 Popen 或 None。

    默认 CREATE_NEW_CONSOLE：每个服务一个自己的窗口，日志原样留在那里，**关窗即停** ——
    跟「手工开两个终端」是同一个心智模型。
    """
    try:
        if 前台:
            print("  → 前台模式：输出汇总到当前终端（Ctrl+C 会一起停）")
            子 = subprocess.Popen(命令, cwd=工作目录)
        else:
            旗标 = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
            子 = subprocess.Popen(命令, cwd=工作目录, creationflags=旗标)
    except FileNotFoundError:
        # 别把 `WinError 2` 以裸 Traceback 甩给用户 —— 它只说明「命令第 0 段找不到」。
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

    # --- 旧实例：默认不静默重起（本脚本最想防的就是「以为重启了」） ---
    要停 = []
    for 角色 in ("问数", "编排"):
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

    # --- ① 问数（编排的下游，必须先起） ---
    if 现场["问数"]["PID"]:
        print("  ① 问数（%d）已在运行 —— 跳过启动" % 问数端口)
    else:
        print("  ① 起问数（%d）…" % 问数端口)
        子 = 起一个("问数", 问数命令(), 仓库根, 前台)
        if 子 is None:
            return 1
        子进程表.append(子)
        print("     等 /health（起手要读语义层定义）…")
        通过, 体, 耗 = 等健康(问数端口, 等问数秒, "问数")
        if not 通过:
            print("     ❌ %g s 内 /health 没通。" % 等问数秒)
            for 一 in 诊断不应答(问数端口):
                print(一)
            return 1
        print("     ✅ 问数就绪（%.1f s，版本 %s）" % (耗, 体.get("版本")))

    # --- ② 编排 ---
    if 现场["编排"]["PID"]:
        print("  ② 编排（%d）已在运行 —— 跳过启动" % 编排端口)
    else:
        print("  ② 起编排（%d）…" % 编排端口)
        子 = 起一个("编排", 编排命令(), 仓库根, 前台)
        if 子 is None:
            return 1
        子进程表.append(子)
        通过, 体, 耗 = 等健康(编排端口, 等编排秒, "编排")
        if not 通过:
            print("     ❌ %g s 内 /health 没通。" % 等编排秒)
            for 一 in 诊断不应答(编排端口):
                print(一)
            return 1
        print("     ✅ 编排就绪（%.1f s，版本 %s）" % (耗, 体.get("版本")))

    # --- ③ 依赖：只检查（各归各家） ---
    for 角色, 端口, 归属 in (("重排", 重排端口, "作品一的启动器"),
                             ("知识库", 知识库端口, "作品一的启动器"),
                             ("生成端", 生成端端口, "Local-AI APP")):
        if 探活(端口):
            print("  ③ %s（%d）✅ 已在跑" % (角色, 端口))
        else:
            print("  ③ %s（%d）❌ 没在跑 —— 去 %s 起它" % (角色, 端口, 归属))
            if 角色 == "知识库":
                print("      （不影响现在起服务；只是 8096 的「知识问答」这条链路会显示不可达）")

    # --- 自证：起完必须验，而且要用**能区分版本**的判据 ---
    print()
    print("=" * 96)
    print("自证（起完不算数，验过才算数）")
    print("=" * 96)
    坏 = 0

    # ① 问数
    通过, 体, 耗 = 等健康(问数端口, 10, "问数")
    if not 通过:
        print("  问数 ❌ /health 不通")
        for 一 in 诊断不应答(问数端口):
            print(一)
        坏 += 1
    else:
        print("  问数 ✅ /health 通了 %.1fs ｜ 版本 %s" % (耗, 体.get("版本")))
        if not 体.get("log_file"):
            print("     ❌ /health 里**没有可用的 log_file** —— 请求日志只上屏的旧实例：")
            print("        窗口一被鼠标拖选，服务就被冻住（端口还在听、CPU 0%、请求全超时）。**重启才生效。**")
            坏 += 1
        else:
            print("     ✅ 请求日志落盘 = %s（屏幕丢行 %s）"
                  % (体.get("log_file"), 体.get("console_dropped")))

    # ② 编排（版本判据 + 下游链路 + 权限白名单）
    通过, 体, 耗 = 等健康(编排端口, 10, "编排")
    if not 通过:
        print("  编排 ❌ /health 不通")
        for 一 in 诊断不应答(编排端口):
            print(一)
        坏 += 1
    else:
        print("  编排 ✅ /health 通了 %.1fs ｜ 版本 %s" % (耗, 体.get("版本")))
        # 🔴 版本判据：banner 不能当判据（新旧 banner 可能一字不差），要验**新代码才有的键**。
        if 编排版本判据键 not in 体:
            print("     ❌ /health 里**没有 `%s` 块** —— 跑的是 1.0 旧代码（权限与审计都没生效！）"
                  % 编排版本判据键)
            print("        判据：包 6 起 `/health` 必有 `权限`；banner 不能当版本判据。**重启才生效。**")
            坏 += 1
        else:
            权 = 体.get(编排版本判据键) or {}
            问题 = 权.get("白名单问题") or []
            if 问题:
                print("     ❌ 权限白名单自检有问题：%s" % "；".join(问题))
                坏 += 1
            else:
                print("     ✅ 版本判据 `%s` 在 ｜ 默认角色=%s ｜ 角色=%s ｜ 白名单无配置问题"
                      % (编排版本判据键, 权.get("默认角色"),
                         "、".join(权.get("角色") or [])))
            审 = 体.get("审计") or {}
            if 审:
                print("     ✅ 审计：总 %s ｜ 被拒 %s ｜ 落盘 %s ｜ 依据不切题 %s"
                      % (审.get("总条数"), 审.get("被拒数"), 审.get("落盘数"), 审.get("依据不切题数")))
        if not 体.get("log_file"):
            print("     ❌ /health 里**没有可用的 log_file** —— 旧实例，窗口拖选会冻住服务")
            坏 += 1
        链 = 体.get("链路") or {}
        if 链:
            缺 = [k for k, v in 链.items() if not v.get("可达")]
            if 缺:
                # 这里**不算失败**：缺的是别的归属没起，不是本脚本的产物。
                print("     ⚠️ 下游链路不可达：%s（按归属去起，不影响本脚本判定）" % "、".join(缺))
            else:
                print("     ✅ 三条链路下游全可达：%s" % "、".join(链))

    # ③ 双实例检查（本脚本最想防的那个症状）
    for 角色 in ("问数", "编排"):
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
        print("    浏览器打开 http://127.0.0.1:%d/ —— 一个入口问三种问题：" % 编排端口)
        print("      · 制度类  「不合格品应该怎么处理？」")
        print("      · 数据类  「上月哪家供应商不良率最高？」")
        print("      · 动作类  「给东莞精诚出一份质量整改通知」（换个角色试「越权被拒」）")
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
                except Exception:                              # noqa: BLE001
                    pass
    return 1 if 坏 else 0


# ======================= 入口 =======================

标题 = "作品二 · ChatBI 本地服务启动器（问数 8095 + 编排 8096；重排/知识库归作品一，生成端归 APP）"


def 主函数(argv):
    参数 = set(argv[1:])

    if not argv[1:]:
        print("=" * 96)
        print(标题)
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
        动作_停([问数端口, 编排端口], 强制=bool({"--强制", "--force", "-f"} & 参数))
        return 0

    前台 = bool({"--前台", "--foreground"} & 参数)
    强制 = bool({"--强制", "--force", "-f"} & 参数)
    重起 = bool({"--重起", "--restart"} & 参数)
    if not (前台 or 强制 or 重起):
        print("未知参数：%s" % " ".join(sorted(参数)))
        print("可用：--状态 / --演练 / --停 / --重起 / --强制 / --前台")
        return 2

    print("=" * 96)
    print(标题)
    print("=" * 96)
    print()
    return 动作_起(前台=前台, 强制=强制, 重起=重起)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")   # 终端 cp936，不要 reconfigure(encoding=...)
    except Exception:                                          # noqa: BLE001
        pass
    sys.exit(主函数(sys.argv))
