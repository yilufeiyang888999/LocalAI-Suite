# -*- coding: utf-8 -*-
"""
诊断-控制台冻结.py —— 「服务会不会被控制台拖选冻死？」的常驻判据
=====================================================================

它验的是一条**不变量**：**请求路径不依赖控制台 I/O**。

为什么需要它
------------
2026-10-10 在 8092 上实测踩到：窗口被鼠标拖选 → conhost 进入「标记」模式 →
进程的**下一次控制台输出永久阻塞** → 服务冻住。表现极具迷惑性：
端口仍在 LISTENING、CPU 0%、**所有请求超时（连 /health 都挂）**，看着像服务崩了。
铁证：同一 PID、没重启，窗口标题从 `Select C:\\...\\python.exe` 变回干净之后，
`/health` 立刻从 6 s 超时变成 0.03 s 返回 —— 进程没死，是卡在控制台 I/O 上。

修法：把「上屏」从请求路径上摘下来 —— 请求线程只**落盘 + 入队**，上屏交给一根后台线程。
本脚本就是那条修法的回归护栏。（写它时的第一版修法就是错的：落盘和上屏放在同一根线程，
控制台一冻，文件也跟着不再增长 —— 实测 6 个请求只落盘 1 行，被这个脚本当场抓出来。）

它怎么造出「控制台冻结」
------------------------
不必真去拖选窗口：把 `sys.stdout` 换成一个 `write` **永不返回**的假对象，
对进程而言与「窗口被拖选」等价（写控制台永久阻塞）。

🔴 关键的一条：**必须同时证明「冻结真的发生了」**
   否则「控制台压根没冻」和「冻了但服务扛住了」表现一模一样 —— 那就是没有判据。
   所以最后一组断言是「假控制台确实被写入过」。

用法：
  python scripts\\诊断-控制台冻结.py            # 默认端口 8093（须空闲）
  python scripts\\诊断-控制台冻结.py 8094
退出码：0 = 全通过；1 = 有失败项。
"""
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

端口 = int(sys.argv[1]) if len(sys.argv) > 1 else 8093
脚本目录 = os.path.dirname(os.path.abspath(__file__))
次数 = 6                       # 冻结期间打几个请求（够看出「一行不少」即可）

结果 = []
监听模式 = re.compile(rb"^TCP\S*\s+\S+:(\d+)\s+\S+\s+LISTENING\s+(\d+)\s*$")


def 记(名称, 通过, 说明=""):
    结果.append((名称, bool(通过), 说明))
    print("  %s %-36s %s" % ("✅" if 通过 else "❌", 名称, 说明))


def 端口在听(号):
    try:
        出 = subprocess.run(["netstat", "-ano"], capture_output=True, timeout=25).stdout
    except Exception:                                          # noqa: BLE001
        return None                                            # 查不了，不当作「空闲」
    for 行 in 出.splitlines():
        匹配 = 监听模式.match(行.strip())
        if 匹配 and int(匹配.group(1)) == 号:
            return int(匹配.group(2))
    return None


class 冻住的控制台:
    """write 永不返回 —— 等价于「窗口被鼠标拖选」。"""

    写次数 = 0

    def write(self, 文):
        type(self).写次数 += 1
        while True:
            time.sleep(60)

    def flush(self):
        pass

    def reconfigure(self, **kw):
        pass


print("=" * 88)
print("  诊断：请求路径是否依赖控制台 I/O —— 冻结控制台后服务还能不能干活")
print("=" * 88)
print()

# ---------- 0. 端口必须空闲，否则起的是「另一个实例」，整场验证无意义 ----------
占用 = 端口在听(端口)
if 占用:
    print("  ❌ 端口 %d 已被 PID %s 占用 —— 请换一个空闲端口再跑（本脚本要自己起服务）。"
          % (端口, 占用))
    sys.exit(1)
print("  · 端口 %d 空闲 ✅" % 端口)

# ---------- 1. 载入服务本体（只取函数，不跑 main） ----------
_载 = importlib.util.spec_from_file_location(
    "知识库服务", os.path.join(脚本目录, "知识库服务.py"))
m = importlib.util.module_from_spec(_载)
_载.loader.exec_module(m)
print("  · 已载入 知识库服务.py（不起 main）")
print()

# ---------- 2. 日志落点：先验它给得出路径，再把落点换到临时文件（别污染真日志）----------
m.监听端口 = 端口
真日志 = m.开日志文件()
记("开日志文件() 给出落点", bool(真日志) and 真日志.endswith(".log"), str(真日志))

临时日志 = os.path.join(tempfile.gettempdir(), "诊断-控制台冻结-日志.log")
open(临时日志, "w", encoding="utf-8").close()
m.日志文件 = 临时日志                 # 让本场测试只写临时文件
记("测试日志改道临时文件", m.日志文件 == 临时日志, 临时日志)

# ---------- 3. 起服务 ----------
m.建基座()
threading.Thread(target=m.上屏线程, name="日志-上屏", daemon=True).start()
服务 = m.容忍断连的服务器(("127.0.0.1", 端口), m.处理器)
threading.Thread(target=服务.serve_forever, name="http", daemon=True).start()
time.sleep(0.4)

真输出 = sys.stdout


def 打健康(超时=5):
    with urllib.request.urlopen("http://127.0.0.1:%d/health" % 端口, timeout=超时) as 应:
        return 应.status, json.loads(应.read().decode("utf-8"))


# 对照组：冻结**之前**必须先能通 —— 否则「冻结后不通」说明不了任何事
码, 体 = 打健康()
记("冻结前 /health 可用（对照基线）", 码 == 200, "200 ｜ console_dropped=%s" % 体.get("console_dropped"))
print()

# ---------- 4. 冻结控制台，然后照常打请求 ----------
假 = 冻住的控制台()
sys.stdout = 假
基线行 = sum(1 for _ in open(临时日志, encoding="utf-8"))
通过数 = 0
最慢 = 0.0
for _ in range(次数):
    t = time.time()
    try:
        打健康()
        通过数 += 1
    except Exception:                                          # noqa: BLE001
        pass
    最慢 = max(最慢, time.time() - t)
sys.stdout = 真输出
新增行 = sum(1 for _ in open(临时日志, encoding="utf-8")) - 基线行

print("[冻结期间] 已把 sys.stdout 换成 write 永不返回的假对象（= 窗口被拖选）")
记("冻结下 /health 全部 200", 通过数 == 次数,
   "%d/%d ｜ 单次最慢 %.3f s" % (通过数, 次数, 最慢))
记("日志文件一行不少", 新增行 == 次数,
   "新增 %d 行 / 期望 %d 行" % (新增行, 次数))
# 🔴 这条是「判据的判据」：=0 说明冻结根本没生效，上面两条等于没检查
记("冻结真的发生了（假控制台被写过）", 假.写次数 > 0,
   "假控制台 write 次数 = %d（=0 表示冻结没生效，本场检查无效）" % 假.写次数)

服务.shutdown()
服务.server_close()
try:
    os.remove(临时日志)
except OSError:
    pass

# ---------- 5. 汇总 ----------
通过 = sum(1 for _, 好, _ in 结果 if 好)
失败 = len(结果) - 通过
print()
print("=" * 88)
print("  通过 %d ｜ 失败 %d" % (通过, 失败))
print("  总判定：%s" % ("✅ 请求路径与控制台已解耦 —— 窗口随便拖选，服务不受影响"
                       if 失败 == 0 else "❌ 仍有失败项 —— 请求路径还可能被控制台冻住"))
print("=" * 88)
print()
print("  说明：本判据验的是**不变量**（请求不依赖控制台），不是某个具体实现。")
print("        若日后有人把落盘挪回上屏线程，第 2 条会立刻变红 —— 那就是它存在的意义。")
sys.exit(1 if 失败 else 0)
