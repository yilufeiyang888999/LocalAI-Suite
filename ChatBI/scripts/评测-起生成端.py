# -*- coding: utf-8 -*-
"""ChatBI · 评测起生成端 —— 在**空闲端口**上按生产口径起一个游离态 llama-server

## 为什么要有这个脚本（与「为什么这不违反一个进程一个主」）

生产生成端（8000）归 Local-AI APP 管 —— 这条边界不动。但评测经常需要：
APP 没跑（作者不在 / 显存被别的模型占着）、或者要换 adapter / 换量化档 /
换 --parallel / 挂 GBNF 语法做对照 —— 这些都是**评测自己的环境变量**，
不该每次都去动 APP 的模型配置。

所以本脚本在**评测专用端口**（默认 18000）起一个游离态生成端，配合
`CHATBI_GEN_URL` 指过去 —— 这与 `lab/推理-服务.py`（自建推理壳）是同一个先例：
**评测环境可以自建，但必须与生产口径逐项对齐，且身份必须可回读**。

## 🔴 口径对齐（对照 `models/DS-R1-Qwen-7B-GGUF/base-Q4_K_M.launch.json`）

    · 模型      base-Q4_K_M.gguf
    · adapter   lora-v3-f16.gguf（--lora 直挂，不 merge —— #5 的生产口径）
    · 模板      tpl-prefill-closed.jinja（--chat-template）
    · 缩放      --lora-scaled 1.0（与 launch.json「适配器缩放」一致）

以上全部**从 launch.json 读**，不在脚本里再写一份默认值 —— 「同一份事实两处定义」
的下场（§9.5）是改了 launch.json 这边还在用旧口径，不报错、只给错结论。

## 🔴 判据纪律

    · **挑空闲端口，被占当场失败** —— 撞上别人的实例 = 连到旧版本，
      与「功能没实现」同形（§9.5）。
    · **「已报 PID」≠「服务起来了」**：起完必须回看端口 + /health + /props，
      并把 /props 的 total_slots（--parallel 生效值）、模型路径回读出来打印。
    · stdout/stderr 落文件（logs/），stderr 是决定性证据。
    · **本脚本起服务后驻留**（前台阻塞），由调用方管理生命周期；
      `--停` 子命令按记录的 PID 杀（PID 记在 logs/评测生成端.pid）。

用法：

    python scripts/评测-起生成端.py                    # 起 18000（base-Q4_K_M + lora-v3，生产口径）
    python scripts/评测-起生成端.py --端口 18001 --无adapter   # 纯基座（微调前对照臂）
    python scripts/评测-起生成端.py --模型 base-Q5_K_M.gguf    # 换量化档
    python scripts/评测-起生成端.py --parallel 2               # 槽位对照
    python scripts/评测-起生成端.py --语法 意图.gbnf           # GBNF 约束解码
    python scripts/评测-起生成端.py --停                       # 停掉（读 pid 文件）

起好之后：

    set CHATBI_GEN_URL=http://127.0.0.1:18000/v1/chat/completions
    python scripts/评测-盲测集.py --题集 问数盲测集-扩样本20题.json
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

脚本目录 = os.path.dirname(os.path.abspath(__file__))
项目目录 = os.path.dirname(脚本目录)
日志目录 = os.path.join(项目目录, "logs")
PID文件 = os.path.join(日志目录, "评测生成端.pid")

# 生产口径的 launch.json（唯一事实来源 —— 参数从这里读，不在脚本里写第二份）
GGUF目录 = r"D:\workspace\10-LocalAI-Installer\models\DS-R1-Qwen-7B-GGUF"
默认模型 = "base-Q4_K_M.gguf"
默认launch = os.path.join(GGUF目录, "base-Q4_K_M.launch.json")
引擎exe = r"D:\workspace\10-LocalAI-Installer\engine\backends\llama-server-cuda.exe"

绕代理 = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def 说(文本=""):
    print(文本)
    sys.stdout.flush()


def 空闲吗(端口):
    s = socket.socket()
    s.settimeout(0.6)
    占用 = s.connect_ex(("127.0.0.1", 端口)) == 0
    s.close()
    return not 占用


def 取端口占用者(端口):
    """(pid, 映像名) 或 (None, None)。netstat 的输出列位随语言变，按 PID 反查任务名最稳。"""
    try:
        出 = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True,
                            timeout=15, encoding="gbk", errors="replace").stdout
        for 行 in 出.splitlines():
            部分 = 行.split()
            if len(部分) >= 5 and 部分[3] == "LISTENING" and 部分[1].endswith(":%d" % 端口):
                pid = int(部分[4])
                任务 = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid, "/FO", "CSV", "/NH"],
                                      capture_output=True, timeout=10, encoding="gbk", errors="replace").stdout
                名 = 任务.split(",")[0].strip('"') if "," in 任务 else "?"
                return pid, 名
    except Exception:                                          # noqa: BLE001
        pass
    return None, None


def 等就绪(端口, 上限秒):
    """轮询 /health + /props。返回 (是否就绪, /props 的 dict 或 None)。
    🔴 /health 200 只证明 HTTP 活；/props 里 total_slots / 模型身份才是「能按这个口径干活」。"""
    起 = time.time()
    上次 = ""
    while time.time() - 起 < 上限秒:
        try:
            健 = 绕代理.open("http://127.0.0.1:%d/health" % 端口, timeout=2)
            if 健.status == 200:
                try:
                    props = json.loads(绕代理.open("http://127.0.0.1:%d/props" % 端口, timeout=4).read().decode())
                    return True, props
                except Exception:                              # noqa: BLE001
                    return True, None
        except Exception:                                      # noqa: BLE001
            pass
        态 = "等待中 %.0fs" % (time.time() - 起)
        if 态 != 上次:
            说("   %s" % 态)
            上次 = 态
        time.sleep(1.0)
    return False, None


def 取适配器(端口):
    """回读 /lora-adapters：返回 list（可能为空 = 裸基座）或 None（取不到）。

    🔴 **必须与「空列表」区分开**（§9.5：未查 ≠ 通过）。
    空列表 = 引擎明确说「一个 adapter 都没有」；None = 这个端点不通/超时，
    什么都不能断言。把 None 当「没挂」会把「没查到」写成「已确认没有」。
    """
    try:
        应 = 绕代理.open("http://127.0.0.1:%d/lora-adapters" % 端口, timeout=4)
        值 = json.loads(应.read().decode("utf-8", "replace"))
        return 值 if isinstance(值, list) else None
    except Exception:                                          # noqa: BLE001
        return None


def 停():
    if not os.path.isfile(PID文件):
        说("没有 pid 记录（%s 不存在）—— 可能没起过，或已被清" % PID文件)
        return 3
    pid = int(open(PID文件).read().strip())
    try:
        # 🔴 走 taskkill 但检查返回码 —— 「证伪脚本自己先错」的教训（§三十七）
        结果 = subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"],
                              capture_output=True, timeout=15)
        if 结果.returncode != 0:
            说("taskkill 返回 %d（进程可能已经死了）：%s" % (
                结果.returncode, 结果.stderr.decode("gbk", "replace").strip()))
        else:
            说("已终止 PID %d" % pid)
        os.remove(PID文件)
        return 0
    except Exception as 错:                                    # noqa: BLE001
        说("停止失败：%s" % 错)
        return 1


def 主():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    参 = argparse.ArgumentParser(description="评测用游离态生成端（生产口径对齐 launch.json）")
    参.add_argument("--端口", type=int, default=18000)
    参.add_argument("--模型", default=None, help="GGUF 文件名（放 GGUF 目录下）；默认取 launch.json 对应的 base-Q4_K_M")
    参.add_argument("--无adapter", action="store_true", help="不挂 LoRA（微调前基线臂）")
    参.add_argument("--缩放", type=float, default=None, help="--lora-scaled（默认取 launch.json 的适配器缩放）")
    参.add_argument("--parallel", type=int, default=1)
    参.add_argument("--ctx", type=int, default=8192)
    参.add_argument("--语法", default=None, help="GBNF 语法文件路径（--grammar）")
    参.add_argument("--GPU层数", default="999")
    参.add_argument("--停", action="store_true", help="停掉已起的（按 pid 文件）")
    参数 = 参.parse_args()

    if 参数.停:
        return 停()

    说("ChatBI · 评测生成端（游离态）")
    说("=" * 76)

    # ---- ① launch.json = 唯一事实来源 ----
    with open(默认launch, encoding="utf-8") as 句柄:
        launch = json.load(句柄)
    模型 = 参数.模型 or 默认模型
    模型路径 = os.path.join(GGUF目录, 模型)
    if not os.path.isfile(模型路径):
        说("🔴 模型不存在：%s" % 模型路径)
        return 1
    模板 = os.path.join(GGUF目录, launch["对话模板文件"])
    适配器 = None if 参数.无adapter else os.path.join(GGUF目录, launch["适配器文件"])
    缩放 = 参数.缩放 if 参数.缩放 is not None else float(launch.get("适配器缩放", 1.0))
    if 适配器 and not os.path.isfile(适配器):
        说("🔴 adapter 不存在：%s" % 适配器)
        return 1

    # ---- ② 挑空闲端口，被占当场失败（撞别人实例 = 连到旧版，与功能没实现同形）----
    if not 空闲吗(参数.端口):
        pid, 名 = 取端口占用者(参数.端口)
        说("🔴 %d 端口已被占用（PID %s ｜ %s）—— 换 --端口，或先处理占用者。" % (参数.端口, pid, 名))
        return 1

    命令 = [
        引擎exe,
        "--model", 模型路径,
        "--host", "127.0.0.1",
        "--port", str(参数.端口),
        "--ctx-size", str(参数.ctx),
        "--parallel", str(参数.parallel),
        "--n-gpu-layers", 参数.GPU层数,
        # 🔴 必须用 --chat-template-**file**（与生产端 service.rs 同参）——
        #    --chat-template 传路径的话引擎把**路径字符串**当模板内容，
        #    模板不生效、模型直接裸答（2026-10-17 实测踩过）。
        "--chat-template-file", 模板,
        "--jinja",
        "--flash-attn", "on",
    ]
    if 适配器:
        命令 += ["--lora", 适配器]
        if 缩放 != 1.0:
            命令 += ["--lora-scaled", 适配器, str(缩放)]
    if 参数.语法:
        # 🔴 必须转绝对路径：llama-server 按**自己的进程工作目录**解析相对路径，
        #    而我们是从项目目录敲命令 → 相对路径在服务侧找不到，
        #    报错是 `failed to open file 'eval/意图JSON.gbnf'`，看起来像文件不存在。
        语法路径 = os.path.abspath(参数.语法)
        if not os.path.isfile(语法路径):
            说("🔴 语法文件不存在：%s" % 语法路径)
            return 1
        命令 += ["--grammar-file", 语法路径]

    说("模型   ：%s" % 模型)
    说("adapter：%s（缩放 %s）" % (os.path.basename(适配器) if 适配器 else "（无 —— 基线臂）", 缩放))
    说("模板   ：%s" % launch["对话模板文件"])
    说("端口 %d ｜ ctx %d ｜ parallel %d ｜ GPU层数 %s%s" % (
        参数.端口, 参数.ctx, 参数.parallel, 参数.GPU层数,
        " ｜ 语法 %s" % os.path.basename(参数.语法) if 参数.语法 else ""))

    os.makedirs(日志目录, exist_ok=True)
    时间戳 = time.strftime("%m%d-%H%M%S")
    出路径 = os.path.join(日志目录, "评测生成端-%s.out.log" % 时间戳)
    错路径 = os.path.join(日志目录, "评测生成端-%s.err.log" % 时间戳)

    # ---- ③ 起服务（后台文件句柄接 stdout/stderr；stderr 是决定性证据）----
    出文件 = open(出路径, "wb")
    错文件 = open(错路径, "wb")
    try:
        进 = subprocess.Popen(命令, stdout=出文件, stderr=错文件, cwd=GGUF目录)
    except Exception as 错:                                    # noqa: BLE001
        说("🔴 起进程失败：%s" % 错)
        说("   （注意 Windows Popen 的 argv[0] 必须是绝对路径 —— 本脚本已用绝对路径）")
        return 1
    说("已报 PID %d（**不等于**起来了 —— 下面回看端口）" % 进.pid)
    with open(PID文件, "w") as 句柄:
        句柄.write(str(进.pid))

    # ---- ④ 回看端口 + /props（「活着」≠「能干这件事」）----
    好, props = 等就绪(参数.端口, 300)
    说("-" * 76)
    if not 好:
        说("🔴 %d 秒内没起来 —— 看 stderr：%s" % (300, 错路径))
        说("   常见原因：显存不够（另一个模型还在卡上）/ 参数不识别 / DLL 缺。")
        return 1
    槽 = (props or {}).get("total_slots")
    # 🔴 键名在**顶层** `model_path`（不是 default_generation_settings.model）——
    #    2026-10-17 实测：取错键时它一直打印 `model=?`，看上去像「引擎没报」，
    #    实际是判据自己写错了字段名（永不触发的钩子）。§9.5
    模名 = (props or {}).get("model_path") or ""
    # 🔴 adapter 身份**必须单独回读**：/props 只报主模型，不报 LoRA。
    #    「引擎 stderr 里没打 lora 字样」**不能**当成「没挂上」（该版本静默）——
    #    只有 /lora-adapters 的返回才是现场。
    已挂 = 取适配器(参数.端口)
    说("✅ 服务就绪 ｜ PID %d ｜ /props 回读：total_slots=%s ｜ model=%s" % (
        进.pid, 槽, os.path.basename(模名) if 模名 else "?"))
    if 已挂 is None:
        说("   🔴 /lora-adapters 取不到 —— adapter 身份**未验证**（不要当「没挂」）")
    elif not 已挂:
        说("   adapter：**无**（/lora-adapters 返回空列表 → 裸基座臂）")
    else:
        for a in 已挂:
            说("   adapter：%s ｜ scale=%s ｜ id=%s" % (
                os.path.basename(a.get("path") or "?"), a.get("scale"), a.get("id")))
    # 声明与实际对不上一律当场说出来，别让人自己比
    if 适配器 and 已挂 is not None and not 已挂:
        说("   🔴 传了 --lora 但 /lora-adapters 为空 —— 加载失败，本臂不能当「微调后」用")
    if not 适配器 and 已挂:
        说("   🔴 没传 --lora 但引擎报有 adapter —— 连到别的实例了")
    说("   stdout 日志：%s" % 出路径)
    说("   stderr 日志：%s" % 错路径)
    说()
    说("接评测（环境变量）：")
    说("    set CHATBI_GEN_URL=http://127.0.0.1:%d/v1/chat/completions" % 参数.端口)
    说("停：python scripts/评测-起生成端.py --停")

    # 驻留 —— 服务生命周期归调用方（本脚本前台阻塞，Ctrl+C 停）
    try:
        进.wait()
    except KeyboardInterrupt:
        说("收到 Ctrl+C，终止生成端")
        进.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(主())
