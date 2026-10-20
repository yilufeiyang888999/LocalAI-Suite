# -*- coding: utf-8 -*-
"""ChatBI · 一键回归 —— 把包 1~13 的自检 + 评测串成一条命令

## 为什么要有这个脚本

回归原来是「按顺序手工跑 8 个脚本」，跑一轮要记住顺序、环境变量、哪些要生成端。
漏一个环境变量（`CHATBI_NO_PROMPT_CACHE` / `CHATBI_NO_INTENT_CACHE`）的后果
不是报错，是**数字不可复现**（§9.5：与「模型很差」同形）。本脚本把顺序、
环境变量、前置检查、超时、落盘一次定死。

## 分两段（不是分两档 —— 段是按「依赖」切，不是按「重要性」切）

**离线段**（不需要 8000 生成端，随时可跑）：

    1. 校验-语义层.py          结构自检 + 逐题对账（15/15）
    2. 自检-拒答与澄清.py      确定性判定 / 词表卫生（纯逻辑）
    3. 自检-B档.py             缓存 / 脱敏 / 身份 / 行级（纯逻辑）

**在线段**（需要 8000 生成端活着，且**必须先过体检**）：

    4. 自检-会话.py            /ask 多轮节真调模型
    5. 自检-编排.py            兜底路由 / 端到端·动作执行（ReAct 调模型）
    6. 自检-权限.py            端到端越权 403：动作链路第一步就调模型，
                               **生成端不可达时连权限关卡都到不到**（2026-10-17 实测：
                               离线跑会假红「越权返回 200」——不是权限 bug，是前置没起）
    7. 评测-链路准确率.py      50 题（先离线对账再在线，脚本内自带）
    8. 评测-盲测集.py          40 题主尺

    ⚠️ 分段依据是「有没有依赖生成端的节」，不是脚本名 —— 判据：
       无生成端时退出码是否仍为 0。分段放错 = 离线段假红
       （2026-10-17 首版把 编排/权限 放离线段，全部假红）。

⚠️ 在线段前置体检复用 `诊断-生成端卡死.py` 的判据（同源不抄第二份）：
   `/slots` 拿不到锁 = 卡死态 —— **端口活不算活**，直接把在线段整段标「跳过」，
   退出码给 3，不许拿卡死端点跑出一个「83.3%」的假数字。

## 环境变量（这里统一设，子进程全继承）

    CHATBI_NO_PROMPT_CACHE=1   生成端前缀缓存关（可复现）
    CHATBI_NO_INTENT_CACHE=1   语义缓存关（评测脚本自己也会硬门槛拦，这里前置省一轮）

## 判据纪律（§9.5）

    · 每个脚本独立子进程，stdout/stderr **各落一个文件** —— 决定性证据常在 stderr。
    · 超时默认 7200 s（> 60 题最坏耗时），**超时单独报**，不与「失败」混同；
      客户端超时小于最坏耗时 = 亲手制造 slot 卡死（五次复现的那条铁律）。
    · 退出码：0 = 全过；1 = 有失败；3 = 有跳过/未跑（含在线段整段跳过）。
      1 与 3 **可叠加语义**：既有失败又有跳过 → 取 1（失败优先），报告里分开列。
    · 报告必须写「已跑 N / 应跑 M」—— 83.3%（10/12）这种分母是已跑题数的报告不许出现。

用法：

    python scripts/一键回归.py               # 离线段 + 在线段（生成端没起 → 在线段自动跳过，退出码 3）
    python scripts/一键回归.py --段 离线     # 只跑离线段
    python scripts/一键回归.py --段 在线     # 只跑在线段（生成端没起 → 直接失败退出码 3）
    python scripts/一键回归.py --演练        # 只打印将跑什么，不执行
"""

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time

脚本目录 = os.path.dirname(os.path.abspath(__file__))
项目目录 = os.path.dirname(脚本目录)
产物目录 = os.path.join(项目目录, "logs", "一键回归")

离线段 = [
    "校验-语义层.py",
    "自检-拒答与澄清.py",
    "自检-B档.py",
]
在线段 = [
    "自检-会话.py",
    "自检-编排.py",
    "自检-权限.py",
    "评测-链路准确率.py",
    "评测-盲测集.py",
]

默认超时秒 = 7200          # 60 题最坏耗时（每题 2 分钟）之上，再宽一倍


def 说(文本=""):
    print(文本)
    sys.stdout.flush()


def 载入诊断模块():
    """与 `重启-生成端.py` 同一个做法：判据同源，不抄第二份。"""
    路径 = os.path.join(脚本目录, "诊断-生成端卡死.py")
    if not os.path.isfile(路径):
        raise SystemExit("找不到同目录的诊断模块：%s" % 路径)
    规格 = importlib.util.spec_from_file_location("诊断生成端卡死", 路径)
    模块 = importlib.util.module_from_spec(规格)
    规格.loader.exec_module(模块)
    return 模块


def 生成端体检(端口=8000):
    """(是否可用, 一句话结论)。判据与 `诊断-生成端卡死.py` 同源：
    /slots 必须拿得到锁 —— 端口活不算活（卡死态 /health 秒回 200）。"""
    诊断 = 载入诊断模块()
    听, _ = 诊断.端口占用者(端口)
    if not 听:
        return False, "%d 端口无监听（生成端没起；去 Local-AI APP 里启动模型服务）" % 端口
    码, 用时, _ = 诊断.探(端口, "/slots", 超时=6)
    if 码 is None:
        return False, "%d 端口有监听但 /slots 拿不到锁 —— **卡死态**（活着但不应答）；先 `python scripts/重启-生成端.py`" % 端口
    码2, _, _ = 诊断.探(端口, "/health", 3)
    if 码2 != 200:
        return False, "%d 端口 /slots 通了但 /health %s —— 正在加载？等半分钟再试" % (端口, 码2)
    return True, "生成端可用（/slots %.2fs 拿到锁）" % 用时


def 判非零退出(出文本):
    """退出码非 0 时，判它是「失败」还是「正常但没全绿」。

    🔴 为什么单独成函数：这段判据要能被自测直接喂字符串验证 ——
       判据本身的正确性不能靠「跑一遍看着像对的」。
    """
    # ① 先读脚本自己的合计行。真实格式：`合计：26 通过 / 2 失败 / 2 未检查（9.6 s）`
    #    「0」在「失败」**前面**，所以旧写法 `失败\\s*0` 永远匹配不到（2026-10-17 实测：
    #    那个「有未查」分支从来没被正确触发过）。判据同源：读脚本自己打的合计行。
    合计 = re.search(r"合计：(\d+) 通过\s*/\s*(\d+) 失败\s*/\s*(\d+) 未检查", 出文本)
    if 合计:
        _, 数失, 数未 = (int(x) for x in 合计.groups())
        if 数失 > 0:
            return "失败"
        return "通过·有未查" if 数未 > 0 else "通过"
    # ② 评测类：退出码 1 = 跑完但有题没过，脚本自己会打这句话（正常结果，不是故障）
    if "跑完但有题没过" in 出文本:
        return "通过·有未过"
    # ③ 兜底：没合计行、但自己说「失败 0」的（评测-链路准确率这类）
    if re.search(r"失败\s*0", 出文本):
        return "通过·有未查"
    return "失败"


def 跑一个(脚本名, 超时秒, 演练):
    """子进程跑一个脚本。返回记录 dict。
    🔴 stdout/stderr 各落文件 —— 决定性证据常在被丢弃的 stderr 里（§9.5）。"""
    os.makedirs(产物目录, exist_ok=True)
    时间戳 = time.strftime("%H%M%S")
    基名 = os.path.splitext(脚本名)[0]
    出路径 = os.path.join(产物目录, "%s-%s.out.txt" % (时间戳, 基名))
    错路径 = os.path.join(产物目录, "%s-%s.err.txt" % (时间戳, 基名))
    记 = {"脚本": 脚本名, "出文件": 出路径, "错文件": 错路径}
    # 🔴 per-脚本环境覆盖：多数脚本要「关缓存」（可复现），但 自检-B档 修的就是
    #   缓存本体 —— NO_INTENT_CACHE=1 会把缓存关掉，让「存得进」假红
    #   （2026-10-17 实测：全绿脚本被回归的环境变量打红）。编排者要懂每个脚本的口味。
    环境 = dict(os.environ)
    if 脚本名 == "自检-B档.py":
        环境.pop("CHATBI_NO_INTENT_CACHE", None)
    if 演练:
        记.update({"状态": "跳过", "说明": "演练", "耗时秒": 0})
        return 记
    python_exe = sys.executable
    if " " in python_exe:
        python_exe = '"%s"' % python_exe
    命令 = [sys.executable, os.path.join(脚本目录, 脚本名)]
    起 = time.time()
    try:
        结果 = subprocess.run(
            命令, capture_output=True,
            timeout=超时秒,
            cwd=脚本目录,
            env=环境,
        )
        出, 错 = 结果.stdout or b"", 结果.stderr or b""
        码 = 结果.returncode
        if 码 == 0:
            记["状态"] = "通过"
        else:
            # 🔴 退出码非 0 有两种含义，必须分开（§9.5：降级/跳过与通过要可区分）：
            #   ① 真失败（合计行「失败 N」且 N>0，或没有合计行 = 崩了）
            #   ② 「通过·有未查」—— 自检协议「失败 0 且未查 0 才退 0」，
            #      离线跑在线节时合法地有未查项。读脚本自己的合计行判定，
            #      不另立判据（同一份事实不两处定义）。
            出文本 = 出.decode("utf-8", "replace")
            记["状态"] = 判非零退出(出文本)
            记["退出码"] = 码
    except subprocess.TimeoutExpired as 错误:
        出 = 错误.stdout or b""
        错 = (错误.stderr or b"") + ("\n[一键回归] 超时 %d s，进程已终止" % 超时秒).encode("utf-8")
        记["状态"] = "超时"
        记["退出码"] = None
    记["耗时秒"] = round(time.time() - 起, 1)
    with open(出路径, "wb") as 句柄:
        句柄.write(出)
    with open(错路径, "wb") as 句柄:
        句柄.write(错)
    # 现场进记录 —— stderr 尾 + stdout 尾各带一点：
    #   决定性证据常在 stderr，但自检类脚本的「合计：N 通过 / M 失败」在 stdout；
    #   且有些 stderr 输出是**预期行为**（如写失败探针的「审计写失败」），
    #   只看 stderr 会把预期当原因（2026-10-17 实测踩过）。
    尾 = 错.decode("utf-8", "replace").strip().splitlines()[-2:]
    if 尾:
        记["stderr尾"] = " | ".join(x.strip() for x in 尾 if x.strip())
    尾2 = 出.decode("utf-8", "replace").strip().splitlines()[-2:]
    if 尾2:
        记["stdout尾"] = " | ".join(x.strip() for x in 尾2 if x.strip())
    return 记


def 段跑(清单, 超时秒, 演练):
    记录 = []
    for 名 in 清单:
        存在 = os.path.isfile(os.path.join(脚本目录, 名))
        if not 存在:
            记录.append({"脚本": 名, "状态": "跳过", "说明": "脚本不存在", "耗时秒": 0})
            continue
        说("▶ %s" % 名)
        记 = 跑一个(名, 超时秒, 演练)
        说("  %s ｜ %.1f s%s" % (
            记["状态"],
            记.get("耗时秒", 0),
            ("｜ " + (记.get("stderr尾") or "")) if 记["状态"] != "通过" and 记.get("stderr尾") else ""))
        if 记["状态"] != "通过" and 记.get("stdout尾"):
            说("    stdout尾：%s" % 记["stdout尾"])
        记录.append(记)
    return 记录


def 汇总(全部记录, 应跑数):
    说()
    说("=" * 76)
    说("汇总：已跑 %d / 应跑 %d" % (len([r for r in 全部记录 if r["状态"] != "跳过"]), 应跑数))
    坏 = 0
    跳 = 0
    未查 = 0
    未过 = 0
    for r in 全部记录:
        if r["状态"] == "通过":
            说("  ✅ %-28s %6.1f s" % (r["脚本"], r.get("耗时秒", 0)))
        elif r["状态"] == "通过·有未过":
            # 评测类：跑完了，只是有题没过。这是**结果**不是**故障** ——
            # 与「失败」必须分开，否则每次盲测都红灯，红灯就不再是信号。
            未过 += 1
            说("  🟡 %-28s %6.1f s ｜ 跑完但有题未过（正常结果，不算故障）"
               % (r["脚本"], r.get("耗时秒", 0)))
        elif r["状态"] == "通过·有未查":
            未查 += 1
            说("  🟡 %-28s %6.1f s ｜ 有未查项（依赖生成端的节）—— 不算失败也不算全绿"
               % (r["脚本"], r.get("耗时秒", 0)))
        elif r["状态"] == "跳过":
            跳 += 1
            说("  ⏭ %-28s %s" % (r["脚本"], r.get("说明", "")))
        else:
            坏 += 1
            说("  ❌ %-28s %s ｜ %s" % (r["脚本"], r["状态"],
                                       r.get("stderr尾") or r.get("说明") or ""))
            说("     现场：%s" % r.get("错文件", ""))
    说("-" * 76)
    if 坏:
        说("结论：❌ %d 项未过（失败/超时 %d ｜ 有未查 %d ｜ 跳过 %d）" % (坏 + 跳, 坏, 未查, 跳))
        return 1
    if 跳:
        说("结论：⏭ 无失败，但 %d 项跳过（多为生成端没起）—— 不算全绿" % 跳)
        return 3
    if 未查 or 未过:
        说("结论：🟡 无失败，%d 项有未查 / %d 项有题未过 —— 都不是故障，但也不是全绿"
           % (未查, 未过))
        return 3
    说("结论：✅ 全绿（%d/%d）" % (应跑数, 应跑数))
    return 0


def 主():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    参 = argparse.ArgumentParser(description="ChatBI 一键回归（离线段 + 在线段）")
    参.add_argument("--段", choices=["离线", "在线", "全部"], default="全部")
    参.add_argument("--演练", action="store_true", help="只打印将跑什么，不执行")
    参.add_argument("--超时秒", type=int, default=默认超时秒,
                    help="单个脚本超时（默认 %d；必须大于最坏耗时，否则是亲手制造卡死）" % 默认超时秒)
    # 🔴 生成端端口可配（2026-10-17 加）。默认 8000 = APP 生产端，行为不变；
    #    游离态评测（`评测-起生成端.py --端口 18000`）时要显式指过来。
    #    这不是「顺手加个便利参数」—— 端口写死会让游离态回归被迫去连 APP，
    #    而「连到别人的实例」与「功能没实现」同形（§9.5）。
    参.add_argument("--生成端端口", type=int, default=8000,
                    help="生成端端口（默认 8000 = Local-AI APP；游离态评测传 18000）")
    参数 = 参.parse_args()

    说("ChatBI · 一键回归 ｜ %s ｜ 段=%s ｜ 生成端 %d" % (
        time.strftime("%Y-%m-%d %H:%M:%S"), 参数.段, 参数.生成端端口))
    说("=" * 76)

    全部记录 = []
    应跑数 = 0

    环境清单 = {"CHATBI_NO_PROMPT_CACHE": os.environ.get("CHATBI_NO_PROMPT_CACHE"),
                "CHATBI_NO_INTENT_CACHE": os.environ.get("CHATBI_NO_INTENT_CACHE")}
    # 非 8000 → 子脚本必须知道生成端在哪（`意图解析` 读 CHATBI_GEN_URL）。
    # 8000 时不设，保持「走默认生产端」的原有行为。
    if 参数.生成端端口 != 8000:
        os.environ["CHATBI_GEN_URL"] = "http://127.0.0.1:%d/v1/chat/completions" % 参数.生成端端口
        环境清单["CHATBI_GEN_URL"] = os.environ["CHATBI_GEN_URL"]
    # 🔴 两个缓存开关在这里统一钉死 —— 子进程全继承。
    #    漏设的后果不是报错，是数字不可复现（§9.5），所以是「设」不是「提醒」。
    os.environ["CHATBI_NO_PROMPT_CACHE"] = "1"
    os.environ["CHATBI_NO_INTENT_CACHE"] = "1"

    跑离线 = 参数.段 in ("离线", "全部")
    跑在线 = 参数.段 in ("在线", "全部")

    if 跑离线:
        应跑数 += len(离线段)
        说("── 离线段（%d 个，不需生成端）──" % len(离线段))
        全部记录 += 段跑(离线段, 参数.超时秒, 参数.演练)

    if 跑在线:
        可用, 结论 = 生成端体检(参数.生成端端口)
        说("── 在线段前置体检：%s ──" % 结论)
        if not 可用:
            for 名 in 在线段:
                全部记录.append({"脚本": 名, "状态": "跳过", "说明": "生成端不可用：%s" % 结论, "耗时秒": 0})
        else:
            应跑数 += len(在线段)
            全部记录 += 段跑(在线段, 参数.超时秒, 参数.演练)

    return 汇总(全部记录, 应跑数)


if __name__ == "__main__":
    raise SystemExit(主())
