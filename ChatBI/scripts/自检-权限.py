# -*- coding: utf-8 -*-
"""ChatBI · 包 6 自检 —— 权限白名单 + 审计日志。

五节：
  一、权限白名单    配置对齐 + **判定矩阵（3 角色 × 3 动作，必放 + 必拒）**
  二、动作关卡      越权硬拒（不落盘）/ 放行落盘 / 「被拒」与「缺参数」分得开
  三、审计          每次调用都留痕（含被拒）/ 依据切题可查 / append-only / 坏行容错
  四、端到端        真起 8096：越权 403、放行 200、未知角色 400
  五、回归          包 5 自检（--跳过HTTP）无回归

🔴 三条判据纪律（都是本项目踩过才写的）

  1. **期望矩阵是**手写的**，不读 `权限.角色表`。**
     拿被测对象自己的数据当判据，等于让它自己给自己打分 ——
     白名单写错了照样「全绿」。判据必须是**独立的一份事实**。
  2. **双侧**：只测「必拒」会把「一律拒绝」当通过；
     只测「必放」会把「一律放行」当通过。两边都要，缺一个判据就失效。
  3. **自检不污染产物目录与审计**：开头把 `动作.产物目录` / `审计.审计目录`
     指到 `outbox/_自检权限/`，结束 `rmtree`（只改两个模块级变量，因为它们是唯一事实来源）。
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

本目录 = os.path.dirname(os.path.abspath(__file__))
项目目录 = os.path.dirname(本目录)
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)

import 审计                                              # noqa: E402
import 权限                                              # noqa: E402
import 动作                                              # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:                                    # noqa: BLE001
        pass

日志行 = []
通过数 = 失败数 = 未查数 = 0
绕代理 = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# 🔴 独立判据：3 角色 × 3 动作的**期望**结果。手写，不从 权限.py 读。
期望矩阵 = {
    ("操作员", "导出异常台账"): True,
    ("操作员", "生成补货申请"): False,
    ("操作员", "生成质量整改通知"): False,
    ("主管", "导出异常台账"): True,
    ("主管", "生成补货申请"): True,
    ("主管", "生成质量整改通知"): False,
    ("管理员", "导出异常台账"): True,
    ("管理员", "生成补货申请"): True,
    ("管理员", "生成质量整改通知"): True,
}

示例依据 = {"列": ["供应商", "到料检验不良率"], "行": [["东莞精诚五金制品有限公司", 4.79]],
          "问题": "东莞精诚的来料不良率"}

原产物目录 = 动作.产物目录
原审计目录 = 审计.审计目录
临时根 = os.path.join(原产物目录, "_自检权限")


def 说(文本=""):
    print(文本)
    日志行.append(文本)


def 判(名称, 好, 说明=""):
    global 通过数, 失败数
    if 好:
        通过数 += 1
        说("  ✅ %-30s %s" % (名称, 说明))
    else:
        失败数 += 1
        说("  🔴 %-30s %s" % (名称, 说明))
    return 好


def 未查(名称, 说明):
    global 未查数
    未查数 += 1
    说("  ⚠️  %-30s %s" % (名称, 说明))


def 产物数():
    if not os.path.isdir(动作.产物目录):
        return 0
    return len([x for x in os.listdir(动作.产物目录) if os.path.isfile(
        os.path.join(动作.产物目录, x))])


# ============================ 一、权限白名单 ============================

def 第一节_白名单():
    说("=" * 74)
    说("一、权限白名单（配置对齐 + 判定矩阵）")
    说("=" * 74)

    问题 = 权限.校验配置()
    判("白名单与真实动作对齐", not 问题,
       "；".join(问题) if 问题 else "%d 角色 / %d 动作" % (
           len(权限.角色表), len(set(动作.动作表))))

    # 判定矩阵：不读 权限.角色表 的判定，只看 权限.检查 的返回
    格子 = []
    漏判 = []
    for 角色 in 权限.角色表:
        for 动作名 in 动作.动作表:
            if (角色, 动作名) not in 期望矩阵:
                漏判.append("%s×%s" % (角色, 动作名))
                continue
            允许, _理由 = 权限.检查(角色, 动作名)
            若 = 期望矩阵[(角色, 动作名)]
            格子.append((角色, 动作名, 若, 允许))
    错 = [(a, b, 期, 实) for (a, b, 期, 实) in 格子 if 期 != 实]
    判("判定矩阵 %d 格（必放 + 必拒双侧）" % len(格子), not 错,
       "全部符合期望" if not 错 else "；".join(
           "%s×%s 期望%s实际%s" % (a, b, "放行" if 期 else "拒绝", "放行" if 实 else "拒绝")
           for (a, b, 期, 实) in 错))
    if 漏判:
        未查("矩阵覆盖", "以下组合未在判据预期里（新增动作需补预期）：%s" % "、".join(漏判))

    # 必放侧与必拒侧各自单独记一笔 —— 混在一起的「全对」看不出是哪一侧在起作用
    必放 = [x for x in 格子 if x[2]]
    必拒 = [x for x in 格子 if not x[2]]
    判("必放侧 %d 格全放行" % len(必放), all(x[3] for x in 必放))
    判("必拒侧 %d 格全拒绝" % len(必拒), not any(x[3] for x in 必拒))

    # 未知角色
    允许, 理由 = 权限.检查("临时工", "导出异常台账")
    判("未知角色被拒且列出可用角色", (not 允许) and "操作员" in 理由, 理由[:52])

    # 拒绝要给「出路」
    _, 理由2 = 权限.检查("操作员", "生成质量整改通知")
    判("拒绝理由给出路（该动作需要谁）", "管理员" in 理由2, 理由2[:52])

    # 内部直调：显式边界，不是静默放行
    允许3, 理由3 = 权限.检查(None, "生成质量整改通知")
    判("内部直调放行且理由写明是直调", 允许3 and "内部直调" in 理由3, 理由3[:52])

    # 死配置：有没有动作谁都不能做
    覆盖 = set()
    for 角色 in 权限.角色表:
        覆盖 |= set(权限.允许动作(角色))
    判("无「谁都不能做」的动作", set(动作.动作表) <= 覆盖,
       "覆盖 %d / %d" % (len(覆盖), len(动作.动作表)))

    # 路径白名单：根目录必须与 动作.产物目录 同源，且真能判定
    里 = 权限.在产物目录内(os.path.join(动作.产物目录, "x.md"))
    外 = 权限.在产物目录内(os.path.join(项目目录, "logs", "x.md"))
    判("路径白名单：内真外假", 里 and not 外, "根=%s" % 动作.产物目录)


# ============================ 二、动作关卡 ============================

def 第二节_动作关卡():
    说("")
    说("=" * 74)
    说("二、动作关卡（权限与审计的唯一必经关口）")
    说("=" * 74)

    前 = 产物数()

    出 = 动作.调用("生成质量整改通知", 角色="操作员", 供应商="东莞精诚", 依据=示例依据)
    判("越权：返回被拒标记（不是「参数错」）", bool(出.get("被拒")) and not 出.get("路径"),
       出.get("拒绝理由", "")[:46])
    判("越权：不落盘", 产物数() == 前, "产物数 %d → %d" % (前, 产物数()))

    出2 = 动作.调用("生成质量整改通知", 角色="管理员", 供应商="东莞精诚", 依据=示例依据)
    判("放行：落盘且在产物目录内",
       bool(出2.get("路径")) and 权限.在产物目录内(出2["路径"]),
       os.path.basename(出2.get("路径") or "—"))
    判("放行：产物数 +1", 产物数() == 前 + 1, "产物数 %d → %d" % (前, 产物数()))

    # 「被拒」与「缺参数」必须分得开 —— 前者换人，后者补参数
    出3 = 动作.调用("生成质量整改通知", 角色="管理员", 依据=示例依据)
    判("有权限但缺参数 → 报「缺参数」而非「被拒」",
       (not 出3.get("被拒")) and bool(出3.get("错误")), 出3.get("错误", "")[:46])

    出4 = 动作.调用("删除全部数据", 角色="管理员", 依据=示例依据)
    判("未知动作：报错且不落盘",
       bool(出4.get("错误")) and 产物数() == 前 + 1, 出4.get("错误", "")[:44])

    出5 = 动作.调用("导出异常台账", 角色="操作员", 依据=示例依据)
    判("操作员可导出台账（必放侧）", bool(出5.get("路径")), os.path.basename(出5.get("路径") or "—"))


# ============================ 三、审计 ============================

def 第三节_审计():
    说("")
    说("=" * 74)
    说("三、审计（留痕必须覆盖「被拒」，且能回答「依据切不切题」）")
    说("=" * 74)

    条 = 审计.读全部()
    判("每次调用都留痕（≥5 条）", len(条) >= 5, "%d 条" % len(条))

    被拒 = [x for x in 条 if x.get("被拒")]
    判("被拒也留痕（≥1 条）", len(被拒) >= 1, "%d 条被拒记录" % len(被拒))
    判("被拒记录带理由且 落盘=false",
       all(x.get("拒绝理由") and x.get("落盘") is False for x in 被拒),
       (被拒[-1].get("拒绝理由", "")[:40] if 被拒 else "—"))

    落盘 = [x for x in 条 if x.get("落盘")]
    判("放行记录带产物路径", all(x.get("产物") and x.get("角色") for x in 落盘),
       "%d 条落盘记录" % len(落盘))

    # 依据切题要能写进审计并查得出来
    动作.调用("生成补货申请", 角色="主管", 依据=示例依据, 依据切题=False)
    不切 = 审计.查询(限=5, 被拒=False)
    判("依据切题=false 可写入审计",
       any(x.get("依据切题") is False for x in 不切),
       "统计=%s" % json.dumps(审计.统计().get("依据不切题数")))

    # append-only：再写一条，旧的必须还在
    前数 = len(审计.读全部())
    审计.记({"角色": "测试", "动作名": "append-only 探针"})
    判("append-only（新写不覆盖旧记录）", len(审计.读全部()) == 前数 + 1,
       "%d → %d" % (前数, len(审计.读全部())))

    # 坏行容错：手动塞一行垃圾，读的时候不能整份崩掉
    路 = 审计.文件()
    with open(路, "a", encoding="utf-8") as 文:
        文.write("{这不是合法 JSON\n")
    读回 = 审计.读全部()
    判("坏行不影响读取（跳过而非崩）", len(读回) == 前数 + 1, "仍读到 %d 条" % len(读回))

    # 写失败不抛异常
    原审计目录2 = 审计.审计目录
    拦路 = os.path.join(临时根, "这不是目录.txt")
    with open(拦路, "w", encoding="utf-8") as 文:
        文.write("x")
    审计.审计目录 = 拦路
    try:
        回 = 审计.记({"角色": "测试", "动作名": "写失败探针"})
        判("写失败不抛异常（业务不受影响）", bool(回.get("_写失败")),
           str(回.get("_写失败"))[:46])
    except Exception as 错:                               # noqa: BLE001
        判("写失败不抛异常（业务不受影响）", False, "抛了 %s" % type(错).__name__)
    finally:
        审计.审计目录 = 原审计目录2

    统 = 审计.统计()
    判("统计能按角色/动作聚合", bool(统.get("按角色")) and bool(统.get("按动作")),
       "总 %d ｜ 被拒 %d ｜ 落盘 %d" % (统.get("总条数", 0), 统.get("被拒数", 0),
                                    统.get("落盘数", 0)))


# ============================ 四、端到端 ============================

def 第四节_HTTP(跳过模型):
    说("")
    说("=" * 74)
    说("四、端到端（真起 8096：越权 403 / 放行 200 / 未知角色 400）")
    说("=" * 74)

    端口 = 8096
    try:
        import socket
        s = socket.socket()
        s.settimeout(0.6)
        占用 = s.connect_ex(("127.0.0.1", 端口)) == 0
        s.close()
        if 占用:
            未查("端到端", "端口 %d 已被占用（先停掉再跑）" % 端口)
            return
    except OSError:
        pass

    脚本 = os.path.join(本目录, "编排服务.py")
    # 🔴 子进程必须**显式**带上覆盖。子进程有自己的模块状态 ——
    #    本进程里 `动作.产物目录 = ...` / `审计.审计目录 = ...` 它根本看不到，
    #    只有环境变量能传过去。（第一次跑就是这么漏的：HTTP 段的「放行」把单据
    #    写进了真实 outbox/、403 探针写进了真实审计，污染了演示记录。）
    环境 = dict(os.environ)
    环境["CHATBI_AUDIT_DIR"] = 审计.审计目录
    环境["CHATBI_OUTBOX_DIR"] = 动作.产物目录
    进 = subprocess.Popen([sys.executable, 脚本, "--端口", str(端口), "--无上屏"],
                         cwd=项目目录, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         env=环境)
    基 = "http://127.0.0.1:%d" % 端口
    try:
        起 = time.time()
        通 = False
        while time.time() - 起 < 20:
            try:
                # 🔴 这个 timeout 必须 > /health 的最坏耗时：编排服务启动初期
                #    /health 实测 ~4 s 才回（链路探活占着锁）—— 2026-10-17 实测
                #    timeout=2 时**每次都超时**，20 秒轮询内「永远不通」，
                #    端到端节稳定假红。判据：客户端超时必须大于最坏耗时（§9.5）。
                绕代理.open(基 + "/health", timeout=6).read()
                通 = True
                break
            except Exception:                              # noqa: BLE001
                time.sleep(0.4)
        判("服务起来（/health 通）", 通, "端口 %d" % 端口)
        if not 通:
            return

        健 = json.loads(绕代理.open(基 + "/health", timeout=5).read().decode())
        判("health 带审计统计", bool(健.get("审计", {}).get("文件")),
           os.path.basename(健.get("审计", {}).get("文件", "")))
        判("health 带权限白名单且无配置问题",
           bool(健.get("权限")) and not 健["权限"].get("白名单问题"),
           "默认角色=%s ｜ 角色=%s" % (健.get("权限", {}).get("默认角色"),
                                   "、".join(健.get("权限", {}).get("角色") or [])))
        目 = json.loads(绕代理.open(基 + "/catalog", timeout=5).read().decode())
        判("catalog 带角色清单", len(目.get("角色") or []) == 3,
           "默认=%s" % 目.get("默认角色"))

        def POST(路0, 体):
            """返回 (状态码, 对象)。🔴 必须接住 HTTPError ——
            判据里就有「越权必须回 403」这一条，helper 只在 2xx 返回的话，
            **服务规规矩矩回了 403，脚本反而抛异常**，「服务对了」被记成「自检失败」。"""
            req = urllib.request.Request(基 + 路0,
                                         data=json.dumps(体, ensure_ascii=False).encode(),
                                         headers={"Content-Type": "application/json"},
                                         method="POST")
            try:
                with 绕代理.open(req, timeout=180) as 应:
                    return 应.status, json.loads(应.read().decode())
            except urllib.error.HTTPError as 错:
                try:
                    return 错.code, json.loads(错.read().decode())
                except Exception:                          # noqa: BLE001
                    return 错.code, {}

        码, 坏 = POST("/ask", {"问题": "给东莞精诚出一份质量整改通知", "角色": "临时工"})
        判("未知角色 → 400", 码 == 400 and "error" in 坏,
           "HTTP %d｜%s" % (码, str(坏.get("error"))[:40]))

        if 跳过模型:
            未查("越权 → 403", "--跳过模型 跳过了（需要 8000 生成端）")
            未查("放行 → 200", "--跳过模型 跳过了")
            return

        码, 应 = POST("/ask", {"问题": "给东莞精诚出一份质量整改通知", "角色": "操作员"})
        判("越权 → HTTP 403", 码 == 403 and 应.get("被拒") is True,
           "HTTP %d｜被拒=%s" % (码, 应.get("被拒")))
        判("越权 → 产物为 None（硬拒不留半成品）", 应.get("产物") is None,
           "产物=%s" % 应.get("产物"))
        判("越权 → 轨迹里有「被拒」一步",
           any(s.get("被拒") for s in (应.get("步骤") or [])),
           "结束原因=%s" % 应.get("结束原因"))

        码, 应2 = POST("/ask", {"问题": "给东莞精诚出一份质量整改通知", "角色": "管理员"})
        判("同问题换管理员 → 200 且产出单据",
           码 == 200 and bool((应2.get("产物") or {}).get("路径")),
           os.path.basename((应2.get("产物") or {}).get("路径", "")))
    finally:
        进.terminate()
        try:
            进.wait(timeout=5)
        except subprocess.TimeoutExpired:
            进.kill()


# ============================ 五、回归 ============================

def 第五节_回归():
    说("")
    说("=" * 74)
    说("五、回归（包 5 自检的**确定性部分** —— 权限层不能改坏已有路由与护栏）")
    说("=" * 74)
    脚本 = os.path.join(本目录, "自检-编排.py")
    if not os.path.isfile(脚本):
        未查("包 5 自检", "脚本缺失")
        return
    # 把回归子进程的审计引到临时目录 —— 回归会真调动作，不该把「（内部）」记录
    # 混进真实审计（真实审计要能回答「谁在什么时候做了什么」，不该掺测试噪声）。
    环境 = dict(os.environ)
    环境["CHATBI_AUDIT_DIR"] = 审计.审计目录
    环境["CHATBI_OUTBOX_DIR"] = 动作.产物目录
    # 🔴 用「--跳过HTTP --跳过模型」，**不是**全量：
    #    全量含**兜底路由题**，那几题要走真模型、每次答案可能不同（包 4 的教训：
    #    同一提示词两种答案）。拿一个不可复现的数字当回归判据，判据自己就先坏了 ——
    #    本文件第一次跑就是这么错的：用全量基线 34/1/0 去比 --跳过HTTP 的 29 项。
    #    确定性部分（规则路由全会 + 假模型护栏 + 动作产物）才是「我有没有改坏它」的证据。
    try:
        进 = subprocess.run([sys.executable, 脚本, "--跳过HTTP", "--跳过模型"],
                            cwd=项目目录, capture_output=True, timeout=600, env=环境)
        文本 = (进.stdout or b"").decode("utf-8", "replace")
    except Exception as 错:                                # noqa: BLE001
        未查("包 5 自检", "%s: %s" % (type(错).__name__, 错))
        return
    找 = re.search(r"合计：(\d+) 通过 / (\d+) 失败 / (\d+) 未检查", 文本)
    if not 找:
        未查("包 5 自检", "没解析到合计行")
        return
    通, 败, 未 = (int(x) for x in 找.groups())
    # 🔴 这两个数字是**基线**，故意写死：它的用途是发现「包 5 的断言少了 / 挂了」。
    #    改成「>= 24」或「失败==0 就行」就等于不判 —— 断言被删掉一半也会通过。
    #    代价是**加断言时要回来同步**（2026-10-17 加「动作嫌疑」9 条：24→33；
    #    2026-10-18 加「口语化处置词」8 条 + 「结束原因传到回答」1 条：33→41）。
    判("包 5 确定性部分无回归（基线 41/0/2）", 通 == 41 and 败 == 0 and 未 == 2,
       "%d 通过 / %d 失败 / %d 未检查" % (通, 败, 未))
    # 「确定性路由全会」的分母不写死 —— 题集加题就会变，那属于题集的事；
    # 这里要钉的是**分子等于分母**（规则可证，必须全会）。分母少了由上一行兜住。
    找2 = re.search(r"确定性合计\s+(\d+)/(\d+)", 文本)
    判("确定性路由全会（X/X）", bool(找2) and 找2.group(1) == 找2.group(2),
       ("确定性路由 %s/%s" % (找2.group(1), 找2.group(2))) if 找2 else "没解析到确定性合计")


# ============================ 主 ============================

def 主():
    sys.stdout.reconfigure(errors="replace")
    参 = argparse.ArgumentParser(description="ChatBI 包 6 自检（权限 + 审计）")
    参.add_argument("--跳过模型", action="store_true", help="跳过需要 8000 生成端的项")
    参.add_argument("--跳过HTTP", action="store_true", help="不跑第四节（不真起 8096）")
    参.add_argument("--跳过回归", action="store_true", help="不跑第五节（不调包 5 自检）")
    参 = 参.parse_args()

    起 = time.time()
    说("=" * 74)
    说("ChatBI 包 6 自检 —— 权限白名单 + 审计日志")
    说("=" * 74)
    说("隔离：产物与审计都改写临时目录，结束清理 —— 不污染真实 outbox/ 与审计。")

    os.makedirs(临时根, exist_ok=True)
    动作.产物目录 = 临时根
    # 🔴 审计放**子目录** —— 与产物同目录的话，`产物数()` 会把审计文件自己数进去，
    #    「越权不落盘」那条判据就被自己的审计搅乱了（判据自己出错）。
    审计.审计目录 = os.path.join(临时根, "审计")
    try:
        第一节_白名单()
        第二节_动作关卡()
        第三节_审计()
        if 参.跳过HTTP:
            未查("端到端", "--跳过HTTP 跳过了")
        else:
            第四节_HTTP(参.跳过模型)
        if 参.跳过回归:
            未查("回归", "--跳过回归 跳过了")
        else:
            第五节_回归()
    finally:
        动作.产物目录 = 原产物目录
        审计.审计目录 = 原审计目录
        shutil.rmtree(临时根, ignore_errors=True)

    用时 = time.time() - 起
    说("")
    说("=" * 74)
    说("合计：%d 通过 / %d 失败 / %d 未检查（%.1f s）"
       % (通过数, 失败数, 未查数, 用时))
    说("=" * 74)

    日志路径 = os.path.join(项目目录, "logs", "自检-权限.log")
    try:
        os.makedirs(os.path.dirname(日志路径), exist_ok=True)
        with open(日志路径, "w", encoding="utf-8") as 文:
            文.write("\n".join(日志行) + "\n")
        说("自检记录：%s" % 日志路径)
    except OSError as 错:
        说("⚠️ 自检记录写失败：%s" % 错)
    return 0 if 失败数 == 0 else 1


if __name__ == "__main__":
    raise SystemExit(主())
