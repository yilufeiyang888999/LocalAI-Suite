# -*- coding: utf-8 -*-
"""评测-口语化动作.py —— 量「口语化动作请求」的**产出率**（不是判通过/失败）

## 为什么单独量这个

`动作嫌疑` 改判（2026-10-17）解决的是「动作请求被静默答成数字」——
它买的是**失败可见**，不是成功率。实测现场：`给东莞精诚发个函让他整改`
改判成功，但前两步吐空动作名、后两步反复查数，4 步用光 → 「未生成任何单据」。

所以这一组题**不是在测改判对不对**（那由 `自检-编排` 的确定性断言管），
而是回答一个数字：**10 句人话里，有几句真能拿到单据？**

## 🔴 判据纪律（三条，都是踩过的）

1. **不设通过线。** 这是「量」，不是「判」。设一个 80% 的线只会让人去调线。
   报告里给的是**分布**：改判了几句 / 有产物几句 / 卡在哪个结束原因。
2. **未跑必须与「跑完且为 0」可区分。** 生成端不可达 ⇒ 退出码 **3**、
   报告头写「未跑」，不许输出一份「产出率 0%」的报告。
3. **口径写死在报告头。** 生成端地址、模型名、是否精简提示词、
   动作嫌疑是否开启 —— 换一个条件，同一个数字就不是一回事。

## 用法

    python scripts/评测-起生成端.py            # 另开一个窗口/后台任务
    set CHATBI_GEN_URL=http://127.0.0.1:18000/v1/chat/completions
    set CHATBI_GEN_MODEL=DS-R1-7B-Q4KM+lora-v3
    python scripts/评测-口语化动作.py
    python scripts/评测-口语化动作.py --逐题      # 打印每一步轨迹

⚠️ **不要加 `CHATBI_LEAN_PROMPT=1`**：精简提示词是评测侧的实验条件，
   不是生产口径；用它量口语化动作，量到的是「精简后的模型」。
"""

import argparse
import json
import os
import sys
import time

脚本目录 = os.path.dirname(os.path.abspath(__file__))
if 脚本目录 not in sys.path:
    sys.path.insert(0, 脚本目录)

import 编排                                              # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:                                    # noqa: BLE001
        pass


# 🔴 题面刻意**不含**「出一份 / 起草 / 帮我生成」这类显式动词 ——
#    那些是规则直接判的（依据=确定性），不构成对「口语化」的检验。
#    `期望动作` 只用于人读归因，**不参与计分**（避免把「量」做成「判」）。
题目 = [
    ("东莞精诚的整改通知",                    "生成质量整改通知", "只有单据名，无动词"),
    ("给东莞精诚发个函让他整改",               "生成质量整改通知", "处置动词「发函/整改」"),
    ("精诚这批货得退货了",                    "生成质量整改通知", "退货，供应商简称"),
    ("东莞精诚最近老出问题，得跟他们谈谈",       "生成质量整改通知", "约谈，无单据名"),
    ("那家供应商的不良率超了，该处理一下",       "生成质量整改通知", "指代+处置，无实体名"),
    ("帮我把这个月的不良情况整理成台账",         "导出异常台账",     "台账，时间词"),
    ("轴承座快没货了，得补",                  "生成补货申请",     "补货，物料名"),
    ("给东莞精诚补个单子",                    "生成补货申请",     "补单，语义模糊"),
    ("这批次来料不合格，通知供应商整改",          "生成质量整改通知", "整改，无实体名"),
    ("上个月那些异常件都记下来",                "导出异常台账",     "台账，口语「记下来」"),
]


def 探生成端(地址):
    """返回 (可达, 版本串)。**探活只证明活着，不证明能生成** —— 所以判据里还要看耗时。"""
    import urllib.request
    import urllib.error
    基 = 地址.rsplit("/v1/", 1)[0]
    try:
        开 = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with 开.open(基 + "/props", timeout=5) as 回:
            数 = json.loads(回.read().decode("utf-8"))
        路 = str(数.get("model_path") or "")
        槽 = 数.get("total_slots")
        return True, "model=%s  slots=%s" % (os.path.basename(路) or "?", 槽)
    except Exception as 错:                              # noqa: BLE001
        return False, "%s: %s" % (type(错).__name__, 错)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--逐题", action="store_true", help="打印每一步轨迹")
    ap.add_argument("--最大步", type=int, default=4)
    ap.add_argument("--超时", type=float, default=180.0)
    ap.add_argument("--报告", default=os.path.join(
        os.path.dirname(脚本目录), "eval", "报告-口语化动作.md"))
    参 = ap.parse_args()

    地址 = os.environ.get("CHATBI_GEN_URL") or "http://127.0.0.1:8000/v1/chat/completions"
    模型 = os.environ.get("CHATBI_GEN_MODEL") or "（未设，用服务端默认）"
    精简 = os.environ.get("CHATBI_LEAN_PROMPT", "")
    可达, 版本 = 探生成端(地址)

    print("=" * 74)
    print("口语化动作 · 产出率测量（**这是量，不是判**）")
    print("=" * 74)
    print("生成端：%s" % 地址)
    print("模型名：%s" % 模型)
    print("精简提示词：%s" % ("⚠️ 开（不是生产口径，数字不可外推）" if 精简 else "关（生产口径）"))
    print("服务自述：%s" % 版本)
    if not 可达:
        print("\n🔴 生成端不可达 ⇒ **本轮未跑**（不是「产出率 0%」）。")
        print("   先起服务：python scripts/评测-起生成端.py")
        return 3

    定义 = 编排.定义对象()
    起 = time.time()
    结果 = []
    for 序, (问, 期望, 说明) in enumerate(题目, 1):
        t0 = time.time()
        try:
            应 = 编排.执行(问, 定义, 最大步=参.最大步)
            错 = None
        except Exception as e:                           # noqa: BLE001
            应, 错 = None, "%s: %s" % (type(e).__name__, e)
        耗时 = time.time() - t0
        条 = {"序号": 序, "问题": 问, "期望动作": 期望, "说明": 说明,
              "类型": (应 or {}).get("类型"),
              "改判": bool((应 or {}).get("改判")),
              "有产物": bool((应 or {}).get("产物")),
              "产物动作": ((应 or {}).get("产物") or {}).get("动作"),
              "结束原因": (应 or {}).get("结束原因"),
              "步数": len((应 or {}).get("步骤") or []),
              "耗时": round(耗时, 1), "错误": 错}
        结果.append(条)
        print("%2d. %-22s → %-6s %s 产物=%-5s %5.1fs  %s" % (
            序, 问[:22], 条["类型"], "改判" if 条["改判"] else "    ",
            "有" if 条["有产物"] else "无", 耗时, 条["结束原因"] or 条["错误"] or ""))
        if 参.逐题:
            for s in (应 or {}).get("步骤") or []:
                print("       %s %s ｜ %s" % (s.get("步"), s.get("动作"),
                                              str(s.get("观察"))[:80].replace("\n", " ")))

    总 = len(结果)
    有产 = sum(1 for r in 结果 if r["有产物"])
    改判数 = sum(1 for r in 结果 if r["改判"])
    对动作 = sum(1 for r in 结果 if r["产物动作"] == r["期望动作"])
    # 🔴 卡点必须**分两类**：没进动作链路 ≠ 进了但没做成。
    #    前者的病灶在**路由词表**（处置动词没覆盖这个词），后者的病灶在**ReAct + 模型**。
    #    混成一个「未知」桶，报告会指向错误的修法（本项目铁律：降级/跳过必须可区分）。
    未进链路 = [r for r in 结果 if r["类型"] != "动作执行"]
    卡点 = {}
    for r in 未进链路:
        键 = "未进动作链路（路由判成「%s」）" % r["类型"]
        卡点[键] = 卡点.get(键, 0) + 1
    for r in 结果:
        if r["有产物"] or r["类型"] != "动作执行":
            continue
        键 = r["结束原因"] or r["错误"] or "动作链路跑了但没记录结束原因"
        卡点[键] = 卡点.get(键, 0) + 1

    print("\n" + "-" * 74)
    print("产出率：**%d/%d = %.0f%%**｜改判：%d/%d｜产物动作对得上：%d/%d"
          % (有产, 总, 100.0 * 有产 / 总, 改判数, 总, 对动作, 总))
    print("卡点分两层（顶层才是主要矛盾）：")
    print("  · 路由层：%d 题没进动作链路" % len(未进链路))
    print("  · 执行层：%d 题进了但没产出" % (总 - 有产 - len(未进链路)))
    if 卡点:
        for k, v in 卡点.items():
            print("    - %s ×%d" % (k, v))
    print("总耗时：%.0f s" % (time.time() - 起))

    行 = ["# 口语化动作 · 产出率测量（%s）" % time.strftime("%Y-%m-%d %H:%M"),
          "",
          "> **这是量，不是判。** 不设通过线 —— 设线只会让人去调线。",
          "",
          "| 口径 | 值 |", "|---|---|",
          "| 生成端 | `%s` |" % 地址,
          "| 模型名 | `%s` |" % 模型,
          "| 精简提示词 | %s |" % ("⚠️ 开（非生产口径）" if 精简 else "**关**（生产口径）"),
          "| 服务自述 | `%s` |" % 版本,
          "| 最大步 | %d |" % 参.最大步,
          "",
          "**产出率 %d/%d = %.0f%%**｜改判 %d/%d｜产物动作对得上 %d/%d"
          % (有产, 总, 100.0 * 有产 / 总, 改判数, 总, 对动作, 总),
          "",
          "**卡点分两层**：路由层 %d 题（没进动作链路）｜执行层 %d 题（进了但没产出）"
          % (len(未进链路), 总 - 有产 - len(未进链路)),
          "",
          "| # | 问题 | 类型 | 改判 | 产物 | 动作 | 卡在 | 步数 | 耗时 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in 结果:
        if r["有产物"]:
            卡 = "—"
        elif r["类型"] != "动作执行":
            卡 = "**未进动作链路**（路由判成「%s」）" % r["类型"]
        else:
            卡 = r["结束原因"] or r["错误"] or "（无记录）"
        行.append("| %d | %s | %s | %s | %s | %s | %s | %d | %.1fs |" % (
            r["序号"], r["问题"], r["类型"] or "—",
            "是" if r["改判"] else "", "**有**" if r["有产物"] else "无",
            r["产物动作"] or "—", 卡, r["步数"], r["耗时"]))
    if 卡点:
        行 += ["", "**卡点分布**："]
        行 += ["- %s ×%d" % (k, v) for k, v in 卡点.items()]
    os.makedirs(os.path.dirname(参.报告), exist_ok=True)
    with open(参.报告, "w", encoding="utf-8") as f:
        f.write("\n".join(行) + "\n")
    print("报告：%s" % 参.报告)
    return 0


if __name__ == "__main__":
    sys.exit(main())
