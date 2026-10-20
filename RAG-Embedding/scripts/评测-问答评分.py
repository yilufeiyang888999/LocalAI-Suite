# -*- coding: utf-8 -*-
"""
作品一 · 知识库问答 · 100 题实测评分（2026-10-10 新建）
=====================================================

干什么
------
把 `eval/问答100题.json` 里的 100 道题**逐题打到 8092 知识库服务**，收集答案，
按**可自动判定**的维度打分，产出：
  · eval/问答100题-结果.json   —— 原始数据（含答案、context、引用、耗时、判据明细）
  · eval/问答100题-评分表.md   —— Markdown 评分表
  · eval/问答100题-评分表.html —— 同内容的可读页面（浏览器直接打开）

为什么只能打「可自动判定」的分（而不是「答案对不对」）
--------------------------------------------------------
本题库是**全新拟题**，没有金标准合取（与 `eval/题目.json` 不同 —— 那是检索实验用的）。
因此这里不判「意思对不对」（那要人工逐题读），只判三件能自动判、且判错就是真错的事：

  ① **要点命中**：答案里有没有出现**本该出现的关键事实**（题库 `参考要点`，已用护栏校验
     每条都能在语料中逐字查到）。这是「答到点上没有」的下界 —— 命中不等于答对，
     但**不命中基本等于没答到**。
  ② **数值无编造**：答案里的数值（百分比 / 文档编号 / **带单位的量**）必须**逐字**
     出现在服务返回的 `context` 里。分母 = 答案里数值的**个数**（不是题数）——
     「一个数值都没写」不算通过，那是「没检查」不是「检查过且通过」（判据铁律 §9.5）。
  ③ **拒答正确**：库外题必须拒答；库内题不得被误拒。

评分口径（写死，别凭感觉调）
--------------------------
库内题（92 道）：
    · 答案里有数值 → 得分 = 70×要点率 + 15×忠实率 + 15×未误拒
    · 答案里无数值 → 得分 = 85×要点率        + 15×未误拒   （忠实度 N/A，权重并给要点）
库外题（8 道）：
    · 得分 = 100（正确拒答） / 0（未拒答）

🔴 生成失败（超时 / 服务不可达）**不计入分母**，并显式标注 —— 把失败当「答错」会把
   分数冲成假的低值（`诊断-库外拒答率.py` 已踩过这个坑）。

用法
----
    python scripts\\评测-问答评分.py                 # 全量 100 题（会真的问服务）
    python scripts\\评测-问答评分.py --复用          # 只重算分：复用上次的答案与 context，不碰服务
    python scripts\\评测-问答评分.py --限 5          # 只跑前 5 题（冒烟）
    python scripts\\评测-问答评分.py --端口 8093     # 服务在别的端口
    python scripts\\评测-问答评分.py --无重试        # 失败不重试

零第三方依赖（只用标准库）。
"""

import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from string import Template as _Template

脚本目录 = os.path.dirname(os.path.abspath(__file__))
根目录 = os.path.dirname(脚本目录)
输出目录 = os.path.join(根目录, "eval")
题库文件 = os.path.join(输出目录, "问答100题.json")
结果文件 = os.path.join(输出目录, "问答100题-结果.json")
表文件MD = os.path.join(输出目录, "问答100题-评分表.md")
表文件HTML = os.path.join(输出目录, "问答100题-评分表.html")

服务 = "http://127.0.0.1:8092"
超时秒 = 240
最多尝试 = 3

# ---------- 数值判定 ----------
# 🔴 只抽「必须先知道事实才能写出」的东西。三类：
#    ① 百分比 ② 文档编号 ③ **带单位的量**（含范围）。
#    第 ③ 类是 2026-10-10 补的 —— 首轮跑完发现只抽 % 和编号时，全 100 题只抽出 17 个数值，
#    「数值无编造」这条判据几乎没有取样，等于没检查（判据铁律：表现相同 = 没有判据）。
#    补上单位后覆盖到温度 / 厚度 / 时长 / 距离等实测值。
百分比模式 = re.compile(r"\d+(?:\.\d+)?\s*%")
编号模式 = re.compile(r"[A-Z]{2,}(?:-[A-Z0-9]+)+")
千分位模式 = re.compile(r"\d{1,3}(?:,\d{3})+")
单位 = r"(?:°C|℃|μm|um|mm|h|min|s|g/L|MPa|kPa|HRC|PPM|ppm|A/dm²|A/dm2|m/s)"
范围单位模式 = re.compile(r"\d+(?:\.\d+)?\s*[~～]\s*\d+(?:\.\d+)?\s*" + 单位)
单值单位模式 = re.compile(r"\d+(?:\.\d+)?\s*" + 单位)


def 规范(文本):
    """规范化：去空白 / 千分位逗号、全角转半角、**单位别名统一**、大小写不敏感。

    🔴 匹配要点与数值必须规范化 —— 语料写「5 ~ 8 μm」，模型常写「5~8μm」；
       语料写「9,265」，模型常写「9265」；语料写「4 h」，模型常写「4 小时」。
       不规范化就是「格式差异当错答案」。
    """
    文本 = str(文本)
    for 旧, 新 in (
            ("\u3000", ""), (" ", ""), ("\t", ""), ("\n", ""), ("\r", ""),
            (",", ""),
            ("～", "~"), ("〜", "~"), ("〜", "~"),
            ("％", "%"), ("\u00b5", "\u03bc"),
            ("℃", "°c"), ("小时", "h"), ("分钟", "min"), ("秒钟", "s"), ("秒", "s"),
            ("微米", "\u03bcm")):
        文本 = 文本.replace(旧, 新)
    return 文本.lower()


def 抽数值(文本):
    """抽三类必须先知道事实才能写出的东西：百分比、文档编号、带单位的量。

    同值去重（保序）—— 同一数值重复出现 3 次不该让它在分母里占 3 份。
    """
    出 = []
    for m in 百分比模式.finditer(文本):
        出.append(规范(m.group(0)))
    for m in 编号模式.finditer(文本):
        s = m.group(0)
        if any(c.isdigit() for c in s):            # 排除纯英文缩写（SCAR / SDS）
            出.append(规范(s))
    for m in 千分位模式.finditer(文本):
        出.append(规范(m.group(0)))
    for 模式 in (范围单位模式, 单值单位模式):
        for m in 模式.finditer(文本):
            出.append(规范(m.group(0)))
    去重 = []
    for v in 出:
        if v not in 去重:
            去重.append(v)
    # 去掉「子串冗余」：范围模式会顺带吐出单值端点（'220°c' ⊂ '190~220°c'、'8μm' ⊂ '5~8μm'），
    # 同一事实算两遍会把分母虚增。
    return [v for v in 去重 if not any(v != w and v in w for w in 去重)]


def 判要点(答案, 要点们):
    """返回 (命中数, 总数, 命中列表, 未中列表)。要点内用 | 表示可接受写法，命中任一即算。"""
    规范答案 = 规范(答案)
    命中, 未中 = [], []
    for 要点 in 要点们:
        alts = [规范(a) for a in 要点.split("|") if a.strip()]
        if alts and any(a in 规范答案 for a in alts):
            命中.append(要点)
        else:
            未中.append(要点)
    return len(命中), len(要点们), 命中, 未中


def 打分(题, 答案, 上下文, 拒答):
    """纯函数：给定答案 / context / 是否判拒答 → 判据明细。复用与实跑都走这里。"""
    答案 = (答案 or "").strip()

    if 题["库外"]:
        return {"判定": "库外拒答" if 拒答 else "库外未拒（❌）",
                "得分": 100.0 if 拒答 else 0.0,
                "要点率": None, "要点命中": [], "要点未中": [],
                "数值": [], "编造": [], "数值状态": "N/A（库外题）"}

    命中, 总数, 命中列, 未中列 = 判要点(答案, 题["参考要点"])
    要点率 = (命中 / 总数) if 总数 else 1.0
    非误拒 = 0.0 if 拒答 else 1.0

    数值 = 抽数值(答案)
    规范上下文 = 规范(上下文)
    编造 = [v for v in 数值 if v not in 规范上下文]

    if 数值:
        忠实率 = (len(数值) - len(编造)) / len(数值)
        得分 = 70 * 要点率 + 15 * 忠实率 + 15 * 非误拒
        数值状态 = "可判 n=%d" % len(数值)
    else:
        得分 = 85 * 要点率 + 15 * 非误拒
        数值状态 = "无数值（该维度 N/A，权重并给要点）"

    if 拒答:
        判定 = "库内误拒（❌）"
    elif 编造:
        判定 = "有编造（❌）"
    elif 要点率 >= 1.0:
        判定 = "要点全中"
    elif 要点率 > 0:
        判定 = "要点部分命中"
    else:
        判定 = "要点未中（❌）"

    return {"判定": 判定, "得分": 得分, "要点率": 要点率,
            "要点命中": 命中列, "要点未中": 未中列,
            "数值": 数值, "编造": 编造, "数值状态": 数值状态}


def 问一题(问句):
    """POST /ask。返回 (响应字典, 错误说明或 None)。"""
    载荷 = json.dumps({"question": 问句}, ensure_ascii=False).encode("utf-8")
    请求 = urllib.request.Request(服务 + "/ask", data=载荷,
                                  headers={"Content-Type": "application/json"})
    try:
        # 🔴 必须读完整 body 再关 —— 只取头就 GC 会让服务端拿到 RST（见 容忍断连的服务器）
        with urllib.request.urlopen(请求, timeout=超时秒) as 响应:
            return json.loads(响应.read().decode("utf-8")), None
    except urllib.error.HTTPError as 错:
        return None, "HTTP %s" % 错.code
    except Exception as 错:                                     # noqa: BLE001
        return None, "%s: %s" % (type(错).__name__, 错)


def 成行(题, 答案, 上下文, 拒答, 拒答类型, 最高分, 库外引用, 引用数, 生耗时, 用时):
    """把一次问答结果整理成一行明细（含复核备注）。"""
    行 = {
        "编号": 题["编号"], "类别": 题["类别"], "文档": 题["文档"],
        "问": 题["问"], "库外": 题["库外"], "参考要点": 题["参考要点"],
        "复核备注": 题.get("复核备注", ""),
        "答案": 答案, "context": 上下文, "耗时": round(用时, 2),
        "失败": False, "错误": None,
        "最高分": 最高分, "拒答": 拒答, "拒答类型": 拒答类型,
        "库外引用": 库外引用, "引用数": 引用数, "答案字数": len(答案),
        "生成耗时": 生耗时,
    }
    行.update(打分(题, 答案, 上下文, 拒答))
    return 行


def 复用行(题, 旧):
    """用上次的答案 + context 重算分（不碰服务）。旧行必须带 context。"""
    行 = dict(旧)
    行["参考要点"] = 题["参考要点"]
    行["复核备注"] = 题.get("复核备注", "")
    行.update(打分(题, 旧.get("答案", ""), 旧.get("context", ""), bool(旧.get("拒答"))))
    return 行


def 跑一题(题, 无重试=False, 缓存=None):
    if 缓存 and 题["编号"] in 缓存:
        旧 = 缓存[题["编号"]]
        if old_ok(旧):
            return 复用行(题, 旧)

    尝试 = 1 if 无重试 else 最多尝试
    响应, 错误, 用时 = None, None, 0.0
    for i in range(1, 尝试 + 1):
        起 = time.time()
        响应, 错误 = 问一题(题["问"])
        用时 = time.time() - 起
        if 响应 is not None:
            应答 = (响应.get("answer") or "")
            if not 应答.startswith("(生成失败") and not 响应.get("error"):
                break
            if i < 尝试:
                print("        ↻ 第 %d 次失败（%s），重试 …"
                      % (i, 应答[:40].replace("\n", " ") or 错误), flush=True)
                time.sleep(2)
        elif i < 尝试:
            print("        ↻ 第 %d 次失败（%s），重试 …" % (i, 错误), flush=True)
            time.sleep(2)

    行 = {
        "编号": 题["编号"], "类别": 题["类别"], "文档": 题["文档"],
        "问": 题["问"], "库外": 题["库外"], "参考要点": 题["参考要点"],
        "复核备注": 题.get("复核备注", ""),
        "答案": "", "context": "", "耗时": round(用时, 2), "失败": False, "错误": 错误,
        "最高分": None, "拒答": False, "拒答类型": None,
        "库外引用": [], "引用数": 0, "答案字数": 0,
    }
    if 响应 is None:
        行.update({"失败": True, "失败原因": 错误 or "无响应"})
        return 行

    答案 = (响应.get("answer") or "").strip()
    if 答案.startswith("(生成失败") or 响应.get("error"):
        行.update({"答案": 答案, "失败": True,
                   "失败原因": (答案[:120] or 响应.get("error"))})
        return 行

    return 成行(题, 答案, 响应.get("context") or "", bool(响应.get("refused")),
               响应.get("refusal_type"), 响应.get("top_score"),
               响应.get("off_library_refs") or [], len(响应.get("citations") or []),
               (响应.get("timings") or {}).get("generate"), 用时)


def old_ok(旧):
    return (not 旧.get("失败")) and ("context" in 旧) and 旧.get("context") is not None


# ---------------------------------------------------------------------------
# 汇总 / 报表
# ---------------------------------------------------------------------------

def 汇总(明细):
    有效 = [x for x in 明细 if not x["失败"]]
    库内 = [x for x in 有效 if not x["库外"]]
    库外 = [x for x in 有效 if x["库外"]]
    总要点 = sum(len(x["参考要点"]) for x in 库内)
    中要点 = sum(len(x["要点命中"]) for x in 库内)
    数值总 = sum(len(x["数值"]) for x in 库内)
    编造总 = sum(len(x["编造"]) for x in 库内)
    有数值题 = sum(1 for x in 库内 if x["数值"])
    库内误拒 = [x for x in 库内 if x["拒答"]]
    库外拒答 = sum(1 for x in 库外 if x["拒答"])
    失败 = [x for x in 明细 if x["失败"]]
    return {
        "题数": len(明细), "有效": len(有效), "失败": len(失败),
        "失败编号": [x["编号"] for x in 失败],
        "平均分": (sum(x["得分"] for x in 有效) / len(有效)) if 有效 else None,
        "库内数": len(库内), "库外数": len(库外),
        "要点总数": 总要点, "要点命中数": 中要点,
        "要点命中率": (中要点 / 总要点) if 总要点 else None,
        "数值总数": 数值总, "编造数": 编造总,
        "编造率": (编造总 / 数值总) if 数值总 else None,
        "有数值题数": 有数值题,
        "库内误拒数": len(库内误拒), "库内误拒编号": [x["编号"] for x in 库内误拒],
        "库外拒答数": 库外拒答,
        "全中题数": sum(1 for x in 库内 if x["判定"] == "要点全中"),
        "满分题数": sum(1 for x in 有效 if x["得分"] >= 99.99),
        "平均字数": (sum(x["答案字数"] for x in 有效) / len(有效)) if 有效 else 0,
         "平均生成秒": (sum((x.get("生成耗时") or 0) for x in 有效) / len(有效)) if 有效 else 0,
    }


def 分组(明细, 键):
    出 = {}
    for x in 明细:
        if x["失败"]:
            continue
        出.setdefault(x[键], []).append(x)
    return 出


def 转义MD(文本, 截断=None):
    t = str(文本).replace("|", "\\|").replace("\n", " ")
    if 截断 and len(t) > 截断:
        t = t[:截断] + "…"
    return t


def 写MD(明细, 汇, 元):
    有备注 = any(x.get("复核备注") for x in 明细)
    行 = []
    行.append("# 企业知识库问答 · 100 题实测评分表")
    行.append("")
    行.append("> 生成时间：%s　｜　服务：%s　｜　检索链：%s"
             % (元["生成时间"], 元["服务"], 元["检索链"]))
    行.append("> 题库：`eval/问答100题.json`（%d 题 = 库内 %d + 库外 %d）　｜　"
               "总用时 %.1f s（平均生成 %.1f s/题）"
             % (汇["题数"], 汇["库内数"], 汇["库外数"], 元["用时秒"], 汇["平均生成秒"]))
    行.append("")
    行.append("## 一、总览")
    行.append("")
    行.append("| 指标 | 值 |")
    行.append("|---|---|")
    行.append("| 有效题数 / 总题数 | %d / %d%s |"
             % (汇["有效"], 汇["题数"],
                "（⚠️ %s 生成失败已剔除分母）" % "、".join(汇["失败编号"]) if 汇["失败"] else ""))
    行.append("| **平均得分** | **%.1f / 100** |" % (汇["平均分"] or 0))
    行.append("| 满分题（≥99.99） | %d / %d |" % (汇["满分题数"], 汇["有效"]))
    行.append("| 要点命中率（库内）| %d / %d = **%.1f%%** |"
             % (汇["要点命中数"], 汇["要点总数"], 100 * (汇["要点命中率"] or 0)))
    行.append("| 要点全中的题 | %d / %d |" % (汇["全中题数"], 汇["库内数"]))
    行.append("| 数值编造率 | %d / %d = **%.1f%%**（%d 题答案含数值） |"
             % (汇["编造数"], 汇["数值总数"], 100 * (汇["编造率"] or 0), 汇["有数值题数"]))
    行.append("| 库内误拒 | %d / %d%s |"
             % (汇["库内误拒数"], 汇["库内数"],
                "：" + "、".join(汇["库内误拒编号"]) if 汇["库内误拒编号"] else ""))
    行.append("| 库外正确答案 | %d / %d |" % (汇["库外拒答数"], 汇["库外数"]))
    行.append("| 平均答案字数 | %.0f 字 |" % 汇["平均字数"])
    行.append("")

    行.append("## 二、按类别")
    行.append("")
    行.append("| 类别 | 题数 | 平均分 | 要点命中 |")
    行.append("|---|---:|---:|---:|")
    for 类, 组 in 分组(明细, "类别").items():
        库内组 = [x for x in 组 if not x["库外"]]
        总 = sum(len(x["参考要点"]) for x in 库内组)
        中 = sum(len(x["要点命中"]) for x in 库内组)
        行.append("| %s | %d | %.1f | %s |"
                 % (类, len(组), sum(x["得分"] for x in 组) / len(组),
                    ("%d/%d" % (中, 总)) if 总 else "—"))
    行.append("")

    行.append("## 三、按文档")
    行.append("")
    行.append("| 文档 | 题数 | 平均分 | 要点命中 |")
    行.append("|---|---:|---:|---:|")
    for 文, 组 in 分组(明细, "文档").items():
        库内组 = [x for x in 组 if not x["库外"]]
        总 = sum(len(x["参考要点"]) for x in 库内组)
        中 = sum(len(x["要点命中"]) for x in 库内组)
        行.append("| %s | %d | %.1f | %s |"
                 % (文, len(组), sum(x["得分"] for x in 组) / len(组),
                    ("%d/%d" % (中, 总)) if 总 else "—"))
    行.append("")

    行.append("## 四、逐题明细（100 题）")
    行.append("")
    行.append("得分口径：库内 = 70×要点率 + 15×数值忠实 + 15×未误拒（答案无数值时，"
               "忠实度的 15 分并入要点，即 85×要点率 + 15×未误拒）；库外 = 正确拒答 100 / 未拒答 0。")
    行.append("")
    表头 = "| # | 编号 | 类别 | 文档 | 问题 | 得分 | 判定 | 要点 | 数值 | 答案 |"
    分隔 = "|---:|---|---|---|---|---:|---|---|---|---|"
    if 有备注:
        表头 = 表头[:-2] + " 人工复核 |"
        分隔 = 分隔[:-2] + "---|"
    行.append(表头)
    行.append(分隔)
    for i, x in enumerate(明细, start=1):
        尾 = ("| " + 转义MD(x.get("复核备注"), 90)) if 有备注 else ""
        if x["失败"]:
            行.append("| %d | %s | %s | %s | %s | — | ⚠️ 生成失败 | — | — | %s %s |"
                     % (i, x["编号"], x["类别"], x["文档"], 转义MD(x["问"]),
                        转义MD(x["失败原因"]), 尾))
            continue
        要点 = "—" if x["库外"] else "%d/%d" % (len(x["要点命中"]), len(x["参考要点"]))
        数值 = "—" if x["库外"] else "%d / 编造 %d" % (len(x["数值"]), len(x["编造"]))
        行.append("| %d | %s | %s | %s | %s | %.1f | %s | %s | %s | %s %s |"
                 % (i, x["编号"], x["类别"], x["文档"], 转义MD(x["问"]), x["得分"],
                    x["判定"], 要点, 数值, 转义MD(x["答案"], 60), 尾))
    行.append("")
    行.append("## 五、口径与局限（必读）")
    行.append("")
    for 段 in 元["口径说明"]:
        行.append("- " + 段)
    with open(表文件MD, "w", encoding="utf-8") as 文:
        文.write("\n".join(行) + "\n")


def 颜色(分):
    if 分 >= 85:
        return "#0a7d3c"
    if 分 >= 70:
        return "#a06b00"
    return "#b3261e"


HTML模板 = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>企业知识库问答 · 100 题实测评分表</title>
<style>
 :root{--bg:#f6f7f9;--fg:#1f2328;--mut:#5b6570;--line:#dfe3e8;--card:#fff;--acc:#1f6feb;}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:14px/1.65 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;}
 .wrap{max-width:1500px;margin:0 auto;padding:28px 20px 60px}
 h1{font-size:24px;margin:0 0 6px}
 h2{font-size:17px;margin:34px 0 12px;padding-left:10px;border-left:4px solid var(--acc)}
 .meta{color:var(--mut);font-size:13px;margin-bottom:18px}
 .mono{font-family:ui-monospace,Consolas,monospace}
 .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:16px 0 4px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
 .card .k{color:var(--mut);font-size:12px}
 .card .v{font-size:24px;font-weight:700;margin:2px 0}
 .card .s{color:var(--mut);font-size:12px}
 table{width:100%;border-collapse:collapse;background:var(--card);
       border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:13px}
 th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
 th{background:#eef1f5;font-weight:600;white-space:nowrap;position:sticky;top:0}
 tr:last-child td{border-bottom:none}
 tr:hover td{background:#fafbfc}
 td.q{min-width:210px}
 td.a{color:#334;min-width:280px}
 td.rm{color:#7a5b00;min-width:200px}
 tr.fail td{background:#fff2f0}
 .note{background:#fffbe6;border:1px solid #ffe58f;border-radius:10px;padding:14px 18px;margin-top:14px}
 .note li{margin:4px 0}
 .scroll{max-height:820px;overflow:auto;border-radius:10px}
</style></head><body><div class="wrap">
<h1>企业知识库问答 · 100 题实测评分表</h1>
<div class="meta">生成时间 $gen_time　｜　服务 <span class="mono">$service</span>　｜　检索链 <span class="mono">$chain</span><br>
题库 <span class="mono">eval/问答100题.json</span>（$n_total 题 = 库内 $n_in + 库外 $n_out）　｜　$elapsed</div>

<div class="cards">
 $c1
 $c2
 $c3
 $c4
 $c5
 $c6
</div>

<h2>按文档分组</h2>
<div class="scroll"><table><thead><tr><th>文档</th><th>题数</th><th>平均分</th><th>要点命中</th></tr></thead>
<tbody>$doc_rows</tbody></table></div>

<h2>按类别分组</h2>
<table><thead><tr><th>类别</th><th>题数</th><th>平均分</th><th>要点命中</th></tr></thead>
<tbody>$cat_rows</tbody></table>

<h2>逐题明细（100 题）</h2>
<div class="scroll"><table><thead><tr>
<th>#</th><th>编号</th><th>类别</th><th>文档</th><th>问题</th><th>得分</th><th>判定</th>
<th>要点</th><th>数值</th><th>答案</th>$remark_head</tr></thead><tbody>$detail_rows</tbody></table></div>

<div class="note"><b>口径与局限（必读）</b><ul>$notes</ul></div>
</div></body></html>
"""


def 写HTML(明细, 汇, 元):
    # 🔴 用 `string.Template`（只认 ASCII 占位名）而不是 `%` 格式化 ——
    #    HTML/CSS 里到处是字面 `%`（width:100%），`%` 格式化会当场炸在 `%;` 上；
    #    而 `str.format` 又会撞上 CSS 的花括号。占位名必须 ASCII，中文名 Template 不认。
    tpl = _Template(HTML模板)
    e = html.escape
    有备注 = any(x.get("复核备注") for x in 明细)

    def 卡片(名, 值, 副=""):
        return ('<div class="card"><div class="k">%s</div><div class="v">%s</div>'
                '<div class="s">%s</div></div>' % (e(名), e(值), e(副)))

    def 组行(组):
        库内组 = [x for x in 组 if not x["库外"]]
        总 = sum(len(x["参考要点"]) for x in 库内组)
        中 = sum(len(x["要点命中"]) for x in 库内组)
        return ("<tr><td>%s</td><td>%d</td><td>%.1f</td><td>%s</td></tr>"
                % (e(组[0].get("_名", "")), len(组),
                   sum(x["得分"] for x in 组) / len(组),
                   ("%d/%d" % (中, 总)) if 总 else "—"))

    行 = []
    for 文, 组 in 分组(明细, "文档").items():
        for x in 组:
            x["_名"] = 文
        行.append(组行(组))
    类别行 = []
    for 类, 组 in 分组(明细, "类别").items():
        for x in 组:
            x["_名"] = 类
        类别行.append(组行(组))

    明细行 = []
    for i, x in enumerate(明细, start=1):
        尾 = ('<td class="rm">%s</td>' % e(x.get("复核备注", "") or "")) if 有备注 else ""
        if x["失败"]:
            明细行.append('<tr class="fail"><td>%d</td><td>%s</td><td>%s</td><td>%s</td>'
                          '<td class="q">%s</td><td>—</td><td>⚠️ 生成失败</td>'
                          '<td>—</td><td>—</td><td class="a">%s</td>%s</tr>'
                          % (i, e(x["编号"]), e(x["类别"]), e(x["文档"]), e(x["问"]),
                             e(x["失败原因"] or ""), 尾))
            continue
        要点 = "—" if x["库外"] else "%d/%d" % (len(x["要点命中"]), len(x["参考要点"]))
        数值 = "—" if x["库外"] else "%d / 编造 %d" % (len(x["数值"]), len(x["编造"]))
        明细行.append('<tr><td>%d</td><td>%s</td><td>%s</td><td>%s</td>'
                      '<td class="q">%s</td>'
                      '<td style="font-weight:700;color:%s">%.1f</td>'
                      '<td>%s</td><td>%s</td><td>%s</td><td class="a">%s</td>%s</tr>'
                      % (i, e(x["编号"]), e(x["类别"]), e(x["文档"]), e(x["问"]),
                         颜色(x["得分"]), x["得分"], e(x["判定"]), 要点, 数值,
                         e(x["答案"]), 尾))

    口径 = "".join("<li>%s</li>" % e(p) for p in 元["口径说明"])

    with open(表文件HTML, "w", encoding="utf-8") as 文:
        文.write(tpl.substitute(
            gen_time=e(元["生成时间"]), service=e(元["服务"]), chain=e(元["检索链"]),
            n_total=汇["题数"], n_in=汇["库内数"], n_out=汇["库外数"],
            elapsed="总用时 %.1f s（平均生成 %.1f s/题）" % (元["用时秒"], 汇["平均生成秒"]),
            c1=卡片("平均得分", "%.1f" % (汇["平均分"] or 0), "/ 100"),
            c2=卡片("有效题数", "%d / %d" % (汇["有效"], 汇["题数"]),
                    ("⚠️ %d 题生成失败已剔除分母" % len(汇["失败"])) if 汇["失败"] else "无失败"),
            c3=卡片("要点命中率", "%.1f%%" % (100 * (汇["要点命中率"] or 0)),
                    "%d / %d" % (汇["要点命中数"], 汇["要点总数"])),
            c4=卡片("数值编造率", "%.1f%%" % (100 * (汇["编造率"] or 0)),
                    "%d / %d（%d 题答案含数值）"
                    % (汇["编造数"], 汇["数值总数"], 汇["有数值题数"])),
            c5=卡片("库内误拒", "%d / %d" % (汇["库内误拒数"], 汇["库内数"]),
                    "、".join(汇["库内误拒编号"]) if 汇["库内误拒编号"] else "无误拒"),
            c6=卡片("库外正确答案", "%d / %d" % (汇["库外拒答数"], 汇["库外数"]), "拒答判定"),
            doc_rows="".join(行), cat_rows="".join(类别行),
            remark_head="<th>人工复核</th>" if 有备注 else "",
            detail_rows="".join(明细行), notes=口径))


# ---------------------------------------------------------------------------

def 主流程():
    global 服务
    sys.stdout.reconfigure(errors="replace")

    if "--端口" in sys.argv:
        服务 = "http://127.0.0.1:%s" % sys.argv[sys.argv.index("--端口") + 1]
    无重试 = "--无重试" in sys.argv
    复用 = "--复用" in sys.argv
    限量 = None
    if "--限" in sys.argv:
        限量 = int(sys.argv[sys.argv.index("--限") + 1])

    健康 = {}
    if 复用 and os.path.exists(结果文件):
        print("· --复用：沿用 %s 里的答案与 context，只重算分（不访问服务）"
              % os.path.relpath(结果文件, 根目录))
    else:
        try:
            with urllib.request.urlopen(服务 + "/health", timeout=8) as r:
                健康 = json.loads(r.read().decode("utf-8"))
        except Exception as 错:                                 # noqa: BLE001
            print("✗ 知识库服务不可达（%s）：%s" % (服务, 错))
            return 1

    缓存 = {}
    复用源 = {}
    if 复用 and os.path.exists(结果文件):
        旧数据 = json.load(open(结果文件, encoding="utf-8"))
        for x in 旧数据.get("明细", []):
            if old_ok(x):
                缓存[x["编号"]] = x
        复用源 = 旧数据.get("元", {})
        健康 = {"chain": 复用源.get("检索链"), "score_threshold": 复用源.get("阈值"),
                "class2_soft": 复用源.get("类2软")}
        if 限量:
            缓存 = {k: v for k, v in list(缓存.items())[:限量]}

    题集 = json.load(open(题库文件, encoding="utf-8"))["题目"]
    if 限量:
        题集 = 题集[:限量]

    print("=" * 100)
    print(" 作品一 · 知识库问答 · %d 题实测评分%s" % (len(题集), "（复用重算）" if 缓存 else ""))
    print(" 服务 %s ｜ 链 %s ｜ 阈值 %s ｜ 类2 %s"
          % (服务, 健康.get("chain"), 健康.get("score_threshold"),
             "软警示" if 健康.get("class2_soft") else "硬拒答"))
    print("=" * 100)

    明细 = []
    起 = time.time()
    for i, 题 in enumerate(题集, start=1):
        行 = 跑一题(题, 无重试, 缓存)
        明细.append(行)
        if 行["失败"]:
            打印 = "  [%3d/%d] %s %-6s ⚠️ 生成失败：%s" % (
                i, len(题集), 行["编号"], 行["类别"], (行.get("失败原因") or "")[:52])
        elif 行["库外"]:
            打印 = "  [%3d/%d] %s %-6s 得分 %5.1f  %s ｜ 最高分 %s ｜ %s" % (
                i, len(题集), 行["编号"], 行["类别"], 行["得分"], 行["判定"],
                行["最高分"], 行["答案"][:36].replace("\n", " "))
        else:
            打印 = ("  [%3d/%d] %s %-6s 得分 %5.1f  %s ｜ 要点 %d/%d ｜ 数值 %d/编造 %d ｜ %s"
                   % (i, len(题集), 行["编号"], 行["类别"], 行["得分"], 行["判定"],
                      len(行["要点命中"]), len(行["参考要点"]),
                      len(行["数值"]), len(行["编造"]),
                      行["答案"][:34].replace("\n", " ")))
        print(打印, flush=True)

    用时 = time.time() - 起
    汇 = 汇总(明细)
    # 🔴 `--复用` 模式下标量耗时应是**答案采集时**的值，不是本次重算的 0.0 s ——
    #    否则报告上写「总用时 0.0 s」，看的人会以为压根没跑过。
    #    专门存一个 `采集用时秒` 并向后传递：链式复用（复用再复用）不会把它冲成 0。
    采集用时 = (round(用时, 1) if not 缓存
                else (复用源.get("采集用时秒") or 复用源.get("用时秒") or None))
    if not 采集用时:
        采集用时 = round(sum((x.get("生成耗时") or 0) for x in 明细), 1)
    元 = {
        "生成时间": 复用源.get("生成时间") if 缓存 else time.strftime("%Y-%m-%d %H:%M:%S"),
        "评分时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "服务": 服务,
        "检索链": 健康.get("chain") or "—",
        "阈值": 健康.get("score_threshold"),
        "类2软": 健康.get("class2_soft"),
        "用时秒": 采集用时,
        "采集用时秒": 采集用时,
        "复用": bool(缓存),
        "口径说明": [
            "本题库为**全新拟题**，没有金标准合取（与 `eval/题目.json` 的检索实验题不同），"
            "因此**不判「意思对不对」**（那要人工逐题读），只判三件可自动判定的事。",
            "**要点命中**：答案里是否出现本该出现的关键事实（题库 `参考要点`，"
            "已用护栏校验每条都能在语料中逐字查到）。命中 ≠ 答对；但**不命中基本等于没答到**。",
            "**数值无编造**：答案里的数值（百分比 / 文档编号 / 带单位的量，如 °C、μm、h）"
            "必须逐字出现在服务返回的 context 里。分母是**数值个数**不是题数 —— "
            "「答案一个数值都没有」记 N/A、不记满分（否则判据分不清「通过」和「没检查」）。",
            "**拒答正确**：库外题必须拒答；库内题不得被误拒。",
            "评分口径：库内题 = 70×要点率 + 15×数值忠实 + 15×未误拒（答案无数值时，"
            "忠实度的 15 分并入要点，即 85×要点率 + 15×未误拒）；库外题 = 拒答 100 / 未拒 0。",
            "生成失败（超时 / 服务不可达）**不计入分母**并在表中标注；"
            "把失败当答错会把平均分冲成假的低值。",
            "**平均分不是「答对率」** —— 它是上面三条自动判据的加权，用于横向比较与回归监控，"
            "不代表「这个百分比的答案就是对的」。",
            "**自动判据查不出的两类问题**：① 答案自相矛盾（如首词与后文相反）；"
            "② 数值「碰巧」与 context 一致但语义错位。这两类靠「人工复核」列与抽样精读。",
        ] + ([
            "🔁 本表由 `--复用` 生成：**答案与 context 取自 %s 的那次实跑**，本次只重算评分"
            "（改了要点定义 / 判据口径后无需再打一遍服务）。" % (复用源.get("生成时间") or "上一次"),
        ] if 缓存 else []),
    }

    with open(结果文件, "w", encoding="utf-8") as 文:
        json.dump({"元": 元, "汇总": 汇, "明细": 明细}, 文, ensure_ascii=False, indent=2)
    写MD(明细, 汇, 元)
    写HTML(明细, 汇, 元)

    print()
    print("=" * 100)
    print(" 汇总（答案采集用时 %.1f s%s）"
          % (元["用时秒"], "；本次为 --复用 重算" if 缓存 else ""))
    print("=" * 100)
    print("  有效题数      ：%d / %d%s"
          % (汇["有效"], 汇["题数"],
             "（⚠️ %s 失败已剔除）" % "、".join(汇["失败编号"]) if 汇["失败"] else ""))
    print("  平均得分      ：%.1f / 100   （满分题 %d）" % (汇["平均分"] or 0, 汇["满分题数"]))
    print("  要点命中率    ：%d / %d = %.1f%%"
          % (汇["要点命中数"], 汇["要点总数"], 100 * (汇["要点命中率"] or 0)))
    print("  要点全中题    ：%d / %d" % (汇["全中题数"], 汇["库内数"]))
    print("  数值编造率    ：%d / %d = %.1f%%（%d 题答案含数值）"
          % (汇["编造数"], 汇["数值总数"], 100 * (汇["编造率"] or 0), 汇["有数值题数"]))
    print("  库内误拒      ：%d / %d%s"
          % (汇["库内误拒数"], 汇["库内数"],
             "（%s）" % "、".join(汇["库内误拒编号"]) if 汇["库内误拒编号"] else ""))
    print("  库外正确答案  ：%d / %d" % (汇["库外拒答数"], 汇["库外数"]))
    print("  平均答案字数  ：%.0f 字　｜　平均生成 %.1f s/题"
          % (汇["平均字数"], 汇["平均生成秒"]))
    print()
    print("  已落盘：")
    for p in (结果文件, 表文件MD, 表文件HTML):
        print("    %s" % os.path.relpath(p, 根目录))
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(主流程())
