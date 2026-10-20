# -*- coding: utf-8 -*-
"""ChatBI · 结果一句话解读（A-2）—— 表与图之外的第三样东西。

为什么要有
----------
外部对标（Vanna 2.0 输出五段，含自然语言摘要；WrenAI 出洞察摘要；北极九章做可解释卡片）
一致的共识：**结果不能只有表和数字**。人看 12 行表格要自己找结论，一句解读把那步省掉。

🔴 本模块唯一致命的风险是「编」—— 所以防编造是它的主要设计，不是附属功能
    · 提示词只喂**已经查出来的**数字，并明令：不许算新数字（环比/均值/占比/差额都不许）、
      不许推测原因（数据里没有原因字段）。
    · **出口还有一道确定性检查**：解读里出现的每个数字，必须在
      「结果集数值 ∪ 行数 ∪ 口径/时间/指标文本里的数字」里找得到；
      找不到就**整条丢弃**并写明理由。
      理由（为什么宁可不说）：一句含凭空数字的解读比没有解读危害大得多 ——
      它长得跟真结论一模一样，读者不会去核。这与「拒答优于硬答」是同一条纪律。

🔴 解读失败**绝不拖垮答案**：模型不可达 / 超时 → 结果照常返回，只是没有解读。
   数据已经查到了，不能因为写不出一句话就不给。

开关：
    · 请求体 `{"解读": false}` → 本次不要解读
    · 环境变量 `CHATBI_NO_INSIGHT=1` → 全局关（评测 / 自检用，省时间也省不确定性）
"""

import argparse
import os
import re
import sys

本目录 = os.path.dirname(os.path.abspath(__file__))
项目目录 = os.path.dirname(本目录)
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)

import 意图解析                                          # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:                                    # noqa: BLE001
        pass

关解读 = bool(os.environ.get("CHATBI_NO_INSIGHT"))
解读超时秒 = 60
最多喂行 = 20                      # 喂太多行，模型会开始「总结趋势」而不是复述事实

_数字 = re.compile(r"\d+(?:\.\d+)?")
# 只像小标题、不像结论的行（`## 解读` / `结论：`）。收拾输出时要整行丢掉 ——
# 见 解读() 里那段注释：留着它会把正文挤掉。
_纯标签 = re.compile(r"^\s*(?:一句话)?\s*(?:解读|结论|摘要|分析|总结|答案)"
                     r"\s*[:：]?\s*$", re.I)


def _表(列, 行, 最多=最多喂行):
    if not 列:
        return "（没有列）"
    线 = []
    线.append(" | ".join(str(c) for c in 列))
    线.append("-" * 24)
    for r in 行[:最多]:
        线.append(" | ".join("" if v is None else str(v) for v in r))
    if len(行) > 最多:
        线.append("…（只给了前 %d 行，共 %d 行）" % (最多, len(行)))
    return "\n".join(线)


_提示 = """你在为一条**已经查出来的**数据结果写一句中文解读。你不是在分析，是在**转述**。

【硬规则，违反即作废】
1. 只允许使用下面给出的数字。**不许计算任何新数字** —— 环比、同比、均值、占比、差额、
   倍数、增长率，一个都不许算。要比较就只用上面已经存在的两个数并列说出来。
2. 不许推测原因（结果里没有原因字段）。不许断言数据里没有的事。
3. 一句话，40 个汉字以内，直接给结论（最大是谁 / 最小是谁 / 集中在哪 / 有几项）。
4. 只有 1 行 1 列时，把那个数字连同单位说清楚。
5. 结果为空（0 行）时，只说「没有匹配到记录」，不要猜为什么。
6. 只输出这一句话本身。不要 Markdown、不要小标题、不要解释你的思路。

【这条问句】%s
【口径】%s
【时间范围】%s
【对象】%s　【单位】%s
【列】%s
【数据】共 %d 行%s
%s

一句话解读："""


def 构造提示(问题, 应):
    列 = 应.get("列") or []
    行 = 应.get("行") or []
    截 = "（结果超上限已截断，下面只反映显示出来的部分）" if 应.get("截断") else ""
    return _提示 % (问题 or "",
                  应.get("口径") or "（无）",
                  应.get("时间") or "（不限）",
                  应.get("指标") or "（未标注）",
                  应.get("单位") or "（无）",
                  "、".join(str(c) for c in 列) or "（无）",
                  len(行), 截, _表(列, 行))


# ============================ 防编造（确定性出口检查） ============================

def 允许数字(应):
    """解读里**允许**出现的数字集合。

    来源限定为「结果本身携带的数字」：结果集单元格、行数、口径/时间/指标文本。
    **刻意不含 SQL** —— SQL 里有 LIMIT / 日期常量，放进来等于把检查放空。
    """
    集 = set()

    def 收(文本):
        for m in _数字.findall(str(文本)):
            try:
                集.add(round(float(m), 2))
            except ValueError:
                pass

    for r in (应.get("行") or []):
        for v in r:
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                集.add(round(float(v), 2))
            else:
                收(v)
    收(应.get("行数") if 应.get("行数") is not None else len(应.get("行") or []))
    收(应.get("口径"))
    收(应.get("时间"))
    收(应.get("指标"))
    收(应.get("单位"))
    for c in (应.get("列") or []):
        收(c)
    收(len(应.get("行") or []))
    return 集


def 查编造(文本, 应):
    """解读里有没有结果集里不存在的数字。返回 (是否可信, 可疑数字列表)。"""
    许 = 允许数字(应)
    可疑 = []
    for m in _数字.findall(str(文本 or "")):
        try:
            值 = round(float(m), 2)
        except ValueError:
            continue
        if not any(abs(值 - x) <= 0.011 for x in 许):
            可疑.append(m)
    return (not 可疑), 可疑


# ============================ 主入口 ============================

def 解读(问题, 应, 调=意图解析.调模型, 超时=解读超时秒):
    """给结果写一句话。返回 {文本, 用时秒, 编造?, 可疑数字, 错误?}。

    **永不抛异常** —— 解读是锦上添花，不能因为它把答案搞没。
    """
    起 = None
    try:
        import time as _t
        起 = _t.time()
        文本, 用时, _ = 调(构造提示(问题, 应), 超时=超时, 最多token=120)
        文本 = (文本 or "").strip()
        # 模型偶尔会带 Markdown 小标题 / 换行 / 项目符号 —— 收拾成一句话。
        # 🔴 只做「去掉开头的 # 再取第一行」是**错的**，实测过：
        #        `## 解读\n东莞精诚 7.66% 最高。` → 剩「解读」两个字，正文整条丢掉
        #    结果是一条语法正常、内容为零的解读，而且它照样通过防编造检查
        #    （没数字可言），界面上看着完全正常。所以先剥记号，再**丢掉只像标签的行**。
        文本 = re.sub(r"^\s*(?:#{1,6}\s*|[-*>]\s+)+", "", 文本)
        行们 = [x.strip() for x in 文本.splitlines()]
        行们 = [x for x in 行们 if x and not _纯标签.match(x)]
        文本 = (行们[0] if 行们 else "").strip("“”\"'`* ")
        if not 文本:
            return {"文本": None, "错误": "模型没有产出内容", "用时秒": round(_t.time() - 起, 2)}
        好, 可疑 = 查编造(文本, 应)
        if not 好:
            # 宁可不说，不可编 —— 含凭空数字的解读比没有解读危险得多。
            return {"文本": None, "编造": True, "可疑数字": 可疑,
                    "原文本": 文本,
                    "错误": "解读里出现了结果集里没有的数字 %s —— 已丢弃（宁可不说，不可编）"
                            % "、".join(可疑[:3]),
                    "用时秒": round(_t.time() - 起, 2)}
        return {"文本": 文本, "用时秒": round(_t.time() - 起, 2)}
    except Exception as 错:                                    # noqa: BLE001
        return {"文本": None, "用时秒": (round(__import__("time").time() - 起, 2) if 起 else None),
                "错误": "%s: %s" % (type(错).__name__, str(错)[:80])}


# ============================ CLI ============================

def 主():
    sys.stdout.reconfigure(errors="replace")
    参 = argparse.ArgumentParser(description="ChatBI 结果一句话解读（A-2）")
    参.add_argument("--问题", required=True)
    参.add_argument("--列", required=True, help="逗号分隔列名")
    参.add_argument("--行", default="", help="JSON 二维数组")
    参.add_argument("--口径", default="")
    参.add_argument("--时间", default="")
    参.add_argument("--指标", default="")
    参.add_argument("--单位", default="")
    参.add_argument("--看提示词", action="store_true")
    参数 = 参.parse_args()

    import json
    应 = {"列": [x for x in 参数.列.split(",") if x],
         "行": json.loads(参数.行) if 参数.行 else [],
         "口径": 参数.口径, "时间": 参数.时间, "指标": 参数.指标, "单位": 参数.单位}
    应["行数"] = len(应["行"])
    if 参数.看提示词:
        print(构造提示(参数.问题, 应))
        return 0
    果 = 解读(参数.问题, 应)
    print(json.dumps(果, ensure_ascii=False, indent=2))
    return 0 if 果.get("文本") else 1


if __name__ == "__main__":
    raise SystemExit(主())
