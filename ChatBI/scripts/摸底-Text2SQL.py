# -*- coding: utf-8 -*-
"""作品二 · 包 0：摸底本地 4B 模型「裸直出 SQL」的水平

测什么：把表结构 + 问题直接丢给 127.0.0.1:8000（Spark-X2.5-4B-Q6_K），
        看它生成的 SQL 执行后能不能和标准答案对上。
为什么测：Text-to-SQL 对模型能力的要求高于 RAG。摸底结论决定架构走
        「模型直出 SQL」还是「模型只出结构化意图 + 语义层模板拼 SQL」。
判据纪律：结论只写实测条数。**不许**用「4B 应该不行」代替测量。

用法：
    python scripts/摸底-Text2SQL.py                 # 全量 20 题
    python scripts/摸底-Text2SQL.py --档 A,B        # 只跑指定档
    python scripts/摸底-Text2SQL.py --限制 1        # 只跑第 1 题（试水）
    python scripts/摸底-Text2SQL.py --只列标准答案  # 不调模型，只校验标准 SQL 能不能跑

注意：动生成端要守分批规则（≤12 次一批，批间验证 8000 真能生成）。
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

本目录 = os.path.dirname(os.path.abspath(__file__))
项目目录 = os.path.dirname(本目录)
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)

import 语义层                                          # noqa: E402

题集文件 = os.path.join(项目目录, "eval", "问数题集-20题.json")
结果文件 = os.path.join(项目目录, "eval", "摸底结果.json")

Demo库 = 语义层.库路径()          # 路径只在定义文件里写一次（曾经这里硬编码过一份）
生成服务 = "http://127.0.0.1:8000/v1/chat/completions"
超时秒 = 180

代理绕过 = urllib.request.build_opener(urllib.request.ProxyHandler({}))

表结构 = """CREATE TABLE 供应商 (供应商编码 TEXT PRIMARY KEY, 供应商名称 TEXT NOT NULL, 供应商类别 TEXT, 所在地 TEXT, 准入状态 TEXT, 供应商评级 TEXT);
CREATE TABLE 物料 (物料编码 TEXT PRIMARY KEY, 物料名称 TEXT NOT NULL, 物料类别 TEXT, 规格型号 TEXT, 单位 TEXT, 安全库存 REAL, 标准单价 REAL);
CREATE TABLE 采购订单 (采购订单号 TEXT PRIMARY KEY, 供应商编码 TEXT NOT NULL, 物料编码 TEXT NOT NULL, 下单日期 TEXT NOT NULL, 约定交期 TEXT NOT NULL, 采购数量 REAL, 采购单价 REAL, 订单金额 REAL, 订单状态 TEXT);
CREATE TABLE 到料检验 (检验单号 TEXT PRIMARY KEY, 采购订单号 TEXT NOT NULL, 物料编码 TEXT NOT NULL, 供应商编码 TEXT NOT NULL, 到料日期 TEXT NOT NULL, 抽检数量 REAL, 不良数量 REAL, 主要不良原因 TEXT, 检验判定 TEXT);
CREATE TABLE 库存 (物料编码 TEXT NOT NULL, 仓库名称 TEXT NOT NULL, 批次号 TEXT, 库存数量 REAL, 更新时间 TEXT, PRIMARY KEY (物料编码, 仓库名称, 批次号));
CREATE TABLE 生产工单 (生产工单号 TEXT PRIMARY KEY, 物料编码 TEXT NOT NULL, 计划数量 REAL, 完工数量 REAL, 计划完工日 TEXT, 实际完工日 TEXT, 工单状态 TEXT);"""

提示模板 = """你是 SQLite 查询助手。下面是一个制造业 ERP 演示库的表结构（列名全中文）：

%s

数据时间范围：2026-04-01 ~ 2026-09-22。

请把用户的问题转成一条 SQLite SQL。

要求：
1. 只输出一条 SELECT 语句，放在 ```sql 代码块里，不要任何解释文字
2. 不要编造数据表或字段
3. 如果问题信息不足、无法确定该怎么查，就不要输出 SQL，而是直接向用户提问需要补充什么（此时不要给代码块）

用户问题：%s"""

反问词 = ("您想", "请问", "指的是", "是指", "哪个环节", "哪家", "哪一种", "补充",
          "具体", "数据来源", "时间范围", "口径", "请说明", "无法确定", "需要明确")


# ======================= 基础工具 =======================

def 连库():
    return sqlite3.connect("file:%s?mode=ro" % Demo库, uri=True)


def 跑SQL(连接, 语句):
    """返回 (列名, 行列表)。异常上抛给调用方。"""
    游标 = 连接.cursor()
    游标.execute(语句)
    列名 = [d[0] for d in (游标.description or [])]
    return 列名, 游标.fetchall()


def 行键(行):
    """把一行压成可比较的字符串元组。浮点保留 2 位 —— 与容差 0.005 等价。"""
    键 = []
    for 格 in 行:
        if 格 is None:
            键.append("None")
        elif isinstance(格, (int, float)):
            键.append("%.2f" % float(格))
        else:
            键.append(str(格).strip())
    return tuple(键)


def 结果一致(基准, 实测):
    """基准/实测 = (列名, 行列表)。忽略列名与行序，比对行集合。"""
    基列, 基行 = 基准
    实列, 实行 = 实测
    if len(基列) != len(实列):
        return False, "列数 %d vs %d" % (len(实列), len(基列))
    if len(基行) != len(实行):
        return False, "行数 %d vs %d" % (len(实行), len(基行))
    if sorted(行键(r) for r in 基行) != sorted(行键(r) for r in 实行):
        return False, "行数相同但取值不同"
    return True, "一致"


def 抽SQL(输出):
    """从模型输出里抽出 SQL。抽不到返回 None。"""
    if not 输出:
        return None
    块 = re.findall(r"```(?:sql|SQL)?\s*(.+?)```", 输出, re.S)
    if 块:
        return 块[0].strip()
    去空 = 输出.strip()
    if re.match(r"^\s*(SELECT|WITH)\b", 去空, re.I):
        return 去空
    return None


def 调模型(问题):
    """调生成端。返回 (文本, 用时秒, 用量)。"""
    体 = {
        "messages": [{"role": "user",
                      "content": 提示模板 % (表结构, 问题)}],
        "max_tokens": 512,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        生成服务, data=json.dumps(体).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    t = time.time()
    r = json.load(代理绕过.open(req, timeout=超时秒))
    用时 = time.time() - t
    return r["choices"][0]["message"].get("content") or "", 用时, r.get("usage")


# ======================= 判定 =======================

def 判一题(连接, 题, 基准缓存):
    """跑一题，返回记录 dict。"""
    记 = {"编号": 题["编号"], "档": 题["档"], "问题": 题["问题"],
          "标准SQL": 题.get("标准SQL")}

    if 题["判据"] == "应反问":
        try:
            文本, 用时, 用量 = 调模型(题["问题"])
        except Exception as 错:                                    # noqa: BLE001
            记.update({"结果": "调用失败", "说明": "生成端异常：%s" % str(错)[:90],
                       "输出": None})
            return 记
        记["输出"] = 文本
        记["用时秒"] = round(用时, 1)
        记["tokens"] = (用量 or {}).get("completion_tokens")
        SQL = 抽SQL(文本)
        if SQL:
            try:
                跑SQL(连接, SQL)
                记.update({"结果": "失败", "说明": "该反问却硬答（给出了可执行 SQL）"})
            except Exception:                                      # noqa: BLE001
                记.update({"结果": "失败", "说明": "硬答且 SQL 执行失败（既没澄清也没查对）"})
            return 记
        if any(词 in 文本 for 词 in 反问词) and len(文本.strip()) >= 10:
            记.update({"结果": "通过", "说明": "正确反问 / 说明，未硬答"})
        else:
            记.update({"结果": "失败", "说明": "既没反问也没给 SQL"})
        return 记

    # ---- 结果集判据 ----
    try:
        文本, 用时, 用量 = 调模型(题["问题"])
    except Exception as 错:                                        # noqa: BLE001
        记.update({"结果": "调用失败", "说明": "生成端异常：%s" % str(错)[:90], "输出": None})
        return 记
    记["输出"] = 文本
    记["用时秒"] = round(用时, 1)
    记["tokens"] = (用量 or {}).get("completion_tokens")

    SQL = 抽SQL(文本)
    记["生成SQL"] = SQL
    if not SQL:
        记.update({"结果": "失败", "说明": "未产出 SQL（模型没给出代码块）"})
        return 记

    try:
        实测 = 跑SQL(连接, SQL)
    except Exception as 错:                                        # noqa: BLE001
        原文 = str(错)
        if "no such table" in 原文 or "no such column" in 原文:
            记.update({"结果": "失败", "说明": "Schema 错：%s" % 原文[:80]})
        else:
            记.update({"结果": "失败", "说明": "语法错：%s" % 原文[:80]})
        return 记

    记["实测行数"] = len(实测[1])
    基准 = 基准缓存.setdefault(题["编号"], 跑SQL(连接, 题["标准SQL"]))
    相同, 原因 = 结果一致(基准, 实测)
    记["基准行数"] = len(基准[1])
    if 相同:
        记.update({"结果": "通过", "说明": "结果集与标准答案一致"})
    elif 原因 == "行数相同但取值不同" and len(实测[0]) == len(基准[0]):
        记.update({"结果": "失败", "说明": "口径错（结构相同、数值不同）"})
    else:
        记.update({"结果": "失败", "说明": "结果不符：%s" % 原因})
    return 记


# ======================= 主流程 =======================

def 主():
    参 = argparse.ArgumentParser()
    参.add_argument("--档", default="", help="只跑指定档，如 A,B")
    参.add_argument("--限制", type=int, default=0, help="只跑前 N 题")
    参.add_argument("--只列标准答案", action="store_true",
                    help="不调模型，只校验题集里的标准 SQL 能否执行")
    参数 = 参.parse_args()

    with open(题集文件, encoding="utf-8") as 句柄:
        题集 = json.load(句柄)
    题目 = 题集["题目"]
    if 参数.档:
        允许 = {x.strip().upper() for x in 参数.档.split(",") if x.strip()}
        题目 = [t for t in 题目 if t["档"] in 允许]
    if 参数.限制:
        题目 = 题目[:参数.限制]

    连接 = 连库()

    if 参数.只列标准答案:
        print("== 校验题集标准 SQL（不调模型）==")
        坏 = 0
        for 题 in 题目:
            if not 题.get("标准SQL"):
                print("  %-6s D 档，无标准 SQL（判据=应反问）" % 题["编号"])
                continue
            try:
                列名, 行 = 跑SQL(连接, 题["标准SQL"])
                首行 = 行[0] if 行 else "（空）"
                print("  %-6s ✅ %2d 列 / %3d 行   首行=%s"
                      % (题["编号"], len(列名), len(行), str(首行)[:66]))
            except Exception as 错:                                # noqa: BLE001
                坏 += 1
                print("  %-6s 🔴 %s: %s" % (题["编号"], type(错).__name__, str(错)[:70]))
        连接.close()
        print("\n标准 SQL 不可用：%d 条" % 坏)
        return 0 if 坏 == 0 else 1

    print("== 摸底：本地 4B 模型裸直出 SQL ==")
    print("模型：Spark-X2.5-4B-Q6_K @ %s" % 生成服务)
    print("题量：%d 条（%s）" % (len(题目), "、".join(sorted({t['档'] for t in 题目}))))
    print()

    基准缓存 = {}
    记录表 = []
    for i, 题 in enumerate(题目, 1):
        记 = 判一题(连接, 题, 基准缓存)
        记录表.append(记)
        标记 = {"通过": "✅", "失败": "🔴", "调用失败": "⚠️ "}[记["结果"]]
        print(" %2d/%d  %-6s [%s] %s" % (i, len(题目), 记["编号"], 记["档"], 记["问题"][:30]))
        print("        %s %s   %.1fs" % (标记, 记["说明"], 记.get("用时秒") or 0))
        if 记.get("生成SQL"):
            print("        生成: %s" % 记["生成SQL"].replace("\n", " ")[:150])
        if 记["结果"] == "失败" and 记["档"] == "D":
            print("        输出: %s" % (记.get("输出") or "").replace("\n", " ")[:110])
        print()

    连接.close()

    # ---- 汇总 ----
    print("=" * 74)
    print("按档统计（分母 = 该档全部题数，不可执行题不从分母剔除）")
    print("=" * 74)
    print("%-6s %-8s %-8s %-8s %s" % ("档", "题数", "通过", "通过率", "主要失败形态"))
    合计通过 = 0
    for 档 in sorted({t["档"] for t in 记录表}):
        子 = [r for r in 记录表 if r["档"] == 档]
        通过 = sum(1 for r in 子 if r["结果"] == "通过")
        合计通过 += 通过
        形态 = {}
        for r in 子:
            if r["结果"] != "通过":
                键 = r["说明"].split("：")[0].split("（")[0][:16]
                形态[键] = 形态.get(键, 0) + 1
        主 = "、".join("%s×%d" % (k, v) for k, v in
                      sorted(形态.items(), key=lambda x: -x[1])[:3]) or "—"
        print("%-6s %-8d %-8d %-8s %s"
              % (档, len(子), 通过, "%.0f%%" % (通过 * 100.0 / len(子)), 主))
    print()
    print("合计：%d/%d 通过（%.1f%%）" % (合计通过, len(记录表),
                                        合计通过 * 100.0 / max(len(记录表), 1)))

    用时总 = sum(r.get("用时秒") or 0 for r in 记录表)
    print("总用时 %.1f s，平均 %.1f s/题" % (用时总, 用时总 / max(len(记录表), 1)))

    # ---- 合并写入：分批跑不能互相覆盖（曾经第二批把第一批冲掉）----
    旧记录 = {}
    if os.path.exists(结果文件):
        try:
            with open(结果文件, encoding="utf-8") as 句柄:
                缓存 = json.load(句柄)
            if 缓存.get("题集") == 题集["版本"]:
                旧记录 = {r["编号"]: r for r in 缓存.get("记录", [])}
        except Exception:                                          # noqa: BLE001
            旧记录 = {}
    合并 = dict(旧记录)
    for r in 记录表:
        合并[r["编号"]] = r
    全部 = [合并[k] for k in sorted(合并)]

    汇总 = {
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "模型": "Spark-X2.5-4B-Q6_K",
        "服务": 生成服务,
        "题集": 题集["版本"],
        "本次批次": {"档": sorted({t["档"] for t in 题目}), "题数": len(记录表),
                     "通过": 合计通过},
        "累计": {"题数": len(全部),
                 "通过": sum(1 for r in 全部 if r["结果"] == "通过")},
        "累计按档": {档: {"题数": len([r for r in 全部 if r["档"] == 档]),
                          "通过": sum(1 for r in 全部
                                      if r["档"] == 档 and r["结果"] == "通过")}
                     for 档 in sorted({t["档"] for t in 全部})},
        "记录": 全部,
    }
    with open(结果文件, "w", encoding="utf-8") as 句柄:
        json.dump(汇总, 句柄, ensure_ascii=False, indent=2)
    print("结果已写入 %s（累计 %d 题 / 通过 %d）"
          % (结果文件, 汇总["累计"]["题数"], 汇总["累计"]["通过"]))
    return 0


if __name__ == "__main__":
    sys.exit(主())
