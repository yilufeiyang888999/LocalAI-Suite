# -*- coding: utf-8 -*-
"""ChatBI 语义层 v1.0 · 渲染器

职责边界（很重要）：
    模型：自然语言 → **结构化意图**（指标 + 维度 + 过滤 + 时间 + 排序）
    本模块：结构化意图 → **SQL**（口径、时间区间、别名、表前缀全部由定义决定）
    → 模型不碰 SQL 语法，也不碰口径。这是包 0 摸底结论确立的分工。

确定性：同样的意图 → 同样的 SQL。可复现、可单测、可 diff。

用法（CLI）：
    python scripts/语义层.py --时间展开
    python scripts/语义层.py --指标 来料不良率 --维度 供应商 --时间 指定月份 --时间值 2026年8月 --限制 1
    python scripts/语义层.py --意图 '{"指标":"来料不良率","维度":["供应商"],"时间":"上月","限制":1}'
    python scripts/语义层.py --意图 '{"基础查询":"分组","目标":"不良原因分布","时间":["指定月份","2026年8月"]}'
    python scripts/语义层.py --场景 C11_主要供应商不良率超标 --参数 '{"阈值":5,"时间":"2026年8月"}'
"""

import argparse
import json
import os
import re
import sys
from datetime import date, timedelta

import yaml

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

本目录 = os.path.dirname(os.path.abspath(__file__))
项目目录 = os.path.dirname(本目录)
定义文件 = os.path.join(项目目录, "eval", "指标口径定义-v1.0.yaml")


# ============================ 载入 ============================

def 载入(路径=None):
    路径 = 路径 or 定义文件
    with open(路径, encoding="utf-8") as 句柄:
        return yaml.safe_load(句柄)


def 库路径(定义=None):
    """演示库的绝对路径。**路径只有这一处定义**，调用方不要再自己拼。

    YAML 里把「相对谁」也写清楚了（`数据源.项目根`），因为 ChatBI/ 与
    学习计划目录是**兄弟目录**，靠目录层级反推会数错层 —— 这是一个已经踩过的坑：
    三个脚本各自硬编码绝对路径，搬一次目录就要改三处。
    """
    定义 = 定义 or 载入()
    路径 = str(定义["数据源"]["路径"])
    if os.path.isabs(路径):
        return 路径
    根说明 = 定义["数据源"].get("项目根")
    基 = os.path.normpath(os.path.join(项目目录, 根说明)) if 根说明 else os.path.dirname(项目目录)
    return os.path.normpath(os.path.join(基, 路径.replace("/", os.sep)))


# ============================ 时间 ============================

def _加月(首日, 偏移):
    """以「月首日」为基准加减月数。偏移可负。"""
    总 = 首日.year * 12 + (首日.month - 1) + 偏移
    return date(总 // 12, 总 % 12 + 1, 1)


def 解析指定月(值):
    """'2026年8月' / '2026-08' / '2026/8' → (2026, 8)"""
    m = re.match(r"^\s*(\d{4})\s*[年\-/]\s*(\d{1,2})\s*月?\s*$", str(值))
    if not m:
        raise ValueError("无法解析指定月份：%r（支持 2026年8月 / 2026-08 / 2026/8）" % 值)
    return int(m.group(1)), int(m.group(2))


def 展开时间(定义, 名):
    """按时间锚点展开为 (起, 止) 的 'YYYY-MM-DD'；'全部' 返回 None。"""
    规范 = 定义["时间表达式"]["窗口定义"]
    当前 = 名
    已见 = set()
    while 当前 in 规范 and "同" in 规范[当前]:
        if 当前 in 已见:
            raise ValueError("时间表达式循环引用：%s" % 当前)
        已见.add(当前)
        当前 = 规范[当前]["同"]
    if 当前 not in 规范:
        raise KeyError("未知时间表达式：%s" % 名)
    规 = 规范[当前]
    类型 = 规["类型"]
    锚 = date.fromisoformat(str(定义["时间锚点"]["值"]))

    if 类型 == "全区间":
        return None
    if 类型 == "月窗口":
        return (str(_加月(锚.replace(day=1), 规["起"])),
                str(_加月(锚.replace(day=1), 规["止"])))
    if 类型 == "季窗口":
        季首 = date(锚.year, ((锚.month - 1) // 3) * 3 + 1, 1)
        return (str(_加月(季首, 3 * 规["起"])), str(_加月(季首, 3 * 规["止"])))
    if 类型 == "年窗口":
        return (str(date(锚.year + 规["起"], 1, 1)), str(date(锚.year + 规["止"], 1, 1)))
    if 类型 == "日窗口":
        return (str(锚 + timedelta(days=规["起"])), str(锚 + timedelta(days=规["止"])))
    raise ValueError("未知窗口类型：%s" % 类型)


def 时间区间(定义, 时间意图):
    """时间意图 = '上月' | ['上月'] | ['指定月份', '2026年8月'] | {'表达式':..., '值':...}
    返回 (起, 止) 或 None（全区间）。"""
    表达式, 值 = None, None
    if isinstance(时间意图, str):
        表达式 = 时间意图
    elif isinstance(时间意图, (list, tuple)):
        表达式 = 时间意图[0]
        值 = 时间意图[1] if len(时间意图) > 1 else None
    elif isinstance(时间意图, dict):
        表达式 = 时间意图.get("表达式")
        值 = 时间意图.get("值")
    else:
        raise ValueError("无法识别的时间意图：%r" % (时间意图,))

    if 表达式 in ("指定月份", "指定日期"):
        年, 月 = 解析指定月(值)
        起 = date(年, 月, 1)
        return (str(起), str(_加月(起, 1)))
    return 展开时间(定义, 表达式)


def 时间谓词(定义, 字段, 时间意图):
    """返回 'X >= 起 AND X < 止'；全区间返回 None。"""
    区间 = 时间区间(定义, 时间意图)
    if 区间 is None:
        return None
    return "%s >= '%s' AND %s < '%s'" % (字段, 区间[0], 字段, 区间[1])


# ---------------------------- 表结构（唯一事实来源是库本身）----------------------------

_列缓存 = {}


def 表列(表):
    """某张表的列名集合。**直接问库**（PRAGMA），不手抄一份字段清单。

    🔴 为什么不去 YAML 里再写一份「每张表有哪些列」：手抄的第二份必然过期，
       而过期**不报错** —— 只是某天某个过滤条件悄悄执行不了（§9.5 唯一事实来源）。
       查不到库时返回 None，**调用方必须把 None 当「不知道」处理**，不能当「没有这列」。
    """
    if 表 in _列缓存:
        return _列缓存[表]
    try:
        import sqlite3
        连接 = sqlite3.connect("file:%s?mode=ro" % 库路径().replace("?", "%3f"), uri=True)
        try:
            列 = {r[1] for r in 连接.execute("PRAGMA table_info(%s)" % 表).fetchall()}
        finally:
            连接.close()
    except Exception:                                    # noqa: BLE001
        return None                                      # 失败不入缓存
    _列缓存[表] = 列
    return 列


def 基础查询表(定义, 意图):
    """基础查询的目标 → (表名, 别名)。别名用于拼过滤谓词（与渲染时的 FROM 一致）。"""
    类, 目标 = 意图.get("基础查询"), 意图.get("目标")
    基 = 定义["基础查询"]
    if 类 == "计数":
        return (基["计数"][目标]["表"], None)
    if 类 == "枚举":
        return (基["枚举字段"][目标]["表"], None)
    if 类 == "求和":
        return (基["原子度量"][目标]["表"], 基["原子度量"][目标].get("别名"))
    if 类 == "分组":
        for 规 in 基["分组聚合"]:
            if 规["名称"] == 目标:
                return (规["表"], 规.get("别名"))
    return (None, None)


def 基础查询过滤条件(定义, 意图):
    """基础查询的 WHERE 片段（不含时间）。返回 (条件串 or "", 执行不了的字段列表)。

    🔴 为什么原来干脆不支持（2026-10-17 改）：原实现**不读「过滤」**，所以意图解析
       一见到过滤就阻断（"基础查询没有过滤槽位"）。实测代价：
         · T47「原料仓的库存总量是多少？」—— 一个完全能答的问题被拒答；
         · M06「三个仓各存了多少货？」—— 模型编了个「仓1」的过滤，同样被拒。
       现在改为：**字段确实是该表的列**就翻译成谓词执行；不是就交回去阻断（不猜）。
    """
    表, 别名 = 基础查询表(定义, 意图)
    if not 表:
        return ("", [])
    列 = 表列(表)
    if 列 is None:
        # 拿不到表结构 → 一律当「执行不了」，让上层阻断（宁可答不出，不可静默丢条件）
        return ("", [x.get("字段") for x in (意图.get("过滤") or [])])
    条, 坏 = [], []
    for 条0 in (意图.get("过滤") or []):
        字 = str(条0.get("字段") or "").strip()
        if 字 not in 列:
            坏.append(字)
            continue
        表达式 = "%s.%s" % (别名, 字) if 别名 else 字
        条.append(过滤谓词(表达式, 条0))
    return (" AND ".join(条), 坏)



# ============================ 值/谓词 ============================

def 引号(值):
    if isinstance(值, bool):
        return "1" if 值 else "0"
    if isinstance(值, (int, float)):
        return repr(值)
    return "'%s'" % str(值).replace("'", "''")


def 过滤谓词(表达式, 条件):
    运 = 条件.get("运算符", 条件.get("运", "="))
    # 🔴 中文运算符先归一：意图解析产出的运算符是中文词，而这里拼的是 SQL。
    #    漏掉这一步的后果是**拼出语法错的 SQL**（`字段 精确 '值'`）——
    #    实测过：筛选条件一用精确匹配就 OperationalError，而错误信息指向 SQLite，
    #    看起来像「数据库有问题」，其实是这一层少认了一个词。
    运 = {"精确": "=", "等于": "=", "是": "=", "==": "="}.get(运, 运)
    值 = 条件["值"]
    if 运 in ("模糊", "包含", "like", "LIKE"):
        return "%s LIKE '%%%s%%'" % (表达式, str(值).replace("'", "''"))
    if 运 in ("属于", "in", "IN"):
        项 = 值 if isinstance(值, (list, tuple)) else [值]
        return "%s IN (%s)" % (表达式, ", ".join(引号(x) for x in 项))
    if 运 in ("范围", "between", "BETWEEN"):
        return "%s BETWEEN %s AND %s" % (表达式, 引号(值[0]), 引号(值[1]))
    if 运 in ("为空", "is null", "IS NULL"):
        return "%s IS NULL" % 表达式
    if 运 in ("非空", "is not null", "IS NOT NULL"):
        return "%s IS NOT NULL" % 表达式
    return "%s %s %s" % (表达式, 运, 引号(值))


# ============================ 指标渲染 ============================

def 找指标(定义, 名):
    for 指 in 定义["指标"]:
        if 指["名称"] == 名 or 名 in (指.get("别名") or []):
            return 指
    raise KeyError("语义层未定义指标：%s（已定义：%s）"
                   % (名, "、".join(x["名称"] for x in 定义["指标"])))


def 渲染指标(定义, 意图):
    名 = 意图.get("指标") or 意图.get("名称")
    指 = 找指标(定义, 名)

    维度名 = list(意图.get("维度") or [])
    需要连接 = []
    选择片段, 分组片段 = [], []
    for 维 in 维度名:
        if 维 not in 指["维度"]:
            raise KeyError("指标「%s」不开放维度「%s」（可用：%s）"
                           % (指["名称"], 维, "、".join(指["维度"])))
        规 = 指["维度"][维]
        需要连接 += list(规.get("连接") or [])
        选择片段.append(规["选择"])
        分组片段.append(规["分组"])

    度量 = 指["度量"]
    表达式 = 度量["表达式"]
    精度 = 度量.get("精度")
    度量列 = ("ROUND(%s, %d) AS %s" % (表达式, 精度, 度量["别名"])
              if 精度 is not None else "%s AS %s" % (表达式, 度量["别名"]))
    选择片段.append(度量列)

    条件 = []
    for 条 in (意图.get("过滤") or []):
        字 = 条["字段"]
        if 字 not in 指["过滤字段"]:
            raise KeyError("指标「%s」不支持过滤字段「%s」（可用：%s）"
                           % (指["名称"], 字, "、".join(指["过滤字段"])))
        规 = 指["过滤字段"][字]
        需要连接 += list(规.get("连接") or [])
        条件.append(过滤谓词(规["表达式"], 条))

    时间说明 = "不限（全区间）"
    if 意图.get("时间"):
        t = 时间谓词(定义, 指["时间字段"], 意图["时间"])
        if t:
            条件.append(t)
            区间 = 时间区间(定义, 意图["时间"])
            时间说明 = "%s ~ %s（字段 %s）" % (区间[0], 区间[1], 指["时间字段"])

    连接子句 = [指["连接"][c] for c in 指["连接顺序"]
                if c in 需要连接 or c in (指.get("必需连接") or [])]

    having = 指.get("聚合后过滤") if 意图.get("聚合后过滤") else None

    # 排序：意图给了就用意图的；没给时——若请求了「月份」维度，按月份升序。
    # 🔴 这条不是美化，是修一个实测过的错：问「近6个月不良率趋势」时模型没给排序槽位，
    #    落回指标默认排序（按度量降序）→ 时间序列被打乱成 09/08/07/04/06/05，
    #    趋势完全读不出来。见 YAML「护栏.呈现」。
    排 = 意图.get("排序")
    if not 排:
        月维 = next((v for v in 维度名 if v in ("月份", "日期")), None)
        排 = {"按": 月维, "方向": "升序"} if 月维 else 指["默认排序"]
    方向词 = "DESC" if 排.get("方向", "降序") == "降序" else "ASC"
    排按 = 排.get("按", "度量")
    if 排按 == "度量":
        排序子句 = "ORDER BY %s %s" % (度量["别名"], 方向词)
    else:
        # 「按维度」= 按意图里第一个维度排序（最常用：按月份看趋势）
        排维 = 维度名[0] if 排按 in ("维度", "第一维度") else 排按
        if not 维度名:
            raise KeyError("意图未给维度，无法按维度排序")
        if 排维 not in 指["维度"]:
            raise KeyError("排序维度「%s」不在本指标维度内（可用：%s）"
                           % (排维, "、".join(指["维度"])))
        排序子句 = "ORDER BY %s %s" % (指["维度"][排维]["分组"], 方向词)

    限制 = int(意图.get("限制") or 0)

    行 = ["SELECT " + ", ".join(选择片段),
          "FROM %s %s" % (指["主表"]["表"], 指["主表"]["别名"])]
    行 += 连接子句
    if 条件:
        行.append("WHERE " + "\n  AND ".join(条件))
    if 分组片段:
        行.append("GROUP BY " + ", ".join(分组片段))
    if having:
        行.append("HAVING " + having)
    行.append(排序子句)
    if 限制:
        行.append("LIMIT %d" % 限制)

    return {"SQL": "\n".join(行), "指标": 指["名称"], "维度": 维度名,
            "时间": 时间说明, "口径": 度量["口径说明"], "单位": 度量.get("单位"),
            "护栏": list(指.get("护栏") or [])}


# ============================ 基础查询渲染 ============================

def 渲染基础查询(定义, 意图):
    类 = 意图.get("基础查询")
    目标 = 意图.get("目标")
    基础 = 定义["基础查询"]

    额外, 坏 = 基础查询过滤条件(定义, 意图)
    if 坏:
        # 校验归一 正常会先拦住；走到这里说明绕过了它 —— **不静默丢条件**。
        raise KeyError("基础查询的过滤字段不在目标表里：%s" % "、".join(x for x in 坏 if x))

    def 条件(时间字段=None):
        片 = []
        if 意图.get("时间") and 时间字段:
            片段 = 时间谓词(定义, 时间字段, 意图["时间"])
            if 片段:
                片.append(片段)
        if 额外:
            片.append(额外)
        return ("\nWHERE " + " AND ".join(片)) if 片 else ""

    if 类 == "计数":
        if 目标 not in 基础["计数"]:
            raise KeyError("未定义计数目标：%s" % 目标)
        规 = 基础["计数"][目标]
        return "SELECT COUNT(*) AS %s数\nFROM %s%s" % (目标, 规["表"], 条件(规.get("时间字段")))

    if 类 == "枚举":
        if 目标 not in 基础["枚举字段"]:
            raise KeyError("未定义枚举目标：%s" % 目标)
        if 额外:
            # 枚举渲染本来就只是「列出全部取值」，加过滤会变成「列出部分取值」——
            # 那不是用户问的东西。交回上层（校验归一 会阻断）。
            raise KeyError("枚举查询不支持过滤条件")
        规 = 基础["枚举字段"][目标]
        return ("SELECT DISTINCT %s\nFROM %s\nORDER BY %s"
                % (规["字段"], 规["表"], 规["字段"]))

    if 类 == "求和":
        if 目标 not in 基础["原子度量"]:
            raise KeyError("未定义原子度量：%s" % 目标)
        规 = 基础["原子度量"][目标]
        if 意图.get("时间") and not 规.get("时间字段"):
            raise KeyError("原子度量「%s」未声明时间字段，不能按时间过滤" % 目标)
        return ("SELECT ROUND(SUM(%s), %d) AS %s合计\nFROM %s %s%s"
                % (规["字段"], 规.get("精度", 2), 目标, 规["表"], 规["别名"],
                   条件(规.get("时间字段"))))

    if 类 == "分组":
        命中 = [x for x in 基础["分组聚合"] if x["名称"] == 目标]
        if not 命中:
            raise KeyError("未定义分组聚合：%s（已定义：%s）"
                           % (目标, "、".join(x["名称"] for x in 基础["分组聚合"])))
        规 = 命中[0]
        方向 = "DESC" if 规.get("排序", "降序") == "降序" else "ASC"
        别名 = 规.get("别名")
        return ("SELECT %s AS %s, %s AS %s\nFROM %s %s%s\nGROUP BY %s\nORDER BY %s %s"
                % (规["分组字段"], 规["分组字段"].split(".")[-1],
                   规["聚合"], 规["聚合别名"], 规["表"], 别名,
                   条件(规.get("时间字段")),
                   规["分组字段"], 规["聚合别名"], 方向))

    raise KeyError("未知基础查询类型：%s" % 类)



# ============================ 场景模板渲染 ============================

def 渲染场景(定义, 名称, 参数=None):
    参数 = dict(参数 or {})
    场 = (定义["关联口径"].get("场景模板") or {}).get(名称)
    if not 场:
        raise KeyError("未定义场景模板：%s（已定义：%s）"
                       % (名称, "、".join(定义["关联口径"].get("场景模板", {}))))
    sql = 场["模板"]
    if "{时间}" in sql:
        # 🔴 三种「没有时间条件」的输入都要当成全区间，不能只判 None：
        #    · 参数里根本没有「时间」键（问句没提时间）→ 值 = None
        #    · 模型给了空串 → ""
        #    · 模型给了「全部」→ 展开成全区间，结果同样是 None
        #    漏判的后果实测过两种：`replace(x, None)` 直接 TypeError（500），
        #    或把空串当时间表达式 → KeyError: 未知时间表达式。都是**服务级崩溃**，
        #    而这两条题（C12 / C15）在包 2、包 3 的自检里都测不到 ——
        #    它们的自检覆盖的是「指标渲染」和「用户问句」，没有覆盖「场景 + 空时间参数」。
        值 = 参数.pop("时间", None)
        区间 = None
        if 值 not in (None, "", "全部"):
            try:
                解析指定月(值)
                时间意图 = ["指定月份", 值]
            except ValueError:
                时间意图 = 值
            区间 = 时间谓词(定义, "到料日期", 时间意图)
        if 区间:
            sql = sql.replace("{时间}", 区间)
        else:
            sql = sql.replace("WHERE {时间}", "").replace("{时间}", "")
    for 键, 值 in 参数.items():
        sql = sql.replace("{%s}" % 键, str(值))
    剩余 = re.findall(r"\{(\w+)\}", sql)
    if 剩余:
        raise KeyError("场景「%s」缺少参数：%s" % (名称, "、".join(sorted(set(剩余)))))
    return sql


# ============================ 统一入口 ============================

def 渲染(定义, 意图):
    """意图里带「指标」走指标渲染，带「基础查询」走基础查询，带「场景」走场景。"""
    if 意图.get("场景"):
        return {"SQL": 渲染场景(定义, 意图["场景"], 意图.get("参数")),
                "类型": "场景模板", "场景": 意图["场景"]}
    if 意图.get("基础查询"):
        return {"SQL": 渲染基础查询(定义, 意图),
                "类型": "基础查询", "目标": 意图.get("目标")}
    if 意图.get("指标") or 意图.get("名称"):
        r = 渲染指标(定义, 意图)
        r["类型"] = "指标"
        return r
    raise ValueError("意图必须含「指标」「基础查询」或「场景」之一：%r" % (意图,))


def 时间展开表(定义):
    """把窗口定义按当前锚点全部展开（供人读 / 断言）。"""
    out = {}
    for 名 in 定义["时间表达式"]["窗口定义"]:
        try:
            out[名] = 展开时间(定义, 名)
        except Exception as 错:                                    # noqa: BLE001
            out[名] = "🔴 %s" % 错
    return out


# ============================ CLI ============================

def 主():
    参 = argparse.ArgumentParser(description="ChatBI 语义层渲染器")
    参.add_argument("--时间展开", action="store_true", help="打印当前锚点下所有时间表达式的展开区间")
    参.add_argument("--意图", help="JSON 结构化意图")
    参.add_argument("--指标", help="指标名（配合 --维度/--时间/--时间值/--排序/--限制 使用）")
    参.add_argument("--维度", default="", help="逗号分隔，如 供应商,月份")
    参.add_argument("--时间", default="", help="时间表达式，如 上月、指定月份")
    参.add_argument("--时间值", default="", help="配合 --时间 指定月份时使用，如 2026年8月")
    参.add_argument("--排序方向", default="", help="升序/降序")
    参.add_argument("--限制", type=int, default=0)
    参.add_argument("--场景", help="场景模板名")
    参.add_argument("--参数", default="", help="场景参数的 JSON")
    参数 = 参.parse_args()

    定义 = 载入()

    if 参数.时间展开:
        print("时间锚点：%s" % 定义["时间锚点"]["值"])
        锚 = 定义["时间锚点"]["值"]
        print("%-10s %-24s %s" % ("表达式", "展开区间", "说明"))
        print("-" * 72)
        for 名, 区 in 时间展开表(定义).items():
            if isinstance(区, str):
                print("%-10s %s" % (名, 区))
            elif 区 is None:
                print("%-10s %-24s %s" % (名, "（不限）", "全区间"))
            else:
                print("%-10s %-24s %s" % (名, "%s ~ %s" % 区, ""))
        return 0

    if 参数.场景:
        意图 = {"场景": 参数.场景, "参数": json.loads(参数.参数) if 参数.参数 else {}}
    elif 参数.指标:
        意图 = {"指标": 参数.指标}
        if 参数.维度:
            意图["维度"] = [x.strip() for x in 参数.维度.split(",") if x.strip()]
        if 参数.时间:
            意图["时间"] = ([参数.时间, 参数.时间值] if 参数.时间值 else 参数.时间)
        if 参数.排序方向:
            意图["排序"] = {"按": "度量", "方向": 参数.排序方向}
        if 参数.限制:
            意图["限制"] = 参数.限制
    elif 参数.意图:
        意图 = json.loads(参数.意图)
    else:
        参.print_help()
        return 2

    结果 = 渲染(定义, 意图)
    print(结果["SQL"])
    if "口径" in 结果:
        print()
        print("-- 口径：%s" % 结果["口径"])
        print("-- 时间：%s" % 结果["时间"])
        print("-- 单位：%s" % 结果.get("单位"))
    return 0


if __name__ == "__main__":
    sys.exit(主())
