# -*- coding: utf-8 -*-
"""ChatBI · 只读护栏（包 2）

定位：数据库前面的**最后一道门**。它只认事实（语句本身），不认来源（谁生成的）。
      语义层渲染出的 SQL 目前一定合法 —— 但护栏的价值恰恰在于**不依赖上游正确**。
      「上游是对的，所以不用检查」不是判据：上游改错了、模型越界了、将来接进
      Agent 编排层让模型自己拼片段了，护栏必须能独立挡住。所以本模块**不 import 语义层**。

三道独立的锁（任意一道单独生效都足以挡住写操作）：
  ① 连接锁 —— `file:<路径>?mode=ro`，SQLite 在**文件层**拒绝写
  ② 语句锁 —— 白名单：只放行单条 SELECT / WITH；写、DDL、会话控制关键字一律拒绝
  ③ 会话锁 —— `PRAGMA query_only = ON` + 进度回调超时
     ③ 兜的不是「写」，而是「合法但跑不完」：一条没写错的 SELECT 也能靠笛卡尔积
     把服务拖死（请求线程占着不返回、后续请求全排队）——那种表现看着像服务挂了。

设计上的两个反直觉点（都写在注释里了，别当可有可无）：
  · **必须先净化再查关键字**（去注释、去字面量）。否则 `WHERE 备注 = '待处理 drop 记录'`
    会被判成写操作 —— 一次误杀就是一次静默故障，和漏判一样糟糕。
  · **`REPLACE` 不能一刀切禁**：`REPLACE INTO` 是写操作，`REPLACE(字段,'旧','新')`
    是标准标量函数。判据要细到「REPLACE + INTO」。
"""

import re
import sqlite3
import time

# 结果行数上限：与 eval/指标口径定义-v1.0.yaml 的「护栏.结果」一致
结果行数上限 = 200
默认超时秒 = 5

# 写操作 / DDL / 会话控制 —— 首次词命中即拒
# 注：`BEGIN/COMMIT/ROLLBACK/SAVEPOINT` 在只读会话里本就无意义，一并拒掉，
#     免得「BEGIN; ...; COMMIT」这种形态绕过分号检查。
禁词 = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH", "DETACH",
    "PRAGMA", "VACUUM", "TRIGGER", "GRANT", "REINDEX", "ANALYZE",
    "BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE",
)

_REPLACE写 = re.compile(r"(?is)\bREPLACE\s+INTO\b")
_首词 = re.compile(r"(?is)^[（(\s]*([A-Za-z]+)")
_表引用 = re.compile(r"(?is)\b(?:FROM|JOIN)\s+([A-Za-z_\u4e00-\u9fff][\w\u4e00-\u9fff]*)")
_CTE名 = re.compile(r"(?is)\b([A-Za-z_\u4e00-\u9fff][\w\u4e00-\u9fff]*)\s+AS\s*\(")


# ============================ 语句净化 ============================

def 去噪(语句):
    """把注释与字符串/标识符字面量整体替换成**等长空格**。

    为什么必须净化后再查关键字（这是实测过的坑，不是洁癖）：
      不净化就会把「字面量里的文本」当语句。
      典型误杀：`WHERE 备注 = '本月要 delete 这批'`、`-- 注释里写了 drop table`。
      误杀与漏判都是静默故障 —— 都会让人以为「服务坏了」或「问题答不出来」。

    等长替换的好处：报错时可以直接拿净化结果与原文对齐定位第几个字符。
    """
    出 = []
    i, n = 0, len(语句)
    while i < n:
        字 = 语句[i]
        # -- 行注释
        if 字 == "-" and i + 1 < n and 语句[i + 1] == "-":
            while i < n and 语句[i] != "\n":
                出.append(" ")
                i += 1
            continue
        # /* 块注释 */
        if 字 == "/" and i + 1 < n and 语句[i + 1] == "*":
            出.append("  ")
            i += 2
            while i < n and not (语句[i] == "*" and i + 1 < n and 语句[i + 1] == "/"):
                出.append(" ")
                i += 1
            if i < n:
                出.append("  ")
                i += 2
            continue
        # '字符串' / "标识符" / `标识符`（SQL 里成对引号 = 转义）
        if 字 in ("'", '"', "`"):
            闭 = 字
            出.append(" ")
            i += 1
            while i < n:
                if 语句[i] == 闭:
                    if i + 1 < n and 语句[i + 1] == 闭:
                        出.append("  ")
                        i += 2
                        continue
                    出.append(" ")
                    i += 1
                    break
                出.append(" ")
                i += 1
            continue
        # [方括号标识符]（T-SQL 风格，SQLite 也认）
        if 字 == "[":
            出.append(" ")
            i += 1
            while i < n and 语句[i] != "]":
                出.append(" ")
                i += 1
            if i < n:
                出.append(" ")
                i += 1
            continue
        出.append(字)
        i += 1
    return "".join(出)


# ============================ 语句白名单 ============================

def 抽表名(净语句):
    """抽出语句引用的表名（CTE 名不算表）。"""
    CTE = set(_CTE名.findall(净语句))
        # ⚠️ 这个正则也会命中 `xxx AS (` 形态的普通别名，但那种写法在本库不存在；
        #    多剔除几个名字只会让白名单**更宽**（漏检方向），不会误杀。
    return [x for x in _表引用.findall(净语句) if x not in CTE]


def 检查(语句, 已知表=None):
    """返回 (是否通过, 原因)。已知表 为 None 时跳过表白名单检查。"""
    if not 语句 or not 语句.strip():
        return False, "空语句"

    净 = 去噪(语句)

    # ① 单条语句：净化后仍出现分号 → 一定是语句分隔符（字面量里的分号已被清空）
    段 = [x.strip() for x in 净.split(";")]
    有效 = [x for x in 段 if x]
    if len(有效) > 1:
        return False, "检到 %d 段语句 —— 只允许单条 SELECT" % len(有效)
    if not 有效:
        return False, "空语句"
    净 = 有效[0]

    # ② 首词必须是 SELECT / WITH
    首 = _首词.match(净)
    词 = 首.group(1).upper() if 首 else ""
    if 词 not in ("SELECT", "WITH"):
        return False, ("只允许 SELECT / WITH 开头的查询，实际以 %r 开头"
                       % (首.group(1) if 首 else 净[:12]))

    # ③ 关键字黑名单
    命中 = [k for k in 禁词 if re.search(r"\b%s\b" % k, 净, re.I)]
    if 命中:
        return False, "含被拒关键字：%s（只读护栏只放行查询）" % "、".join(命中)
    if _REPLACE写.search(净):
        return False, "含被拒关键字：REPLACE INTO（写操作）"

    # ④ 表白名单
    if 已知表 is not None:
        越界 = sorted({x for x in 抽表名(净) if x not in 已知表})
        if 越界:
            return False, ("引用了白名单外的表：%s（白名单：%s）"
                           % ("、".join(越界), "、".join(sorted(已知表))))
    return True, "通过"


# ============================ 只读连接 ============================

def _转义URI(路径):
    """URI 形式只对 `?` 与 `#` 敏感；Windows 反斜杠不需要转（SQLite 自己认）。"""
    return str(路径).replace("?", "%3f").replace("#", "%23")


def 连库(路径, 超时秒=默认超时秒):
    """只读连接。`mode=ro` 是文件层只读，再叠一层 `query_only` 兜住会话层。"""
    连接 = sqlite3.connect("file:%s?mode=ro" % _转义URI(路径), uri=True, timeout=超时秒)
    连接.execute("PRAGMA query_only = ON")
    return 连接


def 读表名(连接):
    """白名单直接来自真实库 —— 不硬编码表清单，避免「库里加了表，护栏还不知道」。"""
    return {r[0] for r in 连接.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}


def 执行(连接, 语句, 上限=结果行数上限, 超时秒=默认超时秒):
    """执行一条已过护栏的 SELECT，返回 {列, 行, 截断, 用时秒}。

    超时用进度回调实现：SQLite 每执行约 5000 条 VM 指令回调一次，
    返回非 0 即中断查询并抛 `interrupted`。
    宁可中断也不让请求线程无限期占着 —— 那种「服务活着但不应答」最难排查。
    """
    起点 = time.time()
    连接.set_progress_handler(
        lambda: 1 if (time.time() - 起点) > 超时秒 else 0, 5000)
    try:
        游标 = 连接.cursor()
        游标.execute(语句)
        列 = [d[0] for d in (游标.description or [])]
        行 = 游标.fetchmany(上限 + 1)          # 多取一行，用来判断「是否被截断」
        截断 = len(行) > 上限
        行 = [tuple(_可序列化(x) for x in r) for r in 行[:上限]]
        return {"列": 列, "行": 行, "截断": 截断, "用时秒": round(time.time() - 起点, 3)}
    finally:
        连接.set_progress_handler(None, 0)


def _可序列化(值):
    """BLOB → 可读字符串。库里现在没有 BLOB，但结果要进 JSON，不能在这里崩。"""
    if isinstance(值, (bytes, bytearray)):
        return bytes(值).decode("utf-8", "replace")
    return 值


# ============================ 自检样例 ============================
# 「必拒」与「必放」两侧都要有样例 —— 只测拒会把「一律拒绝」当成通过。

必拒 = [
    ("写操作-删表", "DROP TABLE 供应商"),
    ("写操作-更新", "UPDATE 物料 SET 安全库存 = 0"),
    ("写操作-插入", "INSERT INTO 供应商 VALUES ('X')"),
    ("写操作-删除行", "DELETE FROM 到料检验"),
    ("多语句拼接", "SELECT 1; DROP TABLE 物料"),
    ("多语句换行", "SELECT 1;\nDELETE FROM 库存"),
    ("PRAGMA 越权", "PRAGMA table_info(供应商)"),
    ("ATTACH 外挂", "ATTACH DATABASE 'x.db' AS x"),
    ("非查询开头", "EXPLAIN SELECT * FROM 供应商"),
    ("REPLACE INTO", "REPLACE INTO 供应商 VALUES ('X')"),
    ("白名单外表", "SELECT * FROM sqlite_master"),
    ("空语句", "   "),
]

必放 = [
    ("普通查询", "SELECT COUNT(*) FROM 供应商"),
    ("带字面量的 delete 字样", "SELECT * FROM 到料检验 WHERE 主要不良原因 = '待 delete'"),
    ("行注释里夹带写操作", "SELECT 1 -- ; DROP TABLE 物料"),
    ("块注释里夹带写操作", "SELECT 1 /* DROP TABLE 物料 */"),
    ("REPLACE 标量函数", "SELECT REPLACE(供应商名称,'公司','') FROM 供应商"),
    ("CTE 多跳", "WITH 主供 AS (SELECT 物料编码 FROM 采购订单 GROUP BY 物料编码) "
                 "SELECT * FROM 主供 JOIN 物料 m ON m.物料编码 = 主供.物料编码"),
    ("末尾分号", "SELECT 1;"),
    ("带注释", "SELECT 1 -- 说明\n"),
    ("窗口函数行号", "SELECT ROW_NUMBER() OVER (PARTITION BY 物料编码 ORDER BY 订单金额 DESC) FROM 采购订单"),
]
