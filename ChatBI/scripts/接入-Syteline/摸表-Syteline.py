# -*- coding: utf-8 -*-
"""
Syteline 摸表工具（阶段 1）
============================================================================
用途：连上 Syteline 的 SQL Server，把「库里有什么」一次性导出来 ——
      表清单（含近似行数）、字段清单、主键外键、以及按业务对象分组的候选表。

定位：**一次性工具，不属于 ChatBI 零依赖主服务**。
      主服务仍然是「标准库 + sqlite3」，这里为了连 SQL Server 才引入 pyodbc。
      ChatBI/scripts/ 下的其它脚本都不要 import 本文件。

用法：
    # 第一步：不知道库名叫什么，先列库
    python 摸表-Syteline.py --主机 172.16.11.254 --用户 sa

    # 第二步：选定库再摸
    python 摸表-Syteline.py --主机 172.16.11.254 --用户 sa --库 SLX_PROD

    # 密码三种给法（优先级从高到低）：
    #   1) --密码 明文（会留在命令历史里，不建议）
    #   2) 环境变量 SYTELINE_PWD（推荐）
    #   3) 交互输入（最安全）
    #   Windows 认证加 --集成认证，此时不需要 --用户 / 密码

产出（默认落在本目录 out/ 下）：
    表清单.csv / 列清单.csv / 约束清单.csv / 摸表结果.json / 摸表日志.txt

设计约束（都是踩过的坑）：
  · 行数取自 sys.partitions 的元数据，**不 COUNT(*)** —— Syteline 的交易表
    动辄百万行，COUNT(*) 全表扫会把 IO 打满，而摸表只需要量级。
  · 驱动只认 ODBC Driver 17/18。老的 "SQL Server" 驱动连新版会栽在 TLS 上，
    这里**明确失败并提示装驱动**，绝不静默降级 —— 降级必须与正常可区分。
  · 只读到底：本脚本不执行任何写语句，也不建临时表。
============================================================================
"""

import argparse
import csv
import getpass
import json
import os
import sys
from datetime import datetime

# 控制台是 cp936，遇到生僻字别崩；不要改成 utf-8，会把中文打成乱码
sys.stdout.reconfigure(errors="replace")
sys.stderr.reconfigure(errors="replace")

本目录 = os.path.dirname(os.path.abspath(__file__))
输出目录 = os.path.join(本目录, "out")

驱动候选 = [
    "ODBC Driver 18 for SQL Server",
    "ODBC Driver 17 for SQL Server",
]

# 按业务对象归组的关键词。这是**启发式**，只用来帮人快速定位，不作为判据。
# 表名以实际库为准 —— 不同版本、不同客户的 Syteline 表名可能有差异。
业务关键词 = {
    "物料": ["item_mst", "item"],
    "客户": ["custaddr", "cust"],
    "供应商": ["vendaddr", "vend"],
    "销售订单": ["co_mst", "coitem"],
    "采购订单": ["po_mst", "pouitem", "pord"],
    "生产工单": ["job_mst", "jobmatl", "job"],
    "库存事务": ["trnitem", "inv"],
    "应收": ["artran", "armast"],
    "应付": ["aptran", "apmast"],
    "站点": ["site"],
}


class 摸表失败(Exception):
    """预期内的失败（驱动缺失、连不上、库不存在）。与代码缺陷区分开。"""


# ---------------------------------------------------------------- 连接

def 选驱动():
    """只返回 17/18。没有就抛错，不退回老驱动。"""
    import pyodbc

    有 = set(pyodbc.drivers())
    for 名 in 驱动候选:
        if 名 in 有:
            return 名
    现况 = "、".join(sorted(有)) or "（无）"
    raise 摸表失败(
        "没有可用的 ODBC Driver 17/18。当前只看到：%s\n"
        "装一个再来：winget install Microsoft.msodbcsql.18\n"
        "（老的 \"SQL Server\" 驱动是 2000 年那版，连新版 SQL Server 会栽在 TLS 上，不要用）"
        % 现况)


def 建连接(参数):
    import pyodbc

    驱动 = 选驱动()
    段 = [
        "DRIVER={%s}" % 驱动,
        "SERVER=%s,%s" % (参数.主机, 参数.端口),
        "DATABASE=%s" % 参数.库,
        "APP=ChatBI-摸表",
    ]
    if 参数.集成认证:
        段.append("Trusted_Connection=yes")
    else:
        if not 参数.用户:
            raise 摸表失败("非集成认证时必须给 --用户")
        段 += ["UID=%s" % 参数.用户, "PWD=%s" % 取密码(参数)]
    # 自签名证书：Syteline 这套用的是自签，不带 TrustServerCertificate 会栽在 08001
    段 += ["Encrypt=yes", "TrustServerCertificate=yes"]

    串 = ";".join(段)
    try:
        # 登录超时走 pyodbc 的 timeout 参数，**不写进连接串**。
        # 实测（2026-10-21，Driver 18.7）：串里带 "Connection Timeout" 或 "LoginTimeout"
        # 会被驱动回一句 "Invalid connection string attribute (0)" —— 但它不报错、
        # 连接照常走，超时其实没设上。这种「看着设了其实没设」比报错更危险。
        连接 = pyodbc.connect(串, readonly=True, timeout=15)
    except pyodbc.Error as 错:
        # pyodbc 的消息里偶尔会带连接串片段，兜一层把密码抹掉
        消息 = str(错).replace(取密码(参数) or "\0", "***")
        raise 摸表失败("连接失败：%s" % 消息)
    连接.timeout = 180      # 大库的元数据查询也可能慢，别用默认的 0（无限等）
    return 连接, 驱动


def 取密码(参数):
    if 参数.密码:
        return 参数.密码
    环 = os.environ.get("SYTELINE_PWD")
    if 环:
        return 环
    return getpass.getpass("SQL Server 密码（不回显）：")


# ---------------------------------------------------------------- 查询

def 查服务器(连接):
    行 = 连接.cursor().execute(
        "SELECT @@VERSION AS v, DB_NAME() AS db, SUSER_SNAME() AS 登录名, "
        "IS_SRVROLEMEMBER('sysadmin') AS 是管理员").fetchone()
    return {
        "版本": str(行.v).splitlines()[0].strip(),
        "当前库": 行.db,
        "登录名": 行.登录名,
        "是sysadmin": bool(行.是管理员) if 行.是管理员 is not None else None,
    }


def 列库清单(连接):
    游标 = 连接.cursor().execute(
        "SELECT name, state_desc FROM sys.databases "
        "WHERE database_id > 4 ORDER BY name")     # 跳过 master/tempdb/model/msdb
    return [(r[0], r[1]) for r in 游标.fetchall()]


def 查表(连接):
    """表 + 视图 + 近似行数 + 占用。

    两个已踩的坑，改动前先读：
    1) **SQL 里不能出现中文字符串字面量**。`'表'` 经 ODBC 到服务端会变成 `'?'`
       （单字节字符集转换），Python 侧再比对 `== "表"` 永远为假，结果是「0 张表」却退出码 0。
       所以类型标记一律用 ASCII 的 T / V，中文只在 Python 侧映射。
    2) **行数不能和 allocation_units 一起 SUM**。一个 partition 可能对应多个
       allocation_unit（IN_ROW_DATA / LOB_DATA），JOIN 后是笛卡尔放大，
       SUM(p.rows) 会虚高（实测 147 万被算成 231 万）。行数与页数分成两个子查询。
    3) 视图必须一起查：列清单覆盖「所有有列的对象」，只查 USER_TABLE 会让
       2435 个视图全部掉进差集，看着像权限缺失。
    """
    游标 = 连接.cursor().execute("""
        SELECT s.name AS 架构, t.name AS 名称, 'T' AS 类型标记,
               ISNULL(行数.rows_sum, 0) AS 近似行数,
               CAST(ISNULL(页数.pages_sum, 0) * 8.0 / 1024 AS DECIMAL(12,1)) AS 占用MB
        FROM sys.tables t
        JOIN sys.schemas s ON t.schema_id = s.schema_id
        LEFT JOIN (SELECT object_id, SUM(rows) AS rows_sum
                   FROM sys.partitions WHERE index_id IN (0, 1)
                   GROUP BY object_id) 行数 ON 行数.object_id = t.object_id
        LEFT JOIN (SELECT p.object_id, SUM(a.total_pages) AS pages_sum
                   FROM sys.partitions p
                   JOIN sys.allocation_units a ON p.partition_id = a.container_id
                   WHERE p.index_id IN (0, 1)
                   GROUP BY p.object_id) 页数 ON 页数.object_id = t.object_id
        UNION ALL
        SELECT s.name, v.name, 'V', 0, 0
        FROM sys.views v
        JOIN sys.schemas s ON v.schema_id = s.schema_id
        ORDER BY 4 DESC
    """)
    return [
        {"架构": r.架构, "表名": r.名称,
         "对象类型": "表" if r.类型标记 == "T" else "视图",
         "近似行数": int(r.近似行数), "占用MB": float(r.占用MB or 0)}
        for r in 游标.fetchall()
    ]


def 查列(连接):
    游标 = 连接.cursor().execute("""
        SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, ORDINAL_POSITION,
               DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, NUMERIC_PRECISION, NUMERIC_SCALE,
               IS_NULLABLE
        FROM INFORMATION_SCHEMA.COLUMNS
        ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
    """)
    return [
        {"架构": r[0], "表名": r[1], "列名": r[2], "序号": r[3], "类型": r[4],
         "长度": r[5], "精度": r[6], "小数位": r[7], "可空": r[8]}
        for r in 游标.fetchall()
    ]


def 查约束(连接):
    """主键 + 外键。Syteline 很多实现不用物理外键，查出来是空的很正常。"""
    主键 = {}
    for r in 连接.cursor().execute("""
            SELECT tc.TABLE_SCHEMA, tc.TABLE_NAME, kcu.COLUMN_NAME, kcu.ORDINAL_POSITION
            FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
              ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
             AND tc.TABLE_SCHEMA = kcu.TABLE_SCHEMA
            WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
            ORDER BY tc.TABLE_NAME, kcu.ORDINAL_POSITION""").fetchall():
        键 = "%s.%s" % (r[0], r[1])
        主键.setdefault(键, []).append(r[2])

    外键 = []
    for r in 连接.cursor().execute("""
            SELECT fk.name,
                   OBJECT_SCHEMA_NAME(fk.parent_object_id), OBJECT_NAME(fk.parent_object_id),
                   OBJECT_SCHEMA_NAME(fk.referenced_object_id), OBJECT_NAME(fk.referenced_object_id)
            FROM sys.foreign_keys fk""").fetchall():
        外键.append({"约束名": r[0], "从": "%s.%s" % (r[1], r[2]), "到": "%s.%s" % (r[3], r[4])})
    return 主键, 外键


# ---------------------------------------------------------------- 汇总

def 归组候选表(表清单):
    """按关键词把表归到业务对象下。一个表可能命中多组，允许重复。"""
    结果 = {}
    for 组名, 关键词们 in 业务关键词.items():
        命中 = [t for t in 表清单 if any(k in t["表名"].lower() for k in 关键词们)]
        if 命中:
            结果[组名] = [{"表名": t["架构"] + "." + t["表名"],
                           "近似行数": t["近似行数"],
                           "占用MB": t["占用MB"]} for t in 命中]
    return 结果


def 写CSV(路径, 表头, 行们):
    # utf-8-sig：Excel 双击打开不乱码（纯 utf-8 会被当成 GBK 读）
    with open(路径, "w", encoding="utf-8-sig", newline="") as 句柄:
        写 = csv.writer(句柄)
        写.writerow(表头)
        写.writerows(行们)


# ---------------------------------------------------------------- 主流程

def 主流程(参数):
    os.makedirs(输出目录, exist_ok=True)
    日志 = []

    def 记(行):
        日志.append(行)
        print(行)

    记("Syteline 摸表 · %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    连接, 驱动 = 建连接(参数)
    记("驱动：%s" % 驱动)

    服务器 = 查服务器(连接)
    记("服务器：%s" % 服务器["版本"])
    记("当前库：%s　登录名：%s　sysadmin：%s"
       % (服务器["当前库"], 服务器["登录名"],
          "是" if 服务器["是sysadmin"] else "否"))

    # 没指定库 → 只列库清单，到此为止
    if 参数.库.lower() == "master":
        库们 = 列库清单(连接)
        记("")
        记("=== 这台实例上的用户库（选一个，用 --库 重跑） ===")
        for 名, 状态 in 库们:
            记("  %-40s %s" % (名, 状态))
        if not 库们:
            记("  （没有用户库 —— Syteline 装了吗？）")
        连接.close()
        return 0

    全部对象 = 查表(连接)
    if not 全部对象:
        raise 摸表失败("库 %s 里一张表都没有 —— 库名对吗？" % 参数.库)
    # 后续所有「行数 / 空表 / 归组」只对实体表有意义；视图没有行数统计
    表清单 = [x for x in 全部对象 if x["对象类型"] == "表"]
    视图们 = [x for x in 全部对象 if x["对象类型"] == "视图"]
    列清单 = 查列(连接)
    主键, 外键 = 查约束(连接)
    连接.close()

    # ---------- 自检：数字之间必须自洽 ----------
    列数表 = {}
    for c in 列清单:
        列数表.setdefault(c["架构"] + "." + c["表名"], 0)
        列数表[c["架构"] + "." + c["表名"]] += 1
    有列的表 = {t["架构"] + "." + t["表名"] for t in 全部对象}
    缺列的 = sorted(有列的表 - set(列数表))
    多余的 = sorted(set(列数表) - 有列的表)
    if 缺列的:
        raise 摸表失败("有 %d 张表查不到字段（权限不足？）：%s"
                       % (len(缺列的), "、".join(缺列的[:5])))

    # ---------- 落盘 ----------
    写CSV(os.path.join(输出目录, "表清单.csv"),
          ["架构", "表名", "近似行数", "占用MB"],
          [[t["架构"], t["表名"], t["近似行数"], t["占用MB"]] for t in 表清单])
    写CSV(os.path.join(输出目录, "视图清单.csv"),
          ["架构", "视图名"],
          [[v["架构"], v["表名"]] for v in 视图们])
    写CSV(os.path.join(输出目录, "列清单.csv"),
          ["架构", "表名", "列名", "序号", "类型", "长度", "精度", "小数位", "可空"],
          [[c["架构"], c["表名"], c["列名"], c["序号"], c["类型"],
            c["长度"], c["精度"], c["小数位"], c["可空"]] for c in 列清单])
    写CSV(os.path.join(输出目录, "约束清单.csv"),
          ["类型", "对象", "明细"],
          [["主键", k, "、".join(v)] for k, v in sorted(主键.items())]
          + [["外键", f["从"], "→ " + f["到"]] for f in 外键])

    # 建一次「对象 → 列名集合」索引：不建的话下面每张表都要遍历 13 万列
    列索引 = {}
    for c in 列清单:
        列索引.setdefault((c["架构"], c["表名"]), set()).add(c["列名"].lower())
    含站点 = [t for t in 表清单
              if "site_ref" in 列索引.get((t["架构"], t["表名"]), ())]
    架构们 = sorted({t["架构"] for t in 表清单})
    空表 = [t for t in 表清单 if t["近似行数"] == 0]

    结果 = {
        "摸表时间": datetime.now().isoformat(timespec="seconds"),
        "主机": "%s,%s" % (参数.主机, 参数.端口),
        "库": 参数.库,
        "驱动": 驱动,
        "服务器": 服务器,
        "规模": {
            "表数": len(表清单),
            "视图数": len(视图们),
            "字段数": len(列清单),
            "架构": 架构们,
            "近似总行数": sum(t["近似行数"] for t in 表清单),
            "空表数": len(空表),
            "含 site_ref 的表数": len(含站点),
            "主键数": len(主键),
            "外键数": len(外键),
        },
        "候选业务表": 归组候选表(表清单),
        "表清单": 表清单,
        "视图清单": 视图们,
        "列清单": 列清单,
        "主键": 主键,
        "外键": 外键,
    }
    with open(os.path.join(输出目录, "摸表结果.json"), "w", encoding="utf-8") as 句柄:
        json.dump(结果, 句柄, ensure_ascii=False, indent=1)

    # ---------- 摘要（控制台只打这个，细节都在文件里） ----------
    记("")
    记("=== 规模 ===")
    记("  表 %d 张｜视图 %d 个｜字段 %d 个｜架构 %s"
       % (len(表清单), len(视图们), len(列清单), "、".join(架构们)))
    记("  含 site_ref 的表 %d 张（多站点，问数时必须显式过滤）" % len(含站点))
    记("  空表 %d 张｜主键 %d 个｜外键 %d 个" % (len(空表), len(主键), len(外键)))
    if not 外键:
        记("  ↳ 没有物理外键是 Syteline 的常态，表间关系要在语义层里手写")

    记("")
    记("=== 行数 Top 10（近似） ===")
    for t in 表清单[:10]:
        记("  %-32s %12s 行  %9.1f MB"
           % (t["架构"] + "." + t["表名"], format(t["近似行数"], ","), t["占用MB"]))

    if 多余的:
        记("")
        记("  注意：%d 个对象只出现在列清单里，既不是表也不是视图（权限或并发变更）：%s"
           % (len(多余的), "、".join(多余的[:3])))

    记("")
    记("=== 候选业务表（启发式，需人工确认） ===")
    for 组名, 表们 in 归组候选表(表清单).items():
        记("  %-8s %s" % (组名, "、".join(x["表名"] for x in 表们[:6])
                          + ("…" if len(表们) > 6 else "")))

    记("")
    记("产出目录：%s" % 输出目录)
    with open(os.path.join(输出目录, "摸表日志.txt"), "w", encoding="utf-8") as 句柄:
        句柄.write("\n".join(日志) + "\n")
    return 0


def 主():
    分析 = argparse.ArgumentParser(description="Syteline 摸表（只读，不写库）")
    分析.add_argument("--主机", required=True, help="如 172.16.11.254")
    分析.add_argument("--端口", default="1433")
    分析.add_argument("--库", default="master", help="不给则只列库清单")
    分析.add_argument("--用户", default="")
    分析.add_argument("--密码", default="")
    分析.add_argument("--集成认证", action="store_true", help="用 Windows 账号认证")
    参数 = 分析.parse_args()
    try:
        return 主流程(参数)
    except 摸表失败 as 错:
        print("\n[失败] %s" % 错, file=sys.stderr)
        return 2
    except Exception as 错:
        import traceback
        print("\n[未预期异常] %r" % 错, file=sys.stderr)
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(主())
