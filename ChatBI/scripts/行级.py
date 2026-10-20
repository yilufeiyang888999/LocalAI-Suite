# -*- coding: utf-8 -*-
"""ChatBI · 行级权限（B-5）—— 按角色限定「能看到哪些行」

要解决的问题：脱敏（B-3）管的是「哪几列看不见」，管不了「看不见别人的数据」。
一个区域主管不该在采购金额榜上看到别的区域的供应商 —— 这不是列的问题，是行的问题。

## 🔴 为什么不在结果层过滤（这是本模块存在的唯一理由）

    结果层过滤 = `[r for r in 行 if r[0] in 允许的供应商]`

    看起来一样是「看不到」，但**聚合已经算完了**：
      · 「不良率最高的供应商」——被隐藏的那家如果最高，删掉它之后
        第二名会被顶上来，用户看到的是「最高的是第二高的那家」。
      · 「一共多少家供应商」——总数里已经算上了看不见的那些。
      · 「各仓库库存占比」——分母包含了看不见的仓库。

    这些**都不报错**，而且数字都是真的（真的从库里算出来的），只是回答的不是
    「你被允许看的那个范围」的问题。所以必须在 **SQL 的 WHERE 层**限定，
    让聚合从一开始就只发生在允许的行上。

## 🔴 怎么注入：Temp View 遮蔽，而不是改 SQL 文本

    想到的第一个办法是「在渲染出的 SQL 里找 WHERE 加条件」。**这条路是错的**，
    而且错得很隐蔽 —— 因为渲染器会产出 CTE（`WITH 主供 AS (...)`，见场景模板 C11/C12/C13）：

        WITH 最高 AS (SELECT 供应商编码 FROM 到料检验 WHERE 到料日期 >= ... GROUP BY ...)
        SELECT ... FROM 最高 h JOIN 采购订单 p ...

    往最外层塞谓词，进不去 CTE 内部；往 CTE 里塞，又会漏掉另一个 CTE。
    而且「找 WHERE 的位置」靠正则数括号 —— 数错一次就是**语义变了但语法合法**。

    改用 SQLite 的 schema 解析顺序：**裸表名先解析 `temp`、再解析 `main`**。
    于是在连接里建一个同名临时视图：

        CREATE TEMP VIEW 供应商 AS SELECT * FROM main.供应商 WHERE 供应商类别 IN ('五金件','铸件')

    之后**任何**引用 `供应商` 的 SQL（指标 / 基础查询 / 场景模板 / 多层 CTE）
    都自动落在被限定的行上 —— 不需要解析 SQL，也不可能漏。

    ⚠️ 前提：渲染器产出的表名**不带 schema 前缀**（写 `main.供应商` 就能绕过遮蔽）。
       这一条由 `探针` / `自检` 盯着（断言渲染出的 SQL 里没有 `main.`）。

## 🔴 建完必须探一下（实测踩到的）

    SQLite 的 `CREATE VIEW` **不校验视图体** —— 实测：

        CREATE TEMP VIEW 没有这张表 AS SELECT * FROM main.没有这张表 WHERE 没有的列 IN ('x')
        → 建视图成功，零报错

    这一条把最危险的配置错从「报错」变成了「静默放行」：
      · **表名写错一个字** → 视图名跟着错 → 真表上根本没有视图 → 该角色看到**全量**，
        而 `应用()` 返回的是「已遮蔽 1 张表」。
      · **字段名写错** → 视图建得成，查询时才报 `no such column`（这个还算响的）。

    所以每建一个视图，紧接着跑一次 `SELECT * FROM "表" LIMIT 0`：`LIMIT 0` 不取一行，
    但语句要完整 prepare，视图体会被展开校验 —— 表/字段不存在当场露出来。
    成本为零，换掉的是「看起来限制生效了、其实一行没限」这个形态。

## 四道锁（比原来多一道）

    1. `mode=ro`          文件层只读
    2. **temp 遮蔽**       行级限定（本模块）
    3. `query_only = ON`  会话层只读 —— 建视图时临时关掉，建完立刻开回来
    4. 语句白名单          护栏.检查

    第 3 道在 `CREATE TEMP VIEW` 期间必须关掉（`query_only` 会拦住 temp 写入），
    所以窗口收得尽可能小：关 → 建 → 立刻开，中间不跑任何用户 SQL。
"""

import argparse
import os
import sys

本目录 = os.path.dirname(os.path.abspath(__file__))
项目目录 = os.path.dirname(本目录)
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)

import 权限                                             # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:                                    # noqa: BLE001
        pass

关行级 = bool(os.environ.get("CHATBI_NO_ROWLEVEL"))      # 自检 / 排障用

# ============================ 数据域（唯一事实来源） ============================
# 🔴 为什么写在**这里**而不是口径定义 YAML：YAML 管「数字怎么算」，
#    这里管「谁看得到哪些行」。改口径不该让某个角色突然少看一批数据，
#    加一个行级规则也不该触发口径变更的评测重跑。
#
# ⚠️ 下面的值是**演示配置**（对着演示库的真实取值写的）。真实部署里，
#    这一份应该来自企业的人员/组织主数据，而不是硬编码 —— 但结构是一样的。

角色数据域 = {
    "操作员": {
        "说明": "只允许看「原料仓」的库存数据",
        "遮蔽": [{"表": "库存", "字段": "仓库名称", "值": ["原料仓"]}],
        "限制在": "仓储",
    },
    "主管": {
        "说明": "只允许看「五金件 / 铸件」两类供应商",
        "遮蔽": [{"表": "供应商", "字段": "供应商类别", "值": ["五金件", "铸件"]}],
        "限制在": "供应商",
    },
    "管理员": {"说明": "不限制（全量）", "遮蔽": []},
}

计数 = {"已遮蔽连接": 0, "建视图": 0, "建视图失败": 0, "无限制": 0, "已关闭": 0}
_最后错 = {"建视图": None}


def 遮蔽清单(角色):
    """这个角色要遮蔽哪些行。返回 [] = 不限制。"""
    if 关行级:
        return []
    规 = 角色数据域.get(角色) or {}
    return [dict(x) for x in (规.get("遮蔽") or [])]


def _谓词(条):
    值 = 条.get("值") or []
    if not 值:
        # 空值列表 = 「一行都不许看」。**不能**当成「不限」——
        # 配置漏填的后果必须是「看不到」而不是「全看到」，这两者在界面上长得一样。
        return "0"
    项 = ", ".join("'%s'" % str(v).replace("'", "''") for v in 值)
    return "%s IN (%s)" % (条["字段"], 项)


def 应用(连接, 角色):
    """在连接上建遮蔽视图。返回 {已遮蔽, 说明, 错误?}。

    **失败必须暴露**：建视图失败若被吞掉，用户看到的就是**全量数据**，
    而界面上一切正常 —— 这是本模块最不能接受的一种失败。
    """
    条 = 遮蔽清单(角色)
    if not 条:
        计数["无限制"] += 1
        return {"已遮蔽": [], "说明": None,
                "角色": 角色, "受限": bool((角色数据域.get(角色) or {}).get("遮蔽"))}

    建过, 失败 = [], []
    连接.execute("PRAGMA query_only = OFF")               # 窗口开始（见模块头「四道锁」）
    try:
        for x in 条:
            表, 字 = x.get("表"), x.get("字段")
            if not (表 and 字):
                失败.append("遮蔽规则缺「表」或「字段」：%r" % (x,))
                continue
            try:
                连接.execute(
                    "CREATE TEMP VIEW %s AS SELECT * FROM main.%s WHERE %s"
                    % (表, 表, _谓词(x)))
                # 🔴 视图体不被 CREATE 校验（见模块头）→ 立刻探一下，不取任何行。
                连接.execute('SELECT * FROM "%s" LIMIT 0' % 表)
                建过.append(x)
                计数["建视图"] += 1
            except Exception as 错:                          # noqa: BLE001
                失败.append("%s：%s: %s" % (表, type(错).__name__, str(错)[:60]))
                计数["建视图失败"] += 1
                _最后错["建视图"] = 失败[-1]
                # 探针失败时视图可能已经建成了（坏的）—— 清掉，别把它留给后续查询。
                try:
                    连接.execute('DROP VIEW IF EXISTS "%s"' % 表)
                except Exception:                            # noqa: BLE001
                    pass
    finally:
        连接.execute("PRAGMA query_only = ON")            # 窗口结束：立刻开回来

    if 失败:
        # 🔴 建不上就不能继续 —— 「部分遮蔽」比「不遮蔽」更危险：
        #    它看起来是生效的（有些表确实受限了），漏掉的那张表没人会去查。
        return {"已遮蔽": 建过, "失败": 失败, "角色": 角色, "受限": True,
                "说明": "🔴 行级权限**没能完全生效**：%s —— 这一类查询已被拒绝，"
                        "请联系管理员，不要在结果上做判断" % "；".join(失败),
                "阻断": True}

    计数["已遮蔽连接"] += 1
    规 = 角色数据域.get(角色) or {}
    名 = "、".join("「%s」表只保留 %s = %s"
                  % (x["表"], x["字段"], "、".join(map(str, x["值"]))) for x in 建过)
    说明 = ("已按角色「%s」限定数据范围：%s（%s）。"
            "限定加在 **SQL 层**（临时视图遮蔽），不是查完再筛 —— "
            "所以聚合、排名、总数都只算你被允许看的那些行。"
            % (角色, 名, 规.get("说明") or ""))
    return {"已遮蔽": 建过, "角色": 角色, "受限": True, "说明": 说明}


def 清单():
    """给页面 / /catalog 用（自描述，不手抄到前端）。"""
    return [{"角色": k, "说明": v["说明"],
             "遮蔽": [{"表": x["表"], "字段": x["字段"], "值": list(x["值"])}
                    for x in (v.get("遮蔽") or [])]}
            for k, v in 角色数据域.items()]


def 校验配置():
    """配置自查。返回问题列表（空 = 干净）。

    🔴 每条规则都要能**打得到东西**：表名或字段名写错一个字，`CREATE TEMP VIEW`
       会直接报错（那种还算好的）；但如果写的是**存在的表 + 不存在的字段**里的
       值（比如「仓库名称 = 华东仓」而库里只有「原料仓」），视图建得成、
       只是**一行都不返回** —— 表现是「这个角色什么都查不到」，
       看着像权限设计，其实是配置写错了。所以这里要拿真库核一遍。
    """
    问题 = []
    真角色 = set(权限.角色表)
    for 角 in 角色数据域:
        if 角 not in 真角色:
            问题.append("数据域里写了未知角色「%s」（可用：%s）" % (角, "、".join(sorted(真角色))))
    for 角 in 真角色:
        if 角 not in 角色数据域:
            # 缺失 = 该角色不受限（`遮蔽清单` 返回空）。这是**放行**，要显式管住。
            问题.append("角色「%s」没有数据域配置 —— 它将是**全量可见**。"
                        "如果这是有意的，请在配置里显式写 `遮蔽: []`" % 角)
    try:
        import sqlite3
        import 语义层 as 语义
        库 = 语义.库路径()
        连接 = sqlite3.connect("file:%s?mode=ro" % 库.replace("?", "%3f"), uri=True)
        try:
            for 角, 规 in 角色数据域.items():
                for x in (规.get("遮蔽") or []):
                    表, 字 = x.get("表"), x.get("字段")
                    try:
                        有 = {r[0] for r in 连接.execute(
                            "SELECT DISTINCT %s FROM %s" % (字, 表)).fetchall()}
                    except Exception as 错:                   # noqa: BLE001
                        问题.append("「%s」的规则 %s.%s 读不出来：%s"
                                    % (角, 表, 字, str(错)[:60]))
                        continue
                    缺 = [v for v in (x.get("值") or []) if v not in 有]
                    if 缺 and 有:
                        # 值不在库里 → 视图建得成但查不到任何行（最隐蔽的一种配置错）
                        问题.append("「%s」限定值 %s 在 %s.%s 里不存在（实际取值：%s）—— "
                                    "该角色将什么都查不到"
                                    % (角, "、".join(map(str, 缺)), 表, 字,
                                       "、".join(sorted(map(str, 有)))))
                    if not (x.get("值") or []):
                        问题.append("「%s」的规则 %s.%s 值为空 —— 该角色一行都看不到"
                                     "（若本意是「不限」，请把整条规则删掉）" % (角, 表, 字))
        finally:
            连接.close()
    except Exception as 错:                                   # noqa: BLE001
        问题.append("拿真库核对时失败（配置**未被验证**，不等于通过）：%s: %s"
                    % (type(错).__name__, str(错)[:70]))
    return 问题


def 统计():
    return {"已关闭": 关行级, **dict(计数),
            **({"最近错误": _最后错["建视图"]} if _最后错["建视图"] else {})}


# ============================ CLI ============================

def 主():
    参 = argparse.ArgumentParser(description="ChatBI 行级权限（B-5）")
    参.add_argument("--清单", action="store_true")
    参.add_argument("--校验", action="store_true")
    参.add_argument("--试", action="store_true", help="拿真库跑一遍对比（受限 vs 全量）")
    参数 = 参.parse_args()

    if 参数.清单:
        for x in 清单():
            print("· %s —— %s" % (x["角色"], x["说明"]))
            for m in x["遮蔽"]:
                print("    遮蔽 %s：只保留 %s ∈ %s"
                      % (m["表"], m["字段"], "、".join(map(str, m["值"]))))
        return 0

    if 参数.校验:
        问题 = 校验配置()
        if not 问题:
            print("✅ 行级配置自洽（%d 个角色，取值都在真库里）" % len(角色数据域))
            return 0
        for x in 问题:
            print("🔴 %s" % x)
        return 1

    if 参数.试:
        import sqlite3
        import 语义层 as 语义
        定义 = 语义.载入()
        库 = 语义.库路径(定义)
        语句集 = [
            ("各仓库库存总量", "SELECT k.仓库名称 AS 仓库, SUM(k.库存数量) AS 总量 "
                          "FROM 库存 k GROUP BY k.仓库名称 ORDER BY 总量 DESC"),
            ("供应商总数", "SELECT COUNT(*) FROM 供应商"),
        ]
        for 角色 in 权限.角色表:
            连接 = sqlite3.connect("file:%s?mode=ro" % 库.replace("?", "%3f"), uri=True)
            连接.execute("PRAGMA query_only = ON")
            try:
                果 = 应用(连接, 角色)
                print()
                print("【%s】%s" % (角色, 果.get("说明") or "无限制"))
                if 果.get("阻断"):
                    print("  → 已阻断（不执行任何查询）")
                    continue
                for 名, SQL in 语句集:
                    行 = 连接.execute(SQL).fetchall()
                    # ⚠️ 不能假定每行都是两列（`COUNT(*)` 只有一列）——
                    #    解包写死成 `a, b` 会在单列结果上直接抛 ValueError。
                    文 = " ｜ ".join(
                        "=".join(str(x) for x in r) if len(r) > 1 else str(r[0])
                        for r in 行[:6]) or "（空）"
                    print("  %s：%s" % (名, 文))
            finally:
                连接.close()
        return 0

    参.print_help()  # parser 本身，对
    return 2


if __name__ == "__main__":
    raise SystemExit(主())
