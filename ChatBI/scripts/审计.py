# -*- coding: utf-8 -*-
"""ChatBI · 审计日志（包 6）—— 每一次动作调用都留痕。

**一行一条 JSONL，只追加不覆盖。**

🔴 为什么是 JSONL 而不是数据库
   审计的价值在「事后能查、能直接看、能 diff」。一行一条、字段自带名字，
   出问题当场 `grep` 就能还原现场。一天几十条的量级不需要索引 ——
   上 SQLite 只是多一层依赖和 schema 维护。

🔴 为什么必须记「依据不切题」
   包 5 真的产出过一张「正文数字都对、但依据不支撑它」的补货申请单
   （依据只有一个「物料数 30」，而补货申请要的是一张缺料**清单**）。
   这类错**不违反任何已有规则** —— 缺参数会报错、越界会报错，它不会。
   白名单拦不住它，只有留痕能事后追查：
   「某时刻有一张单据，它的依据是计数查询」这件事必须查得到。

🔴 写失败不能拖垮业务：`记()` **永不抛异常**。
   磁盘满 / 目录只读时，用户的单据该生成还是要生成；
   但写失败要**能看见**（stderr 一行 + 返回值里带 `_写失败`），不能静默吞掉。
   （「吞异常」和「没异常」表现相同 = 判据没了 —— §9.5 五类错误之二。）
"""

import argparse
import json
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:                                       # noqa: BLE001
        pass

本目录 = os.path.dirname(os.path.abspath(__file__))
项目目录 = os.path.dirname(本目录)

# 🔴 审计目录。**只有这一处定义**（自检把临时目录塞进来就靠改这一个变量）。
#    环境变量 `CHATBI_AUDIT_DIR` 可覆盖 —— 部署时审计可能要落到独立盘 / 共享路径，
#    也方便自检把「回归子进程」的审计引开，不污染真实审计。
审计目录 = os.environ.get("CHATBI_AUDIT_DIR") or os.path.join(项目目录, "logs")


# ============================ 写 ============================

def 文件(时间戳=None):
    return os.path.join(审计目录,
                        "审计-%s.jsonl" % (时间戳 or time.strftime("%Y%m%d")))


def 记(记录):
    """追加一条审计。返回落盘后的记录（补上 `时间`）。**永不抛异常。**"""
    行 = dict(记录)
    行.setdefault("时间", time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        os.makedirs(审计目录, exist_ok=True)
        with open(文件(), "a", encoding="utf-8") as 文:
            文.write(json.dumps(行, ensure_ascii=False) + "\n")
    except OSError as 错:
        行["_写失败"] = "%s: %s" % (type(错).__name__, 错)
        try:
            sys.stderr.write("🔴 审计写失败（业务继续）：%s\n" % 行["_写失败"])
        except Exception:                                   # noqa: BLE001
            pass
    return 行


# ============================ 读 ============================

def 读全部(文件路径=None):
    """读某天的全部记录（按写入顺序）。坏行跳过 —— 一行写坏不该让整份不可读。"""
    路 = 文件路径 or 文件()
    if not os.path.isfile(路):
        return []
    出 = []
    try:
        with open(路, encoding="utf-8") as 文:
            for 原文 in 文:
                原文 = 原文.strip()
                if not 原文:
                    continue
                try:
                    出.append(json.loads(原文))
                except ValueError:
                    continue
    except OSError:
        return []
    return 出


def 查询(限=50, 角色=None, 动作名=None, 被拒=None, 文件路径=None):
    """最近 N 条（**新的在前**），可按 角色 / 动作 / 是否被拒 过滤。"""
    条 = 读全部(文件路径)
    if 角色 is not None:
        条 = [x for x in 条 if x.get("角色") == 角色]
    if 动作名 is not None:
        条 = [x for x in 条 if x.get("动作名") == 动作名]
    if 被拒 is not None:
        条 = [x for x in 条 if bool(x.get("被拒")) == bool(被拒)]
    return 条[-限:][::-1]


def 统计(文件路径=None):
    条 = 读全部(文件路径)
    按角色 = {}
    按动作 = {}
    for x in 条:
        按角色[str(x.get("角色"))] = 按角色.get(str(x.get("角色")), 0) + 1
        if x.get("动作名"):
            按动作[x["动作名"]] = 按动作.get(x["动作名"], 0) + 1
    return {
        "文件": 文件路径 or 文件(),
        "总条数": len(条),
        "被拒数": sum(1 for x in 条 if x.get("被拒")),
        "落盘数": sum(1 for x in 条 if x.get("落盘")),
        "依据不切题数": sum(1 for x in 条 if x.get("依据切题") is False),
        "按角色": 按角色,
        "按动作": 按动作,
    }


# ============================ CLI ============================

def _一行(x):
    头 = "🚫" if x.get("被拒") else ("✅" if x.get("落盘") else "·")
    尾 = ""
    if x.get("被拒"):
        尾 = "  ← %s" % x.get("拒绝理由")
    elif x.get("依据切题") is False:
        尾 = "  ← ⚠️ 依据不切题"
    return "%s %s  %-8s %-16s%s" % (头, x.get("时间", ""), x.get("角色") or "-",
                                   x.get("动作名") or "-", 尾)


def 主():
    sys.stdout.reconfigure(errors="replace")
    参 = argparse.ArgumentParser(description="ChatBI 审计日志（包 6）")
    参.add_argument("--最近", type=int, default=20, help="显示最近 N 条")
    参.add_argument("--统计", action="store_true", help="汇总统计")
    参.add_argument("--被拒", action="store_true", help="只看被拒的记录")
    参.add_argument("--角色")
    参.add_argument("--动作")
    参.add_argument("--文件", help="指定审计文件（默认今天）")
    参 = 参.parse_args()

    if 参.统计:
        print(json.dumps(统计(参.文件), ensure_ascii=False, indent=2))
        return 0

    条 = 查询(限=参.最近, 角色=参.角色, 动作名=参.动作,
             被拒=True if 参.被拒 else None, 文件路径=参.文件)
    print("审计文件：%s" % (参.文件 or 文件()))
    if not 条:
        print("  （没有匹配的记录）")
        return 0
    for x in 条:
        print("  " + _一行(x))
    return 0


if __name__ == "__main__":
    raise SystemExit(主())
