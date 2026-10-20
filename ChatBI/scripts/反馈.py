# -*- coding: utf-8 -*-
"""ChatBI · 用户反馈（A-3）—— 数据飞轮的**唯一入口**。

为什么必须有这个入口
--------------------
外部对标（WrenAI 的 memory / Vanna 的训练数据 / SuperSonic 的 Chat Memory）三家都有同一个东西：
**用户确认或修正过一次的「问法 → 正确意图」，成为后续 few-shot 或微调样本。**
百度那篇给的量化是某物流企业靠「记录用户修正行为 + 沉淀样本」把准确率从 68% 提到 91%。

我们连「用户觉得这次答得对不对」这个信号都没收过 —— 那就没有飞轮，只有飞轮的海报。
本模块收的就是这个信号。

🔴 三条纪律（与 审计.py 同源，理由不再重复）
  1. **一行一条 JSONL，只追加不覆盖**；`记()` **永不抛异常**（写失败要能看见，不能静默吞）。
  2. 落盘目录只有一处定义，环境变量 `CHATBI_FEEDBACK_DIR` 可覆盖（自检把临时目录塞进来）。
  3. **评价词的归一只有这一份**（好评词 / 差评词两个常量）——
     前端传「👍」、curl 传「good」、将来接别的客户端传「有用」，都归到同一处判。

🔴 为什么反馈要带「证据」（意图 / SQL / 列 / 行数）
   只记一句「用户说不对」，事后**无法复现**，也无法判断是模型错、语义层错还是题目本身歧义。
   带上意图与 SQL，这条反馈才是一个可复算的案例 —— 这正是 A-4 盲测题最好的题源。
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

# 落盘目录：**只有这一处定义**（同 审计.py 的纪律）
反馈目录 = os.environ.get("CHATBI_FEEDBACK_DIR") or os.path.join(项目目录, "logs")

# 评价词归一：**唯一事实来源**。前端 / curl / 未来的别的客户端都往这里收。
好评词 = {"✅", "👍", "ok", "OK", "good", "有用", "对", "正确", "没问题"}
差评词 = {"❌", "👎", "bad", "BAD", "错", "不对", "有问题", "差"}


def 归一评价(值):
    """返回 `好评` / `差评` / None（不认识就不记成任何一边 —— 宁可判「没给评价」）。"""
    词 = str(值 or "").strip()
    if 词 in 好评词:
        return "好评"
    if 词 in 差评词:
        return "差评"
    return None


# ============================ 写 ============================

def 文件(时间戳=None):
    return os.path.join(反馈目录,
                        "反馈-%s.jsonl" % (时间戳 or time.strftime("%Y%m%d")))


def 记(记录):
    """追加一条反馈。返回落盘后的记录（补上 `时间`）。**永不抛异常。**

    与 审计.记 同一套打法：磁盘满 / 目录只读时，用户的评价该收还是要收；
    但写失败要**能看见**（stderr 一行 + 返回值带 `_写失败`）。
    """
    行 = dict(记录)
    行.setdefault("时间", time.strftime("%Y-%m-%d %H:%M:%S"))
    评价 = 归一评价(行.get("评价"))
    行["评价归一"] = 评价                     # 归一结果单独一列，原始值原样保留
    if 评价 is None:
        # 不认识就**不记成好评也不记成差评**。这条要能在统计里被看见（见 统计()），
        # 否则「评价词写错了」和「用户没评价」表现完全一样。
        行["_评价未识别"] = str(行.get("评价"))
    try:
        os.makedirs(反馈目录, exist_ok=True)
        with open(文件(), "a", encoding="utf-8") as 文:
            文.write(json.dumps(行, ensure_ascii=False) + "\n")
    except OSError as 错:
        行["_写失败"] = "%s: %s" % (type(错).__name__, 错)
        try:
            sys.stderr.write("🔴 反馈写失败（业务继续）：%s\n" % 行["_写失败"])
        except Exception:                                   # noqa: BLE001
            pass
    return 行


# ============================ 读 ============================

def 读全部(文件路径=None):
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


def 查询(限=50, 评价=None, 文件路径=None):
    """最近 N 条（**新的在前**）。`评价` = `好评` / `差评`。"""
    条 = 读全部(文件路径)
    if 评价:
        条 = [x for x in 条 if x.get("评价归一") == 评价]
    return 条[-限:][::-1]


def 统计(文件路径=None):
    条 = 读全部(文件路径)
    按类型 = {}
    for x in 条:
        if x.get("评价归一") == "差评":
            键 = str(x.get("类型") or "未知")
            按类型[键] = 按类型.get(键, 0) + 1
    return {
        "文件": 文件路径 or 文件(),
        "总条数": len(条),
        "好评数": sum(1 for x in 条 if x.get("评价归一") == "好评"),
        "差评数": sum(1 for x in 条 if x.get("评价归一") == "差评"),
        "评价未识别数": sum(1 for x in 条 if x.get("评价未识别") or x.get("_评价未识别")),
        "带纠正数": sum(1 for x in 条 if str(x.get("纠正") or "").strip()),
        "差评按类型": 按类型,
    }


# ============================ 沉淀（数据飞轮的另一半） ============================

def 正样本(文件路径=None, 限=500):
    """**好评 + 有意图** 的记录 → few-shot 候选。

    🔴 为什么加「有意图」这个条件：好评只说明「用户满意」，不说明「这条能被当教材」。
       一条澄清/拒答也可能被用户点了好评 —— 拿它当 few-shot 样本会教坏模型。
       要求记录里带归一后的意图，等于要求「这是一次**真正走通了**的问数」。
    """
    出 = []
    for x in 读全部(文件路径):
        if x.get("评价归一") != "好评":
            continue
        意图 = x.get("意图")
        if not isinstance(意图, dict) or not 意图:
            continue
        出.append({"问题": x.get("问题"), "意图": 意图,
                   "SQL": x.get("SQL"), "时间": x.get("时间")})
    return 出[-限:]


def 候选盲测题(文件路径=None, 限=200):
    """**差评 + 带纠正** → 候选盲测题（A-4 的题源）。

    这是 A-3 与 A-4 的接口：用户真实问法里「我们答错过的那些」，
    比任何人坐在桌前编的题都更接近真实分布（最佳实践 ⑦ / ⑥）。
    ⚠️ 产出的是**候选**，必须人复核后才能进正式题集 —— 用户说的也可能不对。
    """
    出 = []
    for x in 读全部(文件路径):
        if x.get("评价归一") != "差评":
            continue
        纠正 = str(x.get("纠正") or "").strip()
        if not 纠正 and not x.get("点评"):
            continue
        出.append({"问题": x.get("问题"), "我们当时理解成": x.get("意图"),
                   "用户说应该": 纠正 or x.get("点评"),
                   "实测类型": x.get("类型"), "来源时间": x.get("时间")})
    return 出[-限:]


# ============================ CLI ============================

def 主():
    sys.stdout.reconfigure(errors="replace")
    参 = argparse.ArgumentParser(description="ChatBI 用户反馈（A-3）")
    参.add_argument("--最近", type=int, default=20, help="显示最近 N 条")
    参.add_argument("--统计", action="store_true", help="汇总统计")
    参.add_argument("--好评", action="store_true", help="只看好评")
    参.add_argument("--差评", action="store_true", help="只看差评")
    参.add_argument("--正样本", action="store_true", help="导出 few-shot 候选（好评 + 有意图）")
    参.add_argument("--候选题", action="store_true", help="导出候选盲测题（差评 + 带纠正）")
    参.add_argument("--文件", help="指定反馈文件（默认今天）")
    参数 = 参.parse_args()

    if 参数.统计:
        print(json.dumps(统计(参数.文件), ensure_ascii=False, indent=2))
        return 0

    if 参数.正样本:
        样 = 正样本(参数.文件)
        print(json.dumps(样, ensure_ascii=False, indent=2))
        print("\n共 %d 条 —— 人复核后可作为 few-shot 样本" % len(样))
        return 0

    if 参数.候选题:
        题 = 候选盲测题(参数.文件)
        print(json.dumps(题, ensure_ascii=False, indent=2))
        print("\n共 %d 条 —— ⚠️ 必须人复核才能进正式题集（用户说的也可能不对）" % len(题))
        return 0

    条 = 查询(限=参数.最近,
             评价="好评" if 参数.好评 else ("差评" if 参数.差评 else None),
             文件路径=参数.文件)
    print("反馈文件：%s" % (参数.文件 or 文件()))
    if not 条:
        print("  （没有匹配的记录）")
        return 0
    for x in 条:
        头 = {"好评": "👍", "差评": "👎"}.get(x.get("评价归一"), "·")
        尾 = ("  ← %s" % str(x.get("纠正") or x.get("点评"))[:60]) if x.get("纠正") or x.get("点评") else ""
        print("  %s %s  %s%s" % (头, x.get("时间", ""), str(x.get("问题"))[:36], 尾))
    return 0


if __name__ == "__main__":
    raise SystemExit(主())
