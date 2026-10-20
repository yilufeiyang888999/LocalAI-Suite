# -*- coding: utf-8 -*-
"""ChatBI · 切适配器缩放 —— 运行时改 LoRA scale，**改完必回读**

## 为什么需要运行时改（而不是重启换 --lora-scaled）

#203 要回答的是「scale=0 是不是真的等于裸基座、scale=1 是不是真的等于挂 adapter」。
如果用「重启换参数」做对照，两次之间会混进**权重重新加载、显存重新分配、
采样器重建**三种变化 —— 到时分不清差异是 scale 带来的还是重载带来的。

`POST /lora-adapters` 让我们在**同一个进程、同一份已加载权重**上只改一个数。
这是唯一能把「scale」单独隔离出来的做法。

## 🔴 判据纪律

    · **POST 返回 200 不算改成功** —— 必须 GET 回读，看到 scale 真的是那个数。
      （§9.5：「活着」≠「能干这件事」）
    · **GET 取不到 ⇒ 退 3（不可判）**，不能当成「改失败了」也不能当成「改好了」。
      未查 ≠ 通过，也 ≠ 失败。
    · **引擎报的 scale 和你请求的不一致 ⇒ 退 1**，并把两边的数都打出来。

## 用法

    python scripts/切适配器缩放.py --端口 18000            # 只看当前
    python scripts/切适配器缩放.py --端口 18000 --设为 0     # 改成 0（等于裸基座）
    python scripts/切适配器缩放.py --端口 18000 --设为 1     # 改回 1
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

绕代理 = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def 说(文本=""):
    print(文本)
    sys.stdout.flush()


def 读缩放(端口):
    """返回 list[dict] 或 None（端点不通 —— 与空列表区分开）"""
    try:
        应 = 绕代理.open("http://127.0.0.1:%d/lora-adapters" % 端口, timeout=6)
        值 = json.loads(应.read().decode("utf-8", "replace"))
        return 值 if isinstance(值, list) else None
    except Exception:                                          # noqa: BLE001
        return None


def 设缩放(端口, 编号, 目标):
    体 = json.dumps([{"id": 编号, "scale": 目标}]).encode("utf-8")
    请求 = urllib.request.Request(
        "http://127.0.0.1:%d/lora-adapters" % 端口, data=体,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with 绕代理.open(请求, timeout=15) as 应:
            return True, "HTTP %s ｜ %s" % (应.status, 应.read().decode("utf-8", "replace")[:120])
    except urllib.error.HTTPError as 错:
        return False, "HTTP %s ｜ %s" % (错.code, 错.read().decode("utf-8", "replace")[:160])
    except Exception as 错:                                    # noqa: BLE001
        return False, "%s: %s" % (type(错).__name__, str(错)[:120])


def 主():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    解 = argparse.ArgumentParser(description="运行时切换 LoRA 缩放（改完必回读）")
    解.add_argument("--端口", type=int, default=18000)
    解.add_argument("--设为", type=float, default=None, help="目标 scale（不给 = 只看）")
    解.add_argument("--编号", type=int, default=0)
    参数 = 解.parse_args()

    前 = 读缩放(参数.端口)
    if 前 is None:
        说("🔴 /lora-adapters 取不到（端口 %d）—— 服务没起，或该版本不支持此端点。" % 参数.端口)
        return 3
    if not 前:
        说("当前：**没有任何 adapter**（空列表）—— 裸基座态，没有 scale 可改。")
        return 3
    说("当前：%s" % json.dumps(前, ensure_ascii=False))
    现 = next((a.get("scale") for a in 前 if a.get("id") == 参数.编号), None)
    说("      第 %d 号 scale = %s" % (参数.编号, 现))

    if 参数.设为 is None:
        return 0

    好, 回执 = 设缩放(参数.端口, 参数.编号, 参数.设为)
    说("POST 回执：%s" % 回执)
    if not 好:
        说("🔴 设置请求失败 —— scale **未确认变更**")
        return 1

    后 = 读缩放(参数.端口)
    if 后 is None:
        说("🔴 设置后回读不到 —— 状态未确认（POST 返回过快不代表生效）")
        return 3
    新 = next((a.get("scale") for a in 后 if a.get("id") == 参数.编号), None)
    说("回读：第 %d 号 scale = %s" % (参数.编号, 新))
    if 新 is None or abs(float(新) - 参数.设为) > 1e-6:
        说("🔴 回读值（%s）与目标（%s）不一致 —— 不能当已生效" % (新, 参数.设为))
        return 1
    说("✅ 已生效：%s → %s" % (现, 新))
    return 0


if __name__ == "__main__":
    raise SystemExit(主())
