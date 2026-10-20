# -*- coding: utf-8 -*-
"""ChatBI · 身份（B-4）—— 角色从「调用方自己填的标签」升级到「验过签的令牌」

要解决的问题：现在角色是请求体里的一个字符串（`{"角色":"管理员"}`）。
那**不是身份，是自称** —— 任何人把那一行改掉就是管理员。
B-3 的脱敏、B-5 的行级隔离、包 6 的动作白名单，全都挂在这个字符串上。
底下的东西再严谨，只要这一层是自称，整条链的权限就都是装饰。

## 做法（不引入外部依赖）

    `v1.<base64url(载荷)>.<HMAC-SHA256 前 24 位十六进制>`
    载荷 = {"角色":..., "主体":..., "签发":ts, "过期":ts}

选 HMAC 而不是 JWT：JWT 要引库，而项目纪律是「标准库能做的就不引框架」
（`选型决策.md` D7）。HMAC + base64url 用 `hmac` / `hashlib` / `base64` 就够了，
而且这里不需要 JWT 的跨系统互认能力 —— 签发方和验证方是同一个服务。

## 🔴 密钥从哪来（这一条决定了本模块是真的还是假的）

    `CHATBI_TOKEN_SECRET` 环境变量。**没设就用开发密钥**，并且：
      · `/health` 与签约接口都会标明「开发密钥」
      · `/health` 里 `开发密钥: true` —— 一个能被搜到的字段
    为什么不「没设就随机生成再落盘」：那会让自检**无法构造**一个合法令牌
    （每次跑密钥都不同），于是「验签」这一条永远只能靠读代码相信。
    宁可明确了当地说「这是开发密钥」，也不要造一个看起来更安全、
    其实测不了的东西。

## 🔴 裸角色直传：允许，但必须留痕

    没令牌时仍接受 `{"角色":"主管"}`（否则本地演示的前端没法用），
    但应答里会带 `身份来源: "自称身份（未验证）"`，并且**计数**。
    等 `CHATBI_REQUIRE_TOKEN=1` 一开，裸角色直接 401。
    「能跑」与「能收紧」两个阶段都要有，且**阶段要看得见** ——
    看不见的话，演示环境和生产环境在界面上长得一模一样。
"""

import base64
import hashlib
import hmac
import json
import os
import sys
import threading
import time

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

开发密钥 = "chatbi-dev-secret-do-not-use-in-production"
默认有效秒 = int(os.environ.get("CHATBI_TOKEN_TTL") or 3600)


def 密钥():
    return os.environ.get("CHATBI_TOKEN_SECRET") or 开发密钥


def 用开发密钥():
    return not os.environ.get("CHATBI_TOKEN_SECRET")


def 强制令牌():
    return bool(os.environ.get("CHATBI_REQUIRE_TOKEN"))


计数 = {"签发": 0, "验签通过": 0, "验签失败": 0, "缺令牌被拒": 0, "自称身份": 0, "默认角色": 0}
_锁 = threading.Lock()


class 身份错(Exception):
    """令牌问题。**带原因** —— 「验不过」和「过期了」是两件事，
    混成一句「无效令牌」会让排查方向完全错（前者查密钥，后者查时钟）。"""


# ============================ 签发 / 验签 ============================

def _签(段):
    return hmac.new(密钥().encode("utf-8"), 段.encode("ascii"),
                    hashlib.sha256).hexdigest()[:24]


def 签(角色, 主体=None, 有效秒=None):
    """签一个令牌。角色不在 `权限.角色表` 里直接拒绝 —— 签出一个用不了的角色，
    现象是「这人什么都不让看」，看着像权限策略，其实是个错别字。"""
    if 角色 not in 权限.角色表:
        raise 身份错("未知角色「%s」（可用：%s）" % (角色, "、".join(权限.角色表)))
    现在 = int(time.time())
    载荷 = {"角色": 角色, "主体": 主体, "签发": 现在,
           "过期": 现在 + (有效秒 or 默认有效秒)}
    原 = json.dumps(载荷, ensure_ascii=False, separators=(",", ":"),
                    sort_keys=True).encode("utf-8")
    段 = base64.urlsafe_b64encode(原).decode("ascii").rstrip("=")
    with _锁:
        计数["签发"] += 1
    return "v1.%s.%s" % (段, _签(段))


def 验(令牌, 现在=None):
    """验签 + 查有效期。失败抛 `身份错`（**带具体原因**）。"""
    段 = str(令牌 or "").strip().split(".")
    if len(段) != 3 or 段[0] != "v1":
        with _锁:
            计数["验签失败"] += 1
        raise 身份错("令牌格式不是 v1.<载荷>.<签名>")
    if not hmac.compare_digest(段[2], _签(段[1])):        # 常数时间比较，防时序猜测
        with _锁:
            计数["验签失败"] += 1
        raise 身份错("签名不匹配 —— 令牌被改过，或签名密钥换过（占位符填对了但密钥变了）")
    try:
        原始 = base64.urlsafe_b64decode(段[1] + "=" * (-len(段[1]) % 4))
        载荷 = json.loads(原始.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as 错:
        with _锁:
            计数["验签失败"] += 1
        raise 身份错("载荷解不开：%s" % str(错)[:50]) from None
    过 = 载荷.get("过期")
    if isinstance(过, (int, float)) and (现在 or time.time()) > 过:
        with _锁:
            计数["验签失败"] += 1
        raise 身份错("令牌已过期（过期时间 %s）"
                    % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(过)))
    if 载荷.get("角色") not in 权限.角色表:
        with _锁:
            计数["验签失败"] += 1
        raise 身份错("令牌里的角色「%s」不在角色表里" % 载荷.get("角色"))
    with _锁:
        计数["验签通过"] += 1
    return 载荷


# ============================ 请求 → 角色 ============================

def 解析(令牌=None, 裸角色=None):
    """决定这次请求的角色。返回 {角色, 来源, 主体?, 错误?}。

    🔴 顺序是刻意的：**令牌永远优先于裸角色**。
       反过来的话，一个拿着低权限令牌的人只要在请求体里多写一行
       `"角色":"管理员"` 就能提权 —— 而这不会报错。
    """
    if 令牌:
        try:
            载荷 = 验(令牌)
            return {"角色": 载荷["角色"], "来源": "令牌",
                    "主体": 载荷.get("主体"), "签发": 载荷.get("签发"),
                    "过期": 载荷.get("过期")}
        except 身份错 as 错:
            # 给了令牌但验不过 → **不退回裸角色**。退回等于「拿个假令牌就能降级到自称」，
            # 而现象是「照常工作」，比直接拒绝危险得多。
            return {"角色": None, "来源": "令牌无效", "错误": str(错)}

    if 强制令牌():
        with _锁:
            计数["缺令牌被拒"] += 1
        return {"角色": None, "来源": "缺令牌",
                "错误": "本服务要求携带令牌（CHATBI_REQUIRE_TOKEN=1）—— "
                        "用 scripts/身份.py --签 --角色 管理员 生成一个"}

    if 裸角色:
        if 裸角色 not in 权限.角色表:
            return {"角色": None, "来源": "自称身份",
                    "错误": "未知角色「%s」（可用：%s）" % (裸角色, "、".join(权限.角色表))}
        with _锁:
            计数["自称身份"] += 1
        # ⚠️ 这一条**必须一路传到界面**（应答里的 `身份来源`）——
        #    不留痕的话，自称身份与验过的身份在结果里长得一模一样。
        return {"角色": 裸角色, "来源": "自称身份（未验证）"}

    with _锁:
        计数["默认角色"] += 1
    return {"角色": 权限.默认角色, "来源": "默认（未传身份）"}


def 统计():
    return {"要求令牌": 强制令牌(), "用开发密钥": 用开发密钥(),
            "密钥来源": ("CHATBI_TOKEN_SECRET" if not 用开发密钥() else "内置开发密钥"),
            "令牌有效期秒": 默认有效秒, **dict(计数)}


def 校验配置():
    """配置自查。返回问题列表（空 = 干净）。"""
    问题 = []
    if 强制令牌() and 用开发密钥():
        # 这个组合最危险：**看起来收紧了**（要求令牌），实际每个人都能自己签一个。
        问题.append("CHATBI_REQUIRE_TOKEN=1 但没设 CHATBI_TOKEN_SECRET —— "
                    "开发密钥是公开的，任何人都能签出合法令牌。这两项必须同时设。")
    if 默认有效秒 <= 0:
        问题.append("CHATBI_TOKEN_TTL=%s 不是正整数 —— 签出来的令牌立刻过期" % 默认有效秒)
    return 问题


# ============================ CLI ============================

def 主():
    import argparse
    参 = argparse.ArgumentParser(description="ChatBI 身份令牌（B-4）")
    参.add_argument("--签", action="store_true", help="签发令牌")
    参.add_argument("--角色", help="角色：%s" % "、".join(权限.角色表))
    参.add_argument("--主体", help="谁（工号 / 姓名，仅留痕用）")
    参.add_argument("--有效秒", type=int, help="有效期（默认 %d）" % 默认有效秒)
    参.add_argument("--验", help="验一个令牌")
    参.add_argument("--统计", action="store_true")
    参.add_argument("--校验", action="store_true")
    参数 = 参.parse_args()

    if 参数.校验:
        问题 = 校验配置()
        问题 += 权限.校验配置()
        if not 问题:
            print("✅ 身份配置自洽")
            return 0
        for x in 问题:
            print("🔴 %s" % x)
        return 1

    if 参数.签:
        if not 参数.角色:
            print("--签 需要 --角色（%s）" % "、".join(权限.角色表))
            return 2
        try:
            令牌 = 签(参数.角色, 参数.主体, 参数.有效秒)
        except 身份错 as 错:
            print("🔴 %s" % 错)
            return 1
        print(令牌)
        if 用开发密钥():
            print("⚠️  用的是**内置开发密钥** —— 谁都能签出一样的令牌。"
                  "要让令牌真的有效，先设 CHATBI_TOKEN_SECRET。", file=sys.stderr)
        return 0

    if 参数.验:
        try:
            载荷 = 验(参数.验)
        except 身份错 as 错:
            print("🔴 验不过：%s" % 错)
            return 1
        print(json.dumps({**载荷,
                          "过期可读": time.strftime("%Y-%m-%d %H:%M:%S",
                                                 time.localtime(载荷["过期"]))},
                         ensure_ascii=False, indent=2))
        return 0

    print(json.dumps({**统计(), "角色": list(权限.角色表)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(主())
