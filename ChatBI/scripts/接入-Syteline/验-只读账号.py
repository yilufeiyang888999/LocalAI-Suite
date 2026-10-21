# -*- coding: utf-8 -*-
"""
验-只读账号.py —— 证明「只读账号确实只读」

为什么要单独验：`.local-凭据.txt` 里写着密码，不代表账号建对了；
`readonly=True` 是客户端自觉，挡不住手滑改成 False。唯一可信的判据是
**让服务端拒绝一次写**，并且拒绝理由必须是权限不足，不是语法错。

用法：
    P="C:/Users/Ivan/AppData/Local/Programs/Python/Python314/python.exe"
    "$P" 验-只读账号.py --主机 172.16.11.254 --用户 chatbi_ro --库 Demo

设计约束（都是踩过的）：
  1. 连接串里不写超时键 —— ODBC Driver 18 会静默吞掉，只回一段噪声。
  2. 中文字面量不进 SQL —— 经 ODBC 会变 '?'，判定看起来全对其实全错。
  3. 探针写操作一律 `WHERE 1 = 0` + 显式回滚：SQL Server 在编译阶段就做权限检查，
     不匹配任何行也会报 permission denied，所以探针不碰数据也能验出权限。
"""

import sys
import argparse
import pathlib

# Windows 控制台默认 cp936，模型名/表名里的非 ASCII 会炸；不许改成 utf-8（会乱码）
for 流 in (sys.stdout, sys.stderr):
    try:
        流.reconfigure(errors="replace")
    except Exception:
        流.reconfigure = None

驱动 = "ODBC Driver 18 for SQL Server"
凭据文件 = pathlib.Path(__file__).with_name(".local-凭据.txt")


def 造串(主机, 端口, 用户, 密码, 库):
    """连接串——超时不在这里写。"""
    return (
        f"DRIVER={{{驱动}}};"
        f"SERVER={主机},{端口};"
        f"DATABASE={库};"
        f"UID={用户};"
        f"PWD={密码};"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
    )


def 读密码(用户):
    """从 .local-凭据.txt 按有效行顺序取：第 1 行 sa，第 2 行 chatbi_ro。"""
    行们 = []
    if 凭据文件.exists():
        for 行 in 凭据文件.read_text(encoding="utf-8", errors="replace").splitlines():
            行 = 行.strip()
            if not 行 or 行.startswith("#"):
                continue
            行们.append(行)
    序号 = 1 if 用户.lower() == "sa" else 2
    return 行们[序号 - 1] if len(行们) >= 序号 else ""


结果 = []


def 记(项, 通过, 证据):
    结果.append((项, 通过, 证据))
    旗 = "PASS" if 通过 else "FAIL"
    print(f"[{旗}] {项} —— {证据}")


def 主():
    解 = argparse.ArgumentParser()
    解.add_argument("--主机", default="172.16.11.254")
    解.add_argument("--端口", type=int, default=1433)
    解.add_argument("--用户", default="chatbi_ro")
    解.add_argument("--库", default="Demo")
    参 = 解.parse_args()

    密码 = 读密码(参.用户)
    if not 密码:
        print(f"× .local-凭据.txt 里取不到 {参.用户} 的密码，无法验证")
        return 2

    import pyodbc

    if 驱动 not in pyodbc.drivers():
        print(f"× 未安装 {驱动}，pyodbc 现有：{pyodbc.drivers()}")
        return 2

    串 = 造串(参.主机, 参.端口, 参.用户, 密码, 参.库)

    # ---- 1. 能不能连上 ----
    try:
        连 = pyodbc.connect(串, readonly=True, timeout=15)
        连.timeout = 60
        记("登录", True, f"{参.用户} 连上 {参.主机}:{参.端口}/{参.库}")
    except Exception as 错:
        print(f"[FAIL] 登录 —— {type(错).__name__}: {错}")
        return 2

    游 = 连.cursor()

    # ---- 2. 身份与角色（证明连的确实是这个账号，不是连接的错觉）----
    try:
        游.execute("SELECT SUSER_SNAME(), IS_SRVROLEMEMBER('sysadmin')")
        名, 超管 = 游.fetchone()
        记("身份", 名.lower() == 参.用户.lower(),
           f"SUSER_SNAME={名}，sysadmin={超管}")
        记("不是超管", 超管 in (0, None), f"IS_SRVROLEMEMBER('sysadmin')={超管}")
    except Exception as 错:
        记("身份", False, f"{type(错).__name__}: {错}")

    # ---- 3. 读权限：元数据 + 一张真实业务表 ----
    try:
        游.execute("SELECT COUNT(*) FROM sys.tables")
        表数 = 游.fetchone()[0]
        记("读元数据", 表数 > 0, f"sys.tables = {表数} 张")
    except Exception as 错:
        记("读元数据", False, f"{type(错).__name__}: {错}")
        表数 = 0

    try:
        游.execute("SELECT TOP 1 item FROM dbo.item_mst")
        行 = 游.fetchone()
        记("读业务表", 行 is not None, f"dbo.item_mst 首行 item={行[0] if 行 else '(空)'}")
    except Exception as 错:
        记("读业务表", False, f"{type(错).__name__}: {错}")

    # ---- 4. 写权限必须被拒（核心判据）----
    # 用独立连接且 readonly=False：readonly 连接会先在客户端拦下，测不出服务端权限。
    try:
        可写连 = pyodbc.connect(串, readonly=False, timeout=15)
        可写连.timeout = 60
        可写游 = 可写连.cursor()
        可写游.execute("BEGIN TRAN")
        try:
            可写游.execute("UPDATE dbo.item_mst SET stat = stat WHERE 1 = 0")
            可写游.execute("ROLLBACK")
            写结果 = "写语句被接受（影响 0 行）"
            写被拒 = False
        except Exception as 错:
            try:
                可写游.execute("ROLLBACK")
            except Exception:
                pass
            写结果 = f"{type(错).__name__}: {str(错)[:160]}"
            写被拒 = any(关键词 in str(错) for 关键词 in ("permission", "拒绝", "denied", "229"))
        可写连.close()
        记("写被拒", 写被拒, 写结果)
    except Exception as 错:
        记("写被拒", False, f"连不上可写连接：{type(错).__name__}: {错}")

    连.close()

    # ---- 汇总：降级/失败必须能与通过区分（退出码分级）----
    全过 = all(通过 for _, 通过, _ in 结果)
    print()
    print(f"共 {len(结果)} 项：通过 {sum(1 for _, p, _ in 结果 if p)} / 失败 {sum(1 for _, p, _ in 结果 if not p)}")
    if 全过:
        print(f"结论：{参.用户} 只读账号可用 —— 能读、不能写。正式问数用它，不用 sa。")
        return 0
    print("结论：未达到「只读可用」，逐项看上面的 FAIL。注意区分「连不上」与「连上但权限不对」。")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(主())
    except KeyboardInterrupt:
        sys.exit(130)
