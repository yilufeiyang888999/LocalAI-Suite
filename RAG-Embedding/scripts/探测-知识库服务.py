# -*- coding: utf-8 -*-
"""
知识库服务冒烟（作品一 · D4）
==============================

为什么不用 curl：Windows 上 shell 传中文 JSON 会撞编码（本项目已多次踩过），
用 urllib 直接发 UTF-8 字节最稳。

用法：
    python scripts/探测-知识库服务.py                 # 默认 4 题
    python scripts/探测-知识库服务.py --端口 8092 --题 T01 O03
    python scripts/探测-知识库服务.py --不生成         # 只看检索与拒答
"""

import json
import sys
import time
import urllib.request

默认题 = [
    ("T01", "镀锌件装饰性镀层的厚度要求是多少"),      # 库内·表格 → 应正常答
    ("K01", "判断库存是否低于安全库存该用哪个口径"),   # 库内·短小关键 → 应正常答
    ("O03", "电镀废液具体怎么处理"),                  # 库外·有线索 → 应被类2 拦
    ("O01", "公司今年的营业收入是多少"),              # 库外·没线索 → 应被类1 拦
]


def 取参数(名, 默认=None):
    if 名 in sys.argv:
        i = sys.argv.index(名)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return 默认


def 主流程():
    sys.stdout.reconfigure(errors="replace")
    端口 = 取参数("--端口", "8092")
    基础 = "http://127.0.0.1:%s" % 端口
    要生成 = "--不生成" not in sys.argv

    题表 = 默认题
    if "--题" in sys.argv:
        i = sys.argv.index("--题")
        编号表 = [x for x in sys.argv[i + 1:] if not x.startswith("--")]
        题表 = [t for t in 默认题 if t[0] in 编号表] or 默认题

    print("=" * 96)
    print(" 知识库服务冒烟    %s    生成 %s" % (基础, "开" if 要生成 else "关"))
    print("=" * 96)

    with urllib.request.urlopen(基础 + "/health", timeout=10) as 响应:
        h = json.loads(响应.read().decode("utf-8"))
    print(" /health → " + json.dumps(h, ensure_ascii=False))

    for 编号, 问句 in 题表:
        print()
        print("-" * 96)
        print(" %s  「%s」" % (编号, 问句))
        载荷 = json.dumps({"question": 问句, "generate": 要生成},
                          ensure_ascii=False).encode("utf-8")
        请求 = urllib.request.Request(基础 + "/ask", data=载荷,
                                     headers={"Content-Type": "application/json"})
        起 = time.time()
        try:
            with urllib.request.urlopen(请求, timeout=300) as 响应:
                d = json.loads(响应.read().decode("utf-8"))
        except Exception as 错:                                   # noqa: BLE001
            print("   ✗ 请求失败：%s" % 错)
            continue

        print("   拒答 %s   类型 %s   最高分 %.2f   耗时 %s"
              % ("✅" if d["refused"] else "—", d["refusal_type"] or "—",
                 d["top_score"], d["timings"]))
        if d["refusal_reason"]:
            print("   理由：%s" % d["refusal_reason"])
        if d["off_library_refs"]:
            print("   库外引用：%s" % "、".join(d["off_library_refs"]))
        print("   引用（%d）：" % len(d["citations"]))
        for c in d["citations"]:
            print("     [%d] %-14s %.2f  %s" % (c["rank"], c["doc_no"], c["score"], c["section"]))
        print("   答案：%s" % (d["answer"] or "(空)"))
        print("   （客户端计时 %.2f s）" % (time.time() - 起))
    return 0


if __name__ == "__main__":
    sys.exit(主流程())
