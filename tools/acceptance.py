"""验收跑批：跑十道验收题，把召回块/证据句/对比表原样导出成 JSON，供人工判卷。

验收报告.md 里的判定就是在这份原始数据上做出的；重跑本脚本可复现原始召回
（耗时会变，检索结果不变）：
    python tools/acceptance.py
"""
import io
import json
import sys
import time

sys.path.insert(0, ".")
from app.retrieve import KnowledgeBase

QUESTIONS = [
    ("Q1", "上汽集团2026年上半年营业收入是多少元？同比增长多少？"),
    ("Q2", "比亚迪2026年上半年的归母净利润和经营活动现金流净额是多少？"),
    ("Q3", "长安汽车2026年6月末的资产负债率是多少？"),
    ("Q4", "赛力斯2026年上半年经营活动产生的现金流量净额是多少？"),
    ("Q5", "长城汽车2025年研发投入总额是多少？占营业收入比例多少？"),
    ("Q6", "北汽蓝谷2026年上半年的基本每股收益是多少？"),
    ("Q7", "江淮汽车2026年上半年营业收入是多少元？同比增长多少？"),
    ("Q8", "广汽集团2026年上半年新能源汽车销量是多少？同比增长多少？"),
    ("Q9", "十家公司中2026年上半年营业收入最高的是哪家？"),
    ("Q10", "哪家公司2026年上半年的归母净利润最高？哪家的经营活动现金流净额最高？"),
]


def main():
    kb = KnowledgeBase()
    _ = kb.embedder
    out = []
    for qid, q in QUESTIONS:
        t0 = time.time()
        res = kb.ask(q, mode="hybrid", top_k=8, generate=False)
        entry = {
            "id": qid,
            "question": q,
            "latency_ms": int((time.time() - t0) * 1000),
            "sentences": [
                {
                    "company": s["company"],
                    "page": s["page"],
                    "score": round(s["score"], 3),
                    "text": s["text"],
                }
                for s in ((res.get("answer") or {}).get("sentences") or [])
            ],
            "sources": [
                {
                    "n": s["n"],
                    "company": s["company"],
                    "doc_type": s["doc_type"],
                    "period": s["period"],
                    "page": s["page"],
                    "chapter": s["chapter"],
                    "fused": round(s["fused"], 4) if s["fused"] is not None else None,
                    "dense": round(s["dense"], 4) if s["dense"] is not None else None,
                    "bm25_norm": round(s["bm25_norm"], 4) if s["bm25_norm"] is not None else None,
                    "excerpt": s["excerpt"],
                }
                for s in res["sources"]
            ],
        }
        cmp = res.get("comparison")
        if cmp:
            entry["comparison"] = {
                "metric_label": cmp["metric_label"],
                "unit": cmp["unit"],
                "sort": cmp["sort"],
                "rows": [
                    {
                        "company": r["company_display"],
                        "period": r["period"],
                        "value": round(r["value"], 2),
                        "raw": "%.2f %s" % (r["value_raw"], r["raw_unit"]),
                        "page": r["page"],
                    }
                    for r in cmp["rows"]
                ],
                "missing": [m["company_display"] for m in cmp["missing"]],
            }
        out.append(entry)
        print("%s done %dms sentences=%d sources=%d comparison=%s"
              % (qid, entry["latency_ms"], len(entry["sentences"]),
                 len(entry["sources"]), "Y" if cmp else "N"))
    io.open("tools/acceptance_raw.json", "w", encoding="utf-8").write(
        json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
