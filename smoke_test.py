"""索引冒烟测试：直接调用 KnowledgeBase 跑若干问题，检查检索与证据质量。"""
import os
import time

from app.retrieve import KnowledgeBase

QUESTIONS = [
    "上汽集团2026年上半年营业收入是多少？同比增长多少？",
    "比亚迪2026年上半年的净利润是多少？",
    "长城汽车2025年报的研发投入是多少？",
    "广汽集团的新能源汽车销量怎么样？",
    "哪家公司的资产负债率最高？",
]

if __name__ == "__main__":
    kb = KnowledgeBase()
    print(f"载入 {len(kb.chunks)} 个切块\n")
    qs = QUESTIONS[:1] if os.environ.get("SMOKE_FIRST") else QUESTIONS
    for q in qs:
        t0 = time.time()
        res = kb.ask(q, mode="hybrid", top_k=5)
        ms = int((time.time() - t0) * 1000)
        print("=" * 78)
        print(f"Q: {q}  [{ms} ms]")
        for s in (res["answer"] or {}).get("sentences", []):
            print(f"  [{s['source'] + 1}] ({s['company']} P{s['page']}, {s['score']:.2f}) {s['text'][:90]}")
        gen = (res["answer"] or {}).get("generated")
        err = (res["answer"] or {}).get("llm_error")
        if gen:
            print(f"  [AI] {gen[:700]}")
        elif err:
            print(f"  [AI-ERR] {err}")
        for s in res["sources"][:3]:
            print(f"     src{s['n']}: {s['company']} {s['doc_type']} {s['period']} | {s['chapter'][:38]} | 第{s['page']}页"
                  f" | dense={s['dense']:.3f} bm25={s['bm25_norm']:.3f} fused={s['fused']:.3f}")
        print()
