# retrieval regression check: chunk hit@8, evidence hit, coverage of evidence
# usage: python tools/eval_retrieval.py
import re
import sys
import time

sys.path.insert(0, ".")
from app.retrieve import KnowledgeBase, split_sentences, evidence_tokens
from app.bm25 import tokenize, expand_tokens

CASES = [
    ("上汽集团2026年上半年营业收入是多少？同比增长多少？", [5, 15]),
    ("比亚迪2026年上半年的净利润和经营现金流是多少？", [1351]),
    ("长城汽车2025年报的研发投入是多少？占营收比例多少？", [2908]),
    ("广汽集团新能源汽车销量表现如何？", [973]),
    ("长安汽车的资产负债率是多少？", [3600]),
    ("赛力斯的营业收入构成中，哪些业务占比最高？", [2357, 2097]),
    ("哪家公司的资产负债率最高？", [615]),
    ("江淮汽车的政府补助是多少？", [1738, 1972, 2008]),
    ("金龙汽车的毛利率是多少？", []),
    ("十家公司中2026年上半年营业收入最高的是哪家？", []),
]


def content_tokens(question):
    return [t for t in tokenize(question) if len(t) >= 2]


def covered(question, text, company_tokens):
    lex = set(expand_tokens(evidence_tokens(question, company_tokens)))
    return bool(set(tokenize(text)) & lex)


if __name__ == "__main__":
    kb = KnowledgeBase()
    _ = kb.embedder
    rows = []
    for q, gold in CASES:
        t0 = time.time()
        res = kb.ask(q, mode="hybrid", top_k=8, generate=False)
        ms = int((time.time() - t0) * 1000)
        top = [s["idx"] for s in res["sources"]]
        ev = [s["idx"] for s in ((res.get("answer") or {}).get("sentences", []))]
        ev_texts = [s["text"] for s in ((res.get("answer") or {}).get("sentences", []))]
        cov = [covered(q, txt, kb.company_tokens) for txt in ev_texts]
        hit_top = bool(set(top) & set(gold)) if gold else None
        hit_ev = bool(set(ev) & set(gold)) if gold else None
        rows.append((q[:24], gold, hit_top, hit_ev, cov, len(ev_texts), ms))
        line = "%-26s gold=%-10s top8=%-5s ev=%-5s cov=%s nev=%d %dms"
        print(line % (q[:24], gold, hit_top, hit_ev, cov, len(ev_texts), ms))
        for e in ev_texts[:3]:
            print("      >", e[:88])
    n_top = sum(1 for r in rows if r[2] is True)
    n_gold = sum(1 for r in rows if r[2] is not None)
    n_ev = sum(1 for r in rows if r[3] is True)
    n_evg = sum(1 for r in rows if r[3] is not None)
    cov_all = [c for r in rows for c in r[4]]
    print()
    print("gold hit@8: %d/%d   gold in evidence: %d/%d   evidence with term coverage: %d/%d"
          % (n_top, n_gold, n_ev, n_evg, sum(1 for c in cov_all if c), len(cov_all)))
    print("avg latency ms: %d" % (sum(r[5] for r in rows) / len(rows)))


