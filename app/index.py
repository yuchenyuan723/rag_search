"""索引构建流水线：PDF 解析（可缓存）→ 切块 → 向量化 → BM25。"""
from __future__ import annotations

import json
import pickle
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import config as _cfg
from .bm25 import BM25Index, tokenize
from .chunking import build_chunks
from .config import COMPANY_DISPLAY, MODEL_DIR, W_BM25, W_DENSE
from .ingest import Block, build_period, clean_blocks, discover_docs, parse_pdf
from .qwen3 import Qwen3Embedder


def extract_docs(corpus_dir: Path, extract_dir: Path, verbose: bool = True) -> list[dict]:
    """解析 PDF 并缓存到 data/extract/，重复构建时跳过已解析文档。"""
    extract_dir.mkdir(parents=True, exist_ok=True)
    docs = discover_docs(corpus_dir)
    if verbose:
        print(f"发现 {len(docs)} 份 PDF：{[d['name'] for d in docs]}")
    for doc in docs:
        cache = extract_dir / f"{doc['doc_id']}.json"
        if cache.exists():
            raw = json.loads(cache.read_text(encoding="utf-8"))
            doc["n_pages"] = raw.get("n_pages", 0)
            doc["blocks"] = [vars(b) for b in clean_blocks([Block(**b) for b in raw.get("blocks", [])])]
            doc["company_full"] = raw.get("company_full", "")
            doc["stock_code"] = raw.get("stock_code", "")
            if not doc["period"]:
                doc["period"] = raw.get("period", "")
            if verbose:
                print(f"  缓存命中 {doc['name']}（{doc['n_pages']} 页）")
            continue
        t0 = time.time()
        blocks, n_pages, company_info = parse_pdf(Path(doc["path"]))
        doc.update(company_info)
        blocks = clean_blocks(blocks)
        if not doc["period"]:
            doc["period"] = build_period(doc["doc_type"], company_info.get("period_year", ""))
        if not doc["period"] and company_info.get("period_year"):
            doc["period"] = company_info["period_year"]
        doc["n_pages"] = n_pages
        doc["blocks"] = [vars(b) for b in blocks]
        payload = {
            "n_pages": n_pages,
            "company_full": company_info.get("company_full", ""),
            "stock_code": company_info.get("stock_code", ""),
            "period": doc["period"],
            "blocks": doc["blocks"],
        }
        cache.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        if verbose:
            chars = sum(len(b["text"]) for b in doc["blocks"])
            print(
                f"  解析 {doc['name']}：{n_pages} 页 / {len(doc['blocks'])} 块 / "
                f"{chars} 字符 / {time.time() - t0:.0f}s"
            )
    return docs


def embed_sharded(chunks: list[dict], shards: int, verbose: bool = True) -> np.ndarray:
    """多进程分片向量化：每个 worker 负责一段切块，最后按顺序拼接。"""
    from .config import EMB_DIM

    n = len(chunks)
    bounds = [(i * n // shards, (i + 1) * n // shards) for i in range(shards)]
    bounds = [(s, e) for s, e in bounds if e > s]
    if not bounds:
        return np.zeros((0, EMB_DIM), dtype="float32")
    threads = max(2, (os.cpu_count() or 4) // len(bounds))
    env = dict(os.environ, FIN_EMB_THREADS=str(threads))
    procs = []
    for start, end in bounds:
        if verbose:
            print(f"  分片 worker {start}-{end}（{end - start} 块，{threads} 线程）")
        procs.append((start, end, subprocess.Popen(
            [sys.executable, "-m", "app.embed_worker", str(start), str(end)], env=env,
        )))
    parts = []
    for start, end, proc in procs:
        rc = proc.wait()
        part = _cfg.DATA_DIR / f"partial_emb_{start}_{end}.npy"
        if rc != 0 or not part.exists():
            raise RuntimeError(f"分片 {start}-{end} 向量化失败（exit={rc}）")
        parts.append(np.load(part))
        part.unlink()
    return np.concatenate(parts, axis=0)


def build(corpus_dir=None, model_dir=None, verbose: bool = True) -> dict:
    shards = max(1, int(os.environ.get("FIN_EMB_SHARDS", "1")))

    _cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    docs = extract_docs(Path(corpus_dir or _cfg.CORPUS_DIR), _cfg.EXTRACT_DIR, verbose=verbose)

    chunks: list[dict] = []
    for doc in docs:
        blocks = [Block(**b) for b in doc["blocks"]]
        doc_chunks = build_chunks(doc, blocks)
        chunks.extend(doc_chunks)
        if verbose:
            print(f"  切块 {doc['name']}：{len(doc_chunks)} 块")
    if verbose:
        print(f"共 {len(chunks)} 个切块")

    with open(_cfg.CHUNKS_FILE, "w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            c["id"] = i
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    if shards > 1:
        if verbose:
            print(f"向量化（Qwen3-Embedding，CPU，{shards} 分片并行）…")
        vectors = embed_sharded(chunks, shards=shards, verbose=verbose)
    else:
        embedder = Qwen3Embedder(model_dir)
        if verbose:
            print(f"向量化（Qwen3-Embedding，batch={embedder.batch_size}，CPU）…")

        def prog(done: int, total: int) -> None:
            if verbose:
                print(f"\r  进度 {done}/{total}  ({time.time() - t0:.0f}s)", end="", flush=True)

        vectors = embedder.encode([c["embed_input"] for c in chunks], progress=prog)
        if verbose:
            print()
    np.save(_cfg.EMB_FILE, vectors)

    bm25 = BM25Index().fit([tokenize(c["text"] + " " + c["chapter_full"]) for c in chunks])
    with open(_cfg.BM25_FILE, "wb") as f:
        pickle.dump(bm25, f)

    meta = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": str(Path(model_dir or MODEL_DIR)),
        "embed_dim": int(vectors.shape[1]) if vectors.size else 0,
        "n_chunks": len(chunks),
        "n_docs": len(docs),
        "build_seconds": round(time.time() - t0, 1),
        "weights": {"dense": W_DENSE, "bm25": W_BM25},
        "docs": [
            {
                "doc_id": d["doc_id"],
                "name": d["name"],
                "company": d["company"],
                "company_display": COMPANY_DISPLAY.get(d["company"], d["company"]),
                "company_full": d.get("company_full", ""),
                "doc_type": d["doc_type"],
                "period": d["period"],
                "n_pages": d.get("n_pages", 0),
                "n_chunks": sum(1 for c in chunks if c["doc_id"] == d["doc_id"]),
            }
            for d in docs
        ],
    }
    _cfg.META_FILE.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if verbose:
        print(f"索引完成，用时 {time.time() - t0:.0f}s，写入 {_cfg.DATA_DIR}")
    return meta
