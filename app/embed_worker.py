"""向量化 worker：由索引构建进程按分片启动，写入 partial npy。

用法：python -m app.embed_worker <start> <end>
环境变量：FIN_DATA_DIR、FIN_EMB_THREADS、FIN_EMB_BATCH
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import config as _cfg
from .qwen3 import Qwen3Embedder


def main() -> None:
    start, end = int(sys.argv[1]), int(sys.argv[2])
    data_dir = Path(_cfg.DATA_DIR)
    chunks = []
    with open(data_dir / "chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    texts = [c["embed_input"] for c in chunks[start:end]]
    if not texts:
        return
    threads = int(os.environ.get("FIN_EMB_THREADS", "0")) or None
    emb = Qwen3Embedder(threads=threads)
    t0 = time.time()
    vecs = emb.encode(texts)
    out = data_dir / f"partial_emb_{start}_{end}.npy"
    np.save(out, vecs)
    print(f"worker {start}-{end}: {len(texts)} chunks, {time.time() - t0:.0f}s -> {out.name}")


if __name__ == "__main__":
    main()
