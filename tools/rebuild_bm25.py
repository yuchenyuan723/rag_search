# 只重拟合 BM25 倒排索引（chunks.jsonl -> bm25.pkl），不重算向量。
# 适用于只改了 app/bm25.py 词表/同义词、不想重跑完整构建的场景：
#   python tools/rebuild_bm25.py
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.bm25 import BM25Index, tokenize
from app import config as _cfg


def main() -> None:
    chunks = []
    with open(_cfg.CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print("载入 %d 个切块" % len(chunks))
    t0 = time.time()
    bm25 = BM25Index().fit(
        [tokenize(c["text"] + " " + c["chapter_full"]) for c in chunks]
    )
    with open(_cfg.BM25_FILE, "wb") as f:
        pickle.dump(bm25, f)
    print("重写 %s（%.1fs）" % (_cfg.BM25_FILE, time.time() - t0))


if __name__ == "__main__":
    main()
