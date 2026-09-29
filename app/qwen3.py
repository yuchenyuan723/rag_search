"""本地 Qwen3-Embedding 模型封装：文档向量化 + 查询向量化。"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .config import (
    EMB_BATCH,
    EMB_DIM,
    EMB_MAX_LEN,
    MODEL_DIR,
    QUERY_INSTRUCTION,
)


class Qwen3Embedder:
    def __init__(
        self,
        model_dir=None,
        max_len: int = EMB_MAX_LEN,
        batch_size: int = EMB_BATCH,
        threads: int | None = None,
    ) -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        model_dir = Path(model_dir or MODEL_DIR)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.threads = max(1, threads if threads else (os.cpu_count() or 4))
        torch.set_num_threads(self.threads)
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModel.from_pretrained(str(model_dir), dtype=torch.float32)
        self.model.to(self.device).eval()
        self.max_len = max_len
        self.batch_size = max(1, batch_size)

    def _pool(self, hidden, attention_mask):
        idx = attention_mask.sum(1) - 1
        vec = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
        return torch.nn.functional.normalize(vec, p=2, dim=1)

    def encode(
        self,
        texts: list[str],
        is_query: bool = False,
        progress=None,
        max_len: int | None = None,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, EMB_DIM), dtype="float32")
        prefix = QUERY_INSTRUCTION if is_query else ""
        max_len = max_len or self.max_len
        # 批次按句长排序：tokenizer 会把整批 padding 到批内最长句，
        # 一把长短句混在一起会让短句白白按长句的长度参与计算
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        outs = []
        with torch.no_grad():
            for s in range(0, len(order), self.batch_size):
                batch = [prefix + texts[j] for j in order[s:s + self.batch_size]]
                enc = self.tok(
                    batch, padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**enc).last_hidden_state
                outs.append(self._pool(hidden, enc["attention_mask"]).cpu())
                if progress:
                    progress(min(s + len(batch), len(texts)), len(texts))
        vecs = torch.cat(outs).numpy().astype("float32")
        out = np.empty_like(vecs)
        for pos, j in enumerate(order):
            out[j] = vecs[pos]
        return out

    def encode_one(self, text: str, is_query: bool = False) -> np.ndarray:
        return self.encode([text], is_query=is_query)[0]
