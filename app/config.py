"""项目配置：路径、切块参数、检索权重。"""
from __future__ import annotations

import os
from pathlib import Path

from . import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def env_value(*names: str) -> str:
    """按顺序取第一个非空的环境变量（.env 由 app/__init__.py 载入）。"""
    for name in names:
        value = os.environ.get(name, "")
        if value and value.strip():
            return value.strip()
    return ""


CORPUS_DIR = Path(os.environ.get("FIN_CORPUS_DIR", BASE_DIR / "公司财报"))
MODEL_DIR = Path(
    env_value("MODEL_PATH", "FIN_MODEL_DIR")
    or (BASE_DIR / "models" / "Qwen3-Embedding-0.6B")
)
DATA_DIR = Path(os.environ.get("FIN_DATA_DIR", BASE_DIR / "data"))
EXTRACT_DIR = DATA_DIR / "extract"

CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
EMB_FILE = DATA_DIR / "embeddings.npy"
BM25_FILE = DATA_DIR / "bm25.pkl"
META_FILE = DATA_DIR / "meta.json"

STATIC_DIR = BASE_DIR / "static"

# ---- 切块 ----
CHUNK_TARGET = 480        # 目标长度（字符）
CHUNK_MAX = 720           # 单块上限
CHUNK_MIN = 60            # 低于该长度的块向前合并
CHUNK_OVERLAP = 90        # 相邻块重叠字符数
EMBED_CONTENT_CAP = 800   # 参与向量化的正文上限

# ---- 向量模型 ----
EMB_DIM = 1024
EMB_MAX_LEN = 1024
EMB_BATCH = int(os.environ.get("FIN_EMB_BATCH", "8"))
QUERY_INSTRUCTION = "Instruct: 检索公司财务报告中与问题相关的段落\nQuery: "

# ---- 检索 ----
W_DENSE = 0.6
W_BM25 = 0.4
TOP_CHUNKS = 8
TOP_CHUNKS_RANGE = (3, 12)
EVIDENCE_CHUNKS = 6       # 参与证据句筛选的切块数
TOP_EVIDENCE = 6          # 回答引用的证据句上限
EVIDENCE_CHUNKS_COMPARE = 12  # 比较类问题参与证据筛选的切块数（十家公司都要出场）
TOP_EVIDENCE_COMPARE = 12     # 比较类问题的证据句上限
EVIDENCE_MIN_SCORE = 0.45
EVIDENCE_SENT_CANDIDATES = 28   # 进入向量化粗排的候选句上限
EVIDENCE_MAX_LEN = 384          # 证据句向量化的最大长度
EVIDENCE_DIGIT_BONUS = 0.22     # 数值型问题中含数字证据句的加分
NUMERIC_CHUNK_BONUS = 0.20    # 数值型问题中含多个长数字的切块加分
COMPANY_BOOST = 0.05      # 问题中直接点名公司时的小幅加权
COMPANY_COMPARE_FLOOR = 0.45 # 比较类问题按公司取最优段时的融合分下限

COMPANY_DISPLAY = {
    "上汽": "上汽集团",
    "宇通汽车": "宇通客车",
}

EXAMPLE_QUESTIONS = [
    "上汽集团2026年上半年营业收入是多少？同比增长多少？",
    "比亚迪2026年上半年的净利润和经营现金流是多少？",
    "长城汽车2025年报的研发投入是多少？占营收比例多少？",
    "广汽集团新能源汽车销量表现如何？",
    "长安汽车的资产负债率是多少？",
    "赛力斯的营业收入构成中，哪些业务占比最高？",
    "十家公司中2026年上半年营业收入最高的是哪家？",
    "哪家的归母净利润同比在改善？",
]


