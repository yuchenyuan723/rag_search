"""混合检索与答案组织：向量 + BM25 融合召回，证据句级筛选，出处标注。"""
from __future__ import annotations

import json
import pickle
import re
import urllib.parse
from pathlib import Path

import numpy as np

from . import config as _cfg
from .bm25 import expand_tokens, tokenize
from .compare import available_metrics, comparison_for_question
from .config import (
    COMPANY_BOOST,
    COMPANY_COMPARE_FLOOR,
    CORPUS_DIR,
    NUMERIC_CHUNK_BONUS,
    EVIDENCE_DIGIT_BONUS,
    EVIDENCE_CHUNKS,
    EVIDENCE_CHUNKS_COMPARE,
    EVIDENCE_MAX_LEN,
    EVIDENCE_MIN_SCORE,
    EVIDENCE_SENT_CANDIDATES,
    EXAMPLE_QUESTIONS,
    TOP_CHUNKS,
    TOP_CHUNKS_RANGE,
    TOP_EVIDENCE,
    TOP_EVIDENCE_COMPARE,
    W_BM25,
    W_DENSE,
)
from .qwen3 import Qwen3Embedder
from .llm import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    EVIDENCE_MAX_CHARS,
    LLM_ENABLED,
    LLMError,
    StepFunClient,
)

_NUMERIC_Q = re.compile(r"多少|金额|比例|率|同比|环比|增长|下降|变化|对比|相比|比较|结构|构成|占比")
_COMPARE_Q = re.compile(
    r"哪家|哪些|谁|排名|最高|最低|对比|相比|比较|分别|行业|平均|之间|十大|前十|top"
)
_LONG_NUM = re.compile(r"\d{4,}")
_SENT_SPLIT = re.compile(r"(?<=[。！？；])|\n")
_DIGIT = re.compile(r"\d")
_BOILERPLATE = re.compile(
    r"(半年度报告全文|年度报告全文|季度报告全文|董事会批准报送日期|备查文件目录|"
    r"公司代码[:：]|股票代码[:：]|□适用|□\s*适用|公司负责人[:：]|主管会计工作负责人|"
    r"会计机构负责人|本报告书|释义$)"
)


_CONTEXT_WORDS = {"半年度", "上半年", "年度", "年报", "半年报", "报告期", "公司", "股份"}


def evidence_tokens(question: str, company_tokens: set[str] | None = None) -> list[str]:
    """证据句打分时剔除“上下文词”（公司名、年份、报告期），聚焦问题实体。"""
    # 公司名只用于别名加权与侧栏筛选：留在排序词里，释义表和子公司名录
    # 中反复出现的公司简称会把答案挤出前列；年/上半年等时间词同理，
    # 关联交易预计表满屏都是“2026年上半年预计金额”。
    words = set(_CONTEXT_WORDS)
    for t in tokenize(question):
        if len(t) == 4 and t.isdigit():
            words.add(t)
    words |= set(company_tokens or ())
    return [t for t in tokenize(question) if t not in words and len(t) >= 2]


def normalize01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def soft01(x: np.ndarray) -> np.ndarray:
    """把 BM25 原始分压到 (0,1)，避免最大值归一化让单一冠军吃满权重。"""
    pos = x[x > 0]
    k = float(np.median(pos)) if pos.size else 1.0
    k = max(k, 1.0)
    return x / (x + k)


def split_sentences(text: str) -> list[str]:
    """把切块文本切成可作证据的句子。

    PDF 视觉行会把长句拆断，这里把未以句末标点结尾的行与下一行合并；
    表格数据行统一拼上表头，离开表格上下文后仍可读。
    """
    _END = ("。", "！", "？", "；", "：", "”", "）", ")")
    _LIST_START = re.compile(
        r"^([一二三四五六七八九十]+\s*[、．.]|[（(][一二三四五六七八九十\d]+[)）]|第[一二三四五六七八九十百\d]+[节章])"
    )
    _SEP = set("-| ")

    def is_sep(p: str) -> bool:
        return p.startswith("|") and set(p) <= _SEP

    raw = [p for p in (s.strip() for s in _SENT_SPLIT.split(text)) if p]

    # 先把 [表头, 分隔行, 数据行...] 折叠成“表头 → 数据行”
    # 并把表格前方最近的正文小标题拼到表头前，让数字行带上主题语境。
    # 小标题要累积最近两行：“(2) 营业收入、营业成本的分解信息”后面通常
    # 紧跟一行“单位：元 币种：人民币”，只留一行就会把小标题丢掉。
    # 注意：上下文要拼在行首“|”内侧，行本身仍以 | 开头，
    # 下面的长句合并逻辑才能继续识别出这是表格行
    folded: list[str] = []
    caption = ""
    recent: list[str] = []
    i = 0
    while i < len(raw):
        p = raw[i]
        if i + 1 < len(raw) and p.startswith("|") and is_sep(raw[i + 1]):
            header = p
            if caption and caption not in header:
                header = "|" + caption + " |" + header[1:]
            i += 2
            while i < len(raw) and raw[i].startswith("|"):
                if is_sep(raw[i]):
                    i += 1
                    continue
                folded.append(header + " → " + raw[i])
                i += 1
            continue
        if p.startswith("|"):
            # 无边框表格：孤立行也挂上小标题，离开语境才读得懂
            if caption and caption not in p:
                p = "|" + caption + " →" + p[1:]
            folded.append(p)
            i += 1
            continue
        # “√适用 □不适用”这类选择行不是小标题，不能把它后面的正文标题
        # 清掉：否则“(2) 营业收入、营业成本的分解信息”这一标题就丢了，
        # 数据行失去语境后会被证据句判成“不含问题关键词”
        if "适用" in p:
            i += 1
            continue
        if len(p) <= 40:
            recent.append(p)
            if len(recent) > 2:
                recent.pop(0)
        else:
            recent = []
        caption = " | ".join(recent)
        folded.append(p)
        i += 1

    # 合并被视觉行拆断的长句
    merged: list[str] = []
    for p in folded:
        if (
            merged
            and not merged[-1].endswith(_END)
            and not merged[-1].startswith("|")
            and not p.startswith("|")
            and not _LIST_START.match(p)
        ):
            merged[-1] += p
        else:
            merged.append(p)

    out: list[str] = []
    for p in merged:
        p = p.strip(" |")
        if not p:
            continue
        if p.startswith("-") and set(p) <= _SEP:
            continue
        if len(p) < 6 and not _DIGIT.search(p):
            continue
        if _BOILERPLATE.search(p):
            continue
        out.append(p)
    return out


class KnowledgeBase:
    def __init__(self, data_dir=None, model_dir=None, lazy_model: bool = True):
        self.data_dir = Path(data_dir or _cfg.DATA_DIR)
        self.model_dir = model_dir
        self._embedder: Qwen3Embedder | None = None
        self.meta = self._load_meta()
        self.chunks = self._load_chunks()
        self.vectors = self._load_vectors()
        with open(self.data_dir / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        self.llm = StepFunClient() if LLM_ENABLED else None
        self.company_keys = []
        for c in self.chunks:
            if c["company"] not in self.company_keys:
                self.company_keys.append(c["company"])
        # 公司别名与公司名词集合：
        # - 别名用于问题点名公司时加权，只认长度 >= 2 的完整别名，
        #   不再用 display[:-2] 这种会把“比亚迪”截成“比”的后缀 hack；
        # - 公司名词集合用于把公司名从 BM25 查询词中剔除，
        #   否则释义表/子公司名录里重复的公司简称会淹没答案段。
        self.company_aliases: dict[str, list[str]] = {}
        company_tokens: set[str] = set()
        for key in self.company_keys:
            display = next(c["company_display"] for c in self.chunks if c["company"] == key)
            aliases = {key, display}
            if len(display) >= 4:
                aliases.add(display[:-2])
            aliases = {a for a in aliases if len(a) >= 2}
            self.company_aliases[key] = sorted(aliases, key=len, reverse=True)
            for a in self.company_aliases[key]:
                company_tokens.update(tokenize(a))
        self.company_tokens = company_tokens
        if not lazy_model:
            _ = self.embedder

    # ------------------------------------------------------------ 加载
    def _load_meta(self) -> dict:
        p = self.data_dir / "meta.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def _load_chunks(self) -> list[dict]:
        p = self.data_dir / "chunks.jsonl"
        chunks = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        return chunks

    def _load_vectors(self) -> np.ndarray:
        v = np.load(self.data_dir / "embeddings.npy")
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (v / norms).astype("float32")

    @property
    def embedder(self) -> Qwen3Embedder:
        if self._embedder is None:
            self._embedder = Qwen3Embedder(self.model_dir)
        return self._embedder

    # ------------------------------------------------------------ 元信息
    def meta_info(self) -> dict:
        companies: dict[str, dict] = {}
        for c in self.chunks:
            entry = companies.setdefault(
                c["company"],
                {"key": c["company"], "display": c["company_display"], "docs": [], "n_chunks": 0},
            )
            entry["n_chunks"] += 1
            doc = next((d for d in entry["docs"] if d["doc_id"] == c["doc_id"]), None)
            if doc is None:
                entry["docs"].append({
                    "doc_id": c["doc_id"],
                    "doc_type": c["doc_type"],
                    "period": c["period"],
                })
        return {
            "n_chunks": len(self.chunks),
            "n_docs": len(self.meta.get("docs", [])),
            "built_at": self.meta.get("built_at", ""),
            "model": self.meta.get("model", ""),
            "embed_dim": int(self.vectors.shape[1]) if self.vectors.size else 0,
            "weights": {"dense": W_DENSE, "bm25": W_BM25},
            "companies": list(companies.values()),
            "llm": {
                "enabled": LLM_ENABLED,
                "available": bool(self.llm and self.llm.available),
                "model": DEFAULT_MODEL,
            },
            "metrics": available_metrics(),
            "example_questions": EXAMPLE_QUESTIONS,
            "ready": True,
        }

    # ------------------------------------------------------------ 检索
    def _mask(self, companies, doc_types) -> np.ndarray:
        mask = np.ones(len(self.chunks), dtype=bool)
        if companies:
            mask &= np.array([c["company"] in companies for c in self.chunks])
        if doc_types:
            mask &= np.array([c["doc_type"] in doc_types for c in self.chunks])
        return mask

    def _named_companies(self, question: str) -> list[str]:
        """问题里点名的公司。别名只认长度 >= 2 的完整词，不用后缀截断。"""
        return [k for k, als in self.company_aliases.items()
                if any(a in question for a in als)]

    def search(self, question, companies=None, doc_types=None, mode="hybrid", top_k=TOP_CHUNKS):
        q_vec = self.embedder.encode_one(question, is_query=True)
        dense = self.vectors @ q_vec
        # BM25 排序词只用问题实词（见 evidence_tokens）：剔除公司名、时间词
        # 和单字，避免预测表/释义表靠重复词刷分
        content = evidence_tokens(question, self.company_tokens)
        bm25_raw = self.bm25.score(expand_tokens(content))
        bm25_n = soft01(bm25_raw)
        numeric_q = bool(_NUMERIC_Q.search(question))
        compare_q = bool(_COMPARE_Q.search(question))
        named = self._named_companies(question)

        if mode == "dense":
            fused = dense.copy()
        elif mode == "bm25":
            fused = bm25_n.copy()
        else:
            fused = W_DENSE * dense + W_BM25 * bm25_n
        # 数值型问题优先数字密集的段落：会计政策/释义这类反复解释概念的
        # 文本里几乎没有长数字，答案一定在带数字的表格或摘要行里
        if numeric_q:
            fused = fused + np.array([
                NUMERIC_CHUNK_BONUS if bm25_raw[i] > 0 and len(_LONG_NUM.findall(c["text"])) >= 3 else 0.0
                for i, c in enumerate(self.chunks)
            ], dtype="float32")

        mask = self._mask(companies, doc_types)
        # 只要点名了公司就限定在这些公司内（比较类问题同样如此）：
        # “赛力斯的营业收入构成中哪些业务占比最高”问的是它内部的结构，
        # 别家公司的分解信息表、营业收入表都不该混进来
        if named:
            keys = set(named)
            mask &= np.array([c["company"] in keys for c in self.chunks])
        # 问题中点名公司的小幅加权
        for key, aliases in self.company_aliases.items():
            if any(a in question for a in aliases):
                fused[mask & np.array([c["company"] == key for c in self.chunks])] += COMPANY_BOOST

        fused = np.where(mask, fused, -1.0)
        order = np.argsort(-fused)[: max(1, int(top_k))]

        def _pack(i: int) -> dict:
            return {
                "idx": int(i),
                "chunk": self.chunks[int(i)],
                "dense": float(dense[i]),
                "bm25_norm": float(bm25_n[i]),
                "fused": float(fused[i]),
            }

        results = []
        for i in order:
            if fused[i] < 0:
                continue
            results.append(_pack(i))
        if compare_q:
            # 比较类问题（哪家/最高/对比/哪些业务）先给每家公司留一个
            # 名额——取该公司融合分最高的一段，剩余名额仍按融合分补。
            # 早期版本把各公司代表段提到最前、再把已在全局前列的块判重删掉，
            # 结果排名第一的块反而整体丢失。这里改成“代表段 + 全局排序”合并。
            scope = set(named) if named else set(self.company_keys)
            reps: list[int] = []
            for key in self.company_keys:
                if key not in scope:
                    continue
                sub = np.where(
                    mask & np.array([c["company"] == key for c in self.chunks]),
                    fused, -1.0,
                )
                best = int(np.argmax(sub))
                if sub[best] >= 0.0 and fused[best] >= COMPANY_COMPARE_FLOOR:
                    reps.append(best)
            # 比较题要的是“十家公司各自的数”，默认 top_k=8 装不下十家，
            # 这里把上限抬到参与比较的公司数（不超过滑块最大值 12）
            k = min(max(int(top_k), len(scope)), TOP_CHUNKS_RANGE[1])
            reps.sort(key=lambda i: -fused[i])
            rep_set = set(reps)
            merged = reps + [r["idx"] for r in results if r["idx"] not in rep_set]
            results = [_pack(i) for i in merged[:k]]
        return results, q_vec

    # ------------------------------------------------------------ 证据句
    def _evidence(self, results, q_vec, question: str,
                  max_chunks: int = EVIDENCE_CHUNKS,
                  max_evidence: int = TOP_EVIDENCE) -> list[dict]:
        picks: list[tuple[int, int, str]] = []
        pos = 0
        for r in results[:max_chunks]:
            for s in split_sentences(r["chunk"]["text"]):
                picks.append((r["idx"], pos, s))
                pos += 1
        if not picks:
            return []
        # 相邻切块有重叠，同一表格行会重复出现：先去重再算分，
        # 既省一次向量化，也避免证据列表里出现两条一模一样的句子
        # 比较时去掉“表头 →”前缀：同一行数据在相邻切块里挂的表头可能不同
        uniq: list[tuple[int, int, str]] = []
        seen_text: set[str] = set()
        for p in picks:
            body = p[2].split("→", 1)[1] if "→" in p[2] else p[2]
            if body in seen_text:
                continue
            seen_text.add(body)
            uniq.append(p)
        picks = uniq
        q_tokens_lex = expand_tokens(evidence_tokens(question, self.company_tokens))
        fused_of = {r["idx"]: r["fused"] for r in results}
        # 先用廉价的 BM25 句分 + 切块分粗排，只对候选句做向量化，控制查询耗时
        toks = [tokenize(s) for _, _, s in picks]
        bm = np.array([
            self.bm25.score_text_against(q_tokens_lex, t, b=0.0) for t in toks
        ], dtype="float32")
        prerank = bm + np.array([0.5 * fused_of[ci] for ci, _, _ in picks], dtype="float32")
        # 证据句必须覆盖问题实词，未覆盖的句子（同表格的流动比率、邻表的子公司
        # 名录）最终也会被过滤掉，那就没必要花时间向量化它们：长安“资产负债率”
        # 一问候选句里只有一行命中，编码量从 28 句降到 1 句
        if q_tokens_lex and bm.any():
            cand = [i for i in range(len(picks)) if bm[i] > 0]
        else:
            cand = list(range(len(picks)))
        cand.sort(key=lambda i: -float(prerank[i]))
        shortlist = [picks[i] for i in cand[:EVIDENCE_SENT_CANDIDATES]]
        sel_bm = bm[cand[:EVIDENCE_SENT_CANDIDATES]] if cand else np.zeros(0, dtype="float32")

        sent_vecs = self.embedder.encode(
            [s for _, _, s in shortlist], max_len=EVIDENCE_MAX_LEN
        )
        q_zero = q_vec if q_vec is not None else self.embedder.encode_one(question, is_query=True)
        sent_dense = sent_vecs @ q_zero
        sent_bm25 = sel_bm
        sd_n, sb_n = normalize01(sent_dense), normalize01(sent_bm25)
        chunk_fused = np.array([
            fused_of[ci] for ci, _, _ in shortlist
        ])
        scores = 0.55 * sd_n + 0.45 * sb_n + 0.35 * chunk_fused
        if _NUMERIC_Q.search(question):
            scores += np.array(
                [EVIDENCE_DIGIT_BONUS if _DIGIT.search(s) else 0.0 for _, _, s in shortlist]
            )
        covered = (sel_bm > 0).astype("float32")
        if q_tokens_lex and not covered.any():
            # 没有任何证据句覆盖问题实体：宁可不答，也不要把税率表当答案
            return []

        order_idx = [int(i) for i in np.argsort(-scores)]
        if covered.any():
            # 有句子覆盖问题实词时，同表格的“邻居行”（流动比率、贷款偿还率）
            # 就不该当证据；完全没有覆盖时才退回全局排序
            order_idx = [i for i in order_idx if covered[i] > 0] or order_idx

        kept: list[dict] = []
        for i in order_idx:
            if scores[i] < EVIDENCE_MIN_SCORE:
                if kept or not results:
                    break
                # 没有任何句子过阈值时，至少保留最相关的一句，便于人工判断
            kept.append({
                "idx": shortlist[i][0],
                "text": shortlist[i][2],
                "score": float(scores[i]),
                "order": shortlist[i][1],
            })
            if len(kept) >= max_evidence:
                break

        unique: list[dict] = []
        for e in sorted(kept, key=lambda x: (x["idx"], x["order"])):
            if any(e["text"] in u["text"] or u["text"] in e["text"] for u in unique):
                continue
            unique.append(e)
        return unique

    # ------------------------------------------------------------ 问答
    def compare(self, companies=None, metric: str | None = None, sort: str = "desc") -> dict:
        """跨公司定点取数对比，口径与单位归一化见 app/compare.py。"""
        from .compare import compare_companies
        return compare_companies(self.chunks, companies, metric or "revenue", sort)

    _GEN_SYS = (
        "你是财报问答助手。严格依据下方给出的财报资料回答问题："
        "1. 只使用资料中的数字与事实，禁止推测或引用外部知识；"
        "2. 每个关键数字或结论后用 [编号] 标注出处，编号与资料条目一一对应；"
        "3. 资料无法回答时，明确说明资料中未找到相关信息，不要编造；"
        "4. 使用简体中文，先给结论再分点说明，数字保留原始精度与单位。"
    )
    def _generate_answer(self, question: str, evidence: list[dict], source_order: list[int],
                         max_evidence: int = TOP_EVIDENCE) -> str:
        lines = []
        for e in evidence[:max_evidence]:
            n = source_order.index(e["idx"]) + 1
            c = self.chunks[e["idx"]]
            text = e["text"]
            if len(text) > EVIDENCE_MAX_CHARS:
                text = text[:EVIDENCE_MAX_CHARS] + "…"
            lines.append(f"[{n}] {c['company_display']} {c['period']}{c['doc_type']} 第{c['page']}页：{text}")
        user = "问题：" + question + "\n\n资料：\n" + "\n".join(lines)
        return self.llm.chat(
            self._GEN_SYS,
            user,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )

    def ask(self, question, companies=None, doc_types=None, mode="hybrid",
            top_k=TOP_CHUNKS, generate: bool = True) -> dict:
        results, q_vec = self.search(question, companies, doc_types, mode, top_k)
        q_tokens = tokenize(question)
        # 只有“真的在跨公司比较”时才把每家公司的那一段都拿来当证据：
        # “赛力斯的营业收入构成中哪些业务占比最高”问的是一家内部结构，
        # 放宽到 12 句只会掺进噪声；未点名或点名多家时 AI 生成需要每家都有数字可引
        named = self._named_companies(question)
        cross_company = bool(_COMPARE_Q.search(question)) and len(named) != 1
        if results and cross_company:
            evidence = self._evidence(
                results, q_vec, question,
                max_chunks=min(len(results), EVIDENCE_CHUNKS_COMPARE),
                max_evidence=TOP_EVIDENCE_COMPARE,
            )
        else:
            evidence = self._evidence(results, q_vec, question) if results else []

        source_order: list[int] = []
        for e in evidence:
            if e["idx"] not in source_order:
                source_order.append(e["idx"])
        for r in results:
            if r["idx"] not in source_order:
                source_order.append(r["idx"])

        sources = []
        for pos, idx in enumerate(source_order):
            c = self.chunks[idx]
            pdf_url = "/pdf/" + urllib.parse.quote(c["doc_id"]) + f"#page={c['page']}"
            sources.append({
                "n": pos,
                "idx": idx,
                "doc_id": c["doc_id"],
                "doc_name": c["doc_name"],
                "company": c["company_display"],
                "company_full": c.get("company_full", ""),
                "doc_type": c["doc_type"],
                "period": c["period"],
                "chapter": c["chapter_full"],
                "chapter_short": c["chapter"],
                "page": c["page"],
                "pages": c["pages"],
                "has_table": c["has_table"],
                "n_chars": c["n_chars"],
                "excerpt": c["text"][:300],
                "dense": next((r["dense"] for r in results if r["idx"] == idx), None),
                "bm25_norm": next((r["bm25_norm"] for r in results if r["idx"] == idx), None),
                "fused": next((r["fused"] for r in results if r["idx"] == idx), None),
                "pdf_url": pdf_url,
            })

        generated = None
        llm_error = None
        if generate and self.llm is not None and self.llm.available:
            try:
                generated = self._generate_answer(question, evidence, source_order,
                                                  max_evidence=(TOP_EVIDENCE_COMPARE if cross_company
                                                                else TOP_EVIDENCE))
            except LLMError as exc:
                llm_error = str(exc)

        highlights = sorted(
            set(q_tokens) | set(re.findall(r"\d+(?:\.\d+)?%?", question)),
            key=len, reverse=True,
        )[:15]

        answer_sentences = [
            {
                "idx": e["idx"],
                "text": e["text"],
                "source": source_order.index(e["idx"]),
                "company": self.chunks[e["idx"]]["company_display"],
                "page": self.chunks[e["idx"]]["page"],
                "score": e["score"],
            }
            for e in evidence
        ]

        # 定点取数的对比表（app/compare.py）：检索侧取回的“相关段落”口径不齐，
        # 担保表、母公司附注都混进来，比不出高低；这里按各公司主要会计数据取同一
        # 指标，折算成同一单位并给页码，回答“哪家最高/最低”时有据可依。
        comparison = None
        compare_q = bool(_COMPARE_Q.search(question))
        if compare_q or len(named) != 1:
            if named:
                scope: list[str] = named
            else:
                allowed = set(companies or [])
                scope = [c["company"] for c in self.chunks if not allowed or c["company"] in allowed]
            if len(set(scope)) > 1:
                comparison = comparison_for_question(
                    self.chunks, question, named or companies or None)

        return {
            "question": question,
            "mode": mode,
            "answer": {
                "sentences": answer_sentences,
                "generated": generated,
                "llm_error": llm_error,
            } if answer_sentences else None,
            "sources": sources,
            "comparison": comparison,
            "highlights": highlights,
            "n_total_chunks": len(self.chunks),
        }


