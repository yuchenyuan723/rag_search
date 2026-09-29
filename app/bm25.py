"""BM25（Okapi）倒排索引，中文分词使用 jieba。"""
from __future__ import annotations

import math
import re
from collections import Counter

import jieba

from .textnorm import norm_text

STOPWORDS = set("""
的 了 和 是 在 我 你 他 她 它 我们 你们 他们 这 那 这个 那个 这些 那些
就 都 也 还 与 及 或 而 被 把 对 为 以 于 从 到 等 等等 之一 一个 一些
什么 怎么 怎样 如何 哪些 哪个 多少 几 请 请问 告诉 想 想要 了解 一下
呢 吗 啊 呀 吧 是否 有没有 能否 可以 显示 列出 给出 看看 说明 介绍
以及 并且 但是 因为 所以 如果 虽然 然后 现在 目前 当时 其中 以上 以下
报告 情况 数据 信息 内容 方面 相关 关于 根据 通过 进行 存在 属于
""".split())

FINANCE_WORDS = """
营业收入 营业成本 归母净利润 扣除非经常性损益 扣非净利润 同比增长 同比下降
环比 毛利率 净利率 资产负债率 经营活动产生的现金流量净额 经营现金流
研发投入 研发费用 管理费用 销售费用 财务费用 基本每股收益 稀释每股收益
加权平均净资产收益率 总资产 净资产 营业收入构成 主营业务 乘用车 新能源汽车
产销量 产能 销量 整车 零部件 期间费用 所得税费用 利润总额 营业利润
非经常性损益 政府补助 递延收益 商誉 存货 应收账款 短期借款 长期借款
实际控制人 股东大会 董事会 监事会 独立董事 会计师事务所 审计意见
标准无保留意见 重大资产重组 关联交易 或有事项 期后事项 股东总数
前十大股东 控股股东 股份回购 利润分配 现金分红 半年度报告 年度报告
管理层讨论与分析 公司治理 重要提示 备查文件 现金流量表 资产负债表
利润表 财务报表 附注 会计政策 会计估计 重大风险 核心竞争力 经营计划
发展战略 品牌向上 高端化 智能化 电动化 出海 出口 定点 车型 新车
上市 交付 上险 批发 零售 库存 周转 减值 汇兑 政府补贴 税收优惠
归属于母公司股东的净利润 归属于母公司所有者的净利润 归属于上市公司股东的净利润
扣除非经常性损益的净利润 营业收入构成 收入构成 主营业务分析 研发投入总额
所得税税率 政府补助 现金流量净额 职工薪酬 信用减值损失 分解信息
""".split()

for _w in FINANCE_WORDS:
    jieba.add_word(_w)

_TOKEN_CLEAN = re.compile(r"[^0-9a-zA-Z一-龥%]")


_SYNONYMS_RAW = {
    "营业收入": ["营业总收入", "营收", "收入"],
    "营业总收入": ["营业收入", "营收"],
    "净利润": ["归母净利润", "净利", "归属于母公司股东的净利润", "归属于母公司所有者的净利润",
                "归属于上市公司股东的净利润", "扣除非经常性损益的净利润"],
    "归母净利润": ["净利润", "扣非净利润", "扣除非经常性损益"],
    "研发投入": ["研发费用", "研发支出", "研发投入总额"],
    "销量": ["销售量", "销售数量"],
    "营业收入构成": ["主营业务构成", "收入构成", "主营业务分析", "分解信息"],
    "政府补助": ["政府补贴"],
    "所得税税率": ["税率"],
    "现金流量净额": ["经营现金流"],
    "毛利率": ["销售毛利率"],
    "资产负债率": ["负债率"],
    "经营现金流": ["经营活动产生的现金流量净额", "现金流量净额"],
    "总资产": ["资产总额"],
    "总负债": ["负债总额"],
    "每股收益": ["EPS"],
    "新能源": ["电动汽车", "电动车"],
}

SYNONYMS: dict[str, list[str]] = {}
for _k, _vs in _SYNONYMS_RAW.items():
    SYNONYMS.setdefault(_k, [])
    SYNONYMS[_k].extend(_vs)
    for _v in _vs:
        SYNONYMS.setdefault(_v, [])
        if _k not in SYNONYMS[_v]:
            SYNONYMS[_v].append(_k)


# 同义词表里的词同样要进分词词典：语料里的“营业总收入 / 资产总额”默认会被切成
# “营业 / 总收入”“资产 / 总额”，查询侧即便扩展出这些词也匹配不上，
# BM25 索引里自然也就看不到它们
for _k, _vs in SYNONYMS.items():
    jieba.add_word(_k)
    for _v in _vs:
        jieba.add_word(_v)


def expand_tokens(tokens: list[str]) -> list[str]:
    """为查询词补充财报同义词，缓解“营业收入/营业总收入”这类措辞差异。"""
    out = list(tokens)
    for t in tokens:
        out.extend(SYNONYMS.get(t, []))
    seen: set[str] = set()
    dedup: list[str] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            dedup.append(t)
    return dedup


def tokenize(text: str) -> list[str]:
    text = norm_text(text)
    toks = []
    for w in jieba.lcut(text):
        w = _TOKEN_CLEAN.sub("", w).lower()
        if not w or w in STOPWORDS:
            continue
        toks.append(w)
    return toks


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs: list[list[str]] = []
        self.tf: list[Counter] = []
        self.doc_len: list[int] = []
        self.avgdl = 0.0
        self.df: Counter = Counter()
        self.idf: dict[str, float] = {}
        self.postings: dict[str, list[int]] = {}
        self.n = 0

    def fit(self, docs_tokens: list[list[str]]) -> "BM25Index":
        self.docs = [list(t) for t in docs_tokens]
        self.n = len(self.docs)
        self.doc_len = [len(t) for t in self.docs]
        self.avgdl = sum(self.doc_len) / max(1, self.n)
        self.tf = [Counter(t) for t in self.docs]
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        self.df = df
        self.idf = {
            t: math.log(1 + (self.n - v + 0.5) / (v + 0.5)) for t, v in df.items()
        }
        self.postings = {t: [] for t in df}
        for di, c in enumerate(self.tf):
            for t in c:
                self.postings[t].append(di)
        return self

    def score(self, query_tokens: list[str]):
        import numpy as np

        scores = np.zeros(self.n, dtype="float32")
        if not self.n:
            return scores
        for t in sorted(set(query_tokens)):
            idf = self.idf.get(t)
            if not idf:
                continue
            for di in self.postings.get(t, ()):
                f = self.tf[di][t]
                dl = self.doc_len[di]
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                scores[di] += idf * f * (self.k1 + 1) / denom
        return scores

    def score_text_against(self, query_tokens: list[str], doc_tokens: list[str],
                           b: float | None = None) -> float:
        """用全库 IDF 给单个短文本（证据句）打 BM25 分。

        b=None 时沿用索引自带的长度归一；句子只有一两行的情况下，
        长度归一会让短句靠“短”占便宜，证据句比较时传 b=0 关掉它。
        """
        if not doc_tokens or not self.avgdl:
            return 0.0
        b = self.b if b is None else b
        tf = Counter(doc_tokens)
        dl = len(doc_tokens)
        total = 0.0
        for t in set(query_tokens):
            idf = self.idf.get(t)
            if not idf:
                continue
            f = tf.get(t, 0)
            if not f:
                continue
            denom = f + self.k1 * (1 - b + b * dl / (self.avgdl or 1.0))
            total += idf * f * (self.k1 + 1) / denom
        return total
