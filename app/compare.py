"""跨公司对比：从各公司报告的“主要会计数据和财务指标”章节定点抽取同一指标。

为什么单独做一层：混合检索回答比较题时，十家公司拿到的只是“语义相关”的段落。
有的命中对外担保表里的“为资产负债率超过 70% 的对象提供担保”，有的命中母公司
报表附注，口径不一致，排序自然无从谈起。这里改成确定性抽取：只在“主要会计数据
/主要财务指标”章节里找指标行，就近取单位标记，统一折算成同一单位，再按高低
排序；每条数值都带原文行与页码，可回溯核对。
"""
from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass

UNIT_YUAN = "元"
UNIT_WAN = "万元"

# 这些章节里同名的表格大多不是“公司主要会计数据”：分季度表把年度额拆成四段，
# 母公司附注与合并口径不同，非经常性损益表只列项目金额
CHAPTER_SKIP = ("分季度", "非经常性损益", "财务报表主要项目注释", "审计报告", "备查文件", "关联方")
CHAPTER_PREFER = ("主要会计数据", "主要财务指标")

_UNIT_RE = re.compile(r"单位[：:]\s*(万元|元)")
_YUAN_IN_NAME = re.compile(r"[（(]\s*元\s*[)）]")
_NUMBER_RE = re.compile(r"^[-+]?\d[\d,]*(?:\.\d+)?%?$")
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class MetricSpec:
    name: str                  # 指标 id
    label: str                 # 展示名
    aliases: tuple[str, ...]   # 指标行首单元格名，按优先级前缀匹配
    kind: str                  # amount=金额，per_share=每股指标，percent=百分比
    display_unit: str          # 统一展示单位


METRICS: dict[str, MetricSpec] = {
    "revenue": MetricSpec("revenue", "营业收入", ("营业收入", "营业总收入"), "amount", "亿元"),
    "net_profit": MetricSpec(
        "net_profit", "归属于上市公司股东的净利润",
        ("归属于上市公司股东的净利润",), "amount", "亿元"),
    "op_cashflow": MetricSpec(
        "op_cashflow", "经营活动产生的现金流量净额",
        ("经营活动产生的现金流量净额", "经营活动现金流量净额"), "amount", "亿元"),
    "eps": MetricSpec("eps", "基本每股收益", ("基本每股收益",), "per_share", "元/股"),
    "debt_ratio": MetricSpec("debt_ratio", "资产负债率", ("资产负债率",), "percent", "%"),
}

# 问题识别指标的顺序：越具体的放前面，“收入”这类宽泛词最后兜底
_QUESTION_RULES: tuple[tuple[str, re.Pattern], ...] = (
    ("debt_ratio", re.compile(r"资产负债率|负债率")),
    ("op_cashflow", re.compile(r"经营现金流|现金流量净额|现金流")),
    ("eps", re.compile(r"每股收益|每股股利")),
    ("net_profit", re.compile(r"净利润|净利率|归母净利润")),
    ("revenue", re.compile(r"营业收入|营业总收入|营收|收入")),
)

_SCALE = {UNIT_YUAN: 1e8, UNIT_WAN: 1e4}   # 折算成“亿元”


def available_metrics() -> list[dict]:
    return [{"name": s.name, "label": s.label, "unit": s.display_unit} for s in METRICS.values()]


def metric_from_question(question: str) -> str | None:
    for name, pattern in _QUESTION_RULES:
        if pattern.search(question):
            return name
    return None


def _cells(line: str) -> list[str]:
    # PDF 表格视觉换行会把指标名拆断（"归属于上市公司股 东的净利润"），
    # 去掉单元格内所有空白后再做前缀匹配
    return [_SPACE_RE.sub("", c.strip()) for c in line.split("|") if c.strip()]


def _parse_number(text: str) -> float | None:
    t = text.strip()
    if not t or not _NUMBER_RE.match(t):
        return None
    try:
        return float(t.replace(",", "").rstrip("%"))
    except ValueError:
        return None


def _chunk_order(chunks: list[dict]) -> list[dict]:
    """主要会计数据章节优先，其次按文档原顺序，保证取当期口径而不是附注。"""
    def key(item: tuple[int, dict]):
        pos, chunk = item
        chapter = chunk.get("chapter") or ""
        if any(bad in chapter for bad in CHAPTER_SKIP):
            return (2, pos)
        if any(good in chapter for good in CHAPTER_PREFER):
            return (0, pos)
        return (1, pos)

    return [chunk for _, chunk in sorted(enumerate(chunks), key=key)]


def _find_metric(chunk: dict, spec: MetricSpec) -> tuple[float, str, str, int] | None:
    """在 chunk 内找指标行，返回 (数值, 原文行, 压缩行, 行号)。"""
    for idx, line in enumerate(chunk["text"].splitlines()):
        if "|" not in line:
            continue
        cells = _cells(line)
        if len(cells) < 2:
            continue
        for j, cell in enumerate(cells[:-1]):
            if not any(cell.startswith(alias) for alias in spec.aliases):
                continue
            for value_cell in cells[j + 1:]:
                value = _parse_number(value_cell)
                if value is not None:
                    return value, line.strip(), _SPACE_RE.sub("", line), idx
    return None


def _resolve_unit(company_chunks: list[dict], chunk: dict, idx: int, label_line: str,
                  unit_stat: dict[str, int]) -> str:
    """单位就近推断：行名自带“（元）” > 本 chunk 上方 > 前一 chunk > 公司整体。"""
    if _YUAN_IN_NAME.search(label_line):
        return UNIT_YUAN
    lines = chunk["text"].splitlines()
    # 单位标记一般就写在表头那一行，目标行上方 15 行够用
    for line in reversed(lines[max(0, idx - 15): idx + 3]):
        m = _UNIT_RE.search(line)
        if m:
            return m.group(1)
    index = next(i for i, c in enumerate(company_chunks) if c is chunk)
    for prev in reversed(company_chunks[max(0, index - 2):index]):
        for line in reversed(prev["text"].splitlines()[-6:]):
            m = _UNIT_RE.search(line)
            if m:
                return m.group(1)
    if unit_stat.get(UNIT_WAN, 0) > unit_stat.get(UNIT_YUAN, 0):
        return UNIT_WAN
    return UNIT_YUAN


def extract_company(chunks: list[dict], spec: MetricSpec) -> dict | None:
    """对单家公司定点抽取指标；取不到就返回 None，绝不拿别的章节凑数。"""
    if not chunks:
        return None
    unit_stat = {UNIT_YUAN: 0, UNIT_WAN: 0}
    for c in chunks:
        m = _UNIT_RE.search(c["text"])
        if m:
            unit_stat[m.group(1)] += 1
    for chunk in _chunk_order(chunks):
        found = _find_metric(chunk, spec)
        if not found:
            continue
        value, raw_line, compact, idx = found
        if spec.kind != "amount":
            unit = spec.display_unit          # 元/股、% 本身就是统一单位
        else:
            unit = _resolve_unit(chunks, chunk, idx, compact, unit_stat)
        display = value / _SCALE[unit] if spec.kind == "amount" else value
        return {
            "value": display,
            "value_raw": value,
            "raw_unit": unit if spec.kind == "amount" else spec.display_unit,
            "raw_unit_label": spec.display_unit,
            "raw_row": raw_line[:240],
            "page": chunk["page"],
            "chapter": chunk.get("chapter_full") or chunk.get("chapter") or "",
            "doc_id": chunk["doc_id"],
            "doc_type": chunk.get("doc_type", ""),
            "period": chunk.get("period", ""),
            "company": chunk["company"],
            "company_display": chunk.get("company_display", chunk["company"]),
            "company_full": chunk.get("company_full", ""),
            "pdf_url": "/pdf/" + urllib.parse.quote(chunk["doc_id"]) + "#page=" + str(chunk["page"]),
        }
    return None


def compare_companies(chunks: list[dict], companies: list[str] | None, metric: str,
                      sort: str = "desc") -> dict:
    """跨公司取数。companies 为空表示全部公司。"""
    spec = METRICS.get(metric)
    if spec is None:
        raise ValueError(f"未知指标 {metric!r}，可选：{', '.join(METRICS)}")

    by_company: dict[str, list[dict]] = {}
    order: list[str] = []
    for c in chunks:
        key = c["company"]
        if key not in by_company:
            by_company[key] = []
            order.append(key)
        by_company[key].append(c)
    if companies:
        keys = [k for k in companies if k in by_company]
        unknown = [k for k in companies if k not in by_company]
    else:
        keys, unknown = list(order), []

    rows: list[dict] = []
    missing: list[dict] = []
    for key in keys:
        row = extract_company(by_company[key], spec)
        if row is None:
            missing.append({
                "company": key,
                "company_display": by_company[key][0].get("company_display", key),
                "reason": "该公司报告的主要会计数据章节中未直接披露该指标",
            })
            continue
        rows.append(row)

    reverse = sort != "asc"
    rows.sort(key=lambda r: (r["value"] is None, r["value"] or 0.0), reverse=reverse)
    periods = {r["period"] for r in rows}
    return {
        "metric": spec.name,
        "metric_label": spec.label,
        "unit": spec.display_unit,
        "sort": "asc" if not reverse else "desc",
        "n_companies": len(keys),
        "rows": rows,
        "missing": missing,
        "unknown": unknown,
        "period_mix": len(periods) > 1,
        "periods": sorted(periods),
    }


def comparison_for_question(chunks: list[dict], question: str,
                            companies: list[str] | None) -> dict | None:
    """问答链路用：识别到指标就附一份对比表，识别不到就不返回，不影响原有答案。"""
    metric = metric_from_question(question)
    if not metric:
        return None
    try:
        return compare_companies(chunks, companies, metric)
    except ValueError:
        return None
