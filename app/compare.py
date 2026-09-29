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
from collections import Counter
from dataclasses import dataclass

from .textnorm import half_width

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

# ---------------------------------------------------------------- 自定义指标
MAX_METRIC_LEN = 30          # 自定义指标名长度上限

_PERCENT_HINT = re.compile(r"比例|占比|百分比|%|％|率$")
_PER_SHARE_HINT = re.compile(r"每股|单股")
_COUNT_HINT = re.compile(
    r"销量|产量|数量|辆|台|件|人数|员工|研发人员|股数|股份|股东|门店|网点|专利|次数"
)
_UNIT_ANY_RE = re.compile(r"单位[：:]\s*([^\s|，,；;]{1,8})")


def normalize_text(text: str) -> str:
    """全角转半角再去空白：指标名与表格单元格按同一套规则比较。"""
    return _SPACE_RE.sub("", half_width(text or "").strip())


def _alias_candidates(key: str) -> tuple[str, ...]:
    """可比照的指标名形式：原名优先，再补去掉合计/总额等装饰的短形式。"""
    out = [key]
    for suffix in ("合计", "总额", "净额", "总计", "小计"):
        if key.endswith(suffix) and len(key) > len(suffix) + 1:
            base = key[: -len(suffix)]
            if base not in out:
                out.append(base)
    trimmed = key.strip("：:（()）[]【】%％ ")
    if trimmed and trimmed not in out:
        out.append(trimmed)
    seen: set[str] = set()
    uniq = []
    for a in out:
        if a and a not in seen:
            seen.add(a)
            uniq.append(a)
    return tuple(uniq)


def _infer_kind(key: str) -> str:
    """按指标名推断量纲：百分比、每股指标、计数、金额。"""
    if _PERCENT_HINT.search(key):
        return "percent"
    if _PER_SHARE_HINT.search(key):
        return "per_share"
    if _COUNT_HINT.search(key):
        return "count"
    return "amount"


def custom_spec(text: str) -> MetricSpec | None:
    """把用户输入的指标名变成可抽取的 MetricSpec；空名、超长或含表格分隔符时返回 None。"""
    label = (text or "").strip()
    if not label or len(label) > MAX_METRIC_LEN:
        return None
    key = normalize_text(label)
    if not key or any(ch in key for ch in ("|", "\n", "\t")):
        return None
    kind = _infer_kind(key)
    display_unit = {"amount": "亿元", "percent": "%", "per_share": "元/股"}.get(kind, "")
    return MetricSpec("custom", label, _alias_candidates(key), kind, display_unit)


def resolve_spec(metric: str) -> MetricSpec | None:
    """指标名 -> MetricSpec：先查预设（id / 展示名 / 别名词典），查不到按自定义指标处理。"""
    text = (metric or "").strip()
    if text in METRICS:
        return METRICS[text]
    for spec in METRICS.values():
        if text in (spec.name, spec.label) or text in spec.aliases:
            return spec
    return custom_spec(text)




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


UNIT_YI = "亿元"

_UNIT_WORDS = (
    "元", "万元", "亿元", "户", "辆", "台", "件", "个", "人", "次",
    "吨", "千克", "米", "美元", "港元",
)
_PAREN_UNIT_RE = re.compile(r"[（(]\s*([^\s（）()]{1,6})\s*[)）]")


def _normalize_unit_text(text):
    """只认元 / 万元 / 亿元三种金额单位，其余一律当作没写单位。"""
    unit = (text or "").strip("：:，,；; ")
    return unit if unit in (UNIT_YUAN, UNIT_WAN, UNIT_YI) else None


def _paren_unit(text):
    """行名括号里的单位：股东总数(户)、销量（辆）这类写法。"""
    for m in _PAREN_UNIT_RE.finditer(text or ""):
        if m.group(1).strip() in _UNIT_WORDS:
            return m.group(1).strip()
    return None


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


def _scan_metric(chunk: dict, spec: MetricSpec, containment: bool):
    """containment=False 时指标名必须是行首单元格前缀；True 时允许出现在单元格中间。"""
    aliases = spec.aliases
    if containment:
        aliases = tuple(a for a in aliases if len(a) >= 3)
        if not aliases:
            return None
    for idx, line in enumerate(chunk["text"].splitlines()):
        if "|" not in line:
            continue
        cells = _cells(line)
        if len(cells) < 2:
            continue
        for j, cell in enumerate(cells[:-1]):
            if containment:
                hit = any(alias in cell for alias in aliases)
            else:
                hit = any(cell.startswith(alias) for alias in aliases)
            if not hit:
                continue
            for value_cell in cells[j + 1:]:
                value = _parse_number(value_cell)
                if value is not None:
                    return value, line.strip(), _SPACE_RE.sub("", line), idx
    return None


def _find_metric(chunk: dict, spec: MetricSpec):
    """先按前缀找；自定义指标再放宽一层，允许指标名落在行名中间。"""
    found = _scan_metric(chunk, spec, False)
    if found is not None or spec.name != "custom":
        return found
    return _scan_metric(chunk, spec, True)
_TABLE_SEP_RE = re.compile(r"^[\s|:\-]*$")


def _is_table_sep(line: str) -> bool:
    """`|---|---|` 这类分隔行：它上方的单位标记属于上一张表，不能拿来用。"""
    return "|" in line and bool(_TABLE_SEP_RE.match(line.strip()))


def _unit_from(line: str, any_unit: bool) -> str | None:
    """从一行里把单位标记抽出来；any_unit 为真时返回原文（户、辆、件……）。"""
    m = _UNIT_ANY_RE.search(line)
    if not m:
        return None
    unit = m.group(1).strip("：：，，；； ")
    if any_unit:
        return unit or None
    return _normalize_unit_text(unit)


def _unit_near(lines: list[str], idx: int, any_unit: bool) -> str | None:
    """就近找单位标记：先看目标行及其下方几行，再往上扫；遇上分隔行说明换表了，停。"""
    for i in range(idx + 3, idx - 16, -1):
        if i < 0 or i >= len(lines):
            continue
        if i < idx and _is_table_sep(lines[i]):
            break
        unit = _unit_from(lines[i], any_unit)
        if unit:
            return unit
    return None
def _unit_neighbour(chunks: list[dict], index: int, any_unit: bool) -> str | None:
    """表头常被切到相邻 chunk：取前一块里最后一个单位标记、后一块里第一个。

    只看块尾/块头几行会边上边地错过表头：广汽的「财务报表相关科目变动分析表」
    表头写着「单位：亿元」，但它位于前一块的第 5 行，后面全是数据行。
    """
    if index - 1 >= 0:
        unit = None
        for line in chunks[index - 1]["text"].splitlines():
            found = _unit_from(line, any_unit)
            if found:
                unit = found
        if unit:
            return unit
    if index + 1 < len(chunks):
        for line in chunks[index + 1]["text"].splitlines()[:12]:
            found = _unit_from(line, any_unit)
            if found:
                return found
    return None



def _unit_stat(chunks: list[dict]) -> dict[str, int]:
    """公司层面的单位分布：元 / 万元 / 亿元都要计。

    只用 `单位[：:](万元|元)` 投票时，通篇写「单位：亿元」的公司在统计里是 0 票，
    底层就会退回「元」，把 48.34 亿元算成 0.0000 亿元。
    """
    stat = {UNIT_YUAN: 0, UNIT_WAN: 0, UNIT_YI: 0}
    for c in chunks:
        unit = _unit_from(c["text"], False)
        if unit:
            stat[unit] += 1
    return stat


def _resolve_unit(company_chunks: list[dict], chunk: dict, idx: int, label_line: str,
                  unit_stat: dict[str, int]) -> str:
    """单位就近推断：行名自带“（元）” > 本 chunk > 相邻 chunk > 公司整体。"""
    if _YUAN_IN_NAME.search(label_line):
        return UNIT_YUAN
    unit = _unit_near(chunk["text"].splitlines(), idx, False)
    if unit:
        return unit
    index = next(i for i, c in enumerate(company_chunks) if c is chunk)
    unit = _unit_neighbour(company_chunks, index, False)
    if unit:
        return unit
    if unit_stat.get(UNIT_YI, 0) and unit_stat.get(UNIT_YI, 0) >= unit_stat.get(UNIT_WAN, 0):
        return UNIT_YI
    if unit_stat.get(UNIT_WAN, 0) > unit_stat.get(UNIT_YUAN, 0):
        return UNIT_WAN
    return UNIT_YUAN


def _resolve_unit_any(chunk: dict, idx: int) -> str | None:
    """从表头行抓任意单位文本（户、辆、件、台……），给计数类自定义指标用。"""
    return _unit_near(chunk["text"].splitlines(), idx, True)



def extract_company(chunks: list[dict], spec: MetricSpec) -> dict | None:
    """对单家公司定点抽取指标；取不到就返回 None，绝不拿别的章节凑数。"""
    if not chunks:
        return None
    unit_stat = _unit_stat(chunks)
    for chunk in _chunk_order(chunks):
        found = _find_metric(chunk, spec)
        if not found:
            continue
        value, raw_line, compact, idx = found
        if spec.kind == "amount":
            # 金额：按表头单位折算成亿元，十家公司才在同一把尺子上
            unit = _resolve_unit(chunks, chunk, idx, compact, unit_stat)
            # 表头单位是“亿元”时不再除，否则按元 / 万元折算
            display = value if unit == UNIT_YI else value / _SCALE[unit]
            raw_unit = unit
        elif spec.kind == "count":
            # 计数：单位随各公司原文，不折算；口径不一致时上层会提示
            unit = _paren_unit(raw_line) or _resolve_unit_any(chunk, idx) or ""
            display = value
            raw_unit = unit
        else:
            # 百分比、每股指标本身就是统一单位
            unit = spec.display_unit
            display = value
            raw_unit = spec.display_unit
        return {
            "value": display,
            "value_raw": value,
            "raw_unit": raw_unit,
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
    spec = resolve_spec(metric)
    if spec is None:
        raise ValueError(
            "指标名无效：可填预设指标（营业收入 / 归母净利润 / 经营活动现金流净额 / 基本每股收益 / 资产负债率），"
            "或直接输入财报主要财务指标表里的行名，例如「研发投入合计」「毛利率」「合同负债」"
        )

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
    unit_label = spec.display_unit
    unit_mixed = False
    if spec.kind == "count" and rows:
        units = Counter(r["raw_unit"] for r in rows if r["raw_unit"])
        unit_label = units.most_common(1)[0][0] if units else ""
        unit_mixed = len(units) > 1
    return {
        "metric": spec.name,
        "metric_label": spec.label,
        "custom": spec.name == "custom",
        "unit": unit_label,
        "unit_mixed": unit_mixed,
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
