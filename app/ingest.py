"""财报 PDF 解析：逐页抽取文本行与表格，表格按行列还原为 Markdown。

每个解析块（Block）带页码与当时的章节路径，供切块时作为元数据。
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pdfplumber

from .textnorm import norm_text

# ---------------------------------------------------------------- 文档元信息

_REPORT_WORDS = [
    "半年度报告", "年度报告", "季度报告", "半年度", "中期报告", "半年报",
    "年报", "季报", "中期", "报告",
]
_YEAR = re.compile(r"(19|20)\d{2}")
_YEAR_TAIL = re.compile(r"年+$")


def parse_doc_meta(path: Path) -> dict:
    stem = path.stem
    m = _YEAR.search(stem)
    year = m.group(0)[:4] if m else ""
    name = stem.replace(m.group(0), "") if m else stem
    for w in _REPORT_WORDS:
        name = name.replace(w, "")
    name = re.sub(r"[：:;；\-—_（）()\s]+", "", name) or stem
    name = _YEAR_TAIL.sub("", name) or stem

    if "季" in stem and "半年" not in stem and "年度" not in stem:
        doc_type = "季报"
    elif "半年" in stem or "中期" in stem:
        doc_type = "半年报"
    elif "年报" in stem or "年度" in stem:
        doc_type = "年报"
    else:
        doc_type = "报告"

    if doc_type == "年报":
        period = f"{year}年度" if year else ""
    elif doc_type == "半年报":
        period = f"{year}年上半年" if year else ""
    elif doc_type == "季报":
        period = f"{year}年" if year else ""
    else:
        period = year

    return {
        "doc_id": path.name,
        "name": stem,
        "file": path.name,
        "path": str(path),
        "company": name,
        "doc_type": doc_type,
        "period": period,
    }


def discover_docs(corpus_dir: Path) -> list[dict]:
    docs = []
    for p in sorted(Path(corpus_dir).iterdir()):
        if p.suffix.lower() != ".pdf":
            continue
        docs.append(parse_doc_meta(p))
    return docs


_COMPANY_FULL = re.compile(r"([一-龥]{2,15}股份有限公司)")
_STOCK_CODE = re.compile(r"股票代码[:：\s]*([0-9A-Z]{4,8})")
_PERIOD_YEAR = re.compile(r"(19|20)\d{2}\s*年")


def infer_period_year(text: str) -> str:
    """从封面/目录文本推断报告年度，文件名缺年份时兜底。"""
    from collections import Counter

    counter = Counter(m.group(0)[:4] for m in _PERIOD_YEAR.finditer(text or ""))
    if not counter:
        return ""
    year, _ = counter.most_common(1)[0]
    return year


def build_period(doc_type: str, year: str) -> str:
    if not year:
        return ""
    if doc_type == "年报":
        return f"{year}年度"
    if doc_type == "半年报":
        return f"{year}年上半年"
    if doc_type == "季报":
        return f"{year}年"
    return year


def extract_company_info(first_pages_text: str) -> dict:
    info = {"company_full": "", "stock_code": ""}
    m = _COMPANY_FULL.search(first_pages_text)
    if m:
        info["company_full"] = m.group(1)
    m = _STOCK_CODE.search(first_pages_text)
    if m:
        info["stock_code"] = m.group(1)
    return info


# ---------------------------------------------------------------- 章节识别

_H1 = re.compile(r"^第[一二三四五六七八九十百0-9]{1,4}[节章部分篇]\s*\S")
_H2 = re.compile(r"^[一二三四五六七八九十]{1,3}\s*[、．.]\s*\S")
_H3 = re.compile(r"^[（(][一二三四五六七八九十]{1,3}[)）]\s*\S")
_H4 = re.compile(r"^\d{1,2}\s*[)）.](?!\d)\s*\S")
_HEADINGS = ((1, _H1), (2, _H2), (3, _H3), (4, _H4))
_HEAD_MAX_LEN = {1: 40, 2: 30, 3: 28, 4: 28}
_TOC_TAIL = re.compile(r"(\.{3,}|…{2,}|。{3,})\s*[0-9一二三四五六七八九十]{0,4}\s*$")
_END_PUNCT = set("。；;，,、：:")


def heading_level(line: str) -> int:
    s = line.strip()
    if not s or len(s) > 42:
        return 0
    if s[-1] in _END_PUNCT:
        return 0
    if _TOC_TAIL.search(s):
        return 0
    for lvl, pat in _HEADINGS:
        if pat.match(s) and len(s) <= _HEAD_MAX_LEN[lvl]:
            return lvl
    return 0


class ChapterTracker:
    """按 节/一、/（一）/1. 四级维护章节路径。"""

    def __init__(self) -> None:
        self.levels = ["", "", "", ""]

    def feed(self, line: str) -> bool:
        lvl = heading_level(line)
        if not lvl:
            return False
        self.levels[lvl - 1] = line.strip()
        for i in range(lvl, 4):
            self.levels[i] = ""
        return True

    def path(self) -> str:
        return " / ".join(x for x in self.levels if x)


def short_chapter(path: str) -> str:
    parts = [p for p in path.split(" / ") if p]
    return " / ".join(parts[-2:]) if parts else "未分节"


# ---------------------------------------------------------------- 页眉页脚 / 目录行

_FOOTER_FULL = [
    r"^第?\s*\d{1,4}\s*页$",
    r"^\d{1,4}\s*/\s*\d{1,4}$",
    r"^[\d\s]{1,10}$",
    r"^\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日$",
]
_REPORT_TAIL = re.compile(r"(年度报告|半年度报告|季度报告|中期报告)$")
_REPORT_HEAD = re.compile(r"^(公司)?\s*\d{0,4}\s*(年度|半年度|季度)?报告")


def is_chrome(line: str, top: float, height: float) -> bool:
    s = line.strip()
    if not s:
        return True
    if top < 72 or top > height - 50:
        if any(re.fullmatch(p, s) for p in _FOOTER_FULL):
            return True
        if len(s) <= 36 and _REPORT_TAIL.search(s):
            return True
        if len(s) <= 26 and _REPORT_HEAD.match(s):
            return True
    return False


def is_toc_line(s: str) -> bool:
    return bool(_TOC_TAIL.search(s))


def clean_blocks(blocks: list[Block]) -> list[Block]:
    """剔除页眉页脚噪声行（页码 1/143、页脚日期等），解析层未覆盖的格式在此兜底。"""
    out: list[Block] = []
    for b in blocks:
        if b.kind != "text":
            out.append(b)
            continue
        s = b.text.strip()
        if not s:
            continue
        if any(re.fullmatch(p, s) for p in _FOOTER_FULL):
            continue
        if len(s) <= 36 and _REPORT_TAIL.search(s):
            continue
        out.append(b)
    return out


# ---------------------------------------------------------------- 表格还原

MAX_COLS = 14
MAX_ROWS = 300
MAX_CELL = 60
_NUM_CELL = re.compile(r"^[-+]?[\d,，.]{1,16}%?$")


def _clean_cell(c) -> str:
    s = norm_text(str(c) if c is not None else "")
    s = s.replace("\n", " ").strip()
    s = re.sub(r"\s{2,}", " ", s)
    return s[:MAX_CELL]


def table_to_markdown(rows: list) -> str | None:
    rows = [[_clean_cell(c) for c in row] for row in rows]
    rows = [r for r in rows if any(c for c in r)]
    if len(rows) < 2:
        return None
    ncol = max(len(r) for r in rows)
    if ncol < 2 or ncol > MAX_COLS or len(rows) > MAX_ROWS:
        return None
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    header = rows[0] if any(rows[0]) else [f"列{i + 1}" for i in range(ncol)]
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * ncol) + "|"]
    for r in rows[1:]:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def looks_numeric_table(rows: list) -> bool:
    rows = [r for r in rows if any((c or "").strip() for c in r)]
    if len(rows) < 3:
        return False
    ncol = max(len(r) for r in rows)
    if ncol < 3:
        return False
    cells = [(c or "").strip() for r in rows for c in r]
    cells = [c for c in cells if c]
    if not cells:
        return False
    numeric = sum(1 for c in cells if _NUM_CELL.match(c.replace("%", "")) or _NUM_CELL.match(c))
    return numeric / len(cells) >= 0.35


# ---------------------------------------------------------------- 整页解析

@dataclass
class Block:
    kind: str      # text / table
    text: str
    page: int
    chapter: str


def _page_items(page) -> tuple[list[dict], list[dict]]:
    """返回 (文本行, 表格)。表格优先用线条策略，找不到时回退文本对齐策略。"""
    height = page.height or 842
    lines: list[dict] = []
    try:
        for ln in page.extract_text_lines() or []:
            text = norm_text(ln.get("text") or "")
            if not text:
                continue
            lines.append({
                "text": text,
                "top": float(ln.get("top") or 0),
                "bottom": float(ln.get("bottom") or 0),
                "x0": float(ln.get("x0") or 0),
                "x1": float(ln.get("x1") or 0),
            })
    except Exception:
        lines = []

    tables: list[dict] = []
    try:
        for t in page.find_tables():
            md = table_to_markdown(t.extract())
            if md:
                tables.append({"top": float(t.bbox[1]), "bbox": t.bbox, "text": md})
    except Exception:
        pass

    if not tables:
        # 无框线表格回退：要求至少 3 列 3 行且数值占比足够高
        try:
            for t in page.find_tables(table_settings={
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "snap_tolerance": 4,
                "intersection_tolerance": 5,
                "min_words_vertical": 2,
                "min_words_horizontal": 1,
            }):
                rows = t.extract()
                md = table_to_markdown(rows)
                if md and looks_numeric_table(rows):
                    tables.append({"top": float(t.bbox[1]), "bbox": t.bbox, "text": md})
        except Exception:
            pass
    return lines, tables


def parse_pdf(path: Path) -> tuple[list[Block], int, dict]:
    blocks: list[Block] = []
    tracker = ChapterTracker()
    first_pages_text: list[str] = []
    n_pages = 0
    with pdfplumber.open(str(path)) as pdf:
        n_pages = len(pdf.pages)
        for pno, page in enumerate(pdf.pages, start=1):
            height = page.height or 842
            lines, tables = _page_items(page)
            if pno <= 3:
                first_pages_text.extend(ln["text"] for ln in lines[:40])

            items: list[dict] = []
            for t in tables:
                items.append({"kind": "table", "top": t["top"], "text": t["text"]})
            for ln in lines:
                inside = any(
                    t["bbox"][0] - 2 <= ln["x0"] <= t["bbox"][2] + 2
                    and t["bbox"][1] - 2 <= ln["top"] <= t["bbox"][3] + 2
                    for t in tables
                )
                if inside:
                    continue
                items.append({"kind": "text", "top": ln["top"], "text": ln["text"]})
            items.sort(key=lambda x: x["top"])

            for it in items:
                text = it["text"]
                if it["kind"] == "text":
                    if is_toc_line(text):
                        continue
                    top = it["top"]
                    # 页眉页脚过滤（仅页缘区域）
                    if is_chrome(text, top, height):
                        continue
                    tracker.feed(text)
                blocks.append(Block(it["kind"], text, pno, tracker.path()))

    company_info = extract_company_info("\n".join(first_pages_text))
    company_info["period_year"] = infer_period_year("\n".join(first_pages_text))
    return blocks, n_pages, company_info


def blocks_to_json(blocks: Iterable[Block]) -> list[dict]:
    return [asdict(b) for b in blocks]
