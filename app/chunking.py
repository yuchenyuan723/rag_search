"""切块：按段落边界打包解析块，保留公司/类型/期间/章节/页码元数据。"""
from __future__ import annotations

from dataclasses import dataclass

from .config import (
    CHUNK_MAX,
    CHUNK_MIN,
    CHUNK_OVERLAP,
    CHUNK_TARGET,
    COMPANY_DISPLAY,
    EMBED_CONTENT_CAP,
)
from .ingest import Block, short_chapter


@dataclass
class PackedBlock:
    kind: str
    text: str
    page: int
    chapter: str


def pack_blocks(
    blocks: list[Block],
    target: int = CHUNK_TARGET,
    max_len: int = CHUNK_MAX,
    overlap: int = CHUNK_OVERLAP,
) -> list[list[PackedBlock]]:
    """按段落块贪心打包；相邻切块之间保留少量重叠文本。"""
    packed: list[list[PackedBlock]] = []
    cur: list[PackedBlock] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur, cur_len
        if not cur:
            return
        packed.append(cur)
        tail = ""
        if overlap > 0:
            text = "\n".join(b.text for b in cur)
            if len(text) > overlap:
                tail = text[-overlap:]
                nl = tail.find("\n")
                if nl > 0:
                    tail = tail[nl + 1:]
        last = cur[-1]
        cur = [PackedBlock("text", tail, last.page, last.chapter)] if tail else []
        cur_len = len(cur[0].text) if cur else 0

    for b in blocks:
        pb = PackedBlock(b.kind, b.text, b.page, b.chapter)
        if cur and cur_len + len(b.text) + 1 > max_len:
            flush()
        cur.append(pb)
        cur_len += len(b.text) + 1
        if cur_len >= target:
            flush()
    flush()
    return packed


def merge_small(packed: list[list[PackedBlock]], min_len: int = CHUNK_MIN) -> list[list[PackedBlock]]:
    out: list[list[PackedBlock]] = []
    for buf in packed:
        text = "\n".join(b.text for b in buf)
        if out and len(text) < min_len and buf[0].page == out[-1][-1].page:
            out[-1].extend(buf)
        else:
            out.append(list(buf))
    return out


def build_chunks(doc: dict, blocks: list[Block]) -> list[dict]:
    packed = merge_small(pack_blocks(blocks))
    display = COMPANY_DISPLAY.get(doc["company"], doc["company"])
    chunks: list[dict] = []
    for buf in packed:
        text = "\n".join(b.text for b in buf)
        pages = sorted({b.page for b in buf})
        chapter_full = buf[-1].chapter or buf[0].chapter
        embed_input = (
            f"{display} {doc['doc_type']} {doc['period']}\n{chapter_full}\n"
            f"{text[:EMBED_CONTENT_CAP]}"
        )
        chunks.append({
            "doc_id": doc["doc_id"],
            "doc_name": doc["name"],
            "company": doc["company"],
            "company_display": display,
            "company_full": doc.get("company_full", ""),
            "doc_type": doc["doc_type"],
            "period": doc["period"],
            "chapter": short_chapter(chapter_full),
            "chapter_full": chapter_full,
            "page": pages[0],
            "pages": pages,
            "n_chars": len(text),
            "has_table": any(b.kind == "table" for b in buf),
            "text": text,
            "embed_input": embed_input,
        })
    return chunks
