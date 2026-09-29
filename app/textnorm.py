"""文本归一化：全角转半角、去千分位、折叠空白。"""
from __future__ import annotations

import re

_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_WS = re.compile(r"[ \t\u00a0\u2002\u2003]+")


def half_width(s: str) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        if 0xFF10 <= o <= 0xFF19:          # 全角数字
            out.append(chr(o - 0xFEE0))
        elif 0xFF21 <= o <= 0xFF3A or 0xFF41 <= o <= 0xFF5A:  # 全角字母
            out.append(chr(o - 0xFEE0))
        elif o == 0x3000:                  # 全角空格
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def norm_text(s: str | None) -> str:
    s = half_width(s or "")
    s = _THOUSANDS.sub("", s)
    s = _WS.sub(" ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()
