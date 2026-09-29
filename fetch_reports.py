"""财报 PDF 下载入口：等价于 python tools/fetch_reports.py。

用法和参数完全一致，例如：
    python fetch_reports.py --list                    # 只列出能匹配到的公告，不下载
    python fetch_reports.py                           # 下载缺失的报告
    python fetch_reports.py --year 2024 --overwrite   # 换一年重新抓
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))

from fetch_reports import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
