"""一键入口：缺少索引时自动构建，然后启动问答服务。

用法：
    python run.py                 # 索引已存在则直接启动服务
    python run.py --rebuild       # 强制重建索引（PDF 解析有缓存，增量很快）
    python build_index.py         # 只构建索引，不启动服务
"""
from __future__ import annotations

import argparse

from app.config import BM25_FILE, CHUNKS_FILE, EMB_FILE
from app.index import build
from app.server import create_app


def index_ready() -> bool:
    return all(p.exists() for p in (CHUNKS_FILE, EMB_FILE, BM25_FILE))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--rebuild", action="store_true", help="强制重建索引")
    args = parser.parse_args()

    if args.rebuild or not index_ready():
        print("索引不完整，开始构建 …")
        build()
    app = create_app()
    print(f"问答服务已启动：http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
