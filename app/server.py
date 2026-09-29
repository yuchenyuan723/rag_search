"""Flask 服务：静态问答页面 + /api/ask 检索接口。"""
from __future__ import annotations

import time
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file, send_from_directory

from . import config as _cfg
from .config import TOP_CHUNKS, TOP_CHUNKS_RANGE
from .retrieve import KnowledgeBase


def create_app(kb: KnowledgeBase | None = None) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["JSON_AS_ASCII"] = False
    app.json.ensure_ascii = False
    knowledge = kb or KnowledgeBase()

    @app.get("/")
    def index():
        return send_from_directory(_cfg.STATIC_DIR, "index.html")

    @app.get("/api/meta")
    def meta():
        return jsonify(knowledge.meta_info())

    @app.post("/api/compare")
    def compare():
        """跨公司定点取数：同一指标、同一单位、可回溯页码的横向对比。"""
        data = request.get_json(force=True, silent=True) or {}
        companies = list(data.get("companies") or [])
        metric = str(data.get("metric") or "revenue")
        sort = "asc" if data.get("sort") == "asc" else "desc"
        try:
            result = knowledge.compare(companies=companies, metric=metric, sort=sort)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(result)

    @app.get("/api/health")
    def health():
        llm = getattr(knowledge, "llm", None)
        return jsonify({
            "ok": True,
            "chunks": len(knowledge.chunks),
            "llm": bool(llm and llm.available),
        })

    @app.post("/api/ask")
    def ask():
        data = request.get_json(force=True, silent=True) or {}
        question = (data.get("question") or "").strip()
        if not question:
            return jsonify({"error": "问题不能为空"}), 400
        try:
            top_k = int(data.get("top_k") or TOP_CHUNKS)
        except (TypeError, ValueError):
            top_k = TOP_CHUNKS
        lo, hi = TOP_CHUNKS_RANGE
        top_k = max(lo, min(hi, top_k))

        t0 = time.time()
        try:
            result = knowledge.ask(
                question,
                companies=data.get("companies") or [],
                doc_types=data.get("doc_types") or [],
                mode=data.get("mode") or "hybrid",
                top_k=top_k,
                generate=bool(data.get("generate", True)),
            )
        except Exception as exc:  # 兜底，避免前端拿到 500 白屏
            return jsonify({"error": f"检索失败：{exc}"}), 500
        result["latency_ms"] = int((time.time() - t0) * 1000)
        return jsonify(result)

    @app.get("/pdf/<path:name>")
    def pdf(name: str):
        base = Path(_cfg.CORPUS_DIR).resolve()
        target = (base / name).resolve()
        if not str(target).startswith(str(base)) or not target.is_file():
            abort(404)
        return send_file(target)

    return app


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="公司财报知识库问答服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print("正在加载索引与模型 …")
    kb = KnowledgeBase()
    _ = kb.embedder  # 预热模型，避免首个提问等待加载
    app = create_app(kb)
    print(f"问答服务已启动：http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
