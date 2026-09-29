"""公司财报知识库问答：PDF 解析、混合检索、抽取式作答。"""
# 放在这里的原因：任何 app.* 子模块被导入前都会先执行本文件，
# 因此 .env 一定早于 app.config / app.llm 读取环境变量之前生效。
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path | None = None) -> None:
    """把项目根目录 .env 写进 os.environ；已存在的环境变量优先，不被覆盖。"""
    env_file = Path(path or (Path(__file__).resolve().parent.parent / ".env"))
    if not env_file.is_file():
        return
    try:
        text = env_file.read_text(encoding="utf-8-sig")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip('"').strip("'").strip()
        if name and name not in os.environ:
            os.environ[name] = value


load_dotenv()
