# -*- coding: utf-8 -*-
"""StepFun(阶跃星辰) Chat Completions 客户端：step-5-preview 生成式回答。

密钥只从环境变量读取：LLM_API_KEY 或 STEPFUN_API_KEY。项目根目录的 .env
（模板见 .env.example）即写入这两个键之一，.env 已在 .gitignore 中，不会提交，
仓库里也不留存任何密钥文件。只用标准库 urllib，不引入额外依赖。
任何失败都抛 LLMError，由调用方回退到原文摘句，不阻塞问答。
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_MODEL = os.environ.get("FIN_LLM_MODEL", "step-5-preview")
DEFAULT_URL = os.environ.get(
    "FIN_LLM_URL",
    "https://api.stepfun.ai" + "/step_plan/v1/chat/completions",
)
DEFAULT_TIMEOUT = float(os.environ.get("FIN_LLM_TIMEOUT", "150"))
DEFAULT_MAX_TOKENS = int(os.environ.get("FIN_LLM_MAX_TOKENS", "4096"))
DEFAULT_TEMPERATURE = float(os.environ.get("FIN_LLM_TEMPERATURE", "0.2"))
LLM_ENABLED = os.environ.get("FIN_LLM_ENABLED", "1") not in ("0", "false", "False")
EVIDENCE_MAX_CHARS = int(os.environ.get("FIN_LLM_EVIDENCE_CHARS", "220"))


class LLMError(RuntimeError):
    """LLM 调用失败(无密钥、网络错误、接口返回错误等)。"""


def get_api_key() -> str | None:
    """读取 StepFun API Key；未配置时返回 None(不启用生成)。"""
    for name in ("LLM_API_KEY", "STEPFUN_API_KEY"):
        value = (os.environ.get(name) or "").strip().strip('"').strip("'")
        if value and value.lower() not in ("none", "null", "changeme", "your_key_here"):
            return value
    return None


class StepFunClient:
    """最小化的 chat/completions 封装(OpenAI 兼容格式)。"""

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL,
                 base_url: str = DEFAULT_URL, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.api_key = api_key if api_key is not None else get_api_key()
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat(self, system: str, user: str, temperature: float = 0.2,
             max_tokens: int = 1024) -> str:
        if not self.api_key:
            raise LLMError("未配置 API Key(请在项目根目录 .env 中设置 LLM_API_KEY)")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        req = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")[:300]
            raise LLMError(f"LLM 接口 HTTP {exc.code}: {detail}") from exc
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"LLM 接口请求失败: {exc}") from exc

        try:
            data = json.loads(body)
            choice = data["choices"][0]
            msg = choice["message"]
            text = str(msg.get("content") or "").strip()
            if not text:
                text = str(msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
            if not text:
                raise LLMError("接口返回空内容(finish_reason=" + str(choice.get("finish_reason")) + ")")
            return text
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise LLMError(f"LLM 接口返回格式异常: {body[:300]}") from exc
