"""统一的 LLM 客户端: OpenAI (gpt-*) 和 Anthropic (claude-*)。

- OpenAI: 通过标准 OPENAI_API_KEY env var
- Anthropic: 走 local billing proxy https://api.anthropic.com, api_key=os.environ.get("ANTHROPIC_API_KEY", "")
  - 不能用 ANTHROPIC_API_KEY env var (OAuth token 不符合 sk-ant-api03 格式校验)
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

try:
    import openai  # type: ignore
except ImportError:
    openai = None  # type: ignore

try:
    import anthropic  # type: ignore
except ImportError:
    anthropic = None  # type: ignore


ANTHROPIC_PROXY_URL = "https://api.anthropic.com"


class LLMClient:
    """轻量统一接口, 支持:
    - chat(system, user, model, max_tokens, temperature) -> raw text
    - chat_json(system, user, ...) -> parsed dict (3 次重试解析)
    """

    def __init__(self, model: str = "gpt-4o-mini", timeout: float = 60.0):
        self.model = model
        self.timeout = timeout
        self._openai = None
        self._anthropic = None

    def _backend(self, model: Optional[str] = None) -> str:
        m = (model or self.model or "").lower()
        if m.startswith(("gpt", "o1", "o3", "o4", "openai")):
            return "openai"
        if m.startswith(("claude", "anthropic")):
            return "anthropic"
        # default
        return "openai"

    # ------------- openai -------------
    def _openai_client(self):
        if self._openai is not None:
            return self._openai
        if openai is None:
            raise RuntimeError("openai SDK 未安装")
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY 未设置")
        self._openai = openai.OpenAI(api_key=key, timeout=self.timeout)
        return self._openai

    # ------------- anthropic -------------
    def _anthropic_client(self):
        if self._anthropic is not None:
            return self._anthropic
        if anthropic is None:
            raise RuntimeError("anthropic SDK 未安装")
        # 推荐方案: api_key=os.environ.get("ANTHROPIC_API_KEY", "") + base_url 走 local proxy
        self._anthropic = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            base_url=ANTHROPIC_PROXY_URL,
            timeout=self.timeout,
        )
        return self._anthropic

    # ------------- chat -------------
    def chat(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
        max_tokens: int = 1500,
        temperature: float = 0.2,
        retries: int = 4,
    ) -> str:
        model = model or self.model
        backend = self._backend(model)

        last_err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                if backend == "openai":
                    client = self._openai_client()
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "system", "content": system},
                                  {"role": "user", "content": user}],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    return resp.choices[0].message.content or ""
                else:
                    client = self._anthropic_client()
                    msg = client.messages.create(
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system=system,
                        messages=[{"role": "user", "content": user}],
                    )
                    # content 是 TextBlock 列表
                    parts = []
                    for block in msg.content:
                        if hasattr(block, "text"):
                            parts.append(block.text)
                    return "".join(parts)
            except Exception as e:
                last_err = e
                wait = 2.0 * (1.6 ** attempt)
                time.sleep(min(wait, 15.0))
        raise RuntimeError(f"chat 调用失败 (after {retries} retries): {last_err}")

    # ------------- chat_json -------------
    def chat_json(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
        max_tokens: int = 1500,
        temperature: float = 0.1,
        retries: int = 3,
    ) -> Dict[str, Any]:
        """要求 LLM 返回 JSON, 失败时重试 N 次, 最后还失败抛错。"""
        last_err: Optional[Exception] = None
        cur_user = user
        for attempt in range(retries):
            try:
                text = self.chat(system, cur_user, model=model,
                                 max_tokens=max_tokens, temperature=temperature)
                obj = _extract_json(text)
                if obj is not None:
                    return obj
                last_err = ValueError(f"无法解析为 JSON, raw: {text[:200]}")
                cur_user = user + "\n\nIMPORTANT: 之前的输出无法解析为 JSON。请输出且仅输出一个有效的 JSON 对象, 不要包裹 markdown 代码块。"
            except Exception as e:
                last_err = e
                time.sleep(1.0)
        raise RuntimeError(f"chat_json 失败 (after {retries} retries): {last_err}")


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    # try direct
    try:
        return json.loads(text)
    except Exception:
        pass
    # try fenced
    m = _JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # try first { .. last }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return None
