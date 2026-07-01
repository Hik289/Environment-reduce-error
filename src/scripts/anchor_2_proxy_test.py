"""Anchor_2: Anthropic local proxy 集成测试。

1 次 claude-haiku-4-5 单 message call 应返回有效 JSON。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.api_client import LLMClient, _extract_json


def main():
    out_path = ROOT / "experiments" / "anchor_2_proxy_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = LLMClient(model="claude-haiku-4-5", timeout=30.0)
    sys_prompt = "You are a JSON-only assistant. Reply with exactly one JSON object."
    user = 'Return {"ok": true, "model": "claude-haiku-4-5", "msg": "hello"} as JSON.'

    t0 = time.time()
    try:
        text = client.chat(sys_prompt, user, model="claude-haiku-4-5",
                           max_tokens=80, temperature=0.0)
        elapsed = time.time() - t0
        obj = _extract_json(text)
        parsed_ok = obj is not None
        result = {
            "ok": True, "parsed_ok": parsed_ok, "latency_s": elapsed,
            "raw_text": text, "parsed_obj": obj, "model": "claude-haiku-4-5",
            "base_url": "https://api.anthropic.com",
        }
    except Exception as e:
        result = {"ok": False, "error": repr(e), "model": "claude-haiku-4-5",
                  "base_url": "https://api.anthropic.com"}

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("ok"):
        sys.exit(1)


if __name__ == "__main__":
    main()
