"""JSONL logger 工具。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, IO, Optional


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f: Optional[IO] = None

    def __enter__(self):
        self._f = open(self.path, "a", encoding="utf-8")
        return self

    def __exit__(self, *a):
        if self._f:
            self._f.close()
            self._f = None

    def write(self, record: Dict[str, Any]) -> None:
        if self._f is None:
            self._f = open(self.path, "a", encoding="utf-8")
        self._f.write(json.dumps(record, ensure_ascii=False, default=_default) + "\n")
        self._f.flush()


def _default(o):
    if isinstance(o, set):
        return sorted(list(o), key=lambda x: str(x))
    return str(o)


def append_error(path: str | Path, record: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=_default) + "\n")
