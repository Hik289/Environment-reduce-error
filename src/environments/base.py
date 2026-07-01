"""Environment 抽象接口 + 共享工具函数。

每个环境是 rule-based 模拟器, 拥有:
- 隐藏 gold_state
- task_action (推进任务)
- probe_action (查询局部事实, 不推进任务)

所有随机性必须由 self._rng (seed 播种) 控制, 保证确定性。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StepResult:
    obs: Dict[str, Any]
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbeResult:
    probe_type: str
    target: Any
    answer: Dict[str, Any]
    cost: float = 1.0
    info: Dict[str, Any] = field(default_factory=dict)


class Environment:
    name: str = "BaseEnvironment"

    def reset(self, seed: int, stress_config: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def step_task_action(self, action: str) -> StepResult:
        raise NotImplementedError

    def step_probe_action(self, probe: str) -> ProbeResult:
        raise NotImplementedError

    def get_observation(self) -> Dict[str, Any]:
        raise NotImplementedError

    def get_gold_state(self) -> Dict[str, Any]:
        """返回可外部观测的 gold state (内部 specs 已去除)。用于 scorer / Oracle-Probe。"""
        raise NotImplementedError

    def task_description(self) -> str:
        raise NotImplementedError

    def available_task_actions(self) -> List[str]:
        raise NotImplementedError

    def available_probe_actions(self) -> List[str]:
        raise NotImplementedError

    def is_done(self) -> bool:
        return False

    def step_count(self) -> int:
        return getattr(self, "_step", 0)

    # ---- 用于 anchor_3 确定性 hash 比对 ----
    def canonical_gold(self) -> Dict[str, Any]:
        """返回可哈希的 gold state 副本 (set / tuple key → list / str)。"""
        return _canonicalize(self.get_gold_state())


def _canonicalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k in sorted(obj.keys(), key=lambda x: str(x)):
            out[str(k)] = _canonicalize(obj[k])
        return out
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(x) for x in obj]
    if isinstance(obj, set):
        return sorted([_canonicalize(x) for x in obj], key=lambda x: str(x))
    return obj


# -----------------------------------------------------------------------------
# Belief-vs-gold 相似度 (recursive Jaccard / value overlap)
# -----------------------------------------------------------------------------

def flat_overlap(belief: Any, gold: Any) -> float:
    """对 nested dict / list / scalar 计算 [0, 1] 区间的 fuzzy 相似度。

    - dict: 在 gold 的每个 key 上递归比较, 取平均
    - list: 视为多重集合, 计算 Jaccard
    - scalar: 严格相等返回 1.0, 否则 0.0
    """
    if isinstance(gold, dict):
        if not isinstance(belief, dict) or not gold:
            return 0.0 if gold else 1.0
        scores = []
        for k, gv in gold.items():
            bv = belief.get(k) if isinstance(belief, dict) else None
            scores.append(flat_overlap(bv, gv))
        return sum(scores) / max(1, len(scores))
    if isinstance(gold, (list, set, tuple)):
        gset = set(map(_canon, gold))
        bset = set(map(_canon, belief or [])) if isinstance(belief, (list, set, tuple)) else set()
        if not gset and not bset:
            return 1.0
        if not gset or not bset:
            return 0.0
        return len(gset & bset) / len(gset | bset)
    return 1.0 if _canon(belief) == _canon(gold) else 0.0


def _canon(x: Any) -> str:
    if x is None:
        return "_NONE_"
    if isinstance(x, (str, int, float, bool)):
        return str(x).strip().lower()
    return str(x)
