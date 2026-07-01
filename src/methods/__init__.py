"""Method 选择: 7 种 probe / no-probe / oracle / judge / reset。"""
from __future__ import annotations

from typing import Any, Dict

from .base import Method, MethodContext, MethodDecision
from .no_probe import NoProbeMethod
from .random_probe import RandomProbeMethod
from .periodic_probe import PeriodicProbeMethod
from .self_uncertainty_probe import SelfUncertaintyProbeMethod
from .envprobe_simple import (EnvProbeSimpleMethod,
                              EnvProbeSimpleMinusC, EnvProbeSimpleMinusS,
                              EnvProbeSimpleMinusU, EnvProbeSimpleMinusD)
from .envprobe_simple_cd import EnvProbeSimpleCD
from .envprobe_judge import EnvProbeJudgeMethod
from .oracle_probe import OracleProbeMethod
from .oracle_task_weighted import OracleTaskWeightedMethod


_REGISTRY = {
    "no_probe": NoProbeMethod,
    "random_probe": RandomProbeMethod,
    "periodic_probe": PeriodicProbeMethod,
    "self_uncertainty_probe": SelfUncertaintyProbeMethod,
    "envprobe_simple": EnvProbeSimpleMethod,
    "envprobe_simple_cd": EnvProbeSimpleCD,
    "envprobe_judge": EnvProbeJudgeMethod,
    "oracle_probe": OracleProbeMethod,
    "oracle_task_weighted": OracleTaskWeightedMethod,
    # Ablation variants (Corollary 1)
    "envprobe_simple_minus_c": EnvProbeSimpleMinusC,
    "envprobe_simple_minus_s": EnvProbeSimpleMinusS,
    "envprobe_simple_minus_u": EnvProbeSimpleMinusU,
    "envprobe_simple_minus_d": EnvProbeSimpleMinusD,
}


def get_method(name: str) -> Method:
    if name not in _REGISTRY:
        raise ValueError(f"未知 method: {name}; 可选: {list(_REGISTRY)}")
    return _REGISTRY[name]()


__all__ = ["Method", "MethodContext", "MethodDecision", "get_method"]
