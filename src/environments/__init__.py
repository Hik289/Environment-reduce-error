"""Environment factory + stress configs."""
from __future__ import annotations

from typing import Any, Dict

from .base import Environment, StepResult, ProbeResult, flat_overlap
from .object_state_world import ObjectStateWorld
from .tool_dag_world import ToolDAGWorld
from .graph_nav_world import GraphNavWorld


_ENV_REGISTRY = {
    "ObjectStateWorld": ObjectStateWorld,
    "ToolDAGWorld": ToolDAGWorld,
    "GraphNavWorld": GraphNavWorld,
}


def make_environment(name: str) -> Environment:
    if name not in _ENV_REGISTRY:
        raise ValueError(f"未知环境: {name}; 可选: {list(_ENV_REGISTRY)}")
    return _ENV_REGISTRY[name]()


# Stress 配置预设
STRESS_PRESETS: Dict[str, Dict[str, Any]] = {
    "pilot_low": {
        "horizon": 20,
        "state_cardinality": "low",
        "dependency_density": "low",
        "observation_noise": "clean",
        "state_mutation_rate": "static",
    },
    "pilot_high": {
        "horizon": 40,
        "state_cardinality": "high",
        "dependency_density": "high",
        "observation_noise": "distractor",
        "state_mutation_rate": "volatile",
    },
    "high_stress_h40": {
        "horizon": 40,
        "state_cardinality": "high",
        "dependency_density": "high",
        "observation_noise": "distractor",
        "state_mutation_rate": "volatile",
    },
    "medium_h20": {
        "horizon": 20,
        "state_cardinality": "medium",
        "dependency_density": "medium",
        "observation_noise": "partial",
        "state_mutation_rate": "mild",
    },
    "pilot_med": {
        "horizon": 30,
        "state_cardinality": "medium",
        "dependency_density": "medium",
        "observation_noise": "partial",
        "state_mutation_rate": "mild",
    },
    # === ds stress_grid §B3 final binding (2026-05-27 JST) ===
    # S1/S2/S3 spine + R1-R4 + R6 OAT around S2 (pilot_med)
    "S1": {
        "horizon": 20, "state_cardinality": "low", "dependency_density": "low",
        "observation_noise": "clean", "state_mutation_rate": "static",
        "_mu_bar": 0.02, "_role": "spine_low_boundary",
    },
    "S2": {
        "horizon": 30, "state_cardinality": "medium", "dependency_density": "medium",
        "observation_noise": "partial", "state_mutation_rate": "mild",
        "_mu_bar": 0.10, "_role": "spine_primary",
    },
    "S3": {
        "horizon": 40, "state_cardinality": "high", "dependency_density": "high",
        "observation_noise": "distractor", "state_mutation_rate": "volatile",
        "_mu_bar": 0.30, "_role": "spine_high_boundary",
    },
    "R1": {   # card_low at S2 mu
        "horizon": 30, "state_cardinality": "low", "dependency_density": "medium",
        "observation_noise": "partial", "state_mutation_rate": "mild",
        "_mu_bar": 0.10, "_role": "robustness_card_low",
    },
    "R2": {   # card_high at S2 mu
        "horizon": 30, "state_cardinality": "high", "dependency_density": "medium",
        "observation_noise": "partial", "state_mutation_rate": "mild",
        "_mu_bar": 0.10, "_role": "robustness_card_high",
    },
    "R3": {   # dep_low at S2 mu
        "horizon": 30, "state_cardinality": "medium", "dependency_density": "low",
        "observation_noise": "partial", "state_mutation_rate": "mild",
        "_mu_bar": 0.10, "_role": "robustness_dep_low",
    },
    "R4": {   # dep_high at S2 mu
        "horizon": 30, "state_cardinality": "medium", "dependency_density": "high",
        "observation_noise": "partial", "state_mutation_rate": "mild",
        "_mu_bar": 0.10, "_role": "robustness_dep_high",
    },
    "R6": {   # obs_noisy at S2 mu  (R5 ≡ S2, skipped)
        "horizon": 30, "state_cardinality": "medium", "dependency_density": "medium",
        "observation_noise": "distractor", "state_mutation_rate": "mild",
        "_mu_bar": 0.10, "_role": "robustness_obs_noisy",
    },
}


def default_stress(label: str) -> Dict[str, Any]:
    if label not in STRESS_PRESETS:
        raise ValueError(f"未知 stress preset: {label}; 可选: {list(STRESS_PRESETS)}")
    return dict(STRESS_PRESETS[label])


__all__ = [
    "Environment", "StepResult", "ProbeResult",
    "ObjectStateWorld", "ToolDAGWorld", "GraphNavWorld",
    "make_environment", "default_stress", "STRESS_PRESETS",
    "flat_overlap",
]
