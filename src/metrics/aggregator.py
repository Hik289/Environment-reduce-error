"""Aggregator: 多 episode 汇总到 table。"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def aggregate_episodes(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """按 (env, method, stress) 分组聚合。

    records: List[{env, method, model, stress_label, episode_metrics: {...}}]
    """
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        key = (r["env"], r["method"], r["stress_label"], r.get("model", ""))
        groups[key].append(r["episode_metrics"])

    out: Dict[str, Dict[str, Any]] = {}
    no_probe_collapse: Dict[tuple, float] = {}
    # 第一遍: 找 no_probe baseline
    for (env, method, stress, model), eps in groups.items():
        if method == "no_probe":
            avg = _mean_collapse(eps)
            no_probe_collapse[(env, stress, model)] = avg

    for (env, method, stress, model), eps in groups.items():
        agg = _aggregate(eps)
        cb = no_probe_collapse.get((env, stress, model))
        if cb is not None and agg["collapse_onset_mean"] is not None:
            agg["collapse_delay"] = agg["collapse_onset_mean"] - cb
        else:
            agg["collapse_delay"] = None
        key = f"{env}|{method}|{stress}|{model}"
        out[key] = {
            "env": env, "method": method, "stress": stress, "model": model,
            "n_episodes": len(eps), **agg,
        }
    return out


def _mean_collapse(eps: List[Dict[str, Any]]):
    vals = [e["collapse_onset"] for e in eps if e.get("collapse_onset") is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _aggregate(eps: List[Dict[str, Any]]) -> Dict[str, Any]:
    def safe_mean(key):
        vals = [e[key] for e in eps if e.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "task_success_rate": sum(1 for e in eps if e.get("task_success", 0)) / max(1, len(eps)),
        "task_success_mean": safe_mean("task_success"),
        "collapse_onset_mean": safe_mean("collapse_onset"),
        "action_collapse_onset_mean": safe_mean("action_collapse_onset"),
        "world_state_accuracy_mean": safe_mean("world_state_accuracy"),
        "world_state_accuracy_last_third_mean": safe_mean("world_state_accuracy_last_third"),
        "action_validity_mean": safe_mean("action_validity"),
        "probe_efficiency_mean": safe_mean("probe_efficiency"),
        "useful_probe_rate_mean": safe_mean("useful_probe_rate"),
        "missed_critical_probe_rate_mean": safe_mean("missed_critical_probe_rate"),
        "false_belief_commitment_mean": safe_mean("false_belief_commitment"),
        "self_check_accuracy_mean": safe_mean("self_check_accuracy"),
        "recovery_rate_mean": safe_mean("recovery_rate"),
        "probe_budget_usage_mean": safe_mean("probe_budget_usage"),
        "dependency_accuracy_mean": safe_mean("dependency_accuracy"),
    }


def write_pilot_table(agg: Dict[str, Dict[str, Any]], out_path: str | Path) -> None:
    """生成 markdown 表格。"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["env", "method", "stress", "n_episodes",
            "task_success_rate", "world_state_accuracy_mean", "action_validity_mean",
            "collapse_onset_mean", "probe_efficiency_mean", "useful_probe_rate_mean",
            "probe_budget_usage_mean", "false_belief_commitment_mean", "self_check_accuracy_mean"]
    lines = ["| " + " | ".join(cols) + " |"]
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for k in sorted(agg.keys()):
        row = agg[k]
        vals = []
        for c in cols:
            v = row.get(c)
            if v is None:
                vals.append("—")
            elif isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
