"""R3 P0 ToolDAG floor verification (2026-06-03).

Tests R2 reviewer gemini's strict-trace finding:
    No-Probe + Self-Uncertainty + (c+d) on ToolDAGWorld seeds 42:261
    all yield A_H = 0.7000 sd = 0 task_success = 0%
    → 0.7 is a design floor, not policy-induced gain.

Pulls from existing data (spine_S2 + revision_cd) since 660 ep already exist:
- spine_S2: no_probe + envprobe_simple × ToolDAGWorld × seeds 42:261
- revision_cd: envprobe_simple_cd × ToolDAGWorld × seeds 42:261

Also extends to Task C (3 env × 3 method matrix) to check if floor is
ToolDAG-only or method-level.

Output:
- experiments/r3_floor_verify/summary.json    (per-cell A_H + sd + task_succ)
- experiments/r3_floor_verify/stratified_summary.md
- experiments/r3_floor_verify/results.jsonl   (merged episode records from sources)
- experiments/r3_floor_verify/floor_verdict.json (Task A + Task B verdict)
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "r3_floor_verify"
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = [
    ("spine_S2", ROOT / "experiments" / "spine_S2_episodes.jsonl"),
    ("revision_cd", ROOT / "experiments" / "revision_cd" / "episodes.jsonl"),
]
METHODS = ["no_probe", "envprobe_simple", "envprobe_simple_cd"]
ENVS = ["ObjectStateWorld", "ToolDAGWorld", "GraphNavWorld"]
SEED_LOW, SEED_HIGH = 42, 261


def main():
    # 1) Aggregate
    ah = defaultdict(list)
    succ = defaultdict(list)
    rows = defaultdict(list)
    for src_name, p in SOURCES:
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                m = r.get("method")
                e = r.get("env") or r.get("environment")
                s = r.get("seed")
                if m not in METHODS or e not in ENVS:
                    continue
                if s is None or s < SEED_LOW or s > SEED_HIGH:
                    continue
                em = r.get("episode_metrics", {}) or {}
                a = em.get("world_state_accuracy")
                if a is not None:
                    ah[(m, e)].append(a)
                    rows[(m, e)].append({
                        "seed": s, "A_H": a,
                        "task_success": bool(em.get("task_success")),
                        "source": src_name,
                    })
                succ[(m, e)].append(1 if em.get("task_success") else 0)

    # 2) Merge results.jsonl
    merged = []
    for (m, e), rs in rows.items():
        for r in rs:
            merged.append({"method": m, "env": e, **r})
    merged.sort(key=lambda x: (x["env"], x["method"], x["seed"]))
    with open(OUT / "results.jsonl", "w") as f:
        for r in merged:
            f.write(json.dumps(r) + "\n")

    # 3) Per-cell summary
    summary = {"cells": {}, "task_A_verdict": None, "task_B_verdict": None,
               "task_C_verdict": None}
    for e in ENVS:
        for m in METHODS:
            a_list = ah.get((m, e), [])
            s_list = succ.get((m, e), [])
            n = len(a_list)
            cell = f"{e}_{m}"
            if n == 0:
                summary["cells"][cell] = {"n": 0, "missing": True}
                continue
            am = mean(a_list)
            asd = stdev(a_list) if n > 1 else 0
            amin, amax = min(a_list), max(a_list)
            ts = mean(s_list) if s_list else 0
            is_floor = asd < 0.001 and abs(amax - amin) < 0.001
            counter = Counter([round(v, 4) for v in a_list]).most_common(3)
            summary["cells"][cell] = {
                "n": n,
                "A_H_mean": am, "A_H_sd": asd,
                "A_H_min": amin, "A_H_max": amax,
                "task_success_rate": ts,
                "is_constant_floor": is_floor,
                "A_H_top3_values": [(v, c) for v, c in counter],
            }

    # 4) Task A verdict (ToolDAG no_probe vs CD)
    np_td = ah.get(("no_probe", "ToolDAGWorld"), [])
    cd_td = ah.get(("envprobe_simple_cd", "ToolDAGWorld"), [])
    sp_td = ah.get(("envprobe_simple", "ToolDAGWorld"), [])
    np_floor = len(np_td) > 0 and (max(np_td) - min(np_td)) < 0.001
    cd_floor = len(cd_td) > 0 and (max(cd_td) - min(cd_td)) < 0.001
    np_val = np_td[0] if np_floor else None
    cd_val = cd_td[0] if cd_floor else None
    same_floor = np_floor and cd_floor and abs(np_val - cd_val) < 0.001
    summary["task_A_verdict"] = {
        "no_probe_floor": np_floor, "no_probe_value": np_val,
        "envprobe_simple_cd_floor": cd_floor, "envprobe_simple_cd_value": cd_val,
        "envprobe_simple_floor": (max(sp_td) - min(sp_td) < 0.001) if sp_td else None,
        "envprobe_simple_mean_sd": (mean(sp_td) if sp_td else None,
                                    stdev(sp_td) if len(sp_td) > 1 else None),
        "same_constant_floor": same_floor,
        "conclusion": ("CONFIRMED: ToolDAG A_H=0.7 is design floor, "
                       "No-Probe and (c+d) both saturate to it; 4-dim is the "
                       "outlier that DEVIATES below floor (probe noise).")
                       if same_floor else "Floor NOT confirmed (mixed result)",
    }

    # 5) Task B verdict
    if same_floor:
        summary["task_B_verdict"] = {
            "interpretation": "design_artifact",
            "rec": ("Paper §5/§6: 加诚实段 — ToolDAG A_H=0.7 是 score_belief_state "
                    "weighted-slot ceiling 设计 floor (0.30+0.40+0.30=1.00, "
                    "agent perfect → 0.70 weighted overlap), 不是 policy gain. "
                    "+34pp 反向解释: 4-dim probe 引入扰动 → A_H 偏离 floor; "
                    "(c+d) probe 较少且 conservative → 保持 floor. "
                    "撤回 spine D' '(c+d) > 4-dim by +34pp on procedural' 表述."),
        }
    else:
        summary["task_B_verdict"] = {
            "interpretation": "real_gain",
            "rec": "Paper 保 '+34pp' frame, 但需补充 sd/min/max 报告.",
        }

    # 6) Task C verdict (cross-env)
    cd_sd_by_env = {e: stdev(ah.get(("envprobe_simple_cd", e), [0])) if len(ah.get(("envprobe_simple_cd", e), [])) > 1 else 0 for e in ENVS}
    cd_floor_count = sum(1 for e in ENVS if cd_sd_by_env[e] < 0.001)
    summary["task_C_verdict"] = {
        "envprobe_simple_cd_sd_by_env": cd_sd_by_env,
        "n_envs_at_floor": cd_floor_count,
        "is_method_wide_problem": cd_floor_count >= 2,
        "conclusion": ("ToolDAG-only floor" if cd_floor_count == 1
                       else "Method-wide constant" if cd_floor_count >= 2
                       else "No floor detected"),
    }

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {OUT}/summary.json")
    print(f"Total merged episodes: {len(merged)}")

    # 7) Markdown 3×3 matrix
    md = ["# R3 P0 ToolDAG Floor Verification — 3 env × 3 method Matrix\n",
          f"Data sources: {', '.join(s[0] for s in SOURCES)}\n",
          f"Seeds: 42:261 (n=220 per cell), Stress: S2 (pilot_med), Total: 1980 ep\n\n",
          "| env | method | n | A_H mean | sd | min | max | task_succ | floor? |\n",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|\n"]
    for e in ENVS:
        for m in METHODS:
            d = summary["cells"].get(f"{e}_{m}", {})
            if d.get("missing"):
                md.append(f"| {e} | {m} | 0 | --- | --- | --- | --- | --- | MISSING |\n")
                continue
            floor = "★ YES" if d["is_constant_floor"] else "no"
            md.append(f"| {e} | {m} | {d['n']} | {d['A_H_mean']:.4f} | "
                      f"{d['A_H_sd']:.4f} | {d['A_H_min']:.4f} | {d['A_H_max']:.4f} | "
                      f"{d['task_success_rate']:.4f} | {floor} |\n")

    md.append("\n## Task A: ToolDAG floor verdict\n")
    ta = summary["task_A_verdict"]
    md.append(f"- No-Probe ToolDAG floor: {ta['no_probe_floor']} (value={ta['no_probe_value']})\n")
    md.append(f"- (c+d) ToolDAG floor: {ta['envprobe_simple_cd_floor']} (value={ta['envprobe_simple_cd_value']})\n")
    md.append(f"- 4-dim ToolDAG floor: {ta['envprobe_simple_floor']}; mean/sd = {ta['envprobe_simple_mean_sd']}\n")
    md.append(f"- Same constant floor: **{ta['same_constant_floor']}**\n")
    md.append(f"- Conclusion: {ta['conclusion']}\n\n")

    md.append("## Task B: Interpretation\n")
    tb = summary["task_B_verdict"]
    md.append(f"- {tb['interpretation']}\n- Recommendation: {tb['rec']}\n\n")

    md.append("## Task C: Cross-belief stratum\n")
    tc = summary["task_C_verdict"]
    md.append(f"- (c+d) sd by env: {tc['envprobe_simple_cd_sd_by_env']}\n")
    md.append(f"- Envs at floor: {tc['n_envs_at_floor']} / 3\n")
    md.append(f"- Conclusion: {tc['conclusion']}\n")

    with open(OUT / "stratified_summary.md", "w") as f:
        f.writelines(md)
    print(f"Wrote {OUT}/stratified_summary.md")


if __name__ == "__main__":
    main()
