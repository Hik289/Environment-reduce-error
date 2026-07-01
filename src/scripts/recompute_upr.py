"""重算 useful_probe_count + useful_probe_rate (UPR) on existing JSONL.

读 step records 中 probe_response + belief_table (step k-1 belief), 应用 v2
compare_belief_to_gold_for_probe, 累加 ep 级 useful_probe_count, 重写 episodes.jsonl 字段。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.metrics.scorer import compare_belief_to_gold_for_probe


def recompute(step_path: Path, ep_path: Path, out_ep_path: Path):
    # 按 episode_id 分组 step records
    ep_steps = defaultdict(list)
    with open(step_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            ep_steps[r["episode_id"]].append(r)

    # 重算 useful_probe_count per ep
    new_useful = {}
    new_total = {}
    for ep_id, steps in ep_steps.items():
        steps.sort(key=lambda x: x["step"])
        n_useful = 0
        n_probe = 0
        for s in steps:
            if s.get("decision_type") != "probe":
                continue
            pr = s.get("probe_response")
            if not pr:
                continue
            n_probe += 1
            # belief_before = 此 step 的 agent_belief_state (probe 前 agent 评估的)
            belief_before = {"belief_world_state": s.get("agent_belief_state") or {}}
            if compare_belief_to_gold_for_probe(pr, belief_before):
                n_useful += 1
        new_useful[ep_id] = n_useful
        new_total[ep_id] = n_probe

    # 重写 episodes.jsonl 添加 useful_probe_count_v2 + useful_probe_rate_v2
    n_eps = 0
    with open(ep_path, encoding="utf-8") as f, \
         open(out_ep_path, "w", encoding="utf-8") as fo:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                fo.write(line)
                continue
            ep_id = r["episode_id"]
            v2_useful = new_useful.get(ep_id, 0)
            v2_total = new_total.get(ep_id, r.get("total_probe_count", 0))
            r["useful_probe_count_v2"] = v2_useful
            r["total_probe_count_v2"] = v2_total
            v2_upr = (v2_useful / v2_total) if v2_total > 0 else None
            r["episode_metrics"]["useful_probe_rate_v2"] = v2_upr
            fo.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            n_eps += 1
    print(f"Recomputed UPR for {n_eps} eps; wrote {out_ep_path}")

    # 聚合: UPR by method
    by_method_v2 = defaultdict(list)
    by_method_v1 = defaultdict(list)
    with open(out_ep_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            m = r["method"]
            upr_v2 = r["episode_metrics"].get("useful_probe_rate_v2")
            upr_v1 = r["episode_metrics"].get("useful_probe_rate")
            if upr_v2 is not None:
                by_method_v2[m].append(upr_v2)
            if upr_v1 is not None:
                by_method_v1[m].append(upr_v1)
    print("\n=== UPR by method ===")
    print(f"{'method':28} {'n_v1':>5} {'UPR_v1':>10} {'n_v2':>5} {'UPR_v2':>10}")
    methods = sorted(set(list(by_method_v1.keys()) + list(by_method_v2.keys())))
    for m in methods:
        v1 = by_method_v1.get(m, [])
        v2 = by_method_v2.get(m, [])
        u1 = sum(v1)/len(v1) if v1 else 0
        u2 = sum(v2)/len(v2) if v2 else 0
        print(f"{m:28} {len(v1):>5} {u1:>10.4f} {len(v2):>5} {u2:>10.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--step", default="experiments/spine_S2.jsonl")
    p.add_argument("--ep-in", default="experiments/spine_S2_episodes.jsonl")
    p.add_argument("--ep-out", default="experiments/spine_S2_episodes_v2.jsonl")
    args = p.parse_args()
    recompute(ROOT / args.step, ROOT / args.ep_in, ROOT / args.ep_out)
