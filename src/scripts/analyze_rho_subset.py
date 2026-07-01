"""ANALYSIS 阶段 ρ̂ 子集分析 scaffold。

对 step records 按 belief table 大小分桶, 分别算 ρ-Δ 关联:
- single-belief subset: 1 个 belief, 只能跨 step Pearson (退化)
- multi-belief subset (≥2 belief): within-step Spearman (Theorem 1 完整 regime)
- by belief.type subset: 同 type 内的 ρ-Δ 关联

输出:
- analysis/rho_subset_analysis.json
- analysis/rho_subset_table.md
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def pearson(pairs):
    if len(pairs) < 2:
        return 0.0, len(pairs)
    n = len(pairs)
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    num = sum((p[0]-mx)*(p[1]-my) for p in pairs)
    dx = math.sqrt(sum((p[0]-mx)**2 for p in pairs))
    dy = math.sqrt(sum((p[1]-my)**2 for p in pairs))
    return (num/(dx*dy) if dx*dy else 0.0), n


def ranks(xs):
    sorted_idx = sorted(range(len(xs)), key=lambda i: xs[i])
    rk = [0]*len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j+1 < len(xs) and xs[sorted_idx[j+1]] == xs[sorted_idx[i]]:
            j += 1
        avg = (i+j)/2 + 1
        for k in range(i, j+1):
            rk[sorted_idx[k]] = avg
        i = j + 1
    return rk


def spearman(pairs):
    if len(pairs) < 2:
        return 0.0
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    rx, ry = ranks(xs), ranks(ys)
    return pearson(list(zip(rx, ry)))[0]


def within_step_spearman(records):
    """对每 step (≥2 belief) 算 ρ-Δ rank corr, 然后跨 step 平均。"""
    step_corrs = []
    for r in records:
        rho_list = r.get("probe_score_per_belief") or []
        delta_list = r.get("oracle_delta_per_belief") or []
        if len(rho_list) < 2:
            continue
        d_by_id = {d["id"]: d["delta_i"] for d in delta_list}
        pairs = [(e["rho_i"], d_by_id[e["id"]])
                 for e in rho_list if e["id"] in d_by_id]
        if len(pairs) < 2:
            continue
        c = spearman(pairs)
        step_corrs.append(c)
    if not step_corrs:
        return {"mean": 0.0, "n_steps": 0}
    return {
        "mean": mean(step_corrs),
        "stdev": stdev(step_corrs) if len(step_corrs) > 1 else 0.0,
        "n_steps": len(step_corrs),
    }


def analyze(step_path: Path, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    single_pairs = []
    multi_records = []
    by_type_pairs: Dict[str, list] = defaultdict(list)

    with open(step_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            rho_list = r.get("probe_score_per_belief") or []
            delta_list = r.get("oracle_delta_per_belief") or []
            if not rho_list:
                continue
            d_by_id = {d["id"]: d["delta_i"] for d in delta_list}
            if len(rho_list) == 1 and rho_list[0]["id"] in d_by_id:
                single_pairs.append((rho_list[0]["rho_i"],
                                     d_by_id[rho_list[0]["id"]]))
            elif len(rho_list) >= 2:
                multi_records.append(r)
            for e in rho_list:
                if e["id"] in d_by_id:
                    btype = str(e.get("type", "other"))
                    by_type_pairs[btype].append((e["rho_i"], d_by_id[e["id"]]))

    out = {
        "single_belief": {
            "pearson": pearson(single_pairs),
            "spearman": spearman(single_pairs),
            "n": len(single_pairs),
        },
        "multi_belief_within_step": within_step_spearman(multi_records),
        "by_belief_type": {},
    }
    for t, pairs in by_type_pairs.items():
        out["by_belief_type"][t] = {
            "pearson_pair_and_n": pearson(pairs),
            "spearman": spearman(pairs),
            "n": len(pairs),
        }
    (out_dir / "rho_subset_analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    # Markdown table
    lines = ["| Subset | Pearson r | Spearman ρ | N |", "|---|---:|---:|---:|"]
    sb = out["single_belief"]
    lines.append(f"| single_belief (cross-step) | {sb['pearson'][0]:.3f} | {sb['spearman']:.3f} | {sb['n']} |")
    mb = out["multi_belief_within_step"]
    lines.append(f"| multi_belief (within-step mean) | — | {mb['mean']:.3f} | {mb['n_steps']} steps |")
    for t, v in out["by_belief_type"].items():
        lines.append(f"| type={t} | {v['pearson_pair_and_n'][0]:.3f} | {v['spearman']:.3f} | {v['n']} |")
    (out_dir / "rho_subset_table.md").write_text("\n".join(lines), encoding="utf-8")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--step-jsonl", default="experiments/gate_a_pilot.jsonl")
    p.add_argument("--out-dir", default="analysis")
    args = p.parse_args()
    res = analyze(ROOT / args.step_jsonl, ROOT / args.out_dir)
    print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
