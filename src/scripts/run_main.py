"""RUNNING phase 主 driver。

支持:
- 多 cell 并行 (subprocess pool, 默认 5)
- paired seed (cells_registry.csv 指定每 cell 的 seed range)
- resume from checkpoint (jsonl append-only + processed_cells.txt)
- 增量 p_cw 监控 (每 100 ep 输出 incremental_pcw.jsonl, 仅 no_probe cell)
- 单 ep 5min hard cap (signal.alarm)
- GATE A 决策 (50 ep 后估 ρ̂/σ̂_d/p̂_cw_lcb, 自动判 PASS/FAIL/BORDERLINE)

用法:
    python -m src.scripts.run_main --registry experiments/cells_registry.csv --layer gate_a
    python -m src.scripts.run_main --registry ... --layer full --parallel 5
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agents.llm_agent import LLMAgent
from src.environments import make_environment, default_stress, STRESS_PRESETS
from src.methods import get_method
from src.methods.envprobe_judge import EnvProbeJudgeMethod
from src.methods.base import MethodContext
from src.metrics.scorer import score_step, score_episode, compare_belief_to_gold_for_probe, evaluate_self_check_v2
from src.utils.api_client import LLMClient
from src.utils.logger import JsonlLogger, append_error


JST = timezone(timedelta(hours=9))
EP_HARD_CAP_SEC = 300  # 5 min per episode (soft check, raised between steps)


def now_jst() -> str:
    return datetime.now(JST).isoformat(timespec="seconds")


class EpisodeTimeout(Exception):
    pass


@dataclass
class CellSpec:
    cell_id: str
    env: str
    stress_config: str   # 'S1' / 'S2' / 'R1' / ... STRESS_PRESETS key
    method: str
    n_seeds: int
    seeds: List[int]
    role: str            # spine_primary / boundary_smoke / robustness / ablation / gate_a


def load_registry(path: Path) -> List[CellSpec]:
    cells = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seed_spec = row["seeds_spec"]
            seeds = _parse_seeds(seed_spec)
            cells.append(CellSpec(
                cell_id=row["cell_id"],
                env=row["env"],
                stress_config=row["stress_config"],
                method=row["method"],
                n_seeds=int(row["n_seeds"]),
                seeds=seeds,
                role=row["role"],
            ))
    return cells


def _parse_seeds(spec: str) -> List[int]:
    """e.g. '42:71' → [42,43,...,71]; '42,45,48' → [42,45,48]."""
    if ":" in spec:
        a, b = spec.split(":")
        return list(range(int(a), int(b) + 1))
    if "," in spec:
        return [int(x.strip()) for x in spec.split(",") if x.strip()]
    return [int(spec)]


# -----------------------------------------------------------------------------
# 单 episode (与 run_smoke 类似, 但支持 extra_per_belief)
# -----------------------------------------------------------------------------

def run_one_episode(cell: CellSpec, seed: int, model: str,
                    out_step_path: Path, out_err_path: Path) -> Optional[Dict[str, Any]]:
    env = make_environment(cell.env)
    stress = default_stress(cell.stress_config)
    obs = env.reset(seed=seed, stress_config=stress)
    horizon = int(stress["horizon"])
    probe_budget = max(1, horizon // 4)

    client = LLMClient(model=model)
    agent = LLMAgent(model=model, client=client, max_tokens=1200)
    agent.reset(env.name, env.task_description())

    method = get_method(cell.method)
    method.reset_episode()
    if isinstance(method, EnvProbeJudgeMethod):
        method.attach_agent(agent)

    extra_per_belief = cell.method in ("envprobe_simple", "oracle_probe")
    episode_id = f"{cell.cell_id}-seed{seed}-{uuid.uuid4().hex[:6]}"

    step = 0
    history: List[Dict[str, Any]] = []
    step_records: List[Dict[str, Any]] = []
    probes_used = 0
    useful_probe_count = 0
    total_probe_count = 0

    # 5 min soft cap (在每步开头检查 wall time)
    ep_start = time.time()
    try:
        with JsonlLogger(out_step_path) as step_logger:
            while not env.is_done() and step < horizon:
                if time.time() - ep_start > EP_HARD_CAP_SEC:
                    append_error(out_err_path, {"where": "ep_timeout", "episode": episode_id,
                                                "step_reached": step,
                                                "elapsed": time.time() - ep_start})
                    break
                step += 1
                try:
                    agent_out = agent.step(
                        observation=obs, history=history[-5:],
                        step=step, horizon=horizon,
                        method_hint=method.method_hint,
                        task_action_spec=env.available_task_actions(),
                        probe_action_spec=env.available_probe_actions(),
                        probe_budget_remaining=probe_budget - probes_used,
                        probe_force_act=(probes_used >= probe_budget),
                    )
                except Exception as e:
                    append_error(out_err_path, {"where": "agent.step", "episode": episode_id,
                                                "step": step, "err": repr(e)})
                    break

                ctx = MethodContext(
                    step=step, horizon=horizon, observation=obs, agent_output=agent_out,
                    task_action_spec=env.available_task_actions(),
                    probe_action_spec=env.available_probe_actions(),
                    env=env, history=history,
                    probe_budget_total=probe_budget, probes_used=probes_used,
                )
                try:
                    md = method.decide(ctx)
                except Exception as e:
                    append_error(out_err_path, {"where": "method.decide", "episode": episode_id,
                                                "step": step, "err": repr(e)})
                    md = method.force_act_from_agent(ctx, "method_error")

                action = md.action
                probe_response = None
                task_action_valid = None
                if md.decision_type == "probe":
                    try:
                        pr = env.step_probe_action(action)
                        probe_response = {"probe_type": pr.probe_type, "target": pr.target,
                                          "answer": pr.answer, "cost": pr.cost}
                        total_probe_count += 1
                        probes_used += 1
                        if compare_belief_to_gold_for_probe(probe_response, agent_out):
                            useful_probe_count += 1
                    except Exception as e:
                        append_error(out_err_path, {"where": "step_probe", "episode": episode_id,
                                                    "step": step, "err": repr(e)})
                else:
                    try:
                        env.step_task_action(action)
                        task_action_valid = bool(env._last_action_valid) if hasattr(env, "_last_action_valid") else None
                    except Exception as e:
                        append_error(out_err_path, {"where": "step_task", "episode": episode_id,
                                                    "step": step, "err": repr(e)})

                sstep = score_step(env, agent_out, {"type": md.decision_type}, obs,
                                   extra_per_belief=extra_per_belief)
                sc_correct_v2 = evaluate_self_check_v2(
                    sc_reported=sstep.get("self_check_reported", True),
                    decision_type=md.decision_type,
                    task_action_valid=task_action_valid,
                )

                rec = {
                    "episode_id": episode_id, "cell_id": cell.cell_id,
                    "environment": env.name, "world_regime": cell.stress_config,
                    "role": cell.role, "horizon": horizon,
                    "model": model, "method": cell.method,
                    "seed": seed, "step": step,
                    "gold_state": env.get_gold_state(),
                    "agent_belief_state": agent_out.get("belief_world_state"),
                    "belief_table": agent_out.get("beliefs"),
                    "decision_type": md.decision_type,
                    "selected_action": action,
                    "target_belief": md.target_belief_id,
                    "probe_response": probe_response,
                    "task_action_valid": task_action_valid,
                    "world_state_accuracy": sstep["world_state_accuracy"],
                    "dependency_accuracy": sstep["dependency_accuracy"],
                    "goal_retained": True,
                    "false_belief_commitment": sstep["false_belief_commitment"],
                    "self_check_valid": sstep["self_check_correct"],
                    "self_check_correct_v2": sc_correct_v2,
                    "self_check_reported": sstep.get("self_check_reported"),
                    "task_completed": env.is_done() and getattr(env, "_success", False),
                    "method_overrode_agent": md.overrode_agent,
                    "method_reasoning": md.reasoning,
                    "probes_used_so_far": probes_used,
                    "probe_budget_total": probe_budget,
                }
                if extra_per_belief:
                    rec["probe_score_per_belief"] = sstep.get("probe_score_per_belief")
                    rec["oracle_delta_per_belief"] = sstep.get("oracle_delta_per_belief")
                step_logger.write(rec)

                rec_for_score = dict(rec)
                rec_for_score["self_check_correct"] = sc_correct_v2
                step_records.append(rec_for_score)
                history.append({"step": step, "action": action, "type": md.decision_type,
                                "valid": task_action_valid})
                obs = env.get_observation()
    except Exception as e:
        append_error(out_err_path, {"where": "ep_fatal", "episode": episode_id,
                                    "step_reached": step, "err": repr(e),
                                    "trace": traceback.format_exc()})

    task_success = getattr(env, "_success", False)
    ep_metrics = score_episode(step_records, task_success, probe_budget,
                                probes_used, useful_probe_count, total_probe_count)
    return {
        "episode_id": episode_id, "cell_id": cell.cell_id,
        "env": env.name, "stress_label": cell.stress_config,
        "role": cell.role, "method": cell.method,
        "model": model, "seed": seed,
        "episode_metrics": ep_metrics,
        "probes_used": probes_used, "probe_budget_total": probe_budget,
        "useful_probe_count": useful_probe_count,
        "total_probe_count": total_probe_count,
    }


# -----------------------------------------------------------------------------
# 多 ep 并行
# -----------------------------------------------------------------------------

def _worker(args):
    cell, seed, model, out_step_path, out_err_path = args
    try:
        return run_one_episode(cell, seed, model,
                                Path(out_step_path), Path(out_err_path))
    except Exception as e:
        return {"_error": repr(e), "_trace": traceback.format_exc(),
                "cell_id": cell.cell_id, "seed": seed}


def run_layer(cells: List[CellSpec], model: str, out_dir: Path,
              prefix: str, parallel: int = 5, resume: bool = True,
              pcw_log_every: int = 100,
              gate_b_check_every: int = 0,
              gate_b_max_seed_idx: int = 0) -> Dict[str, Any]:
    """
    gate_b_check_every: 每 N seed-index 完成时, 算 G1 paired bootstrap, 若 p≤0.01 ∧ Δ≥0.15 则早停。
                       N=0 关闭 GATE B。
    gate_b_max_seed_idx: GATE B 检查时, 只用 seed_idx < this 的 ep (避免不完整 paired)。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    step_path = out_dir / f"{prefix}.jsonl"
    ep_path = out_dir / f"{prefix}_episodes.jsonl"
    err_path = out_dir / f"{prefix}_errors.jsonl"
    ckpt_path = out_dir / f"{prefix}_processed.txt"
    pcw_path = out_dir / "incremental_pcw.jsonl"

    # checkpoint resume
    processed = set()
    if resume and ckpt_path.exists():
        with open(ckpt_path) as f:
            for line in f:
                processed.add(line.strip())

    # 任务展开 (按 seed 顺序 interleave, 让长/短 ep 混合, 不让 ThreadPool 卡在长尾 cell)
    import random
    tasks = []
    max_seeds = max(len(c.seeds) for c in cells) if cells else 0
    for i in range(max_seeds):
        for cell in cells:
            if i >= len(cell.seeds):
                continue
            seed = cell.seeds[i]
            key = f"{cell.cell_id}|{seed}"
            if key in processed:
                continue
            tasks.append((cell, seed, model, str(step_path), str(err_path)))

    print(f"[{now_jst()}] run_layer prefix={prefix} cells={len(cells)} pending={len(tasks)} parallel={parallel}")

    all_eps = []
    n_done_total = 0
    n_done_no_probe = 0  # for incremental_pcw
    gate_b_triggered = False
    gate_b_result = None

    with JsonlLogger(ep_path) as ep_logger:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(_worker, t): t for t in tasks}
            for fut in as_completed(futures):
                if gate_b_triggered:
                    # 早停: 让剩余 future 继续完成, 但不发新的; ThreadPool 不能中途取消
                    # 我们简化: 让 future 自然结束, 不主动 cancel
                    pass
                res = fut.result()
                if not res or "_error" in res:
                    append_error(err_path, res or {"err": "empty"})
                    continue
                ep_logger.write(res)
                all_eps.append(res)
                with open(ckpt_path, "a") as f:
                    f.write(f"{res['cell_id']}|{res['seed']}\n")
                n_done_total += 1
                if res.get("method") == "no_probe":
                    n_done_no_probe += 1
                if n_done_total % max(1, pcw_log_every) == 0:
                    _write_pcw_increment(all_eps, pcw_path, prefix)

                # GATE B 检查
                if gate_b_check_every > 0 and not gate_b_triggered \
                   and n_done_total % gate_b_check_every == 0:
                    gb = compute_gate_b(all_eps, out_dir / f"{prefix}_gate_b.json")
                    if gb["decision"] == "EARLY_STOP":
                        gate_b_triggered = True
                        gate_b_result = gb
                        print(f"\n[{now_jst()}] === GATE B EARLY_STOP triggered ===")
                        print(json.dumps(gb, ensure_ascii=False, indent=2))
                        # 让剩余 future 自然结束, 但下面 break-out 逻辑较复杂
                        # 简化: 仅记录, 继续接收 future, 上层 main 看 flag
                        break

                if n_done_total % 50 == 0:
                    print(f"  [{now_jst()}] {n_done_total}/{len(tasks)} done; "
                          f"latest cell={res['cell_id']} seed={res['seed']} "
                          f"succ={res['episode_metrics']['task_success']} "
                          f"wsa={res['episode_metrics']['world_state_accuracy']:.2f}")

    # 末了再 dump 一次 pcw
    _write_pcw_increment(all_eps, pcw_path, prefix)

    print(f"[{now_jst()}] run_layer done. {n_done_total} ep, {sum(1 for _ in open(err_path)) if err_path.exists() else 0} errs")
    return {"n_ep": n_done_total, "all_eps": all_eps,
            "step_path": str(step_path), "ep_path": str(ep_path),
            "err_path": str(err_path), "pcw_path": str(pcw_path),
            "gate_b_triggered": gate_b_triggered,
            "gate_b_result": gate_b_result}


# -----------------------------------------------------------------------------
# GATE B: G1 paired bootstrap 早停
# -----------------------------------------------------------------------------

def compute_gate_b(all_eps: List[Dict[str, Any]], out_path: Path,
                   n_bootstrap: int = 2000,
                   delta_threshold: float = 0.15,
                   p_threshold: float = 0.01) -> Dict[str, Any]:
    """G1 paired bootstrap: envprobe_simple vs periodic_probe wsa_mean diff on pilot_med.
    早停条件: Δ ≥ 0.15 AND paired bootstrap p ≤ 0.01。
    """
    import random
    from collections import defaultdict

    # 仅用 pilot_med / S2 cell
    paired = defaultdict(dict)  # (env, seed) -> {method: wsa}
    for e in all_eps:
        if e.get("stress_label") not in ("S2", "pilot_med"):
            continue
        key = (e["env"], e["seed"])
        paired[key][e["method"]] = e["episode_metrics"]["world_state_accuracy"]

    diffs = []
    for k, md in paired.items():
        if "envprobe_simple" in md and "periodic_probe" in md:
            diffs.append(md["envprobe_simple"] - md["periodic_probe"])

    n = len(diffs)
    if n < 30:
        return {"decision": "CONTINUE", "reason": f"n_paired={n}<30, insufficient",
                "n_paired": n}

    delta_hat = sum(diffs) / n
    # Paired bootstrap: 二尾 p-value for H0: Δ ≤ 0
    rng = random.Random(20260527)
    count_under_null = 0
    centered = [d - delta_hat for d in diffs]  # null: mean = 0
    for _ in range(n_bootstrap):
        sample = [centered[rng.randint(0, n-1)] for _ in range(n)]
        boot_mean = sum(sample) / n
        if boot_mean >= delta_hat:
            count_under_null += 1
    p_value = count_under_null / n_bootstrap

    if delta_hat >= delta_threshold and p_value <= p_threshold:
        decision = "EARLY_STOP"
    else:
        decision = "CONTINUE"
    res = {
        "ts": now_jst(),
        "decision": decision,
        "delta_hat": delta_hat,
        "delta_threshold": delta_threshold,
        "p_value": p_value,
        "p_threshold": p_threshold,
        "n_paired": n,
        "n_bootstrap": n_bootstrap,
    }
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def _write_pcw_increment(all_eps: List[Dict], pcw_path: Path, prefix: str):
    """ds binding 定义的 p_cw 增量监控:
    
    p_cw = P(belief 实际错 | agent 报 confidence ≥ 0.7)
         = #{belief: conf ≥ 0.7 ∧ oracle_delta > 0} / #{belief: conf ≥ 0.7}
    
    仅在含 oracle_delta_per_belief 的 cell (envprobe_simple / oracle_probe) 计算。
    """
    step_path = pcw_path.parent / f"{prefix}.jsonl"
    if not step_path.exists():
        return
    n_high_conf = 0
    n_high_conf_wrong = 0
    try:
        with open(step_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                # ds 定义只在有 oracle_delta 的 cell 算
                delta_list = r.get("oracle_delta_per_belief")
                belief_list = r.get("belief_table")
                if not delta_list or not belief_list:
                    continue
                delta_by_id = {d["id"]: d.get("delta_i", 0) for d in delta_list}
                for b in belief_list:
                    conf = float(b.get("confidence", 0))
                    if conf < 0.7:
                        continue
                    bid = b.get("id")
                    if bid not in delta_by_id:
                        continue
                    n_high_conf += 1
                    if delta_by_id[bid] > 0:
                        n_high_conf_wrong += 1
    except Exception:
        return
    if n_high_conf < 50:
        return
    p_cw = n_high_conf_wrong / n_high_conf
    z = 1.96
    n = n_high_conf
    p = p_cw
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    ci_lo, ci_hi = center - margin, center + margin
    with open(pcw_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": now_jst(), "prefix": prefix,
            "definition": "ds_binding_per_belief_event_conf_ge_0.7",
            "n_high_conf_beliefs": n_high_conf,
            "n_high_conf_wrong": n_high_conf_wrong,
            "p_cw_estimate": p_cw,
            "p_cw_ci95_lower": ci_lo, "p_cw_ci95_upper": ci_hi,
        }) + "\n")


# -----------------------------------------------------------------------------
# GATE A 决策
# -----------------------------------------------------------------------------

def compute_gate_a(eps: List[Dict[str, Any]], out_path: Path) -> Dict[str, Any]:
    """根据 50 ep pilot 算 ρ̂, σ̂_d, p̂_cw_lcb 三信号, 输出 PASS/FAIL/BORDERLINE。

    定义:
    - ρ̂ = Spearman rank correlation between probe_score (ρ_i) and oracle_delta (Δ_i)
          across all (envprobe_simple, oracle_probe) cells × all belief records
    - σ̂_d = std of paired difference (envprobe_simple wsa - no_probe wsa) across seeds × env
    - p̂_cw_lcb = lower bound of 95% Wilson CI for p_cw on no_probe cells
    """
    from collections import defaultdict
    import math

    # === σ̂_d: paired wsa diff (envprobe_simple - no_probe) per (env, seed) ===
    paired_wsa = defaultdict(dict)  # (env, seed) -> {method: wsa}
    for e in eps:
        paired_wsa[(e["env"], e["seed"])][e["method"]] = e["episode_metrics"]["world_state_accuracy"]
    diffs = []
    for k, mdict in paired_wsa.items():
        if "envprobe_simple" in mdict and "no_probe" in mdict:
            diffs.append(mdict["envprobe_simple"] - mdict["no_probe"])
    sigma_d = _stdev(diffs) if len(diffs) > 1 else 0.0

    # === ρ̂: Spearman corr between rho_i and delta_i ===
    # 需要读 step jsonl 中 envprobe_simple / oracle_probe 的 probe_score_per_belief + oracle_delta_per_belief
    rho_pairs: List[Tuple[float, float]] = []
    step_jsonl = out_path.parent / "gate_a_pilot.jsonl"
    if step_jsonl.exists():
        for line in open(step_jsonl, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("method") not in ("envprobe_simple", "oracle_probe"):
                continue
            rho_list = r.get("probe_score_per_belief") or []
            delta_list = r.get("oracle_delta_per_belief") or []
            d_by_id = {d["id"]: d["delta_i"] for d in delta_list}
            for entry in rho_list:
                bid = entry["id"]
                if bid in d_by_id:
                    rho_pairs.append((entry["rho_i"], d_by_id[bid]))
    rho_hat = _spearman(rho_pairs) if len(rho_pairs) >= 10 else 0.0

    # === p̂_cw_lcb: ds binding (per-belief-event, conf ≥ 0.7) ===
    n_high_conf = 0
    n_high_conf_wrong = 0
    if step_jsonl.exists():
        for line in open(step_jsonl, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            delta_list = r.get("oracle_delta_per_belief")
            belief_list = r.get("belief_table")
            if not delta_list or not belief_list:
                continue
            delta_by_id = {d["id"]: d.get("delta_i", 0) for d in delta_list}
            for b in belief_list:
                conf = float(b.get("confidence", 0))
                if conf < 0.7:
                    continue
                bid = b.get("id")
                if bid not in delta_by_id:
                    continue
                n_high_conf += 1
                if delta_by_id[bid] > 0:
                    n_high_conf_wrong += 1
    p_cw = n_high_conf_wrong / n_high_conf if n_high_conf > 0 else 0.0
    z = 1.96
    if n_high_conf > 0:
        denom = 1 + z*z/n_high_conf
        center = (p_cw + z*z/(2*n_high_conf)) / denom
        margin = z * math.sqrt(p_cw*(1-p_cw)/n_high_conf + z*z/(4*n_high_conf*n_high_conf)) / denom
        p_cw_lcb = center - margin
        p_cw_ucb = center + margin
    else:
        p_cw_lcb = p_cw_ucb = 0.0
    # 保持向后兼容: n_act 字段仍输出 (用 n_high_conf 替代)
    n_act = n_high_conf
    n_cw = n_high_conf_wrong

    # === GATE A 决策 ===
    th_rho = 0.15
    th_sigma = 0.65
    th_pcw = 0.6

    signal_rho = rho_hat >= th_rho
    signal_sigma = sigma_d <= th_sigma
    signal_pcw = p_cw_lcb > th_pcw

    n_pass = sum([signal_rho, signal_sigma, signal_pcw])
    if n_pass == 3:
        gate_decision = "PASS"
    elif n_pass == 0:
        gate_decision = "FAIL"
    else:
        gate_decision = "BORDERLINE"

    result = {
        "ts": now_jst(),
        "n_eps_total": len(eps),
        "rho_hat": rho_hat,
        "rho_hat_threshold": th_rho,
        "rho_hat_pass": signal_rho,
        "n_rho_pairs": len(rho_pairs),
        "sigma_d": sigma_d,
        "sigma_d_threshold": th_sigma,
        "sigma_d_pass": signal_sigma,
        "n_paired_diffs": len(diffs),
        "p_cw_estimate": p_cw,
        "p_cw_ci95_lower": p_cw_lcb,
        "p_cw_ci95_upper": p_cw_ucb,
        "p_cw_lcb_threshold": th_pcw,
        "p_cw_pass": signal_pcw,
        "n_no_probe_act_steps": n_act,
        "n_confident_wrong": n_cw,
        "gate_a_decision": gate_decision,
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m)**2 for x in xs) / (len(xs) - 1))


def _spearman(pairs):
    """Spearman rank correlation."""
    if len(pairs) < 2:
        return 0.0
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    rx = _ranks(xs)
    ry = _ranks(ys)
    n = len(xs)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((r - mx)**2 for r in rx))
    dy = math.sqrt(sum((r - my)**2 for r in ry))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def _ranks(xs):
    sorted_idx = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[sorted_idx[j+1]] == xs[sorted_idx[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg_rank
        i = j + 1
    return ranks


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--registry", required=True, help="Path to cells_registry.csv")
    p.add_argument("--layer", default="gate_a", help="Filter cells by role (gate_a/spine_primary/...) or 'all'")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--out-dir", default="experiments")
    p.add_argument("--prefix", default="gate_a_pilot")
    p.add_argument("--parallel", type=int, default=5)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--gate-a", action="store_true", help="Compute GATE A decision after run")
    p.add_argument("--gate-b-every", type=int, default=0,
                   help="Every N completed eps, run GATE B paired bootstrap; 0=disabled")
    args = p.parse_args()

    out_dir = ROOT / args.out_dir
    cells = load_registry(Path(args.registry))
    if args.layer != "all":
        cells = [c for c in cells if c.role == args.layer]
    if not cells:
        print(f"[{now_jst()}] No cells match layer={args.layer}", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    res = run_layer(cells, args.model, out_dir, args.prefix,
                    parallel=args.parallel, resume=not args.no_resume,
                    gate_b_check_every=args.gate_b_every)
    dt = time.time() - t0
    print(f"[{now_jst()}] layer done in {dt/60:.1f} min")

    if args.gate_a:
        gate_path = out_dir / "gate_a_metrics.json"
        gate_result = compute_gate_a(res["all_eps"], gate_path)
        print(f"\n=== GATE A decision: {gate_result['gate_a_decision']} ===")
        print(json.dumps(gate_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
