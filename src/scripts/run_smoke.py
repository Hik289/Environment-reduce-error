"""Anchor_1 smoke test runner.

5 episodes × ObjectStateWorld × {pilot_low, pilot_high} × {no_probe, periodic_probe, envprobe_simple, oracle_probe} × gpt-4o-mini

输出:
- experiments/anchor_1_smoke.jsonl (per-step records)
- experiments/anchor_1_episodes.jsonl (per-episode metrics)
- experiments/anchor_1_summary.md (aggregated table)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agents.llm_agent import LLMAgent
from src.environments import make_environment, default_stress
from src.methods import get_method
from src.methods.envprobe_judge import EnvProbeJudgeMethod
from src.metrics.scorer import score_step, score_episode, compare_belief_to_gold_for_probe, evaluate_self_check_v2
from src.metrics.aggregator import aggregate_episodes, write_pilot_table
from src.utils.api_client import LLMClient
from src.utils.logger import JsonlLogger, append_error


JST = timezone(timedelta(hours=9))


def now_jst() -> str:
    return datetime.now(JST).isoformat(timespec="seconds")


def run_episode(
    env_name: str,
    stress_label: str,
    method_name: str,
    model: str,
    seed: int,
    step_logger: JsonlLogger,
    errors_path: Path,
) -> Optional[Dict[str, Any]]:
    env = make_environment(env_name)
    stress = default_stress(stress_label)
    obs = env.reset(seed=seed, stress_config=stress)
    horizon = int(stress["horizon"])
    probe_budget = max(1, horizon // 4)

    client = LLMClient(model=model)
    agent = LLMAgent(model=model, client=client, max_tokens=1200)
    agent.reset(env.name, env.task_description())

    method = get_method(method_name)
    method.reset_episode()
    if isinstance(method, EnvProbeJudgeMethod):
        method.attach_agent(agent)

    episode_id = f"{env_name}-{method_name}-{stress_label}-{seed}-{uuid.uuid4().hex[:6]}"
    step = 0
    history: List[Dict[str, Any]] = []
    step_records: List[Dict[str, Any]] = []
    probes_used = 0
    useful_probe_count = 0
    total_probe_count = 0

    while not env.is_done() and step < horizon:
        step += 1
        agent_out = agent.step(
            observation=obs,
            history=history[-5:],
            step=step,
            horizon=horizon,
            method_hint=method.method_hint,
            task_action_spec=env.available_task_actions(),
            probe_action_spec=env.available_probe_actions(),
            probe_budget_remaining=probe_budget - probes_used,
            probe_force_act=(probes_used >= probe_budget),
        )

        ctx_kwargs = dict(
            step=step, horizon=horizon, observation=obs, agent_output=agent_out,
            task_action_spec=env.available_task_actions(),
            probe_action_spec=env.available_probe_actions(),
            env=env, history=history,
            probe_budget_total=probe_budget, probes_used=probes_used,
        )
        from src.methods.base import MethodContext
        ctx = MethodContext(**ctx_kwargs)

        try:
            method_decision = method.decide(ctx)
        except Exception as e:
            append_error(errors_path, {"where": "method.decide", "episode": episode_id,
                                       "step": step, "err": repr(e), "trace": traceback.format_exc()})
            method_decision = method.force_act_from_agent(ctx, "method_error")

        # Step the env
        action = method_decision.action
        probe_response: Optional[Dict[str, Any]] = None
        task_action_valid: Optional[bool] = None
        done_after = False

        if method_decision.decision_type == "probe":
            try:
                pr = env.step_probe_action(action)
                probe_response = {"probe_type": pr.probe_type, "target": pr.target,
                                  "answer": pr.answer, "cost": pr.cost}
                total_probe_count += 1
                probes_used += 1
                # useful? — 比较 probe answer 与 agent belief
                if compare_belief_to_gold_for_probe(probe_response, agent_out):
                    useful_probe_count += 1
            except Exception as e:
                append_error(errors_path, {"where": "step_probe", "episode": episode_id,
                                           "step": step, "err": repr(e), "trace": traceback.format_exc()})
                probe_response = {"error": repr(e)}
        else:
            # act
            try:
                sr = env.step_task_action(action)
                task_action_valid = bool(env._last_action_valid) if hasattr(env, "_last_action_valid") else None
                done_after = sr.done
            except Exception as e:
                append_error(errors_path, {"where": "step_task", "episode": episode_id,
                                           "step": step, "err": repr(e), "trace": traceback.format_exc()})

        # 打分
        sstep = score_step(env, agent_out, {"type": method_decision.decision_type}, obs)

        # RF4: 用 v2 (next-act 合法性) 重写 self_check_correct
        sc_correct_v2 = evaluate_self_check_v2(
            sc_reported=sstep.get("self_check_reported", True),
            decision_type=method_decision.decision_type,
            task_action_valid=task_action_valid,
        )

        # 组装 step record (14 metric 之 step 级片段)
        rec = {
            "episode_id": episode_id,
            "environment": env.name,
            "world_regime": stress_label,
            "horizon": horizon,
            "model": model,
            "method": method_name,
            "step": step,
            "gold_state": env.get_gold_state(),
            "agent_belief_state": agent_out.get("belief_world_state"),
            "belief_table": agent_out.get("beliefs"),
            "decision_type": method_decision.decision_type,
            "selected_action": action,
            "target_belief": method_decision.target_belief_id,
            "probe_response": probe_response,
            "task_action_valid": task_action_valid,
            "world_state_accuracy": sstep["world_state_accuracy"],
            "dependency_accuracy": sstep["dependency_accuracy"],
            "goal_retained": True,  # 简化: 未跟踪 goal drift
            "false_belief_commitment": sstep["false_belief_commitment"],
            "self_check_valid": sstep["self_check_correct"],  # 旧 proxy, 保留供回归
            "self_check_correct_v2": sc_correct_v2,            # RF4 新版 (act 步: bool, probe 步: None)
            "self_check_reported": sstep.get("self_check_reported"),
            "task_completed": env.is_done() and getattr(env, "_success", False),
            "method_overrode_agent": method_decision.overrode_agent,
            "method_reasoning": method_decision.reasoning,
            "probes_used_so_far": probes_used,
            "probe_budget_total": probe_budget,
        }
        step_logger.write(rec)
        # 给 score_episode 用 v2 self_check_correct (None for probe steps)
        rec_for_score = dict(rec)
        rec_for_score["self_check_correct"] = sc_correct_v2
        step_records.append(rec_for_score)
        history.append({"step": step, "action": action, "type": method_decision.decision_type,
                        "valid": task_action_valid, "obs_summary": {k: v for k, v in obs.items()
                                                                     if not isinstance(v, (dict, list))}})

        # 更新 observation (act 后 env 已 step, probe 没改 obs)
        obs = env.get_observation()

    # Episode 完结
    task_success = getattr(env, "_success", False)
    ep_metrics = score_episode(step_records, task_success, probe_budget,
                                probes_used, useful_probe_count, total_probe_count)
    return {
        "episode_id": episode_id, "env": env.name, "stress_label": stress_label,
        "method": method_name, "model": model, "seed": seed,
        "episode_metrics": ep_metrics,
        "probes_used": probes_used, "probe_budget_total": probe_budget,
        "useful_probe_count": useful_probe_count, "total_probe_count": total_probe_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", default=["ObjectStateWorld"])
    parser.add_argument("--stress", nargs="+", default=["pilot_low", "pilot_high"])
    parser.add_argument("--methods", nargs="+",
                        default=["no_probe", "periodic_probe", "envprobe_simple", "oracle_probe"])
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--out-dir", default="experiments")
    parser.add_argument("--prefix", default="anchor_1",
                        help="Prefix for output files (e.g. anchor_1_5_mini_validation)")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix
    if prefix == "anchor_1":
        step_path = out_dir / "anchor_1_smoke.jsonl"
        ep_path = out_dir / "anchor_1_episodes.jsonl"
        err_path = out_dir / "anchor_1_errors.jsonl"
        summary_path = out_dir / "anchor_1_summary.md"
    else:
        step_path = out_dir / f"{prefix}.jsonl"
        ep_path = out_dir / f"{prefix.replace('_mini_validation','')}_episodes.jsonl"
        err_path = out_dir / f"{prefix}_errors.jsonl"
        summary_path = out_dir / f"{prefix.replace('_mini_validation','')}_summary.md"

    # 清空旧文件
    for p in [step_path, ep_path, err_path]:
        if p.exists():
            p.unlink()

    print(f"[{now_jst()}] anchor_1 smoke start: envs={args.envs} stress={args.stress} methods={args.methods}")
    all_eps: List[Dict[str, Any]] = []
    grid = [(env, s, m, args.base_seed + i)
            for env in args.envs for s in args.stress for m in args.methods
            for i in range(args.episodes)]
    print(f"  total cells = {len(grid)}")

    with JsonlLogger(step_path) as step_logger, JsonlLogger(ep_path) as ep_logger:
        for idx, (env, s, m, seed) in enumerate(grid, 1):
            t0 = time.time()
            try:
                res = run_episode(env, s, m, args.model, seed, step_logger, err_path)
            except Exception as e:
                append_error(err_path, {"where": "run_episode", "env": env, "method": m,
                                        "stress": s, "seed": seed, "err": repr(e),
                                        "trace": traceback.format_exc()})
                continue
            if res is None:
                continue
            ep_logger.write(res)
            all_eps.append(res)
            elapsed = time.time() - t0
            mt = res["episode_metrics"]
            print(f"  [{idx}/{len(grid)}] {env}|{m}|{s}|seed{seed} succ={mt['task_success']} "
                  f"wsa={mt['world_state_accuracy']:.2f} probes={res['probes_used']}/{res['probe_budget_total']} "
                  f"t={elapsed:.1f}s")

    # 聚合
    agg = aggregate_episodes(all_eps)
    write_pilot_table(agg, summary_path)
    print(f"[{now_jst()}] anchor_1 smoke done. summary -> {summary_path}")
    # 错误率
    n_errors = 0
    if err_path.exists():
        n_errors = sum(1 for _ in open(err_path, encoding="utf-8"))
    total_steps = sum(ep["episode_metrics"]["n_steps"] for ep in all_eps)
    err_rate = n_errors / max(1, total_steps)
    print(f"  total episodes: {len(all_eps)}; total steps: {total_steps}; errors: {n_errors}; "
          f"err_rate: {err_rate:.3%}")


if __name__ == "__main__":
    main()
