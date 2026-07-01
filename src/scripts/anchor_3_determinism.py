"""Anchor_3: 环境确定性自检。

3 env × 20 trajectory × 2 调用 = 60 hash, 100% 匹配。
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.environments import make_environment, default_stress


def canonical_dump(obj) -> str:
    return json.dumps(_normalize(obj), sort_keys=True, default=str, ensure_ascii=False)


def _normalize(o):
    if isinstance(o, dict):
        return {str(k): _normalize(o[k]) for k in sorted(o.keys(), key=str)}
    if isinstance(o, (list, tuple)):
        return [_normalize(x) for x in o]
    if isinstance(o, set):
        return sorted([_normalize(x) for x in o], key=str)
    return o


def hash_trajectory(traj) -> str:
    return hashlib.sha256(canonical_dump(traj).encode("utf-8")).hexdigest()


def random_action(env_name: str, rng: random.Random, obs: dict, gold: dict) -> str:
    if env_name == "ObjectStateWorld":
        choices = []
        if obs.get("objects_in_view"):
            choices.append(f"pick_up({obs['objects_in_view'][0]})")
        rooms = gold.get("rooms", [])
        if rooms:
            choices.append(f"move_to({rng.choice(rooms)})")
        doors = list(gold.get("door_states", {}).keys())
        if doors:
            choices.append(f"unlock({rng.choice(doors)})")
        return rng.choice(choices) if choices else "noop()"
    if env_name == "ToolDAGWorld":
        tools = gold.get("tools", ["t_0"])
        return f"call_tool({rng.choice(tools)})"
    if env_name == "GraphNavWorld":
        choices = []
        neighbors = obs.get("neighbors") or []
        if neighbors:
            choices.append(f"move_to({rng.choice(neighbors)})")
        keys_here = obs.get("keys_in_view") or []
        if keys_here:
            choices.append(f"collect_key({keys_here[0]})")
        return rng.choice(choices) if choices else "move_to(n_0)"
    return "noop()"


def one_trajectory(env_name: str, seed: int, action_seed: int, stress_label: str):
    env = make_environment(env_name)
    stress = default_stress(stress_label)
    obs = env.reset(seed=seed, stress_config=stress)
    rng = random.Random(action_seed)
    traj = [env.canonical_gold()]
    horizon = int(stress["horizon"])
    actions = []
    for step in range(horizon):
        gold = env.get_gold_state()
        action = random_action(env_name, rng, obs, gold)
        actions.append(action)
        sr = env.step_task_action(action)
        traj.append(env.canonical_gold())
        obs = env.get_observation()
        if sr.done:
            break
    return traj, actions


def main():
    out_path = ROOT / "experiments" / "anchor_3_determinism.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    envs = ["ObjectStateWorld", "ToolDAGWorld", "GraphNavWorld"]
    stress_label = "pilot_low"
    n_trajs = 20
    results = {"per_env": {}, "n_trajs_per_env": n_trajs, "total_hashes": 0, "matched": 0}

    for env_name in envs:
        env_block = {"trajs": [], "matched": 0, "total": 0}
        for i in range(n_trajs):
            seed = 1000 + i
            action_seed = 5000 + i

            # 复用同样的 action 序列, 跑两次, hash 必须一致
            traj1, actions1 = one_trajectory(env_name, seed, action_seed, stress_label)
            traj2, actions2 = one_trajectory(env_name, seed, action_seed, stress_label)
            h1 = hash_trajectory(traj1)
            h2 = hash_trajectory(traj2)
            match = (h1 == h2 and actions1 == actions2)
            env_block["trajs"].append({
                "seed": seed, "action_seed": action_seed,
                "hash_1": h1[:16], "hash_2": h2[:16], "n_steps": len(traj1) - 1,
                "match": match,
            })
            env_block["total"] += 1
            if match:
                env_block["matched"] += 1
            results["total_hashes"] += 2
            if match:
                results["matched"] += 2

        env_block["match_rate"] = env_block["matched"] / max(1, env_block["total"])
        results["per_env"][env_name] = env_block
        print(f"  {env_name}: {env_block['matched']}/{env_block['total']} match "
              f"(rate={env_block['match_rate']:.2%})")

    results["overall_match_rate"] = results["matched"] / max(1, results["total_hashes"])
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nOverall: {results['matched']}/{results['total_hashes']} = {results['overall_match_rate']:.2%}")
    if results["overall_match_rate"] < 1.0:
        sys.exit(1)


if __name__ == "__main__":
    main()
