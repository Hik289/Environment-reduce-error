"""GraphNavWorld: 图导航世界, 节点 + 锁定边 + 钥匙。"""
from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Tuple

from .base import Environment, ProbeResult, StepResult, flat_overlap


N_NODES_BY_CARD = {"low": 6, "medium": 12, "high": 20}
N_LOCKED_BY_DEN = {"low": 1, "medium": 3, "high": 6}


class GraphNavWorld(Environment):
    name = "GraphNavWorld"
    _ACT_RE = re.compile(r"^(\w+)\(([^)]*)\)$")

    def __init__(self):
        self._rng = random.Random(0)
        self._step = 0
        self._horizon = 20
        self._stress: Dict[str, Any] = {}
        self._gold: Dict[str, Any] = {}
        self._done = False
        self._success = False
        self._last_action_valid = True
        self._invalid_count = 0
        self._last_obs: Dict[str, Any] = {}
        self._prev_node: Optional[str] = None

    def reset(self, seed: int, stress_config: Dict[str, Any]) -> Dict[str, Any]:
        self._rng = random.Random(int(seed))
        self._step = 0
        self._stress = dict(stress_config or {})
        self._horizon = int(self._stress.get("horizon", 20))
        n = N_NODES_BY_CARD[self._stress.get("state_cardinality", "low")]
        dep = self._stress.get("dependency_density", "low")

        nodes = [f"n_{i}" for i in range(n)]
        edges: Dict[str, Dict[str, Any]] = {}

        def edge_id(a: str, b: str) -> str:
            x, y = sorted([a, b])
            return f"{x}__{y}"

        # 链式 spanning
        for i in range(n - 1):
            eid = edge_id(nodes[i], nodes[i+1])
            edges[eid] = {"a": nodes[i], "b": nodes[i+1], "locked": False, "required_key": None}

        # 额外边
        extra = {"low": n // 4, "medium": n // 2, "high": n}[dep]
        for _ in range(extra):
            a, b = self._rng.sample(nodes, 2)
            eid = edge_id(a, b)
            if eid not in edges:
                edges[eid] = {"a": a, "b": b, "locked": False, "required_key": None}

        keys = [f"key_{c}" for c in ["a", "b", "c", "d", "e"][: max(2, n // 4)]]
        n_locked = min(N_LOCKED_BY_DEN[dep], len(edges), len(keys))
        if n_locked > 0:
            locked_eids = self._rng.sample(list(edges.keys()), k=n_locked)
            chosen_keys = self._rng.sample(keys, k=n_locked) if len(keys) >= n_locked else \
                [self._rng.choice(keys) for _ in range(n_locked)]
            for eid, k in zip(locked_eids, chosen_keys):
                edges[eid]["locked"] = True
                edges[eid]["required_key"] = k

        key_locations = {k: self._rng.choice(nodes) for k in keys}

        self._gold = {
            "nodes": nodes,
            "edges": edges,
            "keys": keys,
            "key_locations": key_locations,
            "inventory": [],
            "agent_node": nodes[0],
            "target_node": nodes[-1],
            "visited": [nodes[0]],
        }
        self._done = False
        self._success = False
        self._last_action_valid = True
        self._invalid_count = 0
        self._prev_node = None
        self._last_obs = self._build_obs(after_action=None)
        return self._last_obs

    def _neighbors(self, n: str) -> List[str]:
        out = []
        for eid, e in self._gold["edges"].items():
            if e["a"] == n:
                out.append(e["b"])
            elif e["b"] == n:
                out.append(e["a"])
        return sorted(out)

    def _edge_lookup(self, a: str, b: str) -> Optional[Dict[str, Any]]:
        x, y = sorted([a, b])
        return self._gold["edges"].get(f"{x}__{y}")

    def _build_obs(self, after_action: Optional[str]) -> Dict[str, Any]:
        noise = self._stress.get("observation_noise", "clean")
        n = self._gold["agent_node"]
        keys_here = sorted([k for k, l in self._gold["key_locations"].items() if l == n])
        neighbors = self._neighbors(n)
        obs: Dict[str, Any] = {
            "current_node": n,
            "neighbors": neighbors,
            "keys_in_view": keys_here,
            "inventory": list(self._gold["inventory"]),
            "target_node": self._gold["target_node"],
            "step": self._step,
            "horizon": self._horizon,
            "after_action": after_action,
            "last_action_valid": self._last_action_valid,
        }
        if noise == "partial" and neighbors and self._rng.random() < 0.4:
            obs["neighbors"] = neighbors[:-1]
        elif noise == "distractor":
            obs["neighbors"] = neighbors + [f"n_fake_{self._rng.randint(0, 99)}"]
        elif noise == "delayed" and self._prev_node and self._rng.random() < 0.3:
            obs["current_node"] = self._prev_node
        self._prev_node = n
        return obs

    def step_task_action(self, action: str) -> StepResult:
        if self._done:
            return StepResult(self._last_obs, 0.0, True, {"already_done": True})
        self._step += 1
        info: Dict[str, Any] = {"action_raw": action}
        valid = False
        m = self._ACT_RE.match((action or "").strip())
        if m:
            verb = m.group(1).lower()
            arg = m.group(2).strip().strip('"').strip("'")
            valid = self._apply(verb, arg, info)
        else:
            info["reason"] = "parse_error"
        self._last_action_valid = valid
        if not valid:
            self._invalid_count += 1

        if self._gold["agent_node"] == self._gold["target_node"]:
            self._done = True
            self._success = True
            info["completed"] = True
        if self._step >= self._horizon and not self._done:
            self._done = True
            info["horizon_reached"] = True

        self._apply_mutation()
        self._last_obs = self._build_obs(after_action=action)
        reward = 1.0 if self._success else (0.0 if valid else -0.05)
        return StepResult(self._last_obs, reward, self._done, info)

    def _apply(self, verb: str, arg: str, info: Dict[str, Any]) -> bool:
        g = self._gold
        if verb == "move_to":
            if arg not in g["nodes"]:
                info["reason"] = "unknown_node"
                return False
            e = self._edge_lookup(g["agent_node"], arg)
            if e is None:
                info["reason"] = "no_edge"
                return False
            if e["locked"] and e["required_key"] not in g["inventory"]:
                info["reason"] = "edge_locked"
                return False
            g["agent_node"] = arg
            if arg not in g["visited"]:
                g["visited"].append(arg)
            return True
        if verb == "collect_key":
            if g["key_locations"].get(arg) != g["agent_node"]:
                info["reason"] = "key_not_here"
                return False
            g["inventory"].append(arg)
            g["key_locations"][arg] = "_inventory_"
            return True
        if verb == "activate_switch":
            info["reason"] = "no_switches"
            return False
        info["reason"] = f"unknown_verb:{verb}"
        return False

    def _apply_mutation(self) -> None:
        rate = self._stress.get("state_mutation_rate", "static")
        if rate == "static":
            return
        prob = 0.10 if rate == "mild" else 0.22
        # 锁定状态翻转 (require_key 不变, 但 locked 翻转)
        for eid, e in self._gold["edges"].items():
            if e.get("required_key") and self._rng.random() < prob:
                e["locked"] = not e["locked"]
        # 钥匙漂移
        loose = [k for k, l in self._gold["key_locations"].items() if l != "_inventory_"]
        for k in loose:
            if self._rng.random() < prob:
                self._gold["key_locations"][k] = self._rng.choice(self._gold["nodes"])

    def step_probe_action(self, probe: str) -> ProbeResult:
        m = self._ACT_RE.match((probe or "").strip())
        if not m:
            return ProbeResult("invalid", probe, {"error": "parse_error"}, cost=1.0)
        verb = m.group(1).lower()
        arg = m.group(2).strip().strip('"').strip("'")
        g = self._gold
        if verb == "check_current_node":
            return ProbeResult("check_current_node", None, {"node": g["agent_node"]}, cost=1.0)
        if verb == "check_edge":
            parts = [p.strip() for p in arg.split(",")]
            if len(parts) != 2:
                return ProbeResult("check_edge", arg, {"error": "bad_args"}, cost=1.0)
            e = self._edge_lookup(parts[0], parts[1])
            if e is None:
                return ProbeResult("check_edge", arg, {"exists": False}, cost=1.0)
            return ProbeResult("check_edge", arg,
                {"exists": True, "locked": e["locked"], "required_key": e["required_key"]}, cost=1.0)
        if verb == "check_node_locked":
            # 节点不锁定本身, 但报告它周围有锁定边
            locks = [eid for eid, e in g["edges"].items() if e["locked"] and (e["a"] == arg or e["b"] == arg)]
            return ProbeResult("check_node_locked", arg, {"locked_neighbor_edges": locks}, cost=1.0)
        if verb == "check_required_key":
            # arg = edge_id or "n_a,n_b"
            e = None
            if "," in arg:
                parts = [p.strip() for p in arg.split(",")]
                if len(parts) == 2:
                    e = self._edge_lookup(parts[0], parts[1])
            else:
                e = g["edges"].get(arg)
            if not e:
                return ProbeResult("check_required_key", arg, {"error": "no_edge"}, cost=1.0)
            return ProbeResult("check_required_key", arg, {"required_key": e["required_key"]}, cost=1.0)
        if verb == "inspect_neighbors":
            return ProbeResult("inspect_neighbors", arg,
                {"neighbors": self._neighbors(arg) if arg in g["nodes"] else []}, cost=1.0)
        if verb == "check_target_distance_hint":
            # BFS hint
            from collections import deque
            start = g["agent_node"]
            target = g["target_node"]
            visited = {start}
            q = deque([(start, 0)])
            dist = -1
            while q:
                v, d = q.popleft()
                if v == target:
                    dist = d
                    break
                for nb in self._neighbors(v):
                    e = self._edge_lookup(v, nb)
                    if e and e["locked"]:
                        continue
                    if nb not in visited:
                        visited.add(nb)
                        q.append((nb, d + 1))
            return ProbeResult("check_target_distance_hint", None, {"hops_to_target": dist}, cost=1.0)
        if verb == "check_location":
            return ProbeResult("check_location", arg,
                {"location": g["key_locations"].get(arg, "unknown")}, cost=1.0)
        return ProbeResult("unknown_probe", arg, {"error": f"unknown probe {verb}"}, cost=1.0)

    def get_observation(self) -> Dict[str, Any]:
        return self._last_obs

    def get_gold_state(self) -> Dict[str, Any]:
        g = dict(self._gold)
        # edges 输出为 list-of-dict 便于序列化, 保留 edge_id key
        g["edges"] = {eid: dict(e) for eid, e in g["edges"].items()}
        return g

    def score_belief_state(self, belief: Dict[str, Any]) -> float:
        gold = self.get_gold_state()
        if not isinstance(belief, dict):
            return 0.0
        bs = belief.get("belief_world_state", belief)
        slots = [
            ("current_location", bs.get("current_location"), gold.get("agent_node"), 0.25),
            ("inventory", bs.get("inventory"), gold.get("inventory"), 0.20),
            ("door_states", bs.get("door_states"),
             {eid: ("locked" if e["locked"] else "open") for eid, e in gold.get("edges", {}).items()}, 0.30),
            ("object_locations", bs.get("object_locations"),
             gold.get("key_locations"), 0.25),
        ]
        return sum(w * flat_overlap(b, g) for _, b, g, w in slots)

    def task_description(self) -> str:
        return (f"You start at {self._gold['nodes'][0]} in a graph with {len(self._gold['nodes'])} nodes. "
                f"Some edges are locked and need specific keys (lying on nodes). "
                f"Goal: reach node '{self._gold['target_node']}'.")

    def available_task_actions(self) -> List[str]:
        return ["move_to(n_X)", "collect_key(key_X)"]

    def available_probe_actions(self) -> List[str]:
        return [
            "check_current_node()", "check_edge(n_A, n_B)",
            "check_node_locked(n_X)", "check_required_key(n_A, n_B)",
            "inspect_neighbors(n_X)", "check_target_distance_hint()",
            "check_location(key_X)",
        ]

    def is_done(self) -> bool:
        return self._done
