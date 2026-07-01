"""ObjectStateWorld: 房间 / 钥匙 / 箱子 / 锁门 的文本世界。

Gold state:
- agent_location, inventory
- object_locations (key -> room | "_inventory_")
- door_states (door_id -> "open"|"locked")
- door_specs (door_id -> {from, to, required_key})
- goal_object (须 pick_up 到 inventory 才算成功)

确定性: 所有随机性都来自 self._rng = random.Random(seed)。
"""
from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional

from .base import Environment, ProbeResult, StepResult, flat_overlap


N_ROOMS_BY_CARD = {"low": 5, "medium": 10, "high": 20}
N_LOCKED_BY_DEP = {"low": 1, "medium": 3, "high": 6}


class ObjectStateWorld(Environment):
    name = "ObjectStateWorld"

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
        self._prev_loc: Optional[str] = None

    # ---------------- reset ----------------
    def reset(self, seed: int, stress_config: Dict[str, Any]) -> Dict[str, Any]:
        self._rng = random.Random(int(seed))
        self._step = 0
        self._stress = dict(stress_config or {})
        self._horizon = int(self._stress.get("horizon", 20))
        n_rooms = N_ROOMS_BY_CARD[self._stress.get("state_cardinality", "low")]
        dep = self._stress.get("dependency_density", "low")

        rooms = [f"room_{i}" for i in range(n_rooms)]
        keys = [f"{c}_key" for c in ["blue", "red", "green", "yellow", "white", "black"][: max(2, n_rooms // 3)]]
        boxes = [f"box_{i}" for i in range(max(1, n_rooms // 4))]
        targets = ["map", "lantern", "scroll"][: max(1, n_rooms // 4)]
        objects = list(keys) + boxes + targets

        # 串行门连接 room_i <-> room_{i+1}
        door_specs: Dict[str, Dict[str, Any]] = {}
        for i in range(n_rooms - 1):
            did = f"door_{i+1}"
            door_specs[did] = {"from": rooms[i], "to": rooms[i+1], "locked": False, "required_key": None}

        # 锁定若干门
        n_locked = min(N_LOCKED_BY_DEP.get(dep, 1), len(door_specs), len(keys))
        if n_locked > 0:
            locked_ids = self._rng.sample(list(door_specs.keys()), k=n_locked)
            chosen_keys = self._rng.sample(keys, k=n_locked)
            for did, k in zip(locked_ids, chosen_keys):
                door_specs[did]["locked"] = True
                door_specs[did]["required_key"] = k

        # 物体随机放置 — 但用 BFS 保证: 任意 locked door 所需 key 必须在该门之前 reachable
        # 简单做法: 按 door 顺序处理, room_i+1 之后的钥匙不放在 room_i 之前不可达的位置
        object_locations: Dict[str, str] = {}
        # 钥匙放置规则: door_k 需要的 key 必须能在到达 door_k 之前 (room_0..room_{k-1}) 拿到
        # 先把锁定门按 from-room 顺序排序, 对每个 locked door, 把它的 required_key 放到 <= from_room 的某个 room
        sorted_locked = sorted(
            [(did, spec) for did, spec in door_specs.items() if spec["locked"]],
            key=lambda x: int(x[0].split("_")[1])
        )
        for did, spec in sorted_locked:
            from_idx = int(did.split("_")[1]) - 1  # door_k connects room_{k-1}-room_k
            req_key = spec["required_key"]
            # 放在 [room_0 .. room_{from_idx}] 之一
            object_locations[req_key] = self._rng.choice(rooms[:from_idx + 1])
        # 剩下的 keys 随机
        for k in keys:
            if k not in object_locations:
                object_locations[k] = self._rng.choice(rooms)
        # boxes / targets 随机
        for obj in boxes + targets:
            object_locations[obj] = self._rng.choice(rooms)

        # 选 goal: 优先非 key 的 target
        non_key_objs = [o for o in objects if not o.endswith("_key")]
        goal_object = self._rng.choice(non_key_objs) if non_key_objs else objects[-1]

        # dependencies: 解锁门需要 key in inventory
        dependencies: Dict[str, List[str]] = {}
        for did, spec in door_specs.items():
            if spec["locked"]:
                dependencies[f"unlock_{did}"] = [f"has_{spec['required_key']}"]
        dependencies[f"pick_up_{goal_object}"] = [f"at_{object_locations[goal_object]}"]

        self._gold = {
            "rooms": rooms,
            "agent_location": rooms[0],
            "inventory": [],
            "object_locations": object_locations,
            "door_states": {did: ("locked" if spec["locked"] else "open") for did, spec in door_specs.items()},
            "door_specs": door_specs,
            "completed_subgoals": [],
            "dependencies": dependencies,
            "goal_object": goal_object,
            "keys": keys,
            "boxes": boxes,
            "targets": targets,
        }
        self._done = False
        self._success = False
        self._last_action_valid = True
        self._invalid_count = 0
        self._prev_loc = None
        self._last_obs = self._build_obs(after_action=None)
        return self._last_obs

    # ---------------- observation ----------------
    def _build_obs(self, after_action: Optional[str]) -> Dict[str, Any]:
        noise = self._stress.get("observation_noise", "clean")
        loc = self._gold["agent_location"]
        objs_here = sorted([o for o, r in self._gold["object_locations"].items() if r == loc])
        doors_here = sorted([did for did, d in self._gold["door_specs"].items()
                             if d["from"] == loc or d["to"] == loc])
        obs: Dict[str, Any] = {
            "current_location": loc,
            "objects_in_view": objs_here,
            "doors_in_view": doors_here,
            "inventory": list(self._gold["inventory"]),
            "step": self._step,
            "horizon": self._horizon,
            "goal_hint": f"obtain '{self._gold['goal_object']}'",
            "after_action": after_action,
            "last_action_valid": self._last_action_valid,
        }
        if noise == "partial" and objs_here and self._rng.random() < 0.4:
            obs["objects_in_view"] = objs_here[:-1]
        elif noise == "distractor":
            obs["objects_in_view"] = objs_here + [f"distractor_{self._rng.randint(0, 999)}"]
        elif noise == "delayed" and self._prev_loc and self._rng.random() < 0.3:
            obs["current_location"] = self._prev_loc
        self._prev_loc = loc
        return obs

    # ---------------- task action ----------------
    def _parse(self, s: str):
        if not isinstance(s, str):
            return None
        m = self._ACT_RE.match(s.strip())
        if not m:
            return None
        return m.group(1).lower(), m.group(2).strip().strip('"').strip("'")

    def step_task_action(self, action: str) -> StepResult:
        if self._done:
            return StepResult(self._last_obs, 0.0, True, {"already_done": True})
        self._step += 1
        info: Dict[str, Any] = {"action_raw": action}
        parsed = self._parse(action or "")
        valid = False
        if parsed:
            verb, arg = parsed
            valid = self._apply(verb, arg, info)
        else:
            info["reason"] = "parse_error"
        self._last_action_valid = valid
        if not valid:
            self._invalid_count += 1

        if self._gold["goal_object"] in self._gold["inventory"]:
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
        gold = self._gold
        loc = gold["agent_location"]
        if verb == "move_to":
            if arg not in gold["rooms"]:
                info["reason"] = "unknown_room"
                return False
            for did, d in gold["door_specs"].items():
                if {d["from"], d["to"]} == {loc, arg}:
                    if gold["door_states"][did] == "locked":
                        info["reason"] = "door_locked"
                        return False
                    gold["agent_location"] = arg
                    return True
            info["reason"] = "no_door"
            return False
        if verb == "pick_up":
            if gold["object_locations"].get(arg) != loc:
                info["reason"] = "object_not_here"
                return False
            gold["object_locations"][arg] = "_inventory_"
            gold["inventory"].append(arg)
            if arg.endswith("_key"):
                gold["completed_subgoals"].append(f"has_{arg}")
            return True
        if verb == "unlock":
            spec = gold["door_specs"].get(arg)
            if not spec:
                info["reason"] = "unknown_door"
                return False
            if spec["from"] != loc and spec["to"] != loc:
                info["reason"] = "door_not_here"
                return False
            if gold["door_states"][arg] != "locked":
                return True
            req = spec["required_key"]
            if req not in gold["inventory"]:
                info["reason"] = "missing_key"
                return False
            gold["door_states"][arg] = "open"
            gold["completed_subgoals"].append(f"unlocked_{arg}")
            return True
        if verb == "open":
            if arg not in gold["boxes"]:
                info["reason"] = "unknown_container"
                return False
            if gold["object_locations"].get(arg) != loc:
                info["reason"] = "container_not_here"
                return False
            return True
        if verb == "place":
            parts = [p.strip() for p in arg.split(",")]
            if len(parts) != 2:
                info["reason"] = "bad_args"
                return False
            obj, target = parts
            if obj not in gold["inventory"]:
                info["reason"] = "not_in_inventory"
                return False
            if target not in gold["rooms"]:
                info["reason"] = "unknown_target"
                return False
            gold["inventory"].remove(obj)
            gold["object_locations"][obj] = target
            return True
        info["reason"] = f"unknown_verb:{verb}"
        return False

    def _apply_mutation(self) -> None:
        rate = self._stress.get("state_mutation_rate", "static")
        if rate == "static":
            return
        prob = 0.15 if rate == "mild" else 0.35
        # 不在 inventory 的 key 可能漂移
        loose_keys = [k for k in self._gold["keys"]
                      if self._gold["object_locations"].get(k) not in ("_inventory_", None)]
        for k in loose_keys:
            if self._rng.random() < prob:
                self._gold["object_locations"][k] = self._rng.choice(self._gold["rooms"])

    # ---------------- probe ----------------
    def step_probe_action(self, probe: str) -> ProbeResult:
        parsed = self._parse(probe or "")
        if not parsed:
            return ProbeResult("invalid", probe, {"error": "parse_error"}, cost=1.0)
        verb, arg = parsed
        gold = self._gold
        if verb == "check_location":
            return ProbeResult("check_location", arg,
                {"location": gold["object_locations"].get(arg, "unknown")}, cost=1.0)
        if verb == "check_inventory":
            return ProbeResult("check_inventory", None,
                {"inventory": list(gold["inventory"])}, cost=1.0)
        if verb == "check_door_status":
            return ProbeResult("check_door_status", arg,
                {"status": gold["door_states"].get(arg, "unknown"),
                 "required_key": gold["door_specs"].get(arg, {}).get("required_key")}, cost=1.0)
        if verb == "check_container_status":
            return ProbeResult("check_container_status", arg,
                {"location": gold["object_locations"].get(arg, "unknown")}, cost=1.0)
        if verb == "inspect_room":
            objs = sorted([o for o, r in gold["object_locations"].items() if r == arg])
            return ProbeResult("inspect_room", arg, {"objects": objs}, cost=1.0)
        if verb == "verify_subgoal":
            return ProbeResult("verify_subgoal", arg,
                {"completed": arg in gold["completed_subgoals"]}, cost=1.0)
        if verb == "check_preconditions":
            missing: List[str] = []
            target = arg
            if target.startswith("unlock_"):
                did = target.split("unlock_", 1)[1]
                spec = gold["door_specs"].get(did)
                if not spec:
                    return ProbeResult("check_preconditions", target, {"error": "unknown_door"}, cost=1.0)
                if spec["locked"] and spec["required_key"] not in gold["inventory"]:
                    missing.append(spec["required_key"])
            elif target.startswith("pick_up_"):
                obj = target.split("pick_up_", 1)[1]
                if gold["object_locations"].get(obj) != gold["agent_location"]:
                    missing.append(f"be_at_{gold['object_locations'].get(obj, 'unknown')}")
            return ProbeResult("check_preconditions", target,
                {"executable": not missing, "missing": missing}, cost=1.0)
        if verb == "check_current_position":
            return ProbeResult("check_current_position", None,
                {"location": gold["agent_location"]}, cost=1.0)
        return ProbeResult("unknown_probe", arg, {"error": f"unknown probe {verb}"}, cost=1.0)

    # ---------------- introspection ----------------
    def get_observation(self) -> Dict[str, Any]:
        return self._last_obs

    def get_gold_state(self) -> Dict[str, Any]:
        g = dict(self._gold)
        g.pop("door_specs", None)
        return g

    def score_belief_state(self, belief: Dict[str, Any]) -> float:
        gold = self.get_gold_state()
        if not isinstance(belief, dict):
            return 0.0
        slots = [
            ("current_location", belief.get("current_location"), gold.get("agent_location"), 0.20),
            ("inventory", belief.get("inventory"), gold.get("inventory"), 0.15),
            ("object_locations", belief.get("object_locations"), gold.get("object_locations"), 0.30),
            ("door_states", belief.get("door_states"), gold.get("door_states"), 0.20),
            ("completed_subgoals", belief.get("completed_subgoals"), gold.get("completed_subgoals"), 0.15),
        ]
        return sum(w * flat_overlap(b, g) for _, b, g, w in slots)

    def task_description(self) -> str:
        return (f"You are in {self._gold['rooms'][0]}. There are {len(self._gold['rooms'])} rooms "
                f"connected by doors (some locked). Your goal: pick up the object '{self._gold['goal_object']}'. "
                f"You may need to collect keys, unlock doors, and traverse rooms.")

    def available_task_actions(self) -> List[str]:
        return [
            "move_to(room_X)", "pick_up(object_name)", "open(box_X)",
            "unlock(door_X)", "place(object_name, room_X)",
        ]

    def available_probe_actions(self) -> List[str]:
        return [
            "check_location(object_name)", "check_inventory()",
            "check_door_status(door_X)", "check_container_status(box_X)",
            "inspect_room(room_X)", "verify_subgoal(name)",
            "check_preconditions(unlock_door_X)", "check_preconditions(pick_up_object)",
            "check_current_position()",
        ]

    def is_done(self) -> bool:
        return self._done
