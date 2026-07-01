"""ToolDAGWorld: 工具依赖图世界。

Gold state:
- tools, variables (t_i 产生 v_i)
- tool_inputs (t_i 需要的 v_j list)
- available_variables (已产生的 set)
- target_variable (终极目标)

任务: call_tool(t_i) 按依赖序产生 target_variable。
"""
from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional

from .base import Environment, ProbeResult, StepResult, flat_overlap


N_TOOLS_BY_CARD = {"low": 5, "medium": 9, "high": 16}
DEPS_BY_DEN = {"low": 1, "medium": 2, "high": 3}


class ToolDAGWorld(Environment):
    name = "ToolDAGWorld"
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
        self._prev_completed: List[str] = []

    def reset(self, seed: int, stress_config: Dict[str, Any]) -> Dict[str, Any]:
        self._rng = random.Random(int(seed))
        self._step = 0
        self._stress = dict(stress_config or {})
        self._horizon = int(self._stress.get("horizon", 20))
        n = N_TOOLS_BY_CARD[self._stress.get("state_cardinality", "low")]
        deps_per = DEPS_BY_DEN[self._stress.get("dependency_density", "low")]

        tools = [f"t_{i}" for i in range(n)]
        variables = [f"v_{i}" for i in range(n)]
        tool_inputs: Dict[str, List[str]] = {}
        for i, t in enumerate(tools):
            if i == 0:
                tool_inputs[t] = []
            else:
                k = min(i, deps_per)
                tool_inputs[t] = sorted(self._rng.sample(variables[:i], k=k))

        self._gold = {
            "tools": tools,
            "variables": variables,
            "tool_inputs": tool_inputs,
            "tool_output_var": {t: v for t, v in zip(tools, variables)},
            "available_variables": [],   # 排序后的 list, 便于 hash 比对
            "tool_outputs": {},           # var -> int value
            "completed_tools": [],
            "target_variable": variables[-1],
        }
        self._done = False
        self._success = False
        self._last_action_valid = True
        self._invalid_count = 0
        self._prev_completed = []
        self._last_obs = self._build_obs(after_action=None)
        return self._last_obs

    def _build_obs(self, after_action: Optional[str]) -> Dict[str, Any]:
        noise = self._stress.get("observation_noise", "clean")
        avail = sorted(self._gold["available_variables"])
        completed = sorted(self._gold["completed_tools"])
        obs: Dict[str, Any] = {
            "available_variables": avail,
            "tool_outputs_visible": dict(self._gold["tool_outputs"]),
            "completed_tools": completed,
            "target_variable": self._gold["target_variable"],
            "step": self._step,
            "horizon": self._horizon,
            "after_action": after_action,
            "last_action_valid": self._last_action_valid,
        }
        if noise == "partial" and avail and self._rng.random() < 0.4:
            obs["available_variables"] = avail[:-1]
        elif noise == "distractor":
            obs["available_variables"] = avail + [f"v_fake_{self._rng.randint(0, 99)}"]
        elif noise == "delayed" and self._prev_completed:
            obs["completed_tools"] = list(self._prev_completed)
        self._prev_completed = completed
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
            arg = m.group(2).strip().strip('"').strip("'").split(",")[0].strip()
            if verb == "call_tool":
                valid = self._call_tool(arg, info)
            else:
                info["reason"] = f"unknown_verb:{verb}"
        else:
            info["reason"] = "parse_error"
        self._last_action_valid = valid
        if not valid:
            self._invalid_count += 1

        if self._gold["target_variable"] in self._gold["available_variables"]:
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

    def _call_tool(self, tool: str, info: Dict[str, Any]) -> bool:
        g = self._gold
        if tool not in g["tools"]:
            info["reason"] = "unknown_tool"
            return False
        # R3 Stage B fix (2026-06-04): reject re-call of already-completed tool.
        # Bug: agent stuck in single-tool loop because re-call returned valid=True
        # despite no progress. Now valid=False with reason="already_completed" so
        # agent gets feedback to pick a different tool.
        if tool in g["completed_tools"]:
            info["reason"] = "already_completed"
            return False
        required = g["tool_inputs"][tool]
        missing = [v for v in required if v not in g["available_variables"]]
        if missing:
            info["reason"] = "missing_inputs"
            info["missing"] = missing
            return False
        out_var = g["tool_output_var"][tool]
        if out_var not in g["available_variables"]:
            g["available_variables"].append(out_var)
            g["available_variables"].sort()
        g["tool_outputs"][out_var] = self._rng.randint(1, 1000)
        if tool not in g["completed_tools"]:
            g["completed_tools"].append(tool)
            g["completed_tools"].sort()
        return True

    def _apply_mutation(self) -> None:
        rate = self._stress.get("state_mutation_rate", "static")
        if rate == "static":
            return
        prob = 0.10 if rate == "mild" else 0.25
        target = self._gold["target_variable"]
        for v in list(self._gold["available_variables"]):
            if v == target:
                continue
            if self._rng.random() < prob:
                self._gold["available_variables"].remove(v)
                self._gold["tool_outputs"].pop(v, None)
                for t, ov in self._gold["tool_output_var"].items():
                    if ov == v and t in self._gold["completed_tools"]:
                        self._gold["completed_tools"].remove(t)

    def step_probe_action(self, probe: str) -> ProbeResult:
        m = self._ACT_RE.match((probe or "").strip())
        if not m:
            return ProbeResult("invalid", probe, {"error": "parse_error"}, cost=1.0)
        verb = m.group(1).lower()
        arg = m.group(2).strip().strip('"').strip("'")
        g = self._gold
        if verb == "inspect_tool_schema":
            if arg not in g["tools"]:
                return ProbeResult("inspect_tool_schema", arg, {"error": "unknown"}, cost=1.0)
            return ProbeResult("inspect_tool_schema", arg,
                {"required_inputs": list(g["tool_inputs"][arg]),
                 "produces": g["tool_output_var"][arg]}, cost=1.0)
        if verb == "check_variable_exists":
            return ProbeResult("check_variable_exists", arg,
                {"exists": arg in g["available_variables"]}, cost=1.0)
        if verb == "validate_tool_output":
            return ProbeResult("validate_tool_output", arg,
                {"value": g["tool_outputs"].get(arg), "exists": arg in g["tool_outputs"]}, cost=1.0)
        if verb in ("check_required_inputs", "check_tool_dependency"):
            if arg not in g["tools"]:
                return ProbeResult(verb, arg, {"error": "unknown"}, cost=1.0)
            req = g["tool_inputs"][arg]
            missing = [v for v in req if v not in g["available_variables"]]
            return ProbeResult(verb, arg,
                {"required_inputs": list(req),
                 "available_inputs": [v for v in req if v not in missing],
                 "missing_inputs": missing}, cost=1.0)
        if verb == "validate_argument":
            return ProbeResult("validate_argument", arg, {"valid": False, "reason": "no_args_in_DAG_world"}, cost=1.0)
        return ProbeResult("unknown_probe", arg, {"error": f"unknown probe {verb}"}, cost=1.0)

    def get_observation(self) -> Dict[str, Any]:
        return self._last_obs

    def get_gold_state(self) -> Dict[str, Any]:
        g = dict(self._gold)
        # tool_inputs / tool_output_var 是结构信息, 保留
        return g

    def score_belief_state(self, belief: Dict[str, Any]) -> float:
        gold = self.get_gold_state()
        if not isinstance(belief, dict):
            return 0.0
        bs = belief.get("belief_world_state", belief)
        slots = [
            ("completed_subgoals", bs.get("completed_subgoals"), gold.get("completed_tools"), 0.30),
            ("tool_outputs", bs.get("tool_outputs"),
             {v: True for v in gold.get("available_variables", [])}, 0.40),
            ("open_dependencies", bs.get("open_dependencies"),
             [t for t in gold.get("tools", []) if t not in gold.get("completed_tools", [])], 0.30),
        ]
        return sum(w * flat_overlap(b, g) for _, b, g, w in slots)

    def task_description(self) -> str:
        return (f"You have {len(self._gold['tools'])} tools and {len(self._gold['variables'])} variables. "
                f"Each tool produces one variable but requires specific input variables. "
                f"Goal: produce '{self._gold['target_variable']}' by calling tools in dependency order.")

    def available_task_actions(self) -> List[str]:
        return ["call_tool(t_X)"]

    def available_probe_actions(self) -> List[str]:
        return [
            "inspect_tool_schema(t_X)", "check_variable_exists(v_X)",
            "validate_tool_output(v_X)", "check_required_inputs(t_X)",
            "check_tool_dependency(t_X)",
        ]

    def is_done(self) -> bool:
        return self._done
