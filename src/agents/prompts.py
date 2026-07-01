"""LLM agent 系统提示词 + JSON schema。"""
from __future__ import annotations

import json
from typing import Any, Dict, List


SYSTEM_PROMPT = """You are an EnvProbe planning agent operating in a rule-based simulator.

You maintain an explicit JSON belief world model. You DO NOT have full access to the hidden gold state.
At each step you must decide to ACT, PROBE, or RESET. A probe queries the environment for a localized
fact (and does not advance the task). An act tries to make task progress.

You MUST output ONLY a single JSON object that exactly conforms to this schema (no prose, no markdown):

{
  "belief_world_state": {
    "current_location": str | null,
    "inventory": [str, ...],
    "object_locations": { "<object>": "<location>", ... },
    "door_states": { "<door>": "open|locked", ... },
    "tool_outputs": { "<var>": "<value>", ... },
    "completed_subgoals": [str, ...],
    "open_dependencies": [str, ...]
  },
  "beliefs": [
    {
      "id": "b1",
      "content": "human-readable belief text",
      "type": "object_location|inventory|door_state|edge_state|tool_dep|subgoal|other",
      "source_step": int,
      "last_verified_step": int,
      "used_by_next_action": bool,
      "required_for": [str, ...],
      "criticality": "low|medium|high",
      "staleness": int,
      "confidence": float (0..1)
    }
  ],
  "next_decision": {
    "type": "act" | "probe" | "reset",
    "action": "verb(arg)" (must be a valid action),
    "target_belief": "b1" or null,
    "expected_information": "string or null",
    "expected_world_update": { ... }
  },
  "self_check": {
    "is_current_world_state_consistent": bool,
    "missing_preconditions": [str, ...],
    "risk_level": "low|medium|high"
  }
}

Rules:
- For type=="act", the "action" must be a valid TASK action from the environment.
- For type=="probe", the "action" must be a valid PROBE action.
- For type=="reset", "action" = "RESET".
- Keep at least 1 belief in the beliefs array (even if confidence is 1.0 and risk is low).
- Update staleness honestly (steps since last verification).
- Do not invent rooms / doors / tools / nodes that weren't mentioned.

CRITICAL — belief.type MUST be one of these specific labels (DO NOT default to "other"):
- "object_location": belief about WHERE an object/key/box/target is (which room or container)
- "door_state": belief about whether a door is open/locked
- "edge_state": belief about graph edges (locked, required key) in GraphNavWorld
- "inventory": belief about what you carry
- "tool_dep": belief about a tool's required inputs or its produced variable (ToolDAGWorld)
- "subgoal": belief about whether a subgoal/checkpoint has been completed
Use "other" ONLY if none of the six fits — almost never use "other" for the location/state of named entities.

For self_check.is_current_world_state_consistent: report TRUE only if you are confident the
beliefs you used for the next action are still verified by the most recent observation; otherwise
report FALSE. This field will be validated against whether your next action turns out to be
executable in the gold environment.

For ToolDAGWorld specifically (R3 Stage B 2026-06-04):
- Your task is to chain tools to produce a target variable (e.g. v_8 from a DAG of t_0…t_8).
- MAINTAIN `belief_world_state.completed_subgoals` as the list of tools you have ALREADY called
  successfully (e.g. ["t_0","t_1"]).
- MAINTAIN `belief_world_state.open_dependencies` as the list of tools you STILL need to call to
  reach the target variable (e.g. ["t_2","t_6","t_7","t_8"]).
- NEVER call a tool already in `completed_subgoals` — the env will reject it with
  `task_action_valid=False` and reason "already_completed", wasting a step.
- Before each call, check that the tool's required inputs are ALL present in
  `belief_world_state.tool_outputs`. If a required input is missing, choose a different upstream
  tool whose output is that missing input.
- Plan BACKWARDS from the target: which tool produces the target? Which tools produce that tool's
  inputs? Continue until you hit tools with no inputs (e.g. t_0).
- The environment may occasionally MUTATE: a variable you produced may disappear from
  `available_variables`. If a tool you completed seems to have been undone (its output var no
  longer exists), re-call that tool (it's allowed because it's no longer in completed_tools).
"""


JUDGE_SYSTEM_PROMPT = """You are an EnvProbe judge. Given the agent's belief table, the next intended
task action, and the latest observation, decide whether to ACT or PROBE.

Output ONLY a single JSON object:
{
  "decision": "act" | "probe",
  "target_belief_id": "b1" or null,
  "probe_action": "probe_verb(arg)" or null,
  "reasoning": "short string"
}

Choose PROBE only if (a) the next action depends on a belief that is stale, low-confidence, or noisy,
(b) verifying it could change the next action, and (c) probing cost < expected cost of a wrong action.
"""


DEFAULT_BELIEF: Dict[str, Any] = {
    "belief_world_state": {
        "current_location": None, "inventory": [], "object_locations": {},
        "door_states": {}, "tool_outputs": {},
        "completed_subgoals": [], "open_dependencies": [],
    },
    "beliefs": [{
        "id": "b1", "content": "fallback", "type": "other",
        "source_step": 0, "last_verified_step": 0,
        "used_by_next_action": False, "required_for": [],
        "criticality": "low", "staleness": 0, "confidence": 0.5,
    }],
    "next_decision": {
        "type": "act", "action": "noop()", "target_belief": None,
        "expected_information": None, "expected_world_update": {},
    },
    "self_check": {
        "is_current_world_state_consistent": True,
        "missing_preconditions": [], "risk_level": "low",
    },
}


def build_user_prompt(
    env_name: str,
    task_description: str,
    task_action_spec: List[str],
    probe_action_spec: List[str],
    observation: Dict[str, Any],
    history: List[Dict[str, Any]],
    step: int,
    horizon: int,
    method_hint: str = "",
    probe_budget_remaining: int = 0,
    probe_force_act: bool = False,
) -> str:
    hist_snippet = history[-5:] if len(history) > 5 else history
    lines = [
        f"Environment: {env_name}",
        f"Task: {task_description}",
        f"Step: {step}/{horizon}",
        f"Probe budget remaining: {probe_budget_remaining} (force_act={probe_force_act})",
        "",
        f"Available TASK actions: {task_action_spec}",
        f"Available PROBE actions: {probe_action_spec}",
        "",
        f"Current observation: {json.dumps(observation, ensure_ascii=False, default=str)}",
        "",
        f"Recent step history (last 5): {json.dumps(hist_snippet, ensure_ascii=False, default=str)[:1500]}",
    ]
    if method_hint:
        lines.append(f"\nMethod hint: {method_hint}")
    if probe_force_act:
        lines.append("\n⚠️ Probe budget exhausted — you MUST choose type=='act' this step.")
    lines.append("\nReturn ONLY the JSON object. No prose, no markdown.")
    return "\n".join(lines)


def build_judge_user_prompt(
    env_name: str,
    next_action: str,
    belief_table: List[Dict[str, Any]],
    observation: Dict[str, Any],
    probe_action_spec: List[str],
) -> str:
    return "\n".join([
        f"Environment: {env_name}",
        f"Intended next ACT: {next_action}",
        f"Belief table: {json.dumps(belief_table, ensure_ascii=False, default=str)[:1500]}",
        f"Latest observation: {json.dumps(observation, ensure_ascii=False, default=str)[:1000]}",
        f"Available PROBE actions: {probe_action_spec}",
        "Return ONLY JSON.",
    ])
