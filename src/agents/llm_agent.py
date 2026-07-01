"""LLMAgent: 调用 LLM 输出结构化 belief + decision JSON。"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from .prompts import (
    DEFAULT_BELIEF, SYSTEM_PROMPT, JUDGE_SYSTEM_PROMPT,
    build_user_prompt, build_judge_user_prompt,
)
from ..utils.api_client import LLMClient


class LLMAgent:
    def __init__(self, model: str = "gpt-4o-mini",
                 client: Optional[LLMClient] = None,
                 max_tokens: int = 1500):
        self.model = model
        self.client = client or LLMClient(model=model)
        self.max_tokens = max_tokens
        self.env_name = ""
        self.task_description = ""

    def reset(self, env_name: str, task_description: str) -> None:
        self.env_name = env_name
        self.task_description = task_description

    def step(self,
             observation: Dict[str, Any],
             history: List[Dict[str, Any]],
             step: int,
             horizon: int,
             method_hint: str = "",
             task_action_spec: List[str] | None = None,
             probe_action_spec: List[str] | None = None,
             probe_budget_remaining: int = 0,
             probe_force_act: bool = False,
             ) -> Dict[str, Any]:
        user = build_user_prompt(
            env_name=self.env_name,
            task_description=self.task_description,
            task_action_spec=task_action_spec or [],
            probe_action_spec=probe_action_spec or [],
            observation=observation,
            history=history,
            step=step,
            horizon=horizon,
            method_hint=method_hint,
            probe_budget_remaining=probe_budget_remaining,
            probe_force_act=probe_force_act,
        )
        try:
            parsed = self.client.chat_json(SYSTEM_PROMPT, user,
                                           model=self.model, max_tokens=self.max_tokens)
        except Exception as e:
            parsed = copy.deepcopy(DEFAULT_BELIEF)
            parsed["_llm_error"] = str(e)[:200]
        return _normalize(parsed)

    def judge_probe(self,
                    next_action: str,
                    belief_table: List[Dict[str, Any]],
                    observation: Dict[str, Any],
                    probe_action_spec: List[str],
                    ) -> Dict[str, Any]:
        user = build_judge_user_prompt(
            env_name=self.env_name,
            next_action=next_action,
            belief_table=belief_table,
            observation=observation,
            probe_action_spec=probe_action_spec,
        )
        try:
            parsed = self.client.chat_json(JUDGE_SYSTEM_PROMPT, user,
                                           model=self.model, max_tokens=300)
        except Exception:
            return {"decision": "act", "target_belief_id": None,
                    "probe_action": None, "reasoning": "judge_failed"}
        return parsed


def _normalize(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """补全缺失字段, 把字段统一成 schema 形状。"""
    if not isinstance(parsed, dict):
        return copy.deepcopy(DEFAULT_BELIEF)
    out = copy.deepcopy(DEFAULT_BELIEF)
    # belief_world_state
    bws = parsed.get("belief_world_state") or {}
    if isinstance(bws, dict):
        for k, v in bws.items():
            out["belief_world_state"][k] = v
    # beliefs
    beliefs = parsed.get("beliefs")
    if isinstance(beliefs, list) and beliefs:
        normalized_beliefs = []
        for i, b in enumerate(beliefs):
            if not isinstance(b, dict):
                continue
            nb = {
                "id": str(b.get("id") or f"b{i+1}"),
                "content": str(b.get("content") or ""),
                "type": str(b.get("type") or "other"),
                "source_step": int(b.get("source_step") or 0),
                "last_verified_step": int(b.get("last_verified_step") or 0),
                "used_by_next_action": bool(b.get("used_by_next_action") or False),
                "required_for": list(b.get("required_for") or []),
                "criticality": str(b.get("criticality") or "low"),
                "staleness": int(b.get("staleness") or 0),
                "confidence": float(b.get("confidence") or 0.5),
            }
            normalized_beliefs.append(nb)
        if normalized_beliefs:
            out["beliefs"] = normalized_beliefs
    # next_decision
    nd = parsed.get("next_decision") or {}
    if isinstance(nd, dict):
        out["next_decision"]["type"] = str(nd.get("type") or "act").lower()
        out["next_decision"]["action"] = str(nd.get("action") or "noop()")
        out["next_decision"]["target_belief"] = nd.get("target_belief")
        out["next_decision"]["expected_information"] = nd.get("expected_information")
        ewu = nd.get("expected_world_update") or {}
        if isinstance(ewu, dict):
            out["next_decision"]["expected_world_update"] = ewu
    # self_check
    sc = parsed.get("self_check") or {}
    if isinstance(sc, dict):
        out["self_check"]["is_current_world_state_consistent"] = bool(
            sc.get("is_current_world_state_consistent", True))
        out["self_check"]["missing_preconditions"] = list(sc.get("missing_preconditions") or [])
        out["self_check"]["risk_level"] = str(sc.get("risk_level") or "low")
    return out
