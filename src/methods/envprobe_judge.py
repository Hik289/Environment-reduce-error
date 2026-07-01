"""EnvProbe-Judge: 用独立 LLM 调用做 probe-vs-act 决策。"""
from __future__ import annotations

from typing import Optional

from .base import Method, MethodContext, MethodDecision


class EnvProbeJudgeMethod(Method):
    name = "envprobe_judge"
    method_hint = "An external judge will decide whether to act or probe. Provide accurate belief table."

    def __init__(self):
        self._agent = None

    def attach_agent(self, agent) -> None:
        self._agent = agent

    def decide(self, ctx: MethodContext) -> MethodDecision:
        if self.is_budget_exhausted(ctx):
            return self.force_act_from_agent(ctx, "budget_exhausted")
        if self._agent is None:
            # 没挂 agent 就降级为 envprobe_simple 思路
            return self.force_act_from_agent(ctx, "judge_unavailable")
        nd = ctx.agent_output.get("next_decision", {})
        next_action = str(nd.get("action", "noop()"))
        beliefs = ctx.agent_output.get("beliefs") or []
        verdict = self._agent.judge_probe(
            next_action=next_action,
            belief_table=beliefs,
            observation=ctx.observation,
            probe_action_spec=ctx.probe_action_spec,
        )
        decision = str(verdict.get("decision", "act")).lower()
        if decision == "probe":
            probe_action = str(verdict.get("probe_action") or "")
            if not probe_action:
                return self.force_act_from_agent(ctx, "judge_empty_probe")
            return MethodDecision(decision_type="probe", action=probe_action,
                                  target_belief_id=verdict.get("target_belief_id"),
                                  reasoning=f"judge:{verdict.get('reasoning','')[:80]}",
                                  overrode_agent=True)
        return self.force_act_from_agent(ctx, "judge_act")
