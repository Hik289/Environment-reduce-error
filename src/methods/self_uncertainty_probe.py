"""Self-Uncertainty-Probe: 仅当 agent 自评 risk_level >= medium 或最低 confidence < 0.5 时 probe。"""
from .base import Method, MethodContext, MethodDecision
from .periodic_probe import _probe_for_belief


class SelfUncertaintyProbeMethod(Method):
    name = "self_uncertainty_probe"
    method_hint = "Probe ONLY when you genuinely judge your belief is uncertain. Set risk_level honestly."

    def decide(self, ctx: MethodContext) -> MethodDecision:
        if self.is_budget_exhausted(ctx):
            return self.force_act_from_agent(ctx, "budget_exhausted")

        agent_out = ctx.agent_output
        beliefs = agent_out.get("beliefs") or []
        sc = agent_out.get("self_check") or {}
        risk = str(sc.get("risk_level", "low"))
        min_conf = min([float(b.get("confidence", 1.0)) for b in beliefs], default=1.0)

        # agent 自报: 高风险 or 任意 belief confidence < 0.5 → probe
        should_probe = (risk in ("medium", "high")) or (min_conf < 0.5)

        if should_probe:
            agent_nd = agent_out.get("next_decision", {})
            if str(agent_nd.get("type")) == "probe":
                return MethodDecision(decision_type="probe",
                                      action=str(agent_nd.get("action", "noop()")),
                                      reasoning="self_uncertainty:agent_probed",
                                      overrode_agent=False)
            # 否则替 agent 选低 confidence belief 探测
            if beliefs:
                target = min(beliefs, key=lambda b: float(b.get("confidence", 1.0)))
                probe = _probe_for_belief(target, ctx)
                return MethodDecision(decision_type="probe", action=probe,
                                      reasoning="self_uncertainty:method_probed",
                                      overrode_agent=True)

        return self.force_act_from_agent(ctx, "self_uncertainty_no_probe")
