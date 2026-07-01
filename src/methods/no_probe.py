"""No-Probe baseline: 总是 act, 永不 probe。"""
from .base import Method, MethodContext, MethodDecision


class NoProbeMethod(Method):
    name = "no_probe"
    method_hint = "Probing is DISABLED. You must always choose type=='act'."

    def decide(self, ctx: MethodContext) -> MethodDecision:
        return self.force_act_from_agent(ctx, "no_probe")
