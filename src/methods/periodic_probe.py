"""Periodic-Probe: 每 K 步 probe 一次。

K = max(1, horizon // probe_budget_total), 保证 budget 用完。
Probe 内容: agent 给的 probe action; 若 agent 没给 probe, 用 heuristic (按 agent belief 表选最 stale).
"""
from .base import Method, MethodContext, MethodDecision


class PeriodicProbeMethod(Method):
    name = "periodic_probe"
    method_hint = "Probing is allowed periodically. The method will trigger probes at fixed intervals."

    def reset_episode(self) -> None:
        self._last_probe_step = 0
        self._period: int = 0

    def decide(self, ctx: MethodContext) -> MethodDecision:
        if not hasattr(self, "_period") or self._period == 0:
            self._period = max(1, ctx.horizon // max(1, ctx.probe_budget_total))
            self._last_probe_step = 0

        if self.is_budget_exhausted(ctx):
            return self.force_act_from_agent(ctx, "budget_exhausted")

        # 每 _period 步触发一次 probe
        if ctx.step - self._last_probe_step >= self._period:
            probe = _select_probe(ctx)
            if probe is not None:
                self._last_probe_step = ctx.step
                return MethodDecision(decision_type="probe", action=probe,
                                      reasoning=f"periodic_step{ctx.step}",
                                      overrode_agent=True)
        return self.force_act_from_agent(ctx, "periodic_skip")


def _select_probe(ctx: MethodContext) -> str | None:
    """Periodic 模式下选 probe target — 优先用 agent 给的, 否则按 staleness 排序选 top 1。"""
    agent_nd = ctx.agent_output.get("next_decision", {})
    if str(agent_nd.get("type")) == "probe":
        return str(agent_nd.get("action"))

    beliefs = ctx.agent_output.get("beliefs") or []
    if beliefs:
        # 选最 stale (staleness 高) 的 belief 来 probe
        target = max(beliefs, key=lambda b: b.get("staleness", 0))
        return _probe_for_belief(target, ctx)

    # 兜底: 用第一个 probe template
    from .random_probe import _fill_probe_template
    if ctx.probe_action_spec:
        return _fill_probe_template(ctx.probe_action_spec[0], ctx)
    return None


def _probe_for_belief(belief: dict, ctx: MethodContext) -> str:
    """根据 belief.type / content 选合适的 probe action。"""
    btype = belief.get("type", "other")
    content = belief.get("content", "")
    env_name = type(ctx.env).__name__

    # 在 content 里抓识别符
    g = ctx.env.get_gold_state()
    if env_name == "ObjectStateWorld":
        # 抓 object name
        for obj in g.get("object_locations", {}):
            if obj in content:
                return f"check_location({obj})"
        for did in g.get("door_states", {}):
            if did in content:
                return f"check_door_status({did})"
        return "check_current_position()"
    if env_name == "ToolDAGWorld":
        # content 提到的 var / tool 优先
        for v in g.get("variables", []):
            if v in content:
                return f"check_variable_exists({v})"
        for t in g.get("tools", []):
            if t in content:
                return f"check_required_inputs({t})"
        # v3: 选最有信息量的 probe — content 没提具体实体时, 用 gold + bws 选
        bws = ctx.agent_output.get("belief_world_state") or {}
        avail = set(g.get("available_variables", []))
        b_outs = set((bws.get("tool_outputs") or {}).keys())
        # 优先: variable 在 gold 里存在但 agent 不知道
        for v in g.get("variables", []):
            if v in avail and v not in b_outs:
                return f"check_variable_exists({v})"
        # 次优: tool 有 missing inputs in gold
        tool_inputs = g.get("tool_inputs", {})
        completed = set(g.get("completed_tools", []))
        for t in g.get("tools", []):
            if t in completed:
                continue
            req = set(tool_inputs.get(t, []))
            if req - avail:  # missing inputs in gold
                return f"check_required_inputs({t})"
        # 次次优: tool 满足条件但 agent 的 open_deps 仍列着它
        b_open = set(bws.get("open_dependencies") or [])
        for t in g.get("tools", []):
            if t in completed:
                continue
            req = set(tool_inputs.get(t, []))
            if not (req - avail) and t in b_open:
                return f"check_required_inputs({t})"
        return f"check_required_inputs({g.get('tools', ['t_0'])[0]})"
    if env_name == "GraphNavWorld":
        for n in g.get("nodes", []):
            if n in content:
                return f"inspect_neighbors({n})"
        for k in g.get("keys", []):
            if k in content:
                return f"check_location({k})"
        return "check_current_node()"
    return "check_current_position()"
