"""Random-Probe: 每步以概率 p 选 probe (从 probe_action_spec 随机选), 否则 act。

p 设为 probe_budget_total / horizon, 期望与其他 method 同等 budget。
"""
import random
from .base import Method, MethodContext, MethodDecision


class RandomProbeMethod(Method):
    name = "random_probe"
    method_hint = "Use random probing roughly within the given probe budget."

    def __init__(self):
        self._rng = random.Random()

    def reset_episode(self) -> None:
        self._rng = random.Random()

    def decide(self, ctx: MethodContext) -> MethodDecision:
        if self.is_budget_exhausted(ctx):
            return self.force_act_from_agent(ctx, "budget_exhausted")
        p = ctx.probe_budget_total / max(1, ctx.horizon)
        if self._rng.random() < p and ctx.probe_action_spec:
            probe_template = self._rng.choice(ctx.probe_action_spec)
            probe = _fill_probe_template(probe_template, ctx)
            return MethodDecision(decision_type="probe", action=probe,
                                  reasoning="random", overrode_agent=True)
        return self.force_act_from_agent(ctx, "random_no_probe")


def _fill_probe_template(template: str, ctx: MethodContext) -> str:
    """把 'check_location(object_name)' 这种模板填一个具体 arg。"""
    obs = ctx.observation or {}
    env_name = type(ctx.env).__name__
    rng = random.Random((ctx.step or 0) * 7919 + hash(template) % 1000)
    if env_name == "ObjectStateWorld":
        if "object_name" in template or "object)" in template:
            objs = obs.get("objects_in_view") or []
            if not objs:
                # fallback: gold state 中 random
                g = ctx.env.get_gold_state()
                objs = list(g.get("object_locations", {}).keys()) or ["red_key"]
            return template.replace("object_name", rng.choice(objs)).replace("object)", rng.choice(objs) + ")")
        if "room_X" in template:
            g = ctx.env.get_gold_state()
            rooms = g.get("rooms", ["room_0"])
            return template.replace("room_X", rng.choice(rooms))
        if "door_X" in template:
            g = ctx.env.get_gold_state()
            doors = list(g.get("door_states", {}).keys()) or ["door_1"]
            return template.replace("door_X", rng.choice(doors))
        if "box_X" in template:
            g = ctx.env.get_gold_state()
            boxes = g.get("boxes", ["box_0"])
            return template.replace("box_X", rng.choice(boxes))
        if "unlock_door_X" in template:
            g = ctx.env.get_gold_state()
            doors = list(g.get("door_states", {}).keys()) or ["door_1"]
            return template.replace("unlock_door_X", f"unlock_{rng.choice(doors)}")
        if "pick_up_object" in template:
            g = ctx.env.get_gold_state()
            objs = list(g.get("object_locations", {}).keys()) or ["red_key"]
            return template.replace("pick_up_object", f"pick_up_{rng.choice(objs)}")
        if "name)" in template:  # verify_subgoal(name)
            return template.replace("name", "find_target")
        return template
    if env_name == "ToolDAGWorld":
        g = ctx.env.get_gold_state()
        if "t_X" in template:
            tools = g.get("tools", ["t_0"])
            return template.replace("t_X", rng.choice(tools))
        if "v_X" in template:
            vars_ = g.get("variables", ["v_0"])
            return template.replace("v_X", rng.choice(vars_))
        return template
    if env_name == "GraphNavWorld":
        g = ctx.env.get_gold_state()
        if "n_A, n_B" in template:
            nodes = g.get("nodes", ["n_0", "n_1"])
            a, b = rng.sample(nodes, 2) if len(nodes) >= 2 else (nodes[0], nodes[0])
            return template.replace("n_A, n_B", f"{a}, {b}")
        if "n_X" in template:
            nodes = g.get("nodes", ["n_0"])
            return template.replace("n_X", rng.choice(nodes))
        if "key_X" in template:
            keys = g.get("keys", ["key_a"])
            return template.replace("key_X", rng.choice(keys))
        return template
    return template
