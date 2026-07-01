"""Method 抽象: 决定每步是 act 还是 probe (在 agent 已输出 decision 后, method 可以覆盖)。

人类研究员决策: probe budget = horizon // 4, 超额必须 force-act。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MethodDecision:
    """Method 最终的决策。"""
    decision_type: str          # "act" / "probe" / "reset"
    action: str                 # 实际要执行的 action 字符串
    target_belief_id: Optional[str] = None
    reasoning: str = ""
    overrode_agent: bool = False  # method 是否覆盖了 agent 原始 decision


@dataclass
class MethodContext:
    """传给 method.decide 的上下文。"""
    step: int
    horizon: int
    observation: Dict[str, Any]
    agent_output: Dict[str, Any]    # agent JSON
    task_action_spec: List[str]
    probe_action_spec: List[str]
    env: Any                        # 环境 (Oracle / Judge 用)
    history: List[Dict[str, Any]] = field(default_factory=list)
    probe_budget_total: int = 0
    probes_used: int = 0


class Method:
    name: str = "base"
    method_hint: str = ""           # 加到 agent prompt 里, 帮 LLM 调整行为

    def reset_episode(self) -> None:
        pass

    def decide(self, ctx: MethodContext) -> MethodDecision:
        """根据 agent 输出 + 环境状态 + budget 决定本步执行什么。"""
        raise NotImplementedError

    # ---- 公共: probe budget 检查 ----
    @staticmethod
    def is_budget_exhausted(ctx: MethodContext) -> bool:
        return ctx.probes_used >= ctx.probe_budget_total

    @staticmethod
    def take_agent_action(ctx: MethodContext) -> MethodDecision:
        """直接遵照 agent 决策, 不覆盖。"""
        nd = ctx.agent_output.get("next_decision", {})
        return MethodDecision(
            decision_type=str(nd.get("type", "act")),
            action=str(nd.get("action", "noop()")),
            target_belief_id=nd.get("target_belief"),
            reasoning="agent_decision",
            overrode_agent=False,
        )

    @staticmethod
    def force_act_from_agent(ctx: MethodContext, reason: str) -> MethodDecision:
        """强制 act: 若 agent 给的是 probe, 用一个 fallback act。"""
        nd = ctx.agent_output.get("next_decision", {})
        agent_type = str(nd.get("type", "act"))
        agent_action = str(nd.get("action", "noop()"))
        if agent_type == "act":
            return MethodDecision(decision_type="act", action=agent_action,
                                  reasoning=reason, overrode_agent=False)
        # 找一个 valid 的 task action 作为 fallback
        fallback = _heuristic_act(ctx)
        return MethodDecision(decision_type="act", action=fallback,
                              reasoning=f"{reason}|force_act_fallback",
                              overrode_agent=True)


def _heuristic_act(ctx: MethodContext) -> str:
    """简易 fallback action — 选第一个 task_action_spec 模板, 用观察填参。"""
    obs = ctx.observation or {}
    env_name = type(ctx.env).__name__
    if env_name == "ObjectStateWorld":
        # 优先拿场内物体, 否则向相邻房间移动
        if obs.get("objects_in_view"):
            return f"pick_up({obs['objects_in_view'][0]})"
        doors = obs.get("doors_in_view") or []
        if doors:
            # 找门对端 — 但 obs 里没给, 只能开锁尝试 / 用 move_to
            cur = obs.get("current_location", "room_0")
            # 解析 cur idx, 尝试相邻
            if isinstance(cur, str) and cur.startswith("room_"):
                try:
                    idx = int(cur.split("_")[1])
                    return f"move_to(room_{idx + 1})"
                except Exception:
                    pass
        return "check_current_position()" if False else "move_to(room_1)"
    if env_name == "ToolDAGWorld":
        completed = set(obs.get("completed_tools") or [])
        avail = set(obs.get("available_variables") or [])
        # 找一个未完成且可执行的 tool
        env_gold = ctx.env.get_gold_state() if hasattr(ctx.env, "get_gold_state") else {}
        for t in env_gold.get("tools", []):
            if t in completed:
                continue
            req = env_gold.get("tool_inputs", {}).get(t, [])
            if all(v in avail for v in req):
                return f"call_tool({t})"
        return "call_tool(t_0)"
    if env_name == "GraphNavWorld":
        neighbors = obs.get("neighbors") or []
        if neighbors:
            return f"move_to({neighbors[0]})"
        keys_here = obs.get("keys_in_view") or []
        if keys_here:
            return f"collect_key({keys_here[0]})"
        return "move_to(n_1)"
    return "noop()"
