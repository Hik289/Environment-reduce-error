"""Oracle-Probe: 直接读 gold state, 比较 agent belief 与 gold, 选 mismatch 最大的 belief probe。

这是 upper bound, 也用于 anchor_4 self-check (oracle traces 给 perfect score)。
"""
from typing import Any, Dict, List, Optional, Tuple

from .base import Method, MethodContext, MethodDecision
from .periodic_probe import _probe_for_belief


class OracleProbeMethod(Method):
    name = "oracle_probe"
    method_hint = "An oracle scheduler may force probes on critically mismatched beliefs."

    def decide(self, ctx: MethodContext) -> MethodDecision:
        if self.is_budget_exhausted(ctx):
            return self.force_act_from_agent(ctx, "budget_exhausted")

        gold = ctx.env.get_gold_state()
        beliefs = ctx.agent_output.get("beliefs") or []
        bws = ctx.agent_output.get("belief_world_state") or {}

        # 评估每个 belief 与 gold 的不一致, 选最严重的
        scored = []
        for b in beliefs:
            mismatch = _belief_mismatch(b, bws, gold, type(ctx.env).__name__)
            if mismatch > 0:
                scored.append((mismatch, b))

        if not scored:
            return self.force_act_from_agent(ctx, "oracle_no_mismatch")

        scored.sort(key=lambda x: x[0], reverse=True)
        _, top_b = scored[0]
        probe = _probe_for_belief(top_b, ctx)
        return MethodDecision(decision_type="probe", action=probe,
                              target_belief_id=top_b.get("id"),
                              reasoning="oracle_mismatch",
                              overrode_agent=True)


def _belief_mismatch(belief: Dict[str, Any], bws: Dict[str, Any],
                     gold: Dict[str, Any], env_name: str) -> float:
    """返回 [0, ∞) 的 mismatch 分数 (0=完全 ok, 高=明显错)。

    RF3 修复: 不再因 type='other' 退化。对每个 belief 都尝试用 content keyword
    匹配 gold 各结构化字段, 取最大 mismatch; 如果完全没 keyword 命中, 再退到
    bws (整体 belief 字典) vs gold 的"字段级 oracle 差异分"。
    """
    btype = belief.get("type", "other")
    content = str(belief.get("content", "")).lower()

    # 1) 按 type 精确路径 (保留原行为)
    typed_score = _typed_mismatch(belief, bws, gold, env_name)
    if typed_score > 0:
        return typed_score

    # 2) RF3: content keyword 匹配 — 哪怕 type='other', 只要 content 提到具体实体, 就比对
    keyword_score = _keyword_mismatch(content, bws, gold, env_name)
    if keyword_score > 0:
        return keyword_score

    # 3) RF3: 整体 bws-vs-gold 字段级 oracle 差异 — 给一个 base "oracle 仍可看出有差"
    #    的兜底分, 让 oracle 在 LLM 没结构化 belief 时也能识别"belief schema 与 gold 不一致"
    bws_score = _bws_vs_gold_score(bws, gold, env_name)
    if bws_score > 0:
        # 缩放: bws_score 来自整体相似度, 给一个温和的 oracle score
        return 0.5 * bws_score

    # 4) 最后回退: criticality × low confidence × staleness
    crit = {"high": 1.0, "medium": 0.5, "low": 0.0}.get(belief.get("criticality", "low"), 0.0)
    conf = float(belief.get("confidence", 1.0))
    stale = min(1.0, belief.get("staleness", 0) / 10.0)
    return crit * (1 - conf) * stale  # ∈ [0, 1]


def _typed_mismatch(belief, bws, gold, env_name) -> float:
    """原 type-driven mismatch 函数 (RF3 修复前的行为, 仍是首选 path)。"""
    btype = belief.get("type", "other")
    content = str(belief.get("content", ""))

    if env_name == "ObjectStateWorld":
        if btype == "object_location":
            gold_locs = gold.get("object_locations", {})
            belief_locs = bws.get("object_locations", {}) or {}
            mismatches = 0
            for obj, gloc in gold_locs.items():
                if obj in content:
                    bloc = belief_locs.get(obj)
                    if bloc is not None and str(bloc) != str(gloc):
                        mismatches += 1
            return float(mismatches)
        if btype == "door_state":
            gold_doors = gold.get("door_states", {})
            belief_doors = bws.get("door_states", {}) or {}
            mismatches = 0
            for did, gst in gold_doors.items():
                if did in content:
                    bst = belief_doors.get(did)
                    if bst is not None and str(bst) != str(gst):
                        mismatches += 1
            return float(mismatches)
        if btype == "inventory":
            return 1.0 if set(bws.get("inventory", []) or []) != set(gold.get("inventory", []) or []) else 0.0
    if env_name == "ToolDAGWorld":
        if btype in ("tool_dep", "subgoal"):
            gold_avail = set(gold.get("available_variables", []))
            bel_outputs = set((bws.get("tool_outputs") or {}).keys())
            return float(len(gold_avail.symmetric_difference(bel_outputs)))
    if env_name == "GraphNavWorld":
        if btype in ("edge_state", "door_state"):
            edges = gold.get("edges", {})
            bel_doors = bws.get("door_states", {}) or {}
            mismatches = 0
            for eid, e in edges.items():
                gst = "locked" if e["locked"] else "open"
                bst = bel_doors.get(eid)
                if bst is not None and str(bst) != gst:
                    mismatches += 1
            return float(mismatches)
    return 0.0


def _keyword_mismatch(content: str, bws, gold, env_name: str) -> float:
    """RF3: 用 content 中的实体名匹配 gold 字段, 找 mismatch。"""
    if not content:
        return 0.0
    score = 0.0
    if env_name == "ObjectStateWorld":
        gold_locs = gold.get("object_locations", {})
        bws_locs = bws.get("object_locations", {}) or {}
        for obj, gloc in gold_locs.items():
            if obj.lower() in content:
                bloc = bws_locs.get(obj)
                if bloc is None or str(bloc).lower() != str(gloc).lower():
                    score += 1.0
        gold_doors = gold.get("door_states", {})
        bws_doors = bws.get("door_states", {}) or {}
        for did, gst in gold_doors.items():
            if did.lower() in content:
                bst = bws_doors.get(did)
                if bst is None or str(bst).lower() != str(gst).lower():
                    score += 1.0
    if env_name == "ToolDAGWorld":
        gold_avail = set(gold.get("available_variables", []))
        bws_outputs = set((bws.get("tool_outputs") or {}).keys())
        for v in gold.get("variables", []):
            if v.lower() in content:
                if (v in gold_avail) != (v in bws_outputs):
                    score += 1.0
        for t in gold.get("tools", []):
            if t.lower() in content:
                req = set(gold.get("tool_inputs", {}).get(t, []))
                missing_in_avail = req - gold_avail
                if missing_in_avail:
                    score += 0.5
    if env_name == "GraphNavWorld":
        edges = gold.get("edges", {})
        bws_doors = bws.get("door_states", {}) or {}
        bws_locs = bws.get("object_locations", {}) or {}
        for eid, e in edges.items():
            gst = "locked" if e["locked"] else "open"
            if eid.lower() in content or e["a"].lower() in content and e["b"].lower() in content:
                bst = bws_doors.get(eid)
                if bst is None or str(bst).lower() != gst:
                    score += 1.0
        for k, gloc in (gold.get("key_locations", {}) or {}).items():
            if k.lower() in content:
                bloc = bws_locs.get(k)
                if bloc is None or str(bloc).lower() != str(gloc).lower():
                    score += 1.0
    return score


def _bws_vs_gold_score(bws, gold, env_name: str) -> float:
    """RF3 兜底: 把 bws 整体 vs gold 的字段级"oracle 差"算出来, 让 oracle 至少能识别"belief 不完整"。"""
    if not isinstance(bws, dict):
        return 0.0
    score = 0.0
    if env_name == "ObjectStateWorld":
        gold_locs = gold.get("object_locations", {})
        bws_locs = bws.get("object_locations", {}) or {}
        score += sum(1 for o, gl in gold_locs.items() if str(bws_locs.get(o, "_")).lower() != str(gl).lower())
        gold_doors = gold.get("door_states", {})
        bws_doors = bws.get("door_states", {}) or {}
        score += sum(1 for d, gs in gold_doors.items() if str(bws_doors.get(d, "_")).lower() != str(gs).lower())
        if set(bws.get("inventory", []) or []) != set(gold.get("inventory", []) or []):
            score += 1.0
    if env_name == "ToolDAGWorld":
        score += abs(len(gold.get("available_variables", [])) - len(bws.get("tool_outputs") or {}))
    if env_name == "GraphNavWorld":
        bws_doors = bws.get("door_states", {}) or {}
        for eid, e in (gold.get("edges") or {}).items():
            gst = "locked" if e["locked"] else "open"
            if str(bws_doors.get(eid, "_")).lower() != gst:
                score += 1.0
    return score
