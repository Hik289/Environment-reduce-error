"""Oracle-Task-Weighted: 改进的 Oracle baseline (REVISION P0 Cell B).

REVISION 背景: 3 reviewer 一致质疑原 Oracle 用 uniform utility gold-mismatch argmax,
导致 Oracle UPR=0%, Judge > Oracle +26pp (artifact of bad baseline). 本 cell 用 task
contribution weight 加权每个 belief field, 而非 uniform mismatch count.

权重 (从 env config 直接读 gold trace, 不用 LLM):
- ToolDAGWorld: variable/tool 权重 = downstream tool dependency count (归一化)
- ObjectStateWorld: goal_object=1.0, critical path 上 door/key=0.7, 其他=0.1
- GraphNavWorld: shortest path edge=1.0, 解锁 path 上 locked edge 的 key=0.7
"""
from collections import defaultdict, deque
from typing import Any, Dict

from .base import Method, MethodContext, MethodDecision
from .periodic_probe import _probe_for_belief


class OracleTaskWeightedMethod(Method):
    name = "oracle_task_weighted"
    method_hint = ("Oracle scheduler weighting each belief field by task contribution "
                   "(critical-path / dependency / goal-reachability), probing highest "
                   "weighted mismatch.")

    def decide(self, ctx: MethodContext) -> MethodDecision:
        if self.is_budget_exhausted(ctx):
            return self.force_act_from_agent(ctx, "budget_exhausted")

        gold = ctx.env.get_gold_state()
        beliefs = ctx.agent_output.get("beliefs") or []
        bws = ctx.agent_output.get("belief_world_state") or {}
        env_name = type(ctx.env).__name__

        weights = _compute_task_weights(gold, env_name)
        if not weights:
            return self.force_act_from_agent(ctx, "oracle_tw_no_weights")

        scored = []
        for b in beliefs:
            score = _weighted_mismatch(b, bws, gold, env_name, weights)
            if score > 0:
                scored.append((score, b))

        if not scored:
            return self.force_act_from_agent(ctx, "oracle_tw_no_mismatch")

        scored.sort(key=lambda x: x[0], reverse=True)
        _, top_b = scored[0]
        probe = _probe_for_belief(top_b, ctx)
        return MethodDecision(decision_type="probe", action=probe,
                              target_belief_id=top_b.get("id"),
                              reasoning=f"oracle_tw_score={scored[0][0]:.2f}",
                              overrode_agent=True)


# ---------------------------------------------------------------------------
# Task weight computation
# ---------------------------------------------------------------------------

def _compute_task_weights(gold: Dict[str, Any], env_name: str) -> Dict[str, Dict[str, float]]:
    if env_name == "ToolDAGWorld":
        return _weights_tool_dag(gold)
    if env_name == "ObjectStateWorld":
        return _weights_object_state(gold)
    if env_name == "GraphNavWorld":
        return _weights_graph_nav(gold)
    return {}


def _weights_tool_dag(gold):
    tool_inputs = gold.get("tool_inputs", {}) or {}
    tools = gold.get("tools", []) or []
    completed = set(gold.get("completed_tools", []) or [])
    tool_output_var = gold.get("tool_output_var", {}) or {}
    var_consumers = defaultdict(set)
    for t, ins in tool_inputs.items():
        if t in completed:
            continue
        for v in ins:
            var_consumers[v].add(t)
    tool_downstream = defaultdict(int)
    for t in tools:
        if t in completed:
            tool_downstream[t] = 0
            continue
        visited = {t}
        q = deque([t])
        cnt = 1
        while q:
            cur = q.popleft()
            v_cur = tool_output_var.get(cur)
            for t2 in var_consumers.get(v_cur, set()):
                if t2 not in visited:
                    visited.add(t2)
                    cnt += 1
                    q.append(t2)
        tool_downstream[t] = cnt
    max_t = max(tool_downstream.values()) if tool_downstream else 1
    tool_w = {t: c / max_t for t, c in tool_downstream.items()} if max_t > 0 else {}
    var_w = {}
    for v, consumers in var_consumers.items():
        var_w[v] = max((tool_w.get(t, 0.0) for t in consumers), default=0.0)
    for t in completed:
        v = tool_output_var.get(t)
        if v is not None:
            var_w[v] = 0.1
    return {"tools": tool_w, "variables": var_w}


def _weights_object_state(gold):
    goal_obj = gold.get("goal_object")
    obj_locs = gold.get("object_locations", {}) or {}
    door_states = gold.get("door_states", {}) or {}
    rooms = gold.get("rooms", []) or []
    agent_loc = gold.get("agent_location")
    goal_room = obj_locs.get(goal_obj) if goal_obj else None
    obj_w = {o: 0.1 for o in obj_locs}
    door_w = {d: 0.1 for d in door_states}
    if goal_obj:
        obj_w[goal_obj] = 1.0
    if agent_loc and goal_room and goal_room in rooms:
        adj = defaultdict(list)
        for did in door_states:
            if "__" in did:
                a, b = did.split("__", 1)
                if a in rooms and b in rooms:
                    adj[a].append((b, did))
                    adj[b].append((a, did))
        parent = {agent_loc: (None, None)}
        q = deque([agent_loc])
        found = False
        while q and not found:
            cur = q.popleft()
            if cur == goal_room:
                found = True
                break
            for nxt, did in adj.get(cur, []):
                if nxt not in parent:
                    parent[nxt] = (cur, did)
                    q.append(nxt)
        if found:
            cur = goal_room
            while parent.get(cur, (None, None))[0] is not None:
                prev, did = parent[cur]
                door_w[did] = 0.7
                if door_states.get(did, "").lower() == "locked":
                    key_name = f"key_{did}"
                    if key_name in obj_w:
                        obj_w[key_name] = 0.7
                cur = prev
    return {"object_locations": obj_w, "door_states": door_w}


def _weights_graph_nav(gold):
    agent_node = gold.get("agent_node")
    goal_node = gold.get("goal_node")
    edges = gold.get("edges", {}) or {}
    key_locs = gold.get("key_locations", {}) or {}
    edge_w = {eid: 0.1 for eid in edges}
    key_w = {k: 0.1 for k in key_locs}
    if not (agent_node and goal_node):
        return {"door_states": edge_w, "object_locations": key_w}
    adj = defaultdict(list)
    for eid, e in edges.items():
        a, b = e.get("a"), e.get("b")
        if a and b:
            adj[a].append((b, eid))
            adj[b].append((a, eid))
    parent = {agent_node: (None, None)}
    q = deque([agent_node])
    while q:
        cur = q.popleft()
        if cur == goal_node:
            break
        for nxt, eid in adj.get(cur, []):
            if nxt not in parent:
                parent[nxt] = (cur, eid)
                q.append(nxt)
    if goal_node not in parent:
        return {"door_states": edge_w, "object_locations": key_w}
    cur = goal_node
    while parent.get(cur, (None, None))[0] is not None:
        prev, eid = parent[cur]
        edge_w[eid] = 1.0
        if edges.get(eid, {}).get("locked"):
            req_key = edges[eid].get("required_key")
            if req_key and req_key in key_w:
                key_w[req_key] = 0.7
        cur = prev
    return {"door_states": edge_w, "object_locations": key_w}


# ---------------------------------------------------------------------------
# Weighted mismatch
# ---------------------------------------------------------------------------

def _weighted_mismatch(belief, bws, gold, env_name, weights) -> float:
    content = str(belief.get("content", "")).lower()
    score = 0.0

    if env_name == "ObjectStateWorld":
        obj_w = weights.get("object_locations", {})
        door_w = weights.get("door_states", {})
        gold_locs = gold.get("object_locations", {}) or {}
        bws_locs = bws.get("object_locations", {}) or {}
        for o, gloc in gold_locs.items():
            if o.lower() not in content:
                continue
            bloc = bws_locs.get(o)
            if bloc is None or str(bloc).lower() != str(gloc).lower():
                score += obj_w.get(o, 0.1)
        gold_doors = gold.get("door_states", {}) or {}
        bws_doors = bws.get("door_states", {}) or {}
        for d, gst in gold_doors.items():
            if d.lower() not in content:
                continue
            bst = bws_doors.get(d)
            if bst is None or str(bst).lower() != str(gst).lower():
                score += door_w.get(d, 0.1)

    elif env_name == "ToolDAGWorld":
        tool_w = weights.get("tools", {})
        var_w = weights.get("variables", {})
        gold_avail = set(gold.get("available_variables", []) or [])
        bws_outputs = set((bws.get("tool_outputs") or {}).keys())
        for v in gold.get("variables", []) or []:
            if v.lower() not in content:
                continue
            if (v in gold_avail) != (v in bws_outputs):
                score += var_w.get(v, 0.1)
        completed = set(gold.get("completed_tools", []) or [])
        b_completed = set(bws.get("completed_subgoals", []) or [])
        for t in gold.get("tools", []) or []:
            if t.lower() not in content:
                continue
            if (t in completed) != (t in b_completed):
                score += tool_w.get(t, 0.1)

    elif env_name == "GraphNavWorld":
        edge_w = weights.get("door_states", {})
        key_w = weights.get("object_locations", {})
        edges = gold.get("edges", {}) or {}
        bws_doors = bws.get("door_states", {}) or {}
        for eid, e in edges.items():
            a = (e.get("a") or "").lower()
            b = (e.get("b") or "").lower()
            if eid.lower() not in content and not (a in content and b in content):
                continue
            gst = "locked" if e.get("locked") else "open"
            bst = bws_doors.get(eid)
            if bst is None or str(bst).lower() != gst:
                score += edge_w.get(eid, 0.1)
        bws_locs = bws.get("object_locations", {}) or {}
        for k, gloc in (gold.get("key_locations", {}) or {}).items():
            if k.lower() not in content:
                continue
            bloc = bws_locs.get(k)
            if bloc is None or str(bloc).lower() != str(gloc).lower():
                score += key_w.get(k, 0.1)

    if score == 0.0:
        crit = {"high": 1.0, "medium": 0.5, "low": 0.0}.get(belief.get("criticality", "low"), 0.0)
        conf = float(belief.get("confidence", 1.0))
        score = crit * (1 - conf) * 0.5

    return score
