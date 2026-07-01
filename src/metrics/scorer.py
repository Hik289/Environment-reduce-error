"""Scorer: 计算 14 项 metric。

每步打分 score_step, 每集汇总 score_episode。
14 metric:
1. task_success
2. collapse_onset (世界状态首次跌破 0.6 的 step)
3. world_state_accuracy (mean)
4. action_validity (有效 task action 比例)
5. probe_efficiency (probe / total task action)
6. useful_probe_rate (probe answer 修正 belief 比例)
7. missed_critical_probe_rate (failure 时 critical belief 没被 probe 的比例)
8. false_belief_commitment (基于错 belief 的 action 数)
9. self_check_accuracy (agent self-check 正确率)
10. collapse_delay (vs no_probe, 在 aggregator 算)
11. recovery_rate (drift 后 K 步内恢复)
12. probe_budget_usage (probes_used / budget)
13. mean_world_state_accuracy_last_third (后 1/3 步)
14. dependency_accuracy (door/dep mismatch 比例)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


COLLAPSE_THRESHOLD = 0.6
RECOVERY_K = 3


def score_step(env, agent_output: Dict[str, Any], decision: Dict[str, Any],
               obs_before: Dict[str, Any],
               extra_per_belief: bool = False) -> Dict[str, Any]:
    """每步打分: 调用 env.score_belief_state, 比对 action 是否合法。

    RF4: self_check_correct 在 score_step 中只给"老版本 wsa-threshold proxy"作初值;
    主 loop (run_smoke) 在 act 执行后会用 task_action_valid 重写为 H3-纯净版。

    Theorist 信号 3: 当 extra_per_belief=True, 在 envprobe_simple/oracle_probe cell 中
    额外输出 probe_score_per_belief + oracle_delta_per_belief, 用于 ANALYSIS 算
    α Spearman rank correlation。
    """
    bws = agent_output.get("belief_world_state", {})
    wsa = float(env.score_belief_state(bws or {}))

    dep_acc = _dependency_accuracy(env, bws)

    # 旧 proxy: 仅作 fallback (probe step 没 task_action_valid 时用这个)
    sc = agent_output.get("self_check", {})
    sc_self = bool(sc.get("is_current_world_state_consistent", True))
    sc_correct_proxy = (sc_self == (wsa >= COLLAPSE_THRESHOLD))

    fbc = _false_belief_commitment(env, agent_output)

    out = {
        "world_state_accuracy": wsa,
        "dependency_accuracy": dep_acc,
        "self_check_correct": sc_correct_proxy,
        "self_check_reported": sc_self,
        "false_belief_commitment": fbc,
    }

    if extra_per_belief:
        beliefs = agent_output.get("beliefs") or []
        env_name = type(env).__name__
        # rho_i (复用 envprobe_simple._belief_score)
        from ..methods.envprobe_simple import _belief_score, CRIT_MAP
        # delta_i (复用 oracle_probe._belief_mismatch)
        from ..methods.oracle_probe import _belief_mismatch
        gold = env.get_gold_state()
        rho_list = []
        delta_list = []
        for b in beliefs:
            rho_i = _belief_score(b)
            stale = min(1.0, float(b.get("staleness", 0)) / 10.0)
            conf = float(b.get("confidence", 1.0))
            used = 1.0 if b.get("used_by_next_action") else 0.0
            rf_len = len(b.get("required_for") or [])
            rho_list.append({
                "id": b.get("id"), "rho_i": rho_i,
                "c": CRIT_MAP.get(str(b.get("criticality", "low")), 0.0),
                "s": stale, "u": max(0.0, 1.0 - conf),
                "d": 0.5 * used + 0.5 * min(1.0, rf_len / 3.0),
                "criticality_raw": b.get("criticality"),
                "type": b.get("type"),
            })
            delta_i = _belief_mismatch(b, bws, gold, env_name)
            delta_list.append({
                "id": b.get("id"), "delta_i": float(delta_i),
                "type": b.get("type"),
            })
        out["probe_score_per_belief"] = rho_list
        out["oracle_delta_per_belief"] = delta_list

    return out


def evaluate_self_check_v2(sc_reported: bool, decision_type: str,
                            task_action_valid: Optional[bool]) -> Optional[bool]:
    """RF4: H3 纯净评分语义。

    定义:
    - TP: sc.consistent=True  且 act 实际 valid  (agent 自信对 → 真对)
    - TN: sc.consistent=False 且 act 实际 invalid (agent 自疑 → 真错)
    - FP (false-confident, H3 现象): sc.consistent=True  且 act invalid
    - FN (false-doubt):                sc.consistent=False 且 act valid

    self_check_correct = (TP or TN) → True
                       = (FP or FN) → False

    在 probe / reset 步 (没 task_action_valid), 返回 None (不计入 accuracy 分母)。
    """
    if decision_type != "act" or task_action_valid is None:
        return None
    return bool(sc_reported) == bool(task_action_valid)


def _dependency_accuracy(env, bws: Dict[str, Any]) -> float:
    if not isinstance(bws, dict):
        return 0.0
    gold = env.get_gold_state()
    env_name = type(env).__name__
    if env_name == "ObjectStateWorld":
        # door_states match
        bg = bws.get("door_states") or {}
        gg = gold.get("door_states") or {}
        if not gg:
            return 1.0
        matches = sum(1 for d, st in gg.items() if bg.get(d) == st)
        return matches / max(1, len(gg))
    if env_name == "ToolDAGWorld":
        # completed tools / available vars
        bg = set((bws.get("completed_subgoals") or []))
        gg = set(gold.get("completed_tools") or [])
        if not gg and not bg:
            return 1.0
        inter = len(bg & gg)
        union = len(bg | gg)
        return inter / max(1, union) if union else 1.0
    if env_name == "GraphNavWorld":
        edges = gold.get("edges") or {}
        bdoor = bws.get("door_states") or {}
        if not edges:
            return 1.0
        matches = 0
        for eid, e in edges.items():
            gst = "locked" if e["locked"] else "open"
            bst = bdoor.get(eid)
            if bst is not None and str(bst) == gst:
                matches += 1
        return matches / max(1, len(edges))
    return 0.0


def _false_belief_commitment(env, agent_output: Dict[str, Any]) -> bool:
    nd = agent_output.get("next_decision", {})
    if str(nd.get("type")) != "act":
        return False
    tb_id = nd.get("target_belief")
    if not tb_id:
        return False
    beliefs = agent_output.get("beliefs") or []
    tb = next((b for b in beliefs if b.get("id") == tb_id), None)
    if not tb:
        return False
    # 若 belief 的 content 与 gold 显然不一致, 视为 fbc=True
    bws = agent_output.get("belief_world_state", {})
    gold = env.get_gold_state()
    return _belief_inconsistent_with_gold(tb, bws, gold, type(env).__name__)


def _belief_inconsistent_with_gold(b, bws, gold, env_name) -> bool:
    btype = b.get("type", "other")
    content = b.get("content", "")
    if env_name == "ObjectStateWorld":
        if btype == "object_location":
            gold_locs = gold.get("object_locations", {})
            bws_locs = bws.get("object_locations", {}) or {}
            for obj, gloc in gold_locs.items():
                if obj in content and obj in bws_locs and str(bws_locs[obj]) != str(gloc):
                    return True
    return False


# -----------------------------------------------------------------------------
# Episode 级聚合
# -----------------------------------------------------------------------------

def score_episode(step_records: List[Dict[str, Any]],
                  task_success: bool,
                  probe_budget_total: int,
                  probes_used: int,
                  useful_probe_count: int,
                  total_probe_count: int,
                  ) -> Dict[str, Any]:
    if not step_records:
        return _empty_episode(task_success, probe_budget_total, probes_used)

    wsa_list = [r.get("world_state_accuracy", 0.0) for r in step_records]
    da_list = [r.get("dependency_accuracy", 0.0) for r in step_records]
    # RF4: 只把 self_check_correct in {True, False} 的 step 计入分母 (None=probe/reset 跳过)
    sc_list = [r.get("self_check_correct") for r in step_records if r.get("self_check_correct") is not None]
    fbc_list = [r.get("false_belief_commitment", False) for r in step_records]
    action_valid_list = [r.get("task_action_valid") for r in step_records
                         if r.get("decision_type") == "act" and r.get("task_action_valid") is not None]

    # collapse_onset: 首次 wsa < 0.6 且后续连续 <=0.6 至少 2 步
    collapse_onset = _detect_collapse(wsa_list)
    action_collapse_onset = _detect_collapse_bool([not v for v in [r.get("task_action_valid", True)
                                                                    for r in step_records]])

    # action_validity: 在所有 act 动作中合法的比例
    action_validity = sum(1 for v in action_valid_list if v) / max(1, len(action_valid_list))

    n = len(step_records)
    last_third_start = max(0, 2 * n // 3)
    wsa_last_third = sum(wsa_list[last_third_start:]) / max(1, n - last_third_start)

    # probe efficiency / useful probe rate
    probe_efficiency = total_probe_count / max(1, n)  # probe / total step
    useful_probe_rate = (useful_probe_count / max(1, total_probe_count)) if total_probe_count > 0 else None

    # recovery_rate
    recovery_rate = _recovery_rate(wsa_list, k=RECOVERY_K)

    # missed_critical_probe_rate: 在 failure (final wsa < 0.5) 时, 有 high-criticality belief 未被 probe 的比例
    missed_critical = _missed_critical(step_records, task_success)

    return {
        "task_success": int(bool(task_success)),
        "collapse_onset": collapse_onset,
        "action_collapse_onset": action_collapse_onset,
        "world_state_accuracy": sum(wsa_list) / max(1, n),
        "world_state_accuracy_last_third": wsa_last_third,
        "action_validity": action_validity,
        "probe_efficiency": probe_efficiency,
        "useful_probe_rate": useful_probe_rate,
        "missed_critical_probe_rate": missed_critical,
        "false_belief_commitment": sum(1 for x in fbc_list if x),
        "self_check_accuracy": (sum(1 for x in sc_list if x) / len(sc_list)) if sc_list else None,
        "self_check_n_act_steps": len(sc_list),
        "collapse_delay": None,  # 由 aggregator vs no_probe 计算
        "recovery_rate": recovery_rate,
        "probe_budget_usage": probes_used / max(1, probe_budget_total) if probe_budget_total > 0 else 0.0,
        "dependency_accuracy": sum(da_list) / max(1, n),
        "n_steps": n,
    }


def _empty_episode(task_success, budget, used):
    return {
        "task_success": int(bool(task_success)),
        "collapse_onset": None,
        "action_collapse_onset": None,
        "world_state_accuracy": 0.0,
        "world_state_accuracy_last_third": 0.0,
        "action_validity": 0.0,
        "probe_efficiency": 0.0,
        "useful_probe_rate": None,
        "missed_critical_probe_rate": None,
        "false_belief_commitment": 0,
        "self_check_accuracy": None,
        "self_check_n_act_steps": 0,
        "collapse_delay": None,
        "recovery_rate": None,
        "probe_budget_usage": used / max(1, budget) if budget > 0 else 0.0,
        "dependency_accuracy": 0.0,
        "n_steps": 0,
    }


def _detect_collapse(wsa_list: List[float]) -> Optional[int]:
    for i in range(len(wsa_list) - 1):
        if wsa_list[i] < COLLAPSE_THRESHOLD and wsa_list[i+1] < COLLAPSE_THRESHOLD:
            return i + 1  # 1-indexed step
    if wsa_list and wsa_list[-1] < COLLAPSE_THRESHOLD:
        return len(wsa_list)
    return None


def _detect_collapse_bool(bool_list: List[bool]) -> Optional[int]:
    for i in range(len(bool_list) - 1):
        if bool_list[i] and bool_list[i+1]:
            return i + 1
    return None


def _recovery_rate(wsa_list, k: int = 3) -> Optional[float]:
    drift_events = 0
    recovered = 0
    i = 0
    while i < len(wsa_list):
        if wsa_list[i] < COLLAPSE_THRESHOLD:
            drift_events += 1
            # 看后 K 步内是否恢复
            for j in range(i + 1, min(i + 1 + k, len(wsa_list))):
                if wsa_list[j] >= COLLAPSE_THRESHOLD:
                    recovered += 1
                    break
            i += k + 1
        else:
            i += 1
    if drift_events == 0:
        return None
    return recovered / drift_events


def _missed_critical(step_records, task_success) -> Optional[float]:
    if task_success:
        return 0.0
    # 收集所有 high-criticality belief 与是否被 probe 过
    total_critical = 0
    probed_critical = 0
    probed_ids = set()
    for r in step_records:
        if r.get("decision_type") == "probe" and r.get("target_belief"):
            probed_ids.add(r.get("target_belief"))
    for r in step_records:
        for b in (r.get("belief_table") or []):
            if str(b.get("criticality")) == "high":
                total_critical += 1
                if b.get("id") in probed_ids:
                    probed_critical += 1
    if total_critical == 0:
        return None
    return 1.0 - (probed_critical / total_critical)


# -----------------------------------------------------------------------------
# Probe usefulness: probe answer 修正了 belief 吗?
# -----------------------------------------------------------------------------

def compare_belief_to_gold_for_probe(probe_response: Dict[str, Any],
                                     belief_before: Dict[str, Any]) -> bool:
    """判定 probe answer 是否揭示了与 belief 不同的事实 (useful)。

    修复 v2 (2026-05-28):
    - 缺失 belief field (b_x = None) 但 probe 返回了有效值 → useful (agent 不知道, probe 提供新知识)
    - 加全所有 probe types: check_current_position / inspect_room / verify_subgoal /
      check_node_locked / check_required_key / inspect_neighbors / check_target_distance_hint /
      check_current_node / inspect_tool_schema / validate_tool_output / check_container_status
    - check_required_inputs / check_tool_dependency: missing_inputs 非空 = useful
    """
    if not probe_response or not isinstance(probe_response, dict):
        return False
    ans = probe_response.get("answer") or {}
    if not ans or ans.get("error"):
        return False
    target = probe_response.get("target")
    bws = belief_before.get("belief_world_state", {}) if isinstance(belief_before, dict) else {}
    ptype = probe_response.get("probe_type", "")

    # --- ObjectStateWorld probes ---
    if ptype == "check_location" and target:
        ans_loc = ans.get("location")
        bws_locs = bws.get("object_locations") or {}
        b_loc = bws_locs.get(str(target))
        if ans_loc is None:
            return False
        if str(ans_loc) == "unknown":
            return False
        # belief 缺失 OR belief 与 probe 答案不同 → useful
        if b_loc is None or str(b_loc).lower() != str(ans_loc).lower():
            return True
        return False

    if ptype == "check_door_status" and target:
        ans_st = ans.get("status")
        b_st = (bws.get("door_states") or {}).get(str(target))
        if ans_st is None or str(ans_st) == "unknown":
            return False
        if b_st is None or str(b_st).lower() != str(ans_st).lower():
            return True
        return False

    if ptype == "check_container_status" and target:
        ans_loc = ans.get("location")
        b_loc = (bws.get("object_locations") or {}).get(str(target))
        if ans_loc is None or str(ans_loc) == "unknown":
            return False
        if b_loc is None or str(b_loc).lower() != str(ans_loc).lower():
            return True
        return False

    if ptype == "check_inventory":
        ans_inv = set(ans.get("inventory") or [])
        b_inv = set(bws.get("inventory") or [])
        return ans_inv != b_inv

    if ptype == "check_current_position":
        ans_loc = ans.get("location")
        b_loc = bws.get("current_location")
        if ans_loc is None:
            return False
        if b_loc is None or str(b_loc).lower() != str(ans_loc).lower():
            return True
        return False

    if ptype == "inspect_room" and target:
        ans_objs = set(ans.get("objects") or [])
        # 把 belief 里在该 room 的 objs 取出
        bws_locs = bws.get("object_locations") or {}
        b_objs = {o for o, r in bws_locs.items() if str(r) == str(target)}
        return ans_objs != b_objs

    if ptype == "verify_subgoal" and target:
        ans_done = ans.get("completed")
        b_done = str(target) in set(bws.get("completed_subgoals") or [])
        if ans_done is None:
            return False
        return bool(ans_done) != b_done

    if ptype == "check_preconditions":
        # executable=False 揭示行动会失败 = useful
        if ans.get("executable") is False:
            return True
        # missing 非空也是 useful
        if ans.get("missing"):
            return True
        return False

    # --- ToolDAGWorld probes ---
    if ptype == "check_variable_exists" and target:
        ans_ex = bool(ans.get("exists"))
        b_outs = bws.get("tool_outputs") or {}
        b_ex = str(target) in b_outs
        return ans_ex != b_ex

    if ptype == "validate_tool_output" and target:
        ans_ex = bool(ans.get("exists"))
        b_outs = bws.get("tool_outputs") or {}
        b_ex = str(target) in b_outs
        return ans_ex != b_ex

    if ptype in ("check_required_inputs", "check_tool_dependency"):
        missing = ans.get("missing_inputs") or []
        if missing:
            return True  # tool has unmet deps = useful
        # v3: missing=[] (tool IS ready) — useful if agent's open_dependencies still lists this tool
        if isinstance(target, str):
            b_open_deps = bws.get("open_dependencies") or []
            if target in b_open_deps:
                return True  # agent thinks tool has open deps but actually it's ready
        # v3: useful if avail_inputs reveals vars that agent's tool_outputs doesn't have
        avail = ans.get("available_inputs") or []
        b_outs = bws.get("tool_outputs") or {}
        for v in avail:
            if str(v) not in b_outs:
                return True  # agent didn't know this variable was available
        return False

    if ptype == "inspect_tool_schema" and target:
        # 揭示了 required_inputs / produces, 几乎总是 informative (除非 agent 已知)
        produces = ans.get("produces")
        b_outs = bws.get("tool_outputs") or {}
        if produces is not None and str(produces) not in b_outs:
            return True
        # v3: useful if any required_input is not in agent's tool_outputs
        req_inputs = ans.get("required_inputs") or []
        for v in req_inputs:
            if str(v) not in b_outs:
                return True
        return False

    # --- GraphNavWorld probes ---
    if ptype == "check_current_node":
        ans_node = ans.get("node")
        b_node = bws.get("current_location")
        if ans_node is None:
            return False
        if b_node is None or str(b_node).lower() != str(ans_node).lower():
            return True
        return False

    if ptype == "check_edge":
        # belief 里 door_states 记录 edge_id; 若 belief 没记或不一致 → useful
        if ans.get("exists") is False:
            return True  # belief 期望 edge 存在但实际不存在
        ans_locked = ans.get("locked")
        if ans_locked is None:
            return False
        # target 是 "n_A, n_B" 字符串, edge_id 需查 (我们用 target 字串 fuzzy 匹配 door_states key)
        b_doors = bws.get("door_states") or {}
        # 找到对应 edge_id (sorted alphabetic join with __)
        if isinstance(target, list) and len(target) == 2:
            a, b = sorted(map(str, target))
            eid = f"{a}__{b}"
        elif isinstance(target, str) and "," in target:
            parts = [p.strip() for p in target.split(",")]
            if len(parts) == 2:
                a, b = sorted(parts)
                eid = f"{a}__{b}"
            else:
                eid = target
        else:
            eid = str(target)
        b_st = b_doors.get(eid)
        ans_st = "locked" if ans_locked else "open"
        if b_st is None or str(b_st).lower() != ans_st:
            return True
        return False

    if ptype == "check_node_locked":
        # 揭示 locked_neighbor_edges; 若 belief 中无对应 door_state → useful
        locks = ans.get("locked_neighbor_edges") or []
        return bool(locks)

    if ptype == "check_required_key":
        return ans.get("required_key") is not None

    if ptype == "inspect_neighbors":
        return bool(ans.get("neighbors"))  # 任何邻居信息都是新的 (we don't track belief.neighbors)

    if ptype == "check_target_distance_hint":
        return ans.get("hops_to_target") is not None and ans.get("hops_to_target") >= 0

    # 未知 probe_type → False (保守)
    return False
