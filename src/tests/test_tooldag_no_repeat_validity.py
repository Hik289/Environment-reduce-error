"""R3 Stage B invariant tests for ToolDAGWorld no_repeat fix (2026-06-04).

Tests:
- T1: call_tool(t_0) on fresh env → valid=True
- T2: call_tool(t_0) twice → second call valid=False with reason='already_completed'
- T3: After t_0 mutated away (variable removed), re-call t_0 valid=True again
- T4: call_tool(t_unknown) → valid=False with reason='unknown_tool'
- T5: call_tool with missing inputs → valid=False with reason='missing_inputs'
- T6: Scoring is unchanged (R3 β fix removed): empty-empty=0.7 const back
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.environments import make_environment, default_stress


def setup_env(seed=42, mutation="static"):
    """Default static (no mutation) for deterministic invariant tests."""
    env = make_environment("ToolDAGWorld")
    stress = default_stress("S2")
    stress["state_mutation_rate"] = mutation
    env.reset(seed=seed, stress_config=stress)
    return env


def test_T1_first_call_valid():
    env = setup_env()
    res = env.step_task_action("call_tool(t_0)")
    assert env._last_action_valid is True, f"first call_tool(t_0) should be valid, info={res.info}"
    print(f"  T1: first call_tool(t_0) → valid=True ✓")


def test_T2_repeat_call_rejected():
    env = setup_env()
    env.step_task_action("call_tool(t_0)")
    assert env._last_action_valid is True
    res2 = env.step_task_action("call_tool(t_0)")
    assert env._last_action_valid is False, f"repeat call should be rejected"
    assert res2.info.get("reason") == "already_completed", f"reason should be 'already_completed', got {res2.info.get('reason')}"
    print(f"  T2: repeat call_tool(t_0) → valid=False reason='already_completed' ✓")


def test_T3_mutation_re_enables_recall():
    """volatile mutation may remove v_0, then re-call t_0 should succeed."""
    env = setup_env(seed=42, mutation="static")
    env.step_task_action("call_tool(t_0)")
    assert env._last_action_valid is True
    # Manually simulate mutation: remove v_0 + t_0 from completed
    env._gold["available_variables"] = [v for v in env._gold["available_variables"] if v != "v_0"]
    env._gold["tool_outputs"].pop("v_0", None)
    env._gold["completed_tools"] = [t for t in env._gold["completed_tools"] if t != "t_0"]
    # Now re-call should succeed
    env.step_task_action("call_tool(t_0)")
    assert env._last_action_valid is True, f"after mutation re-call should succeed"
    print(f"  T3: post-mutation re-call_tool(t_0) → valid=True ✓")


def test_T4_unknown_tool():
    env = setup_env()
    res = env.step_task_action("call_tool(t_999)")
    assert env._last_action_valid is False
    assert res.info.get("reason") == "unknown_tool"
    print(f"  T4: call_tool(t_999) → reason='unknown_tool' ✓")


def test_T5_missing_inputs():
    env = setup_env()
    # t_1 needs v_0; before calling t_0, v_0 doesn't exist
    res = env.step_task_action("call_tool(t_1)")
    assert env._last_action_valid is False
    assert res.info.get("reason") == "missing_inputs"
    print(f"  T5: call_tool(t_1) before t_0 → reason='missing_inputs' ✓")


def test_T6_scoring_floor_restored():
    """R3 β fix was rolled back; init agent vs empty gold → 0.7 const (intentional, paper uses raw scoring)."""
    env = setup_env()
    init_belief = {
        "belief_world_state": {
            "completed_subgoals": [],
            "tool_outputs": {},
            "open_dependencies": [],
        }
    }
    score = env.score_belief_state(init_belief)
    # 0.30 (completed empty-empty=1.0) + 0.40 (tool_outputs empty-empty=1.0) + 0.30 (open_deps [] vs all_tools=0) = 0.70
    assert abs(score - 0.70) < 0.01, f"init score should be 0.70 (R3 β rollback), got {score}"
    print(f"  T6: init scoring = {score:.4f} (R3 β fix rolled back to 0.70 const) ✓")


if __name__ == "__main__":
    print("=== R3 Stage B no_repeat invariant tests ===")
    test_T1_first_call_valid()
    test_T2_repeat_call_rejected()
    test_T3_mutation_re_enables_recall()
    test_T4_unknown_tool()
    test_T5_missing_inputs()
    test_T6_scoring_floor_restored()
    print()
    print("✓ All Stage B invariant tests passed.")
