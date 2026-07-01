"""EnvProbe-Simple: 4 维评分 (criticality × staleness × uncertainty × dependency_role)。

probe_score(belief) = criticality_score + staleness_score + uncertainty_score + dependency_role_score
每项归一化到 [0, 1], 总分阈值 1.5 (4 项中至少有 1~2 项达到 0.5+ 的强信号)。

若 max(score) > threshold → probe 该 belief; 否则 act。

Ablation 变体 (Corollary 1): _minus_{c, s, u, d} 关掉一维, threshold 按比例降。
"""
from .base import Method, MethodContext, MethodDecision
from .periodic_probe import _probe_for_belief


CRIT_MAP = {"high": 1.0, "medium": 0.5, "low": 0.0}
PROBE_SCORE_THRESHOLD = 1.5


class EnvProbeSimpleMethod(Method):
    name = "envprobe_simple"
    method_hint = ("Maintain accurate criticality / staleness / confidence / required_for fields. "
                   "Probing is triggered when belief score is high. "
                   "Score = criticality + staleness + (1-confidence) + dependency_role.")
    # 用于 ablation: 哪些维度启用 (默认全部)
    _use_c = True
    _use_s = True
    _use_u = True
    _use_d = True

    @property
    def threshold(self) -> float:
        n_dims = sum([self._use_c, self._use_s, self._use_u, self._use_d])
        # 保持比例: 默认 4 维 threshold=1.5, 砍一维降到 3/4 × 1.5 = 1.125
        return PROBE_SCORE_THRESHOLD * n_dims / 4.0

    def decide(self, ctx: MethodContext) -> MethodDecision:
        if self.is_budget_exhausted(ctx):
            return self.force_act_from_agent(ctx, "budget_exhausted")

        beliefs = ctx.agent_output.get("beliefs") or []
        if not beliefs:
            return self.force_act_from_agent(ctx, "no_beliefs")

        scored = [(b, _belief_score(b, self._use_c, self._use_s, self._use_u, self._use_d))
                  for b in beliefs]
        best_b, best_score = max(scored, key=lambda x: x[1])

        if best_score >= self.threshold:
            probe = _probe_for_belief(best_b, ctx)
            return MethodDecision(decision_type="probe", action=probe,
                                  target_belief_id=best_b.get("id"),
                                  reasoning=f"{self.name}:score={best_score:.2f}",
                                  overrode_agent=True)

        return self.force_act_from_agent(ctx, f"{self.name}:max_score={best_score:.2f}")


def _belief_score(b: dict, use_c: bool = True, use_s: bool = True,
                  use_u: bool = True, use_d: bool = True) -> float:
    """4 项加和评分, 每项 [0, 1]。ablation 通过 use_* flag 关一维。"""
    score = 0.0
    if use_c:
        score += CRIT_MAP.get(str(b.get("criticality", "low")), 0.0)
    if use_s:
        stale_raw = float(b.get("staleness", 0))
        score += min(1.0, stale_raw / 10.0)
    if use_u:
        conf = float(b.get("confidence", 1.0))
        score += max(0.0, 1.0 - conf)
    if use_d:
        used = 1.0 if b.get("used_by_next_action") else 0.0
        rf = b.get("required_for") or []
        score += 0.5 * used + 0.5 * min(1.0, len(rf) / 3.0)
    return score


# ---- Ablation 变体 (Corollary 1) ----

class EnvProbeSimpleMinusC(EnvProbeSimpleMethod):
    name = "envprobe_simple_minus_c"
    _use_c = False
    method_hint = "Probing 评分忽略 criticality 维度 (ablation)。"


class EnvProbeSimpleMinusS(EnvProbeSimpleMethod):
    name = "envprobe_simple_minus_s"
    _use_s = False
    method_hint = "Probing 评分忽略 staleness 维度 (ablation)。"


class EnvProbeSimpleMinusU(EnvProbeSimpleMethod):
    name = "envprobe_simple_minus_u"
    _use_u = False
    method_hint = "Probing 评分忽略 uncertainty 维度 (ablation)。"


class EnvProbeSimpleMinusD(EnvProbeSimpleMethod):
    name = "envprobe_simple_minus_d"
    _use_d = False
    method_hint = "Probing 评分忽略 dependency_role 维度 (ablation)。"
