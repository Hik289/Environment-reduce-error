"""EnvProbe-Simple-CD: 仅用 criticality + dependency_role 两维 (REVISION P0 Cell A).

REVISION 背景: N1 ablation 显示 -s/-u 在 procedural 上 Δ_AH = +16.9pp / +23.8pp,
即砍掉 staleness 和 uncertainty 反而提升 procedural A_H。3 reviewer 一致质疑 4 维
score 不是 minimal sufficient policy, (c+d)-only 才是。本 cell 把它独立 baseline 化
跑 spine n=220 × 3 env, 与 envprobe_simple (4-dim) 和 periodic_probe paired。

probe_score(belief) = criticality + dependency_role, threshold = 0.75 (= 1.5 × 2/4)
"""
from .envprobe_simple import EnvProbeSimpleMethod


class EnvProbeSimpleCD(EnvProbeSimpleMethod):
    """仅用 criticality + dependency_role 维度 (REVISION main proposal)。"""
    name = "envprobe_simple_cd"
    method_hint = ("Maintain accurate criticality / required_for fields. "
                   "Probing triggered when (criticality + dependency_role) score is high.")
    _use_c = True
    _use_s = False
    _use_u = False
    _use_d = True
