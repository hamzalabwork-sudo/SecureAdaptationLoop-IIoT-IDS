"""Writes configs/atk_<attack>_<dataset>.json for the Phase-1 attack study.

Every attack runs against the threshold gate TG (balanced net benefit, tau_D = 1%, hard floor
30%, per-class cumulative budget 30%, soft escalation 10%, 2 of 3 validators with 300 clean flows
per class). Stream: P1 0-59, P2 60-119 (covariate shift), P3 120-199 (new attacks B, A absent),
P4 200-279 (A returns).
"""
import json
from pathlib import Path

DATASETS = ["wustl_iiot", "xiiotid", "edge_iiotset", "forge_iiot", "mu_iot"]
GATE = {"policy": "net_bal", "threshold": 2, "max_recall_drop": 0.3, "max_detect_drop": 0.01,
        "max_cum_drop": 0.3, "soft_drop": 0.1}
TRIG = {"frac": 0.15, "feat_frac": 0.2, "offset": 3.0}
ATTACKS = {
    "clean": {},
    # naive attack: every attack label flipped at 80% while known attacks return
    "naive": {"poison": {"target": "all", "rate": 0.8, "windows": [[200, 280]]}},
    # strong burst on one known class in a stable phase, with and without forced adaptation
    "poisononly": {"poison": {"target": "A0", "rate": 1.0, "inject": 0.1, "windows": [[20, 60]]}},
    "hijack": {"poison": {"target": "A0", "rate": 1.0, "inject": 0.1, "windows": [[20, 60]]},
               "trigger": {**TRIG, "windows": [[20, 60]], "on": 4, "off": 8}},
    # many forced episodes with a low flip rate: each update loses little, damage accumulates
    "slow": {"poison": {"target": "A0", "rate": 0.5, "inject": 0.05, "windows": [[20, 60], [210, 280]]},
             "trigger": {**TRIG, "windows": [[20, 60], [210, 280]], "on": 3, "off": 9}},
    # gate-aware: only a 30% variant of the class, during genuine drift whose benefit masks the loss
    "variant": {"poison": {"target": "A0", "rate": 1.0, "variant_q": 0.3, "inject": 0.05,
                           "windows": [[60, 120], [200, 280]]}},
    # surgical zero-day poisoning of a new class (no injection, so known classes are untouched),
    # validators one phase behind, and the control without lag
    "zeroday": {"poison": {"target": "B0", "rate": 1.0, "windows": [[120, 280]]}, "lag": 1},
    "zerodaynolag": {"poison": {"target": "B0", "rate": 1.0, "windows": [[120, 280]]}, "lag": 0},
}

# gate-optimal variant: wiping out a share q of one class lowers the balanced detectability
# D = (spec + mean attack detectability)/2 by q/(2K), K = attack classes the validators hold, so a
# variant with q < 2 K tau_D passes every per-update check. Attack in P4 (K = |A| + |B|) at 0.9x
# the bound, and at 1.8x the bound (relies on the benefit of relearning A to mask the loss)
TAU_D = GATE["max_detect_drop"]
K_P4 = {"wustl_iiot": 2, "xiiotid": 3, "edge_iiotset": 8, "forge_iiot": 6, "mu_iot": 8}


def variant_k(d, f):
    return {"poison": {"target": "A0", "rate": 1.0, "variant_q": round(f * 2 * K_P4[d] * TAU_D, 4),
                       "inject": 0.02, "windows": [[200, 280]]}}


def salami(d):
    # e-value-gate-aware erosion: the gate-optimal variant poisoned at a low rate over long windows,
    # so each checkpoint loses less than the per-cell tolerance and the loss accumulates
    return {"poison": {"target": "A0", "rate": 0.15, "variant_q": round(0.9 * 2 * K_P4[d] * TAU_D, 4),
                       "inject": 0.005, "windows": [[20, 60], [200, 280]]}}


def auditmem(d):
    # compromised gateway: gate-optimal variant poisoning + memorization of published audit flows
    return {**variant_k(d, 0.9), "memorize": 0.5}


PER_DATASET = {"variantK": lambda d: variant_k(d, 0.9), "variantK2": lambda d: variant_k(d, 1.8),
               "auditmem": auditmem,
               "salami": salami,
               # a variant smaller than the class-level tolerance (2.5% < eps = 3%)
               "variantS": lambda d: {"poison": {"target": "A0", "rate": 1.0, "variant_q": 0.025,
                                                 "inject": 0.01, "windows": [[200, 280]]}}}

for a, fn in PER_DATASET.items():
    for d in DATASETS:
        cfg = {"extends": f"{d}.json", "name": f"atk_{a}_{d}",
               "scenario": {"validators": {"n": 3, "per_class": 300, "lag_phases": 0}},
               "learners": {"gate": GATE}, "attack": fn(d)}
        Path(f"configs/atk_{a}_{d}.json").write_text(json.dumps(cfg, indent=2))

for a, spec in ATTACKS.items():
    spec = dict(spec)
    lag = spec.pop("lag", 0)
    for d in DATASETS:
        cfg = {"extends": f"{d}.json", "name": f"atk_{a}_{d}",
               "scenario": {"validators": {"n": 3, "per_class": 300, "lag_phases": lag}},
               "learners": {"gate": GATE}}
        if spec:
            cfg["attack"] = spec
        Path(f"configs/atk_{a}_{d}.json").write_text(json.dumps(cfg, indent=2))
print((len(ATTACKS) + len(PER_DATASET)) * len(DATASETS), "configs")


# ----------------------------------------------------------------------------- Phase 2
# threshold gate (tg) vs the e-value gate (eg) and its ablations, same validators
# (600 clean flows per class, 3 private sub-clusters per class for eg)
EV = {"alpha": 0.05, "eps": 0.03, "eps_normal": 0.08, "n_max": 1000, "f": 1, "anchors": 4,
      "phi": 0.5, "coverage_after": 3, "spending": True, "highwater": True, "fixed_audit": False,
      "defer": True}
GATES = {
    "tg": ({"clusters": 0}, GATE),
    "eg": ({"clusters": 3}, {"policy": "evalue", "evalue": EV}),
    "egnocl": ({"clusters": 0}, {"policy": "evalue", "evalue": EV}),
    "egnohw": ({"clusters": 3}, {"policy": "evalue", "evalue": {**EV, "highwater": False}}),
    "egfixed": ({"clusters": 3}, {"policy": "evalue", "evalue": {**EV, "fixed_audit": True}}),
    "egnospend": ({"clusters": 3}, {"policy": "evalue", "evalue": {**EV, "spending": False}}),
    "egnodefer": ({"clusters": 3}, {"policy": "evalue", "evalue": {**EV, "defer": False}}),
    # competitor: validate-before-commit sequential gate (balanced accuracy, commit/reject/defer)
    "vbc": ({"clusters": 0}, {"policy": "vbc", "evalue": EV}),
    # audit-randomness study: each decision audits a third of every cell, fresh or fixed
    "egsub": ({"clusters": 3}, {"policy": "evalue", "evalue": {**EV, "audit_subset": 0.34}}),
    "egsubfixed": ({"clusters": 3}, {"policy": "evalue",
                                     "evalue": {**EV, "audit_subset": 0.34, "fixed_audit": True}}),
}
P2_ATTACKS = ["clean", "naive", "hijack", "slow", "variantK", "variantK2", "zeroday", "zerodaynolag",
              "variantS", "salami", "auditmem"]
n2 = 0
for a in P2_ATTACKS:
    for gname, (vext, gate) in GATES.items():
        for d in DATASETS:
            base = json.loads(Path(f"configs/atk_{a}_{d}.json").read_text())
            cfg = {"extends": f"{d}.json", "name": f"p2_{a}_{gname}_{d}",
                   "scenario": {"validators": {**base["scenario"]["validators"], "per_class": 600, **vext}},
                   "learners": {"gate": gate}}
            if "attack" in base:
                cfg["attack"] = base["attack"]
            Path(f"configs/p2_{a}_{gname}_{d}.json").write_text(json.dumps(cfg, indent=2))
            n2 += 1
print(n2, "phase-2 configs")
