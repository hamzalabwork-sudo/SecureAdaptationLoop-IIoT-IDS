"""Phase-2 report: threshold gate (tg) vs the e-value gate (eg) and its ablations.

  python -m dtl.p2_report --root results/p2main results/p2abl --out results/p2_report

Per (attack, gate, dataset, seed) for DTL+Gate-dual (and DTL without a gate as reference):
target detectability (end, worst phase end), lifetime macro-F1, gate decisions judged by the
ground-truth oracle (harmful-accept and useful-admission rates, as in attack_report), coverage
escalations (on the attacker's target class / any class) and audit samples per decision.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .attack_report import classify

GATE_ORDER = ["tg", "vbc", "eg", "egnocl", "egnohw", "egfixed", "egnospend", "egnodefer", "egsub",
              "egsubfixed"]
ATTACK_ORDER = ["clean", "naive", "hijack", "slow", "variantK", "variantK2", "variantS", "salami", "auditmem",
                "zeroday", "zerodaynolag"]


def load(roots):
    rows = []
    for root in roots:
        for f in glob.glob(f"{root}/p2_*/*/s*/phase_eval.jsonl"):
            p = Path(f)
            _, attack, gate, ds = p.parts[-4].split("_", 3)
            key = "variant_detect" if attack.startswith(("variant", "salami", "auditmem")) else "target_detect"
            meta = json.loads((p.parent / "meta.json").read_text())
            cfg = json.loads((p.parent / "config_used.json").read_text())
            tgt = cfg.get("attack", {}).get("poison", {}).get("target")
            tgt_idx = (meta["classes"].index(cfg["scenario"]["groups"][tgt[0]][int(tgt[1:])])
                       if tgt and tgt != "all" else None)
            for line in open(f):
                r = json.loads(line)
                m = r["method"]
                pe = [x.get(key, np.nan) for x in r["phase_eval"]]
                tl = pd.read_csv(p.parent / "timelines" / f"{m}_s{r['seed']}.csv")
                row = {"attack": attack, "gate": gate if m != "DTL" else "none", "dataset": ds,
                       "seed": r["seed"], "method": m, "target_end": pe[-1],
                       "target_min": np.nanmin(pe) if np.isfinite(pe).any() else np.nan,
                       "f1": tl.f1.mean(), "fpr": tl.fpr.mean(), "missed": tl.missed_attacks.sum()}
                gl, cov = r["gate_log"], r.get("coverage_log", [])
                if gl:
                    cls = [classify(g, key) for g in gl]
                    acc = np.array([g["accepted"] for g in gl])
                    harm = np.array([c[0] for c in cls])
                    use = np.array([c[1] for c in cls])
                    row.update({"proposals": len(gl), "accepted": int(acc.sum()),
                                "harmful": int(harm.sum()), "harmful_acc": int((harm & acc).sum()),
                                "useful": int(use.sum()), "useful_acc": int((use & acc).sum()),
                                "audit": np.mean([g.get("audit_samples", np.nan) for g in gl])})
                if m != "DTL":
                    row["escalations"] = len(cov)
                    row["escalated_target"] = int(tgt_idx is not None and any(e["class"] == tgt_idx for e in cov))
                    row["first_escalation_t"] = min((e["t"] for e in cov), default=np.nan)
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", nargs="*", default=["results/p2main", "results/p2eg", "results/p2abl", "results/p2targeted", "results/p2vbc", "results/p2rand"])
    ap.add_argument("--out", default="results/p2_report")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = load(a.root)
    runs.to_csv(out / "all_runs.csv", index=False)
    # the no-gate reference appears in both tg and eg configs (identical streams): keep one copy
    runs = runs.drop_duplicates(["attack", "gate", "dataset", "seed", "method"])
    gates = [g for g in ["none"] + GATE_ORDER if g in set(runs.gate)]
    lines = ["# Phase 2: threshold gate (tg) vs e-value gate (eg) (MLP, 5 datasets x 5 seeds)", ""]
    for col, title in [("target_end", "Attacker's target detectability at the end (%)"),
                       ("target_min", "Worst target detectability over phase ends (%)"),
                       ("f1", "Lifetime macro-F1 (%)"), ("fpr", "Mean FPR (%)")]:
        t = (runs.groupby(["attack", "gate"])[col].mean().unstack()
             .reindex(index=ATTACK_ORDER, columns=gates) * 100).round(1)
        lines += [f"## {title}", "", t.to_markdown(), ""]
        t.to_csv(out / f"{col}.csv")
    g = runs[runs.gate != "none"].groupby(["attack", "gate"])[
        ["proposals", "accepted", "harmful", "harmful_acc", "useful", "useful_acc"]].sum()
    g["harmful_accept_rate"] = (g.harmful_acc / g.harmful.replace(0, np.nan) * 100).round(1)
    g["useful_admission_rate"] = (g.useful_acc / g.useful.replace(0, np.nan) * 100).round(1)
    for col, title in [("harmful_accept_rate", "Harmful updates accepted (% of harmful candidates, oracle)"),
                       ("useful_admission_rate", "Useful updates admitted (% of useful candidates, oracle)")]:
        t = g[col].unstack().reindex(index=ATTACK_ORDER, columns=[x for x in GATE_ORDER if x in gates])
        lines += [f"## {title}", "", t.to_markdown(), ""]
        t.to_csv(out / f"{col}.csv")
    e = runs[runs.gate != "none"].groupby(["attack", "gate"]).agg(
        escalated_target=("escalated_target", "mean"), runs_with_escalation=("escalations", lambda x: (x > 0).mean()),
        audit=("audit", "mean"))
    e[["escalated_target", "runs_with_escalation"]] *= 100
    lines += ["## Coverage escalations (% of runs) and audit samples per decision", "",
              e.round(1).to_markdown(), ""]
    per_ds = (runs[runs.attack.isin(["variantK", "variantS", "salami", "zeroday"])].groupby(["attack", "dataset", "gate"])
              .target_end.mean().unstack().reindex(columns=gates) * 100).round(1)
    lines += ["## Target detectability at the end, per dataset (%)", "", per_ds.to_markdown(), ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
