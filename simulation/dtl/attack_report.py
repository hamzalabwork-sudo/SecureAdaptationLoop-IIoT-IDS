"""Phase-1 report: adaptive attacks on the adaptation loop vs. the threshold gate (TG).

  python -m dtl.attack_report --root results/attack1 --out results/attack1_report

Per (attack, dataset, method, seed):
  attack success   held-out detectability of the attacker's target (the variant for 'variant')
                   at the end of the stream and its minimum over phase ends; prequential
                   detectability of target flows inside the poisoning windows
  utility          lifetime macro-F1 and missed attacks
  gate decisions   judged against the ground-truth oracle (large held-out test sets, never seen
                   by the gate):
                     harmful   the candidate lowers the target's detectability by > 5 points or
                               the balanced detectability D by > 1 point
                     useful    the candidate raises D by > 0.5 point and harms no target
                   harmful-accept rate = accepted harmful / harmful candidates
                   useful-admission rate = accepted useful / useful candidates
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

EPS_T, EPS_D, EPS_U = 0.05, 0.01, 0.005


def classify(g, key):
    oc, op = g["oracle_cand"], g["oracle_parent"]
    dt = op.get(key, np.nan) - oc.get(key, np.nan)
    dD = op["D"] - oc["D"]
    harmful = (not np.isnan(dt) and dt > EPS_T) or dD > EPS_D
    useful = (not harmful) and (oc["D"] - op["D"] > EPS_U)
    return harmful, useful


def load(root):
    rows, decisions = [], []
    for f in glob.glob(f"{root}/atk_*/*/s*/phase_eval.jsonl"):
        p = Path(f)
        attack, ds = p.parts[-4][len("atk_"):].split("_", 1)
        cfg = json.loads((p.parent / "config_used.json").read_text())
        win = cfg.get("attack", {}).get("poison", {}).get("windows", [])
        key = "variant_detect" if attack.startswith("variant") else "target_detect"
        summ = pd.read_csv(p.parent / "summary_per_seed.csv").set_index("method")
        for line in open(f):
            r = json.loads(line)
            m, seed = r["method"], r["seed"]
            pe = [x.get(key, np.nan) for x in r["phase_eval"]]
            tl = pd.read_csv(p.parent / "timelines" / f"{m}_s{seed}.csv")
            inwin = np.zeros(len(tl), bool)
            for lo, hi in win:
                inwin |= (tl.batch >= lo) & (tl.batch < hi)
            row = {"attack": attack, "dataset": ds, "method": m, "seed": seed,
                   "target_end": pe[-1], "target_min": np.nanmin(pe) if np.isfinite(pe).any() else np.nan,
                   "target_window": tl.loc[inwin, "target_detect"].mean() if inwin.any() else np.nan,
                   "f1": tl.f1.mean(), "missed": tl.missed_attacks.sum(),
                   "alarms": len(r["alarm_info"]), "injected": r["attack"].get("n_injected", 0),
                   "flipped": r["attack"].get("n_flipped", 0)}
            gl = r["gate_log"]
            if gl:
                cls = [classify(g, key) for g in gl]
                acc = np.array([g["accepted"] for g in gl])
                harm = np.array([c[0] for c in cls])
                use = np.array([c[1] for c in cls])
                row.update({"proposals": len(gl), "accepted": int(acc.sum()),
                            "harmful": int(harm.sum()), "harmful_acc": int((harm & acc).sum()),
                            "useful": int(use.sum()), "useful_acc": int((use & acc).sum())})
                for g, (h, u) in zip(gl, cls):
                    decisions.append({"attack": attack, "dataset": ds, "seed": seed, "t": g["t"],
                                      "accepted": g["accepted"], "harmful": h, "useful": u})
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(decisions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", nargs="*", default=["results/attack1", "results/attack1b"])
    ap.add_argument("--out", default="results/attack1_report")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    parts = [load(r) for r in a.root]
    runs = pd.concat([p[0] for p in parts], ignore_index=True)
    dec = pd.concat([p[1] for p in parts], ignore_index=True)
    runs.to_csv(out / "all_runs.csv", index=False)
    dec.to_csv(out / "gate_decisions.csv", index=False)

    order = ["clean", "naive", "poisononly", "hijack", "slow", "variant", "variantK", "variantK2",
             "zeroday", "zerodaynolag"]
    meth = ["Static", "ER", "DTL", "DTL+Gate-dual"]
    lines = ["# Phase 1: adaptive attacks on the adaptation loop (MLP, 5 datasets x 5 seeds)", ""]
    for col, title in [("target_end", "Target detectability at the end of the stream (%)"),
                       ("target_min", "Worst target detectability over phase ends (%)"),
                       ("target_window", "Prequential target detectability inside the poisoning windows (%)"),
                       ("f1", "Lifetime macro-F1 (%)")]:
        t = (runs.groupby(["attack", "method"])[col].mean().unstack()
             .reindex(index=order, columns=meth) * 100).round(1)
        lines += [f"## {title}", "", t.to_markdown(), ""]
        t.to_csv(out / f"{col}.csv")
    per_ds = (runs[runs.method.isin(["DTL", "DTL+Gate-dual"])]
              .groupby(["attack", "dataset", "method"])["target_end"].mean().unstack() * 100).round(1)
    lines += ["## Target detectability at the end, per dataset (%)", "", per_ds.to_markdown(), ""]
    g = runs[runs.method == "DTL+Gate-dual"].groupby("attack")[
        ["proposals", "accepted", "harmful", "harmful_acc", "useful", "useful_acc"]].sum().reindex(order)
    g["accept_rate"] = (g.accepted / g.proposals * 100).round(1)
    g["harmful_accept_rate"] = (g.harmful_acc / g.harmful.replace(0, np.nan) * 100).round(1)
    g["useful_admission_rate"] = (g.useful_acc / g.useful.replace(0, np.nan) * 100).round(1)
    lines += ["## Gate decisions judged by the ground-truth oracle (sum over datasets and seeds)", "",
              g.to_markdown(), ""]
    g.to_csv(out / "gate_decisions_summary.csv")
    (out / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
