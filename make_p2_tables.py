"""Generates every results table of the paper from the experiment outputs (no number typed by hand).

  python make_p2_tables.py          (from the repository root) -> tables/*.tex
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE / "simulation" / "results"
TAB = HERE / "tables"
TAB.mkdir(parents=True, exist_ok=True)
DS = ["wustl_iiot", "xiiotid", "edge_iiotset", "forge_iiot", "mu_iot"]
DSL = {"wustl_iiot": "WUSTL", "xiiotid": "X-IIoTID", "edge_iiotset": "Edge-IIoTset",
       "forge_iiot": "ForgeIIOT", "mu_iot": "MU-IoT"}
ATK = {"poisononly": "Known-class burst", "hijack": "Trigger hijacking", "slow": "Slow poisoning",
       "variant": "Variant (30\\%)", "variantK": "Gate-optimal variant", "variantK2": "Masked variant",
       "variantS": "Small variant (2.5\\%)", "salami": "Salami erosion", "zeroday": "Denial of learning",
       "zerodaynolag": "Denial of learning (no lag)", "naive": "Naive label flipping",
       "clean": "Clean stream", "auditmem": "Audit memorization"}


def write(name, body):
    (TAB / f"{name}.tex").write_text(body + "\n")
    print("wrote", name)


def pct(x, d=1):
    return "--" if pd.isna(x) else f"{100 * x:.{d}f}"


def rate(num, den):
    return "--" if den == 0 else f"{100 * num / den:.1f}"


# ------------------------------------------------------------------ attack study: threshold gate
def t_phase1():
    r = pd.read_csv(R / "attack1_report" / "all_runs.csv")
    rows = []
    for a in ["poisononly", "hijack", "slow", "variant", "variantK", "variantK2", "zeroday"]:
        x = r[r.attack == a]
        g = lambda m, c: x[x.method == m][c].mean()
        gt = x[x.method == "DTL+Gate-dual"]
        rows.append(f"{ATK[a]} & {pct(g('ER', 'target_end'))} ({pct(g('ER', 'target_min'))}) & "
                    f"{pct(g('DTL', 'target_end'))} ({pct(g('DTL', 'target_min'))}) & "
                    f"{pct(g('DTL+Gate-dual', 'target_end'))} ({pct(g('DTL+Gate-dual', 'target_min'))}) & "
                    f"{rate(gt.harmful_acc.sum(), gt.harmful.sum())} & {int(gt.proposals.sum())} \\\\")
    clean = r[(r.attack == "clean") & (r.method == "DTL+Gate-dual")]
    rows.append(f"\\midrule {ATK['clean']} & -- & -- & -- & {rate(clean.harmful_acc.sum(), clean.harmful.sum())} & "
                f"{int(clean.proposals.sum())} \\\\")
    write("phase1", "\n".join(rows))


# ------------------------------------------------------------------ Phase 2: gates
def p2():
    r = pd.read_csv(R / "p2_report" / "all_runs.csv")
    return r.drop_duplicates(["attack", "gate", "dataset", "seed", "method"])


def t_main():
    r = p2()
    rows = []
    for a in ["clean", "naive", "hijack", "slow", "variantK", "variantK2", "variantS", "zeroday"]:
        cells = []
        for g in ["none", "tg", "vbc", "eg"]:
            x = r[(r.attack == a) & (r.gate == g)]
            cells.append(pct(x.target_end.mean()))
        for g in ["tg", "vbc", "eg"]:
            x = r[(r.attack == a) & (r.gate == g)]
            cells.append(rate(x.harmful_acc.sum(), x.harmful.sum()))
        for g in ["none", "tg", "vbc", "eg"]:
            x = r[(r.attack == a) & (r.gate == g)]
            cells.append(pct(x.f1.mean()))
        # bold the best (lowest) harmful-accept rate among gates
        h = [cells[4], cells[5], cells[6]]
        num = [float(v) if v != "--" else np.inf for v in h]
        if np.isfinite(min(num)):
            best = min(num)
            for i, v in enumerate(num):
                if v == best:
                    cells[4 + i] = f"\\textbf{{{h[i]}}}"
        rows.append(f"{ATK[a]} & " + " & ".join(cells) + " \\\\")
    write("main", "\n".join(rows))


def t_escalation():
    r = p2()
    rows = []
    for d in DS:
        cells = []
        for a in ["clean", "zeroday"]:
            for g in ["tg", "vbc", "eg"]:
                x = r[(r.attack == a) & (r.gate == g) & (r.dataset == d)]
                cells.append("--" if g != "eg" else f"{100 * (x.escalations > 0).mean():.0f}")
        x = r[(r.attack == "zeroday") & (r.dataset == d)]
        cells += [pct(x[x.gate == g].target_end.mean()) for g in ["none", "tg", "eg"]]
        rows.append(f"{DSL[d]} & " + " & ".join([cells[2], cells[5]] + cells[6:]) + " \\\\")
    write("escalation", "\n".join(rows))


def t_ablation():
    r = p2()
    spec = [("eg", "Full e-value gate"), ("egnocl", "w/o private sub-clusters"),
            ("egnohw", "w/o high-water reference"), ("egfixed", "w/o fresh audit randomness"),
            ("egnospend", "w/o $\\alpha$ spending"), ("egnodefer", "w/o deferral")]
    rows = []
    for g, lab in spec:
        cells = []
        for a in ["variantK", "zeroday", "slow"]:
            x = r[(r.attack == a) & (r.gate == g)]
            cells.append(rate(x.harmful_acc.sum(), x.harmful.sum()))
        x = r[(r.attack == "variantS") & (r.gate == g)]
        cells.append(rate(x.harmful_acc.sum(), x.harmful.sum()) if len(x) else "--")
        x = r[(r.attack == "clean") & (r.gate == g)]
        cells += [pct(x.f1.mean()), rate(x.useful_acc.sum(), x.useful.sum()), f"{x.audit.mean() / 1000:.1f}"]
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    write("ablation", "\n".join(rows))


def t_randomness():
    r = p2()
    rows = []
    for g, lab in [("tg", "TG (whole fixed set)"), ("egsubfixed", "EG, fixed audit subset"),
                   ("egsub", "EG, fresh audit subset")]:
        cells = []
        for a in ["variantK", "auditmem"]:
            x = r[(r.attack == a) & (r.gate == g)]
            cells += [pct(x.target_end.mean()), rate(x.harmful_acc.sum(), x.harmful.sum())]
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    write("randomness", "\n".join(rows))


def t_chain():
    s = json.loads((R / "chain_ev" / "summary.json").read_text())
    lat, gas = s["latency_s"], s["gas"]
    rows = []
    for k, lab in [("propose", "proposeUpdate (commit)"), ("seed_wait", "wait for audit seed (next block)"),
                   ("attest", "attest (per validator)"), ("coverage", "reportCoverage (per validator)")]:
        g = f"{gas[k]['mean'] / 1000:.0f}k" if k in gas else "--"
        rows.append(f"{lab} & {g} & {lat[k]['mean']:.2f} & {lat[k]['p95']:.2f} \\\\")
    rows.append(f"\\midrule Decision (commit, seed, 2 attestations) & -- & "
                f"{np.mean(list(s['decision_s'].values())):.2f} & -- \\\\")
    rows.append(f"Deployment ({len(json.loads((HERE / 'simulation' / 'chain' / 'ModelUpdateGateEV.json').read_text())['bin']) // 2 / 1000:.1f}\\,kB) & "
                f"{gas['deploy']['mean'] / 1e6:.2f}M & -- & -- \\\\")
    write("chain", "\n".join(rows))
    live = []
    for name, lab in [("variantK_xiiotid", "Gate-optimal variant, X-IIoTID"), ("zeroday_wustl_iiot", "Denial of learning, WUSTL")]:
        f = R / "p2live" / name / "phase_eval.jsonl"
        if not f.exists():
            continue
        r = json.loads(f.read_text().splitlines()[0])
        gl = r["gate_log"]
        key = "variant_detect" if name.startswith("variant") else "target_detect"
        live.append(f"{lab} & {len(gl)} & {sum(g['accepted'] for g in gl)} & {sum(g.get('deferred', False) for g in gl)} & "
                    f"{len(gl) - sum(g['accepted'] for g in gl) - sum(g.get('deferred', False) for g in gl)} & "
                    f"{len(r.get('coverage_log', []))} & {pct(r['phase_eval'][-1].get(key))} \\\\")
    write("live", "\n".join(live))


if __name__ == "__main__":
    t_phase1()
    t_main()
    t_escalation()
    t_ablation()
    t_randomness()
    t_chain()
