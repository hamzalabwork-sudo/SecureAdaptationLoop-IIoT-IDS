"""Figures of the paper (run from simulation/ after the grids).

  python make_p2_figures.py --out results/p2_figures

  fig_validity     e-process pass rate vs. true per-cell loss (Monte Carlo): validity at eps, power below
  fig_blindspot    threshold gate vs. variant share relative to the bound 2 K tau_D (Prop. 1)
  fig_tradeoff     harmful-accept vs. useful-admission rate per gate and attack (oracle)
  fig_variant_ts   variant detectability over the stream, X-IIoTID, gate-optimal variant, per gate
  fig_zeroday_ts   zero-day class detectability with the coverage escalation, WUSTL
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dtl.learners import eprocess_pass  # noqa: E402

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False})
COL = {"none": "#7f7f7f", "tg": "#d62728", "vbc": "#ff7f0e", "eg": "#2ca02c"}
LAB = {"none": "No gate", "tg": "Threshold gate (TG)", "vbc": "Validate-before-commit (VBC)", "eg": "E-value gate (EG, ours)"}


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig_validity(out, eps=0.03, n=1000, reps=4000, seed=0):
    rng = np.random.default_rng(seed)
    drops = np.linspace(0, 0.08, 17)
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for k, ls in [(1, "-"), (5, "--"), (20, ":")]:
        thr = 2 / (0.05 * 6 / (np.pi ** 2 * k ** 2))
        rate = []
        for d in drops:
            ok = 0
            for _ in range(reps // 4):
                base = rng.random(n) < 0.97
                cand = base & (rng.random(n) > d / 0.97)
                ok += eprocess_pass(base.astype(float) - cand.astype(float), eps, thr)[0]
            rate.append(ok / (reps // 4))
        ax.plot(drops * 100, rate, ls, color="#2ca02c", label=f"update k={k} (threshold {thr:.0f})")
        ax.axhline(1 / thr, color="#2ca02c", ls=ls, lw=0.5, alpha=0.6)
    ax.axvline(eps * 100, color="k", lw=0.6)
    ax.text(eps * 100 + 0.1, 0.55, r"$\varepsilon$", fontsize=8)
    ax.set_xlabel("true loss of the candidate in the cell (%)")
    ax.set_ylabel("P(cell passes)")
    ax.legend(frameon=False, fontsize=6)
    save(fig, out, "fig_validity")


def fig_blindspot(out, runs):
    # variant share relative to the bound 2 K tau_D: variantK = 0.9x, variantK2 = 1.8x; 'variant' = 30%
    K = {"wustl_iiot": 2, "xiiotid": 3, "edge_iiotset": 8, "forge_iiot": 6, "mu_iot": 8}
    rows = []
    for a, mult in [("variantK", 0.9), ("variantK2", 1.8)]:
        for g in ("none", "tg", "vbc", "eg"):
            x = runs[(runs.attack == a) & (runs.gate == g)]
            rows.append({"mult": mult, "gate": g, "det": x.target_end.mean() * 100,
                         "harm": (x.harmful_acc.sum() / max(x.harmful.sum(), 1) * 100) if g != "none" else np.nan})
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.8))
    w = 0.2
    for i, g in enumerate(("none", "tg", "vbc", "eg")):
        z = d[d.gate == g]
        ax[0].bar(np.arange(2) + (i - 1.5) * w, z.det, w, color=COL[g], label=LAB[g])
        if g != "none":
            ax[1].bar(np.arange(2) + (i - 2) * w, z.harm, w, color=COL[g])
    for a_, t in zip(ax, ("variant detected (%)", "harmful accepted (%)")):
        a_.set_xticks(range(2))
        a_.set_xticklabels([r"$0.9\times$", r"$1.8\times$"])
        a_.set_xlabel(r"variant share / $2K\tau_D$")
        a_.set_ylabel(t)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False,
               fontsize=6, bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    save(fig, out, "fig_blindspot")


def fig_tradeoff(out, runs):
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    mk = {"hijack": "^", "slow": "v", "variantK": "o", "variantK2": "s", "variantS": "D",
          "zeroday": "X", "clean": "P", "naive": "*"}
    for g in ("tg", "vbc", "eg"):
        for a, m in mk.items():
            x = runs[(runs.attack == a) & (runs.gate == g)]
            if not len(x) or x.harmful.sum() == 0:
                continue
            h = x.harmful_acc.sum() / x.harmful.sum() * 100
            u = x.useful_acc.sum() / max(x.useful.sum(), 1) * 100
            ax.scatter(u, h, marker=m, color=COL[g], s=22, edgecolor="k", lw=0.3)
    for g in ("tg", "vbc", "eg"):
        ax.scatter([], [], color=COL[g], label=LAB[g], s=20)
    for a, m in mk.items():
        ax.scatter([], [], marker=m, color="w", edgecolor="k", label=a, s=18)
    ax.set_xlabel("useful updates admitted (%)")
    ax.set_ylabel("harmful updates accepted (%)")
    ax.legend(frameon=False, fontsize=5.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    save(fig, out, "fig_tradeoff")


def _timeline(root, cfg, method, seed_glob="s*"):
    out = []
    for f in sorted(glob.glob(f"{root}/{cfg}/mlp/{seed_glob}/timelines/{method}_s*.csv")):
        out.append(pd.read_csv(f, usecols=["batch", "target_detect"]).set_index("batch").target_detect)
    return pd.concat(out, axis=1).mean(1) if out else None


def fig_variant_ts(out, ds="xiiotid"):
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    series = {"none": _timeline("results/p2main", f"p2_variantK_tg_{ds}", "DTL"),
              "tg": _timeline("results/p2main", f"p2_variantK_tg_{ds}", "DTL+Gate-dual"),
              "vbc": _timeline("results/p2vbc", f"p2_variantK_vbc_{ds}", "DTL+Gate-dual"),
              "eg": _timeline("results/p2eg", f"p2_variantK_eg_{ds}", "DTL+Gate-dual")}
    for g, s in series.items():
        if s is not None:
            s = s.loc[200:].rolling(5, min_periods=1).mean() * 100
            ax.plot(s.index, s.values, color=COL[g], lw=1.4 if g == "eg" else 0.9, label=LAB[g])
    ax.set_xlabel("batch (phase P4, variant poisoned)")
    ax.set_ylabel("variant flows detected (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=6, loc="lower left")
    save(fig, out, "fig_variant_ts")


def fig_zeroday_ts(out, ds="wustl_iiot"):
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    clean = _timeline("results/p2main", f"p2_clean_tg_{ds}", "DTL")
    pois = {"none": _timeline("results/p2main", f"p2_zeroday_tg_{ds}", "DTL"),
            "tg": _timeline("results/p2main", f"p2_zeroday_tg_{ds}", "DTL+Gate-dual"),
            "eg": _timeline("results/p2eg", f"p2_zeroday_eg_{ds}", "DTL+Gate-dual")}
    # zero-day class detectability is not logged on clean runs (no target); use the B-group recall
    if clean is None or clean.isna().all():
        fs = sorted(glob.glob(f"results/p2main/p2_clean_tg_{ds}/mlp/s*/timelines/DTL_s*.csv"))
        clean = pd.concat([pd.read_csv(f).set_index("batch").B_recall for f in fs], axis=1).mean(1)
    ax.plot(clean.loc[120:].rolling(5, min_periods=1).mean() * 100, color="k", lw=0.8, ls="--",
            label="DTL, clean labels")
    for g, s in pois.items():
        if s is not None:
            ax.plot(s.loc[120:].rolling(5, min_periods=1).mean() * 100, color=COL[g],
                    lw=1.4 if g == "eg" else 0.9, label=LAB[g] + ", poisoned")
    esc = []
    for f in glob.glob(f"results/p2eg/p2_zeroday_eg_{ds}/mlp/s*/phase_eval.jsonl"):
        for line in open(f):
            esc += [e["t"] for e in json.loads(line).get("coverage_log", [])]
    if esc:
        ax.axvline(np.median(esc), color=COL["eg"], ls=":", lw=1)
        ax.text(np.median(esc) + 2, 55, "on-chain coverage\nescalation", fontsize=6, color=COL["eg"])
    ax.text(160, 6, "all poisoned runs: 0%", fontsize=6, color="0.3")
    ax.set_xlabel("batch (new attack appears at 120)")
    ax.set_ylabel("new-attack flows detected (%)")
    ax.set_ylim(-3, 105)
    ax.legend(frameon=False, fontsize=5.5, loc="center left")
    save(fig, out, "fig_zeroday_ts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/p2_figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv("results/p2_report/all_runs.csv").drop_duplicates(
        ["attack", "gate", "dataset", "seed", "method"])
    fig_validity(out)
    fig_blindspot(out, runs)
    fig_tradeoff(out, runs)
    fig_variant_ts(out)
    fig_zeroday_ts(out)
    print("figures in", out)


if __name__ == "__main__":
    main()
