"""Generic experiment grid: one job per (config, backbone, seed), spread over devices, resumable.

  python run_grid.py stage2 --archs mlp --devices cuda:0 cuda:1 cuda:2 --jobs 12
  python run_grid.py gate   --devices cuda:0 cuda:1 cuda:2 --jobs 12
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATASETS = ["wustl_iiot", "xiiotid", "edge_iiotset", "forge_iiot", "mu_iot"]
ATTACKS = ["clean", "naive", "poisononly", "hijack", "slow", "variant", "zeroday", "zerodaynolag"]
P2_ATTACKS = ["clean", "naive", "hijack", "slow", "variantK", "variantK2", "zeroday", "zerodaynolag",
              "variantS", "salami"]
GRIDS = {
    # Phase 1: adaptive attacks on the adaptation loop, against the threshold gate (TG)
    "attack1": {"configs": [f"atk_{a}_{d}" for a in ATTACKS for d in DATASETS],
                "methods": ["Static", "ER", "DTL", "DTL+Gate-dual"]},
    # gate-optimal variant attacks (q just below / twice the blind-spot bound 2 K tau_D)
    "attack1b": {"configs": [f"atk_{a}_{d}" for a in ("variantK", "variantK2") for d in DATASETS],
                 "methods": ["Static", "ER", "DTL", "DTL+Gate-dual"]},
    # Phase 2: threshold gate vs the e-value gate, every attack (DTL = no-gate reference)
    "p2main": {"configs": [f"p2_{a}_{g}_{d}" for a in P2_ATTACKS for g in ("tg", "eg") for d in DATASETS],
               "methods": ["DTL", "DTL+Gate-dual"]},
    # e-value gate with deferral (re-run after adding the defer outcome)
    "p2eg": {"configs": [f"p2_{a}_eg_{d}" for a in P2_ATTACKS for d in DATASETS],
             "methods": ["DTL+Gate-dual"]},
    # one targeted attack per gate component: sub-clusters (variantS), high-water mark (salami)
    "p2targeted": {"configs": [f"p2_{a}_{g}_{d}" for a, gs in (("variantS", ("tg", "eg", "egnocl")),
                                                              ("salami", ("tg", "eg", "egnohw")))
                               for g in gs for d in DATASETS],
                   "methods": ["DTL", "DTL+Gate-dual"]},
    # audit randomness: compromised gateway memorizing published audit flows, fresh vs fixed audits
    "p2rand": {"configs": [f"p2_{a}_{g}_{d}" for a in ("auditmem", "variantK", "clean")
                           for g in ("tg", "egsub", "egsubfixed") for d in DATASETS],
               "methods": ["DTL", "DTL+Gate-dual"]},
    # competitor baseline: validate-before-commit gate on every attack
    "p2vbc": {"configs": [f"p2_{a}_vbc_{d}" for a in P2_ATTACKS for d in DATASETS],
              "methods": ["DTL+Gate-dual"]},
    # Phase 2 ablations of the e-value gate
    "p2abl": {"configs": [f"p2_{a}_{g}_{d}" for a in ("clean", "slow", "variantK", "zeroday")
                          for g in ("egnocl", "egnohw", "egfixed", "egnospend", "egnodefer")
                          for d in DATASETS],
              "methods": ["DTL+Gate-dual"]},
}


def run(job, devices, root, methods):
    i, (cfg, arch, seed) = job
    out = root / cfg / arch / f"s{seed}"
    if (out / "summary_per_seed.csv").exists():
        return 0
    log = root / "logs" / f"{cfg}_{arch}_s{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with open(log, "w") as f:
        rc = subprocess.call([sys.executable, "run_experiment.py", "--config", f"configs/{cfg}.json",
                              "--arch", arch, "--seed-list", str(seed), "--methods", *methods,
                              "--device", devices[i % len(devices)], "--out", str(out)],
                             stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUTF8": "1"})
    print(f"[{time.strftime('%H:%M:%S')}] {cfg:22s} {arch:12s} s{seed} rc={rc} {time.time() - t:.0f}s",
          flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid", choices=GRIDS)
    ap.add_argument("--archs", nargs="*", default=["mlp"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--devices", nargs="*", default=["cpu"])
    args = ap.parse_args()
    g = GRIDS[args.grid]
    root = Path("results") / args.grid
    jobs = [(c, a, s) for c in g["configs"] for a in args.archs for s in range(args.seeds)]
    with ThreadPoolExecutor(args.jobs) as ex:
        rcs = list(ex.map(lambda j: run(j, args.devices, root, g["methods"]), enumerate(jobs)))
    print(f"failed jobs: {sum(r != 0 for r in rcs)}")


if __name__ == "__main__":
    main()
