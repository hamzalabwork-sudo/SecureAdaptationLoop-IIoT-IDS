"""Live test of ModelUpdateGateEV on PureChain: every outcome path, both protections, gas and latency.

  python -m chain.ev_live_test --out results/chain_ev
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from chain.gate import ACCEPTED, DEFERRED, REJECTED, PureChainEGate


def h(s):
    return hashlib.sha256(s.encode()).digest()


def expect_revert(fn, label):
    try:
        fn()
    except Exception as e:  # the node returns the revert reason in the error
        return f"{label}: reverted ({str(e)[:80]})"
    return f"{label}: NOT reverted"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/chain_ev")
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    g = PureChainEGate(n=3, f=1, alpha=0.05, deploy_new=True)
    print("deployed", g.address)
    g._tx(g.owner, g.c.functions.registerGateway(g.gateway.address, h("genesis")), "register")
    g.active = h("genesis")
    log, checks = [], []

    def cycle(tag, verdicts, expect):
        t0 = time.perf_counter()
        uid = g.propose(h(f"{tag}-{len(g.updates)}"), 0, 0.5)
        seed = g.audit_seed(uid)
        thr = g.e_threshold(uid)
        status = None
        for v, verdict in enumerate(verdicts):
            e = thr * 1.01 if verdict == "pass" else thr * 0.1
            status = g.attest(v, uid, verdict, e)
            if status in (ACCEPTED, REJECTED, DEFERRED):
                break
        dt = time.perf_counter() - t0
        log.append({"tag": tag, "uid": uid, "k": g.updates[uid]["k"], "seed": seed, "threshold": thr,
                    "status": int(status), "expected": expect, "decision_s": dt})
        print(f"{tag:8s} k={g.updates[uid]['k']} thr={thr:9.1f} status={status} (expected {expect}) {dt:.2f}s")
        return uid

    for r in range(a.rounds):
        cycle("accept", ["pass", "pass"], ACCEPTED)
        cycle("reject", ["harm", "harm"], REJECTED)
        cycle("defer", ["unsure", "unsure"], DEFERRED)

    # protection 1: attesting before the audit seed exists must revert
    uid = g.propose(h("early"), 0, 0.5)
    u = g.updates[uid]
    checks.append(expect_revert(lambda: g.c.functions.attest(u["id"], 0, 2 ** 200).call(
        {"from": g.validators[0].address, "block_identifier": u["commit"]}), "attest before seed"))
    # protection 2: a Pass whose e-value is below the threshold must revert
    g.audit_seed(uid)
    low = int(g.e_threshold(uid) * 1000 * 0.5)
    checks.append(expect_revert(lambda: g.c.functions.attest(u["id"], 0, low).call(
        {"from": g.validators[0].address}), "pass below threshold"))
    g.attest(0, uid, "pass", g.e_threshold(uid) * 1.01)
    g.attest(1, uid, "pass", g.e_threshold(uid) * 1.01)
    # coverage escalation needs f + 1 validator reports
    g.escalate(3, 230)
    rc = g.latency
    print("\n".join(checks))

    lat = {}
    for label, dt in g.latency:
        lat.setdefault(label, []).append(dt)
    gas = {}
    for label, gu in g.gas:
        gas.setdefault(label, []).append(gu)
    summ = {"address": g.address,
            "latency_s": {k: {"mean": float(np.mean(v)), "p95": float(np.percentile(v, 95)), "n": len(v)}
                          for k, v in lat.items()},
            "gas": {k: {"mean": float(np.mean(v)), "n": len(v)} for k, v in gas.items()},
            "decision_s": {t: float(np.mean([x["decision_s"] for x in log if x["tag"] == t]))
                           for t in ("accept", "reject", "defer")},
            "checks": checks, "cycles": log}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps({k: summ[k] for k in ("latency_s", "gas", "decision_s")}, indent=1))


if __name__ == "__main__":
    main()
