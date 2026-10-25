# Attacking and Provably Securing the Adaptation Loop of Self-Updating IIoT Intrusion Detection

Code, smart contract, configurations and per-run results for the paper

> H. Ibrahim, L. A. C. Ahakonye, J.-M. Lee, D.-S. Kim, "Attacking and Provably Securing the Adaptation Loop of Self-Updating IIoT Intrusion Detection," submitted to *IEEE Internet of Things Journal*.

A self-updating IDS validates every model update before it may serve traffic. This repository contains:
- **the attacks** against such a gate: trigger hijacking, the gate-optimal variant, checkpoint erosion and denial of learning;
- **the e-value gate**, which fixes them. Per-cell anytime-valid tests run against a high-water reference. Acceptance uses a Byzantine-robust threshold (n − f)/α_k with α-spending and three outcomes: accept, defer or reject. A coverage obligation turns denial of learning into an on-chain escalation.
- **`ModelUpdateGateEV.sol`**, which enforces the protocol on the PureChain blockchain. The gateway must commit its candidate before the audit randomness exists, and the contract checks every e-value against the threshold.

## Repository layout

```
simulation/
  dtl/                      reference self-updating IDS and experiment machinery
    data.py, natural.py       datasets, injected-drift stream, validators' private cells (class x sub-cluster)
    drift.py                  hybrid CUSUM drift detector
    learners.py               drift-triggered continual learner (DTL) and baselines; gates:
                                threshold gate (TG), validate-before-commit (VBC), e-value gate (EG)
                                (eprocess_pass, _gate_check_ev, _coverage_audit, _gate_check_vbc)
    adversary.py              adaptive adversary: label flipping, mislabelled injection, forged drift,
                                secret variants, compromised gateway with audit memorization
    runner.py, metrics.py     prequential loop, ground-truth oracle for every gate decision
    attack_report.py          report of the attack study (TG)
    p2_report.py              report of the gate comparison and ablations
  chain/
    ModelUpdateGateEV.sol     e-value gate contract (+ compiled ABI/bytecode .json)
    ModelUpdateGate.sol       threshold-gate contract (baseline)
    gate.py                   local and live (PureChain) backends of both gates
    ev_live_test.py           live test of every outcome path, both protections, gas and latency
    compile.py                solc 0.8.20 compilation
    deployment_ev.json        address of the deployed e-value gate contract (public data only)
  configs/                  atk_* (attack study), p2_* (gate comparison), p2live_* (live streams)
  make_attack_configs.py    generates all attack and gate configurations
  run_experiment.py         runs one configuration
  run_grid.py               runs a full grid (resumable, multi-GPU)
  make_p2_figures.py        figures of the paper
  results/                  per-run results and reports used in the paper (see below)
make_p2_tables.py           regenerates every table of the paper into tables/
```

## Setup

Python 3.10 or later:

```
pip install -r requirements.txt
```

## Datasets

The datasets are not redistributed here. Place them in the **repository root**, at the paths used by `simulation/configs/*.json`:

| Dataset | Expected path | Source |
|---|---|---|
| WUSTL-IIoT-2021 | `wustl_iiot_2021.csv` | IEEE DataPort, doi:10.21227/yftq-n229 |
| X-IIoTID | `X-IIoTID/X-IIoTID dataset.csv` | IEEE IoT J. 2022, doi:10.1109/JIOT.2021.3102056 |
| Edge-IIoTset | `Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv` | IEEE Access 2022, doi:10.1109/ACCESS.2022.3165809 |
| ForgeIIOT (IoTForge Pro) | `Forge_IIOT_Pro.csv` | IEEE IoT J., doi:10.1109/JIOT.2024.3501017 |
| MU-IoT | `MU-IoT.csv` | IEEE Access 2024, doi:10.1109/ACCESS.2024.3494052 |

## Reproducing the paper

Run the grids and reports from `simulation/`, and the tables from the repository root:

```
python make_attack_configs.py                       # (re)generate all configurations

python run_grid.py attack1  --devices cuda:0 cuda:1 # attack study against TG (Table II)
python run_grid.py attack1b --devices cuda:0 cuda:1 # gate-optimal variants (Proposition 1)
python run_grid.py p2main   --devices cuda:0 cuda:1 # TG and the no-gate reference, every attack
python run_grid.py p2eg     --devices cuda:0 cuda:1 # e-value gate, every attack
python run_grid.py p2vbc    --devices cuda:0 cuda:1 # validate-before-commit baseline
python run_grid.py p2abl    --devices cuda:0 cuda:1 # ablations of the e-value gate
python run_grid.py p2targeted --devices cuda:0 cuda:1  # small variant and salami attacks
python run_grid.py p2rand   --devices cuda:0 cuda:1 # compromised gateway, fixed vs fresh audits

python -m dtl.attack_report                         # -> results/attack1_report
python -m dtl.p2_report                             # -> results/p2_report
python make_p2_figures.py                           # -> results/p2_figures
cd .. && python make_p2_tables.py                   # -> tables/*.tex
```

Every gate decision is also scored by a ground-truth oracle on large held-out test sets that the gate never sees. A candidate is *harmful* if the attacker's target loses more than 5 points of detectability, or if balanced detectability drops by more than 1 point. It is *useful* if balanced detectability rises by more than 0.5 point without harm.

### Included results

For every run (`results/<grid>/<config>/mlp/s<seed>/`), the repository includes:
- `summary_per_seed.csv`: metrics per method;
- `phase_eval.jsonl`: held-out evaluation per phase, every gate decision with its oracle verdict, coverage escalations and chain latency;
- `meta.json` and `config_used.json`.

It also includes the aggregated reports, the figures, the live-chain results (`chain_ev`, `p2live`) and the per-batch timelines used by the figures. From these files, `make_p2_tables.py` regenerates every table without rerunning any experiment.

## PureChain (live blockchain)

The grids use a local implementation of the contract with the same decision logic. The live experiments use `ModelUpdateGateEV` on PureChain:
- RPC: `https://purechainnode.com`
- Chain ID: 900520900520
- Consensus: PoA²
- Gas price: 0

Test accounts are generated on first use in `simulation/.secrets/`. This folder is ignored by git; never commit it.

```
python -m chain.compile ModelUpdateGateEV           # optional, the compiled .json is included
python -m chain.ev_live_test --out results/chain_ev
python run_experiment.py --config configs/p2live_variantK_xiiotid.json --seed-list 0 --methods DTL+Gate-dual
```

## License

MIT. See `LICENSE`. If you use this code, please cite the paper above.
