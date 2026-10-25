"""Compiles chain/<Contract>.sol into chain/<Contract>.json (abi, bin) with solc 0.8.20.

  python -m chain.compile ModelUpdateGateEV
"""
import json
import sys
from pathlib import Path

import solcx

HERE = Path(__file__).resolve().parent


def main(name):
    out = solcx.compile_files([str(HERE / f"{name}.sol")], output_values=["abi", "bin"],
                              solc_version="0.8.20", optimize=True, optimize_runs=200)
    art = next(v for k, v in out.items() if k.endswith(f":{name}"))
    (HERE / f"{name}.json").write_text(json.dumps({"abi": art["abi"], "bin": art["bin"]}))
    print(f"{name}: {len(art['bin']) // 2} bytes of bytecode")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ModelUpdateGateEV")
