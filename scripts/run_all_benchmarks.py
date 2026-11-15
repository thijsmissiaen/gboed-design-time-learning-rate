"""Run all three full GPU benchmarks and save unified machine-readable results."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
for _p in (_SRC_DIR, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_location_benchmark import run_benchmark as run_location
from run_nonlinear_benchmark import run_benchmark as run_nonlinear
from run_regression_benchmark import run_benchmark as run_regression


def main() -> None:
    t_all = time.time()
    out_path = Path("results/benchmark_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== Running Benchmark 1: Location Finding (d=4, K=2, T=30, R=90, M=1024) ===")
    loc_res = run_location(R=90, T=30, M=1024, seed=1000)

    print("\n=== Running Benchmark 2: Bayesian Linear Regression (T=10, R=270) ===")
    reg_res = run_regression(R=270, T=10, seed=2000)

    print("\n=== Running Benchmark 3: Well-Specified-Prior Nonlinear Design (T=20, R=90, M=1024) ===")
    nonlin_res = run_nonlinear(R=90, T=20, M=1024, seed=3000)

    payload = {
        "total_elapsed_sec": float(time.time() - t_all),
        "benchmarks": {
            "location_finding": loc_res,
            "linear_regression": reg_res,
            "nonlinear_radial_sinusoidal": nonlin_res,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nAll benchmarks completed in {payload['total_elapsed_sec']:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
