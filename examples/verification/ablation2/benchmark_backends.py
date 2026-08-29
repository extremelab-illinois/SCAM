# SPDX-License-Identifier: MIT
"""Benchmark NumPy and experimental JAX kernels on SCAM ablation2.

Examples
--------
python3 examples/verification/ablation2/benchmark_backends.py --backend numpy
python3 examples/verification/ablation2/benchmark_backends.py --backend jax --repeat 3
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import time

CASES = {
    "base": ("compare_pato_ablation2", ("table",), {}),
    "chemistry-off": ("compare_pato_ablation2_chemistryOff", (), {}),
    "multi-mat": ("compare_pato_ablation2_multiMat", (), {}),
    "element-transport": (
        "compare_pato_ablation2_equilibriumElementConservation",
        (),
        {"mode": "equilibrium", "verbose": False},
    ),
}


def _run_once(case_name: str, backend: str):
    module_name, run_args, run_kwargs = CASES[case_name]
    case = importlib.import_module(module_name)
    original_run = case.run

    def quiet_run(*args, **kwargs):
        options = args[6]
        options.array_backend = backend
        kwargs["verbose"] = False
        return original_run(*args, **kwargs)

    case.run = quiet_run
    try:
        start = time.perf_counter()
        results = case.run_scam(*run_args, **run_kwargs)
        elapsed = time.perf_counter() - start
    finally:
        case.run = original_run

    times = results.times_array()
    return {
        "case": case_name,
        "seconds": elapsed,
        "outputs": int(times.size),
        "final_time": float(times[-1]),
        "final_T_wall": float(results.T_wall_array()[-1]),
        "final_recession": float(results.s_array()[-1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(CASES), default="base")
    parser.add_argument("--backend", choices=("numpy", "jax"), required=True)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    runs = []
    for index in range(args.repeat):
        result = _run_once(args.case, args.backend)
        result["run"] = index + 1
        runs.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    summary = {
        "case": args.case,
        "backend": args.backend,
        "repeat": args.repeat,
        "median_seconds": statistics.median(run["seconds"] for run in runs),
        "runs": runs,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
