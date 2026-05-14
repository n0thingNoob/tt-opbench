"""CPU-only matmul experiment for TT-OpBench v0.1."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_M = 128
DEFAULT_N = 128
DEFAULT_K = 128
DEFAULT_SEED = 0
DEFAULT_WARMUP = 3
DEFAULT_REPEAT = 10
DEFAULT_DTYPE = "float32"
DEFAULT_RTOL = 1e-5
DEFAULT_ATOL = 1e-6


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the TT-OpBench v0.1 CPU-only matmul experiment."
    )
    parser.add_argument("--m", type=positive_int, default=DEFAULT_M)
    parser.add_argument("--n", type=positive_int, default=DEFAULT_N)
    parser.add_argument("--k", type=positive_int, default=DEFAULT_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--warmup", type=non_negative_int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeat", type=positive_int, default=DEFAULT_REPEAT)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args(argv)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def make_inputs(m: int, n: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((m, k), dtype=np.float32)
    b = rng.standard_normal((k, n), dtype=np.float32)
    return a, b


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b


def measure_ms(
    fn: Any, warmup: int, repeat: int
) -> tuple[Any, list[float], dict[str, float]]:
    result = None
    for _ in range(warmup):
        result = fn()

    run_ms = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        end = time.perf_counter()
        run_ms.append((end - start) * 1000.0)

    summary = {
        "mean_ms": float(np.mean(run_ms)),
        "min_ms": float(np.min(run_ms)),
        "max_ms": float(np.max(run_ms)),
    }
    return result, run_ms, summary


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    a, b = make_inputs(args.m, args.n, args.k, args.seed)

    baseline_output, baseline_runs, baseline_summary = measure_ms(
        lambda: matmul(a, b), args.warmup, args.repeat
    )
    variant_output, variant_runs, variant_summary = measure_ms(
        lambda: matmul(a, b), args.warmup, args.repeat
    )

    max_abs_error = float(np.max(np.abs(baseline_output - variant_output)))
    passed = bool(
        np.allclose(
            baseline_output,
            variant_output,
            rtol=DEFAULT_RTOL,
            atol=DEFAULT_ATOL,
        )
    )

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case": {
            "name": "matmul",
            "m": args.m,
            "n": args.n,
            "k": args.k,
            "dtype": DEFAULT_DTYPE,
            "seed": args.seed,
        },
        "baseline": {
            "name": "cpu_numpy_baseline",
            "runtime": "numpy",
            "operation": "a @ b",
        },
        "variant": {
            "name": "cpu_numpy_variant",
            "runtime": "numpy",
            "operation": "a @ b",
        },
        "runtime": {
            "kind": "cpu",
            "library": "numpy",
            "numpy_version": np.__version__,
        },
        "timing": {
            "protocol": {
                "timer": "time.perf_counter",
                "warmup": args.warmup,
                "repeat": args.repeat,
                "unit": "ms",
            },
            "baseline": {
                "runs_ms": baseline_runs,
                **baseline_summary,
            },
            "variant": {
                "runs_ms": variant_runs,
                **variant_summary,
            },
        },
        "correctness": {
            "passed": passed,
            "max_abs_error": max_abs_error,
            "rtol": DEFAULT_RTOL,
            "atol": DEFAULT_ATOL,
        },
        "config": {
            "output_dir": str(args.output_dir),
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
    }


def write_result(result: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    case = result["case"]
    filename = (
        f"{timestamp}_matmul_m{case['m']}_n{case['n']}_k{case['k']}"
        f"_seed{case['seed']}.json"
    )
    output_path = output_dir / filename
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output_path


def print_summary(result: dict[str, Any], output_path: Path) -> None:
    case = result["case"]
    correctness = result["correctness"]
    timing = result["timing"]
    baseline = timing["baseline"]
    variant = timing["variant"]
    speedup = baseline["mean_ms"] / variant["mean_ms"]

    print("TT-OpBench result")
    print("-----------------")
    print(f"case:        {case['name']} m={case['m']} n={case['n']} k={case['k']} dtype={case['dtype']}")
    print(f"baseline:    {result['baseline']['name']} ({result['baseline']['runtime']})")
    print(f"variant:     {result['variant']['name']} ({result['variant']['runtime']})")
    print(f"correctness: {'pass' if correctness['passed'] else 'fail'}")
    print(f"max error:   {correctness['max_abs_error']:.6g}")
    print(
        f"timing:      baseline={baseline['mean_ms']:.6f} ms, "
        f"variant={variant['mean_ms']:.6f} ms, speedup={speedup:.3f}x"
    )
    print(f"protocol:    warmup={timing['protocol']['warmup']} repeat={timing['protocol']['repeat']}")
    print(f"result:      {output_path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_result(args)
    output_path = write_result(result, args.output_dir)
    print_summary(result, output_path)
    return 0 if result["correctness"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
