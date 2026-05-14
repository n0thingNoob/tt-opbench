"""TT-NN builtin matmul experiment for TT-OpBench v0.2-min."""

from __future__ import annotations

import argparse
import json
import os
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
DEFAULT_DEVICE_ID = 0
DEFAULT_RTOL = 1e-1
DEFAULT_ATOL = 5e-1


def positive_multiple_of_32(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    if parsed % 32 != 0:
        raise argparse.ArgumentTypeError("value must be a multiple of 32 for this TT-NN v0.2-min path")
    return parsed


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the TT-OpBench v0.2-min TT-NN builtin matmul experiment."
    )
    parser.add_argument("--m", type=positive_multiple_of_32, default=DEFAULT_M)
    parser.add_argument("--n", type=positive_multiple_of_32, default=DEFAULT_N)
    parser.add_argument("--k", type=positive_multiple_of_32, default=DEFAULT_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--warmup", type=non_negative_int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeat", type=positive_int, default=DEFAULT_REPEAT)
    parser.add_argument("--device-id", type=non_negative_int, default=DEFAULT_DEVICE_ID)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args(argv)


def import_ttnn_and_torch() -> tuple[Any, Any]:
    os.environ.setdefault(
        "TTNN_CONFIG_OVERRIDES",
        json.dumps({"root_report_path": "/tmp/tt-opbench-ttnn-generated"}),
    )
    try:
        import torch
        import ttnn
    except ModuleNotFoundError as exc:
        missing = exc.name or "ttnn/torch"
        raise SystemExit(
            f"Missing {missing}. Run with the TT-Metal Python environment, for example: "
            "/home/yijia/tt-metal/python_env/bin/python -m tt_opbench.ttnn_matmul"
        ) from exc
    return ttnn, torch


def make_inputs(torch: Any, m: int, n: int, k: int, seed: int) -> tuple[Any, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    a = torch.randn((m, k), dtype=torch.float32, generator=generator)
    b = torch.randn((k, n), dtype=torch.float32, generator=generator)
    return a, b


def measure_ms(fn: Any, sync: Any, warmup: int, repeat: int) -> tuple[Any, list[float], dict[str, float]]:
    result = None
    for _ in range(warmup):
        result = fn()
        sync()

    run_ms = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        sync()
        end = time.perf_counter()
        run_ms.append((end - start) * 1000.0)

    summary = {
        "mean_ms": float(np.mean(run_ms)),
        "min_ms": float(np.min(run_ms)),
        "max_ms": float(np.max(run_ms)),
    }
    return result, run_ms, summary


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    ttnn_work_dir = Path("/tmp/tt-opbench-ttnn-work")
    ttnn_work_dir.mkdir(parents=True, exist_ok=True)
    original_cwd = Path.cwd()
    os.chdir(ttnn_work_dir)
    try:
        ttnn, torch = import_ttnn_and_torch()
    finally:
        os.chdir(original_cwd)

    torch_a, torch_b = make_inputs(torch, args.m, args.n, args.k, args.seed)
    baseline_output = torch_a @ torch_b

    os.chdir(ttnn_work_dir)
    device = ttnn.open_device(device_id=args.device_id)
    try:
        tt_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        def run_variant() -> Any:
            return ttnn.matmul(tt_a, tt_b)

        def sync() -> None:
            ttnn.synchronize_device(device)

        tt_output, variant_runs, variant_summary = measure_ms(
            run_variant, sync, args.warmup, args.repeat
        )
        torch_output = ttnn.to_torch(tt_output).to(torch.float32)
    finally:
        ttnn.close_device(device)
        os.chdir(original_cwd)

    diff = torch.abs(baseline_output - torch_output)
    max_abs_error = float(torch.max(diff).item())
    passed = bool(
        torch.allclose(
            baseline_output,
            torch_output,
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
            "input_dtype": "float32",
            "ttnn_dtype": "bfloat16",
            "layout": "TILE_LAYOUT",
            "seed": args.seed,
        },
        "baseline": {
            "name": "cpu_torch_baseline",
            "runtime": "torch_cpu",
            "operation": "a @ b",
        },
        "variant": {
            "name": "ttnn_builtin_matmul",
            "runtime": "ttnn",
            "operation": "ttnn.matmul",
        },
        "runtime": {
            "kind": "tenstorrent",
            "library": "ttnn",
            "device_id": args.device_id,
            "ttnn_version": getattr(ttnn, "__version__", "unknown"),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
        "timing": {
            "protocol": {
                "timer": "time.perf_counter",
                "warmup": args.warmup,
                "repeat": args.repeat,
                "unit": "ms",
                "includes_device_synchronize": True,
                "excludes_input_transfer": True,
                "excludes_output_transfer": True,
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
            "ttnn_work_dir": str(ttnn_work_dir),
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
        f"{timestamp}_ttnn_matmul_m{case['m']}_n{case['n']}_k{case['k']}"
        f"_seed{case['seed']}.json"
    )
    output_path = output_dir / filename
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output_path


def print_summary(result: dict[str, Any], output_path: Path) -> None:
    case = result["case"]
    correctness = result["correctness"]
    timing = result["timing"]
    variant = timing["variant"]

    print("TT-OpBench result")
    print("-----------------")
    print(
        f"case:        {case['name']} m={case['m']} n={case['n']} k={case['k']} "
        f"input_dtype={case['input_dtype']} ttnn_dtype={case['ttnn_dtype']}"
    )
    print(f"baseline:    {result['baseline']['name']} ({result['baseline']['runtime']})")
    print(f"variant:     {result['variant']['name']} ({result['variant']['runtime']})")
    print(f"device:      {result['runtime']['kind']} device_id={result['runtime']['device_id']}")
    print(f"correctness: {'pass' if correctness['passed'] else 'fail'}")
    print(f"max error:   {correctness['max_abs_error']:.6g}")
    print(f"timing:      variant={variant['mean_ms']:.6f} ms")
    print(
        f"protocol:    warmup={timing['protocol']['warmup']} repeat={timing['protocol']['repeat']} "
        "sync=yes transfers=excluded"
    )
    print(f"result:      {output_path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    result = build_result(args)
    output_path = write_result(result, args.output_dir)
    print_summary(result, output_path)
    return 0 if result["correctness"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
