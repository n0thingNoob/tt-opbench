# Design

TT-OpBench is built around a small experiment model. The goal is to make each operator study easy to understand, repeat, and compare.

The design is Tenstorrent-first but accelerator-aware. Tenstorrent runtimes are the first priority, while the result schema should be able to describe other accelerators later without turning the project into a universal benchmark.

## Core Terms

- **Case**: the operator or kernel being tested, including input shape, dtype, layout, and other relevant configuration.
- **Baseline**: the reference implementation used for comparison.
- **Variant**: an optimized implementation being tested against the baseline.
- **Runtime**: how an implementation is executed, such as NumPy, TT-NN, TT-Lang, TT-Metal, CUDA, ROCm, or Triton.
- **Accelerator**: the hardware family used by a runtime, such as Tenstorrent, NVIDIA, AMD, or CPU.
- **Result**: one recorded run with timing, correctness, config, and environment information.
- **Artifact**: an external output linked from a result, such as profiler traces, CSV reports, or logs.
- **Comparison**: a baseline-vs-variant view under the same input and timing protocol.

## Flow

```text
case
  -> baseline
  -> variant
  -> correctness check
  -> timing/result
  -> comparison report
```

## Development Control Rules

- Work on one case at a time.
- Every variant must have an explicit baseline.
- Compare only under the same input and timing protocol.
- Every result should be self-describing enough to inspect later.
- Prefer flat files before databases or dashboards.
- Do not add a leaderboard, scoring system, or web server unless explicitly planned later.
- Do not reimplement TT-Metal profiler or Tracy.
- Keep the core schema accelerator-aware, but do not claim cross-accelerator fairness by default.

Future code should map back to:

```text
Case -> Baseline -> Variant -> Result -> Comparison
```

## Performance Model

TT-OpBench should use a two-level performance model:

- **Quick timing**: lightweight wall-clock timing around a single operator run, with synchronization when needed. This is useful for sanity checks and rough comparisons.
- **Profiler-backed analysis**: TT-Metal profiler or Tracy artifacts for serious performance work.

TT-OpBench owns the experiment record:

- what case was run
- which baseline and variant were compared
- what timing protocol was used
- whether correctness passed
- where profiler artifacts are stored

TT-Metal profiler and Tracy own detailed performance data. TT-OpBench should not duplicate device timelines, kernel event formats, or profiler CSV schemas.

For non-Tenstorrent runtimes, the same rule applies: TT-OpBench may record artifact paths from tools such as Nsight Systems, rocprof, or framework profilers, but it should not replace them.

## Accelerator-Aware Schema

Results should identify the runtime and accelerator explicitly:

```json
"runtime": {
  "name": "ttnn",
  "accelerator": "tenstorrent",
  "device": "wormhole_b0"
}
```

Profiler outputs should be linked as artifacts:

```json
"artifacts": [
  {
    "tool": "tracy",
    "kind": "timeline",
    "path": "..."
  }
]
```

Adding another accelerator should mean adding a new runtime path and artifact links, not changing the core `Case -> Baseline -> Variant -> Result -> Comparison` model.

## v0.1 CPU Matmul

The first executable experiment is intentionally small:

- Case: one matrix multiplication with `M`, `N`, `K`, dtype `float32`, and a deterministic seed.
- Baseline: NumPy `a @ b`.
- Variant: NumPy `a @ b`, named `cpu_numpy_variant`, to prove the harness path before real optimized variants exist.
- Runtime: CPU with NumPy.
- Result: one JSON file with case, baseline, variant, timing, correctness, config, and environment information.
- Comparison: baseline and variant use the same generated inputs and timing protocol.

The default output directory is `results/`.

## v0.2-min TT-NN Matmul

The first Tenstorrent path stays narrow:

- Case: matmul with `M`, `N`, and `K` as multiples of 32.
- Baseline: CPU Torch `a @ b` using deterministic inputs.
- Variant: TT-NN builtin `ttnn.matmul` on device `0`.
- Runtime: TT-NN from the local TT-Metal Python environment.
- Timing: measures repeated `ttnn.matmul` calls with device synchronization; input and output transfers are outside the timed section.
- Result: one JSON file with the same case, variant, correctness, timing, config, and environment structure.
- Control: the TT-NN device section runs from `/tmp/tt-opbench-ttnn-work` so generated inspector files do not land in the repo.

The timing in this path is a quick signal, not a replacement for TT-Metal profiler output. Future profiler integration should record profiler artifact paths in the JSON result.
