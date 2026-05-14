# TT-OpBench

TT-OpBench is a small Tenstorrent-first, accelerator-aware experiment harness for operator-level optimization studies.

It is meant for local research experiments where one implementation is compared against another under the same operator case. A typical use is comparing a baseline implementation with optimized variants such as TT-NN builtins, custom TT-Lang kernels, and later TT-Metal kernels.

The project is Tenstorrent-first in priorities and early examples, but its result records should be able to describe other runtimes or accelerators later.

## Problem

Low-level optimization work can become hard to track when each experiment uses different inputs, timing methods, or correctness checks.

TT-OpBench should help keep these experiments comparable by recording:

- the operator case being tested
- the baseline and optimized variant
- correctness information
- timing information
- configuration and environment details

## What This Is Not

TT-OpBench is not a universal accelerator benchmark and should not claim fair cross-accelerator rankings.

It is also not a leaderboard, scoring system, dashboard, database, web server, or full benchmark framework. It is mainly a small research harness for controlled operator experiments.

It is also not a replacement for TT-Metal profiling tools. TT-OpBench should organize experiments and record result metadata. Detailed Tenstorrent performance analysis should come from the existing TT-Metal profiler and Tracy tooling.

## Current Scope

The current `v0.1` scope is a minimal CPU-only matmul experiment.

It uses deterministic NumPy inputs, compares a NumPy baseline against a NumPy variant, records basic timing and correctness, and writes one JSON result file.

Install locally and run it with:

```bash
python -m pip install .
```

```bash
tt-opbench-cpu-matmul
```

For local development without installing the package, run:

```bash
PYTHONPATH=src python -m tt_opbench.cpu_matmul
```

Future versions may add TT-NN builtin baselines, TT-Lang variants, comparison reports, more operators, and later TT-Metal microbenchmarks.

## TT-NN Matmul

If the TT-Metal Python environment is available, run the minimal TT-NN matmul path with:

```bash
PYTHONPATH=src /home/yijia/tt-metal/python_env/bin/python -m tt_opbench.ttnn_matmul
```

This compares a CPU Torch baseline against `ttnn.matmul` on device `0` and writes a JSON result.
The TT-NN device section runs from `/tmp/tt-opbench-ttnn-work` so TT-Metal generated inspector files do not land in the repo.

The command prints a short experiment summary with the case, baseline, variant, correctness, quick timing, timing protocol, and JSON result path.

## Performance Workflow

Use TT-OpBench timing as a quick signal only:

- Did the case run?
- Did correctness pass?
- Is the variant roughly faster or slower?
- Which result JSON describes this run?

Use TT-Metal profiler or Tracy when the question needs real performance detail:

- device program timing
- host dispatch overhead
- kernel timeline
- NoC, L1, DRAM, or synchronization behavior

The preferred long-term model is that TT-OpBench records links to profiler artifacts instead of inventing its own profiler format.

## Accelerator-Aware Records

Future result records should describe the runtime and accelerator explicitly instead of baking one platform into the schema. For example:

```json
"runtime": {
  "name": "ttnn",
  "accelerator": "tenstorrent",
  "device": "wormhole_b0"
}
```

Other accelerators can be added later as runtimes, but TT-OpBench should still compare only runs that share the same case, inputs, and timing protocol.
