# Roadmap

TT-OpBench should grow in small stages. Each version should add one main capability and avoid becoming a full framework before the experiments need it.

The project is Tenstorrent-first, but result records should stay accelerator-aware so other runtimes can be described later without turning this into a universal benchmark.

## v0.0

- Documentation only

## v0.1

- CPU-only matmul experiment
- Deterministic input generation
- Basic timing
- JSON result output
- Minimal Python package and one console entry point

## v0.2

- TT-NN builtin matmul baseline
- Minimal TT-NN matmul path against a CPU reference
- Treat built-in timing as quick sanity timing only

## v0.3

- First TT-Lang custom matmul variant

## v0.4

- Baseline-vs-variant comparison report
- Record TT-Metal profiler or Tracy artifact paths when profiling is enabled
- Keep profiler artifacts generic enough to support other tools later

## v0.5

- More operators such as eltwise add, layernorm/rmsnorm, and softmax

## v0.6

- TT-Metal microbenchmarks such as NoC transfer, L1 circular buffer, and synchronization variants

## Later

- Optional runtime paths for other accelerators, if they are useful for local research
- Keep comparisons scoped to matching cases, inputs, and timing protocols
- Do not add leaderboard-style cross-accelerator rankings by default
