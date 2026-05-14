# TT-OpBench

TT-OpBench is a small experiment harness for Tenstorrent operator-level optimization studies.

The goal is to make it easier to run, reproduce, and compare low-level Tenstorrent experiments.

This repo is not intended to be a public benchmark suite, leaderboard, or universal accelerator benchmark. It is mainly for local research experiments.

## Purpose

When testing a new optimization, such as a custom TT-Lang kernel, synchronization change, buffering strategy, layout change, or NoC communication pattern, this repo should help answer:

- Does the optimized version still produce correct results?
- Is it faster than the baseline?
- Which part became faster or slower?
- Does the optimization work across different input sizes?

## Basic Idea

Each experiment compares one or more implementation variants under the same operator case.
