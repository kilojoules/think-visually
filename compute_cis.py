"""Exact 95% binomial CIs (Clopper–Pearson) on the 4×3 heatmap.

Maze row uses n=50 (20 original + 30 extra seeds). Earlier versions
bootstrapped these cells; on n=20 Bernoulli outcomes the bootstrap is
strictly worse than the exact interval and degenerates to [0, 0] at
0/n successes. Clopper–Pearson is computed by inverting the binomial
CDF directly — pure Python, no dependencies.
"""
from __future__ import annotations
import csv
import json
import os
from math import comb

MODELS = ["Qwen2.5-1.5B", "Qwen2.5-3B", "Llama-3.2-1B", "Llama-3.2-3B"]
TASKS = ["fold1", "fold2", "maze"]

FOLD1_FILES = {
    "Qwen2.5-1.5B": ("data/results_qwen15_vg64_n20_fold1_txt.csv", "scaffold", "verifier_guided"),
    "Qwen2.5-3B":   ("data/results_ksweep_qwen25_3b_fold1_verifier_guided.csv", None, None),
    "Llama-3.2-1B": ("data/results_ksweep_llama32_1b_fold1_verifier_guided.csv", None, None),
    "Llama-3.2-3B": ("data/results_ksweep_llama32_3b_fold1_verifier_guided.csv", None, None),
}
FOLD2_FILES = {
    "Qwen2.5-1.5B": ("data/results_ksweep_qwen25_15b_fold2_verifier_guided.csv", None, None),
    "Qwen2.5-3B":   ("data/results_ksweep_qwen25_3b_fold2_verifier_guided.csv", None, None),
    "Llama-3.2-1B": ("data/results_ksweep_llama32_1b_fold2_verifier_guided.csv", None, None),
    "Llama-3.2-3B": ("data/results_ksweep_llama32_3b_fold2_verifier_guided.csv", None, None),
}
MAZE_FILES_BASE = {
    "Qwen2.5-1.5B": "data/results_ksweep_qwen25_15b_maze_verifier_guided.csv",
    "Qwen2.5-3B":   "data/results_ksweep_qwen25_3b_maze_verifier_guided.csv",
    "Llama-3.2-1B": "data/results_ksweep_llama32_1b_maze_verifier_guided.csv",
    "Llama-3.2-3B": "data/results_ksweep_llama32_3b_maze_verifier_guided.csv",
}
MAZE_FILES_EXTRA = {
    "Qwen2.5-1.5B": "data/results_maze_extra_qwen25_15b.csv",
    "Qwen2.5-3B":   "data/results_maze_extra_qwen25_3b.csv",
    "Llama-3.2-1B": "data/results_maze_extra_llama32_1b.csv",
    "Llama-3.2-3B": "data/results_maze_extra_llama32_3b.csv",
}


def load_k64(path: str, scaffold_col: str | None, scaffold_val: str | None) -> list[int]:
    """Return list of 0/1 outcomes at K=64."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for row in csv.DictReader(f):
            if scaffold_col and row.get(scaffold_col) != scaffold_val:
                continue
            if "K" in row and int(row["K"]) != 64:
                continue
            out.append(int(row["correct"]))
    return out


def _binom_cdf(x: int, n: int, p: float) -> float:
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(x + 1))


def _bisect(f, lo: float, hi: float, tol: float = 1e-10) -> float:
    """Root of f on [lo, hi], assuming f(lo) > 0 > f(hi)."""
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson(x: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Return (point estimate, low, high) as percentages."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    if x == 0:
        lo = 0.0
    else:
        # largest p with P(X >= x | n, p) <= alpha/2
        lo = _bisect(lambda p: alpha / 2 - (1 - _binom_cdf(x - 1, n, p)), 0.0, 1.0)
    if x == n:
        hi = 1.0
    else:
        # smallest p with P(X <= x | n, p) <= alpha/2
        hi = _bisect(lambda p: _binom_cdf(x, n, p) - alpha / 2, 0.0, 1.0)
    return x / n * 100, lo * 100, hi * 100


def main():
    table: dict = {"models": MODELS, "tasks": TASKS, "cells": []}
    print(f"{'Model':14s}  {'Task':6s}  {'Mean':>5s}  {'CI 95%':>16s}  {'n':>4s}")
    for model in MODELS:
        for task in TASKS:
            if task == "fold1":
                path, c, v = FOLD1_FILES[model]
                outcomes = load_k64(path, c, v)
            elif task == "fold2":
                path, c, v = FOLD2_FILES[model]
                outcomes = load_k64(path, c, v)
            else:  # maze
                outcomes = load_k64(MAZE_FILES_BASE[model], None, None)
                extra = load_k64(MAZE_FILES_EXTRA[model], None, None)
                outcomes = outcomes + extra
            mean, lo, hi = clopper_pearson(sum(outcomes), len(outcomes))
            print(f"{model:14s}  {task:6s}  {mean:5.1f}  [{lo:5.1f}, {hi:5.1f}]   {len(outcomes):4d}")
            table["cells"].append({
                "model": model, "task": task,
                "n": len(outcomes), "mean": mean, "ci_lo": lo, "ci_hi": hi,
            })
    with open("data/ci_table.json", "w") as f:
        json.dump(table, f, indent=2)
    print("\nWrote data/ci_table.json")


if __name__ == "__main__":
    main()
