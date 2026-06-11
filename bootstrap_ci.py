"""Bootstrap 95% CIs on the 4×3 heatmap, with maze using n=50 (20 original + 30 extra)."""
from __future__ import annotations
import csv
import json
import os
import numpy as np

RNG_SEED = 42
N_BOOT = 10000

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


def bootstrap_ci(outcomes: list[int], n_boot: int = N_BOOT, alpha: float = 0.05) -> tuple[float, float, float]:
    """Return (point estimate, low, high) as percentages."""
    if not outcomes:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(RNG_SEED)
    arr = np.array(outcomes)
    n = len(arr)
    boots = rng.choice(arr, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [alpha/2, 1 - alpha/2])
    return arr.mean() * 100, lo * 100, hi * 100


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
            mean, lo, hi = bootstrap_ci(outcomes)
            print(f"{model:14s}  {task:6s}  {mean:5.1f}  [{lo:5.1f}, {hi:5.1f}]   {len(outcomes):4d}")
            table["cells"].append({
                "model": model, "task": task,
                "n": len(outcomes), "mean": mean, "ci_lo": lo, "ci_hi": hi,
            })
    with open("data/ci_table.json", "w") as f:
        json.dump(table, f, indent=2)
    print("\nWrote ci_table.json")


if __name__ == "__main__":
    main()
