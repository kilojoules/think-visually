"""Orchestrate the maze-row n=50 extension: 4 models × maze × K=64 × seeds 20-49.

Sequential to keep memory safe. Each model loaded, run, unloaded before next.
"""
from __future__ import annotations
import subprocess
import sys

MODELS = ["qwen2.5:1.5b", "qwen2.5:3b", "llama3.2:1b", "llama3.2:3b"]
N_EXTRA = 30  # seeds 20-49

for m in MODELS:
    cmd = [
        sys.executable, "run_k_sweep.py",
        "--model", m,
        "--task", "maze",
        "--ks", "64",
        "--n", str(N_EXTRA),
        "--start-seed", "20",
        "--out", f"results_maze_extra_{m.replace(':','_').replace('.','')}.csv",
    ]
    print(f"\n{'='*70}\nRunning: {' '.join(cmd)}\n{'='*70}")
    subprocess.run(cmd, check=True)

print("\n=== ALL DONE ===")
