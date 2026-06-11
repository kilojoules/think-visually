# think-visually — A monitor cannot rescue what the model cannot produce

Three procedural spatial-reasoning tasks × four small open-weight LLMs ×
verifier-guided rejection sampling at K=64, run end-to-end on a 16 GB
MacBook for $0. The headline: **no model wins more than one task**, and a
1B-parameter Llama beats every larger model on the task with the largest
answer space.

**Read order**

1. [`BLOG_POST.md`](BLOG_POST.md) — the writeup. Frames the result as a
   monitoring-as-selection upper bound.
2. [`matrix_with_cis.png`](matrix_with_cis.png) — the lead chart. 4×3
   accuracy matrix with bootstrap 95% CIs in every cell.
3. [`REPORT.md`](REPORT.md) — the full technical report (methods,
   per-task analysis, raw numbers).

## Headline matrix (K=64, bootstrap 95% CIs)

| Model | fold1 (n=20) | fold2 (n=20) | maze (n=50) |
|---|---:|---:|---:|
| Qwen2.5-1.5B | **55%** [35, 75] | 0% [0, 0] | 34% [22, 48] |
| Qwen2.5-3B | 10% [0, 25] | 0% [0, 0] | 0% [0, 0] |
| Llama-3.2-1B | 5% [0, 15] | 10% [0, 25] | **54%** [40, 68] |
| Llama-3.2-3B | 15% [0, 30] | **20%** [5, 40] | 30% [18, 42] |

## Setup

```bash
# Ollama and the four models (text-only, ~7.6 GB total on disk)
brew install ollama
ollama serve &
ollama pull qwen2.5:1.5b qwen2.5:3b llama3.2:1b llama3.2:3b

# Python deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Reproduce a single K-sweep cell (Qwen-1.5B on fold1, K=1..256)
python run_k_sweep.py --model qwen2.5:1.5b --task fold1 \
    --ks 1,2,4,8,16,32,64,128,256 --n 20 \
    --out results_ksweep_qwen25_15b_fold1_verifier_guided.csv

# Rebuild the headline chart from cached CSVs
python bootstrap_ci.py     # writes ci_table.json
python plot_matrix_ci.py   # writes matrix_with_cis.png
```

## Layout

- `run_k_sweep.py` — main runner. Sweeps K, supports
  `--start-seed` for incremental n extensions.
- `run_maze_n50.py` — orchestrator for the n=50 maze row.
- `scaffolds.py` — `bare`, `self_consistency`, `verifier_guided`,
  `best_partial`, `whiteboard_of_thought`.
- `tasks/folding.py`, `tasks/maze.py` — task generators and
  deterministic physics verifiers (no precomputed answers).
- `memsafe.py` — `available_gb()` abort check; the local-Mac
  ceiling is ~14 GB resident before swap starts hurting.
- `bootstrap_ci.py`, `plot_matrix_ci.py`, `plot_*.py` — analysis
  and figures.
- `results_*.csv` — every raw run (one row per instance × K).

## Caveats

n=20 for the fold tasks, n=50 for the maze row. The cell-level
winners (Qwen-1.5B fold1, Llama-3B fold2, Llama-1B maze) are
statistically separable from runners-up at these sample sizes;
the rest of the matrix is suggestive. See "What would change my
mind" in `BLOG_POST.md` for the full list of follow-ups (third
model family, frontier-model row, prompt-sensitivity study).

Total cash: $0. Total wall-clock: ~14 hours including two
near-OOM incidents.
