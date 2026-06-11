# A monitor cannot rescue what the model cannot produce

> Three procedural spatial-reasoning tasks × four small open-weight LLMs ×
> verifier-guided rejection sampling at K=64, end-to-end on a 16 GB
> MacBook for $0. Each task has a different point-estimate winner, and a
> 1B-parameter Llama posts the best score on the task with the largest
> answer space — though see the [caveats](#caveats): only one of the three
> winners is statistically separable, and the maze row has a known
> prompt confound awaiting a re-run.

📄 **[Read the writeup →](BLOG_POST.md)** &nbsp;·&nbsp;
📑 [Full technical report](REPORT.md) &nbsp;·&nbsp;
📊 [Lead chart](figures/matrix_with_cis.png)

*(Why "think-visually"? The project began as a study of vision-language
models that reason by drawing. The first finding was that no VLM with
interleaved image generation fits in 16 GB — so the question became what
text-only models can do on spatial tasks, and the name stuck.)*

---

![One real verifier-guided run: llama3.2:1b solves a 5×5 maze on attempt 7](figures/maze_thinking.gif)

*A real run from the harness: `llama3.2:1b` attempts a 5×5 maze under
verifier-guided rejection sampling. The model reasons in text only (left); the
verifier walks each proposed path (right) and rejects until a path reaches G —
here on attempt 7. The model never sees the picture: scaffolding can only
select from what the model's text-token distribution already contains.
Frames are unedited, but the instance was selected for pacing (another
collected run took 52 attempts). Re-render exactly from the committed trace
with `make_animation.py`; collecting a fresh trace via
`make_animation_data.py` samples at temperature 0.8 and will differ.*

![Three tasks, three point-estimate winners](figures/matrix_with_cis.png)

*Verifier-guided rejection sampling at K=64 with exact 95% binomial CIs
(Clopper–Pearson; n=20 for the fold tasks, n=50 for the maze row). Each task
has a different point-estimate winner, and Qwen2.5-3B never wins (though it
edges Llama-1B on fold1). Of the three winners, only fold1 is statistically
separable from its runner-up (Fisher exact p=0.019); fold2 (p=0.66) and maze
(p=0.069) are suggestive, not conclusive.*

## Headline

| Model | fold1 (n=20) | fold2 (n=20) | maze (n=50) |
|---|---:|---:|---:|
| Qwen2.5-1.5B | **55%** [32, 77] | 0% [0, 17] | 34% [21, 49] |
| Qwen2.5-3B | 10% [1, 32] | 0% [0, 17] | 0% [0, 7] |
| Llama-3.2-1B | 5% [0, 25] | 10% [1, 32] | **54%** [39, 68] |
| Llama-3.2-3B | 15% [3, 38] | **20%** [6, 44] | 30% [18, 45] |

The implication for monitoring-as-selection: a verifier-as-monitor cannot
rescue behavior the model never produces, and what a model produces depends
on the task in a way that doesn't correlate with parameter count. Model
choice is upstream of what monitoring can fix.

(A note on the headline pattern itself: with four models and three tasks,
three *distinct* winners arise 37.5% of the time even under a null where
all models are identical. The pattern is descriptive; the inferential
content lives in the individual cells and the K-sweep curves.)

## Coverage rises with K — with strongly diminishing returns

![Qwen-family K-sweeps](figures/all_ksweeps.png)

*Accuracy vs K for the five Qwen-family sweeps (the Llama models were
measured per-task at fewer K values — see `data/`). On fold1 with
Qwen-1.5B, accuracy rises from 5% at K=1 to 65% at K=256, but the last
two doublings buy only ~5pp each, versus ~20pp per doubling earlier in
the curve. We did not measure beyond K=256; the curve is still rising at
the final point. The coverage argument rests on the diminishing-returns
shape plus the soft-scorer comparison, not on a literal plateau.*

## Same task, different family, different ceiling

![Cross-family on fold1](figures/cross_family_fold1.png)

*Qwen2.5 family vs Llama-3.2 family on fold1. Same task, same K, very
different shapes — and Qwen-3B sits at the bottom of the Qwen family on
the easiest task in the matrix.*

## Reproduce

```bash
# Ollama + the four text-only models (~7.6 GB on disk)
brew install ollama
ollama serve &
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:3b
ollama pull llama3.2:1b
ollama pull llama3.2:3b

# Python deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Rebuild figures from cached CSVs (no model calls — pure analysis)
python compute_cis.py      # exact binomial CIs → data/ci_table.json
python plot_matrix_ci.py   # writes figures/matrix_with_cis.png
python plot_all_curves.py  # writes figures/all_ksweeps.png

# Run one K-sweep cell from scratch (Qwen-1.5B on fold1, K=1..256).
# Write to a fresh path — sampling at temperature 0.8 means your numbers
# will differ from the shipped CSVs.
python run_k_sweep.py --model qwen2.5:1.5b --task fold1 \
    --ks 1,2,4,8,16,32,64,128,256 --n 20 \
    --out my_rerun_fold1.csv
```

## Layout

```
.
├── BLOG_POST.md           the writeup (read this first)
├── REPORT.md              full technical report
├── figures/               all charts (.png) + the animation (.gif)
├── data/                  raw CSVs + ci_table.json + animation trace
├── tasks/                 folding.py, maze.py + their physics verifiers
├── scaffolds.py           bare, self_consistency, verifier_guided,
│                          best_partial, whiteboard_of_thought
├── memsafe.py             available_gb() abort check
├── models.py              thin Ollama wrapper
├── make_animation_data.py per-attempt trace of one real maze run
├── make_animation.py      renders figures/maze_thinking.gif (Pillow only)
├── run_k_sweep.py         main runner (sweeps K, --start-seed for
│                          incremental n)
├── run_maze_n50.py        orchestrator for the n=50 maze row
├── run_one_case.py        single-instance runner (used for debugging)
├── compute_cis.py         exact binomial CIs → data/ci_table.json
└── plot_*.py              figure builders
```

## Caveats

**Known confound in the maze row (re-run planned).** The maze prompt ends
with `Example: D,D,R,R,U,R`. Because every maze fixes start at the top-left
open cell and goal at the bottom-right, and the verifier accepts as soon as
a walk reaches G (trailing moves are not checked), that example string
*verbatim* solves 24 of the 50 test mazes (48%). Any model with a nonzero
chance of approximately echoing the example gets free wins, so the maze
accuracies — including Llama-1B's 54% — sit only modestly above a ~48%
parrot baseline, and the "smallest model wins through output diversity"
interpretation is not safe. The fix (a syntactically valid, never-solving
example string + full maze-row re-run) is queued; numbers here will be
updated when it lands.

**Statistical strength varies by cell.** n=20 for the fold tasks, n=50 for
the maze row. Of the three per-task winners, only fold1 (Qwen-1.5B,
11/20 vs 3/20, Fisher exact p=0.019) is separable from its runner-up.
fold2 (4/20 vs 2/20, p=0.66) and maze (27/50 vs 17/50, p=0.069) are
point-estimate leads only. And three distinct winners across three tasks
occurs 37.5% of the time under an identical-models null — treat the
"three winners" framing as descriptive. See **"What would change my
mind"** in [`BLOG_POST.md`](BLOG_POST.md) for the full follow-up list
(third model family, frontier-model row, prompt-sensitivity study).

Total cash: $0. Total wall-clock: ~14 hours, including two near-OOM
incidents that taught me the resident-size ceiling on a 16 GB Mac
(`memsafe.py` is where those lessons live).
