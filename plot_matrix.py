"""Cross-model × cross-difficulty heatmap at K=64.

Shows the no-free-lunch / pattern-flip finding directly.
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def acc_at_K(path, target_K=64):
    by_K: dict[int, list[int]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            K = int(row["K"])
            if K == target_K:
                by_K.setdefault(K, []).append(int(row["correct"]))
    if target_K in by_K:
        v = by_K[target_K]
        return sum(v) / len(v) * 100
    return float("nan")


def qwen15_fold1_at_K(K):
    """qwen-1.5b's K=64 fold1 number lives in a different file."""
    if K != 64:
        return float("nan")
    by_K = []
    with open("results_qwen15_vg64_n20_fold1_txt.csv") as f:
        for row in csv.DictReader(f):
            if row["scaffold"] == "verifier_guided":
                by_K.append(int(row["correct"]))
    return sum(by_K) / len(by_K) * 100


models = ["Qwen2.5-1.5B", "Qwen2.5-3B", "Llama-3.2-1B", "Llama-3.2-3B"]
tasks = ["fold1", "fold2", "maze"]

fold1_files = {
    "Qwen2.5-1.5B": None,  # special
    "Qwen2.5-3B":   "results_ksweep_qwen25_3b_fold1_verifier_guided.csv",
    "Llama-3.2-1B": "results_ksweep_llama32_1b_fold1_verifier_guided.csv",
    "Llama-3.2-3B": "results_ksweep_llama32_3b_fold1_verifier_guided.csv",
}
fold2_files = {
    "Qwen2.5-1.5B": "results_ksweep_qwen25_15b_fold2_verifier_guided.csv",
    "Qwen2.5-3B":   "results_ksweep_qwen25_3b_fold2_verifier_guided.csv",
    "Llama-3.2-1B": "results_ksweep_llama32_1b_fold2_verifier_guided.csv",
    "Llama-3.2-3B": "results_ksweep_llama32_3b_fold2_verifier_guided.csv",
}
maze_files = {
    "Qwen2.5-1.5B": "results_ksweep_qwen25_15b_maze_verifier_guided.csv",
    "Qwen2.5-3B":   "results_ksweep_qwen25_3b_maze_verifier_guided.csv",
    "Llama-3.2-1B": "results_ksweep_llama32_1b_maze_verifier_guided.csv",
    "Llama-3.2-3B": "results_ksweep_llama32_3b_maze_verifier_guided.csv",
}

mat = np.zeros((len(models), len(tasks)))
for i, m in enumerate(models):
    if m == "Qwen2.5-1.5B":
        mat[i, 0] = qwen15_fold1_at_K(64)
    else:
        mat[i, 0] = acc_at_K(fold1_files[m])
    mat[i, 1] = acc_at_K(fold2_files[m])
    mat[i, 2] = acc_at_K(maze_files[m])

print("Accuracy at K=64:")
print(f"{'Model':18s}  {'fold1':>6s}  {'fold2':>6s}  {'maze':>6s}")
for i, m in enumerate(models):
    print(f"{m:18s}  {mat[i,0]:5.0f}%  {mat[i,1]:5.0f}%  {mat[i,2]:5.0f}%")

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=60, aspect="auto")

ax.set_xticks(range(len(tasks)))
ax.set_xticklabels(["fold1\n(2 holes,\n120 ans-space)", "fold2\n(4 holes,\n1,820 ans-space)", "maze\n(5×5 path,\n~4^7 ans-space)"], fontsize=10)
ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=11)
ax.set_title("Three tasks, three different winners — and one universal loser\n"
             "verifier-guided rejection sampling at K=64, n=20 per cell",
             fontsize=12)

# Annotate each cell
for i in range(len(models)):
    for j in range(len(tasks)):
        v = mat[i, j]
        color = "white" if v < 25 else "black"
        ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                fontsize=14, color=color, fontweight="bold")

cbar = plt.colorbar(im, ax=ax, shrink=0.7)
cbar.set_label("Accuracy (%)", fontsize=10)
plt.tight_layout()
out = "matrix_fold1_fold2.png"
plt.savefig(out, dpi=140)
print(f"\nSaved {out}")
