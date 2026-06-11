"""Heatmap with exact 95% binomial CIs (Clopper–Pearson) shown in each cell."""
from __future__ import annotations
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


with open("data/ci_table.json") as f:
    data = json.load(f)

models = data["models"]
tasks = data["tasks"]

mat = np.zeros((len(models), len(tasks)))
los = np.zeros_like(mat)
his = np.zeros_like(mat)
ns = np.zeros_like(mat, dtype=int)

for cell in data["cells"]:
    i = models.index(cell["model"])
    j = tasks.index(cell["task"])
    mat[i, j] = cell["mean"]
    los[i, j] = cell["ci_lo"]
    his[i, j] = cell["ci_hi"]
    ns[i, j] = cell["n"]

fig, ax = plt.subplots(figsize=(9, 5.5))
im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=60, aspect="auto")

ax.set_xticks(range(len(tasks)))
ax.set_xticklabels([
    f"fold1\n(2 holes, ans-space ~120)\nn={ns[0,0]}",
    f"fold2\n(4 holes, ans-space 1,820)\nn={ns[0,1]}",
    f"maze\n(5×5, ans-space ~4⁷)\nn={ns[0,2]}",
], fontsize=10)
ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=11)
ax.set_title(
    "Verifier-guided rejection sampling, K=64 — exact 95% binomial CIs (Clopper–Pearson)\n"
    "Three point-estimate winners; only fold1's is statistically separable (Fisher p=0.019)",
    fontsize=12,
)

for i in range(len(models)):
    for j in range(len(tasks)):
        v = mat[i, j]
        lo, hi = los[i, j], his[i, j]
        color = "white" if v < 25 else "black"
        ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                fontsize=15, color=color, fontweight="bold")
        ax.text(j, i + 0.28, f"[{lo:.0f}, {hi:.0f}]", ha="center", va="center",
                fontsize=9, color=color)

cbar = plt.colorbar(im, ax=ax, shrink=0.7)
cbar.set_label("Accuracy (%) at K=64", fontsize=10)
plt.tight_layout()
out = "figures/matrix_with_cis.png"
plt.savefig(out, dpi=140)
print(f"Saved {out}")

# Also print a clean table
print("\nFinal table (K=64, exact 95% binomial CIs):")
print(f"{'Model':14s}  {'fold1':>20s}  {'fold2':>20s}  {'maze':>20s}")
for i, m in enumerate(models):
    cells = []
    for j in range(len(tasks)):
        cells.append(f"{mat[i,j]:5.0f}%  [{los[i,j]:4.0f},{his[i,j]:4.0f}]")
    print(f"{m:14s}  {'  '.join(cells)}")
