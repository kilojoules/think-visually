"""Cross-family model comparison: Qwen vs Llama, 1.5B-class vs 3B-class.

This is the chart the post should lead with.
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    by_K: dict[int, list[int]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            K = int(row["K"])
            by_K.setdefault(K, []).append(int(row["correct"]))
    return [(K, sum(v)/len(v)*100) for K, v in sorted(by_K.items())]


def load_qwen15():
    """Qwen-1.5B data comes from two files (K=64 is in a separate run)."""
    pts = load("data/results_qwen15_ksweep_fold1.csv")
    # Append K=64 from the n=20 vg=64 run
    by_K: dict[int, list[int]] = {}
    with open("data/results_qwen15_vg64_n20_fold1_txt.csv") as f:
        for row in csv.DictReader(f):
            if row["scaffold"] == "verifier_guided":
                by_K.setdefault(64, []).append(int(row["correct"]))
    if 64 in by_K:
        pts.append((64, sum(by_K[64])/len(by_K[64])*100))
    return sorted(pts)


sweeps = {
    "Qwen2.5-1.5B": (None, "C3", "o"),  # loaded via load_qwen15
    "Qwen2.5-3B":   ("data/results_ksweep_qwen25_3b_fold1_verifier_guided.csv", "C0", "s"),
    "Llama-3.2-1B": ("data/results_ksweep_llama32_1b_fold1_verifier_guided.csv", "C2", "^"),
    "Llama-3.2-3B": ("data/results_ksweep_llama32_3b_fold1_verifier_guided.csv", "C1", "D"),
}

fig, ax = plt.subplots(figsize=(9, 5.5))
for label, (path, color, marker) in sweeps.items():
    pts = load_qwen15() if path is None else load(path)
    # restrict to K=1,4,16,64 for the cross-family plot
    pts = [(K, a) for (K, a) in pts if K in (1, 4, 16, 64)]
    Ks = [p[0] for p in pts]
    accs = [p[1] for p in pts]
    ax.plot(Ks, accs, marker=marker, color=color, lw=2.5, markersize=11, label=label)
    # annotate the K=64 point
    if 64 in Ks:
        idx = Ks.index(64)
        ax.annotate(f"{accs[idx]:.0f}%", (64, accs[idx]),
                    textcoords="offset points", xytext=(8, 0),
                    ha="left", va="center", fontsize=10, color=color, fontweight="bold")

ax.set_xscale("log")
ax.set_xlabel("K (verifier-guided rejection-sampling budget, log scale)", fontsize=11)
ax.set_ylabel("Accuracy on 1-fold paper folding (%)", fontsize=11)
ax.set_title("Coverage-limited plateau across model families\n"
             "fold1, n=20 per cell, qwen2.5 vs llama3.2 at 1.5B-class and 3B-class",
             fontsize=12)
ax.grid(True, alpha=0.3, which="both")
ax.set_ylim(-3, 75)
ax.set_xticks([1, 4, 16, 64])
ax.set_xticklabels(["1", "4", "16", "64"])
ax.legend(loc="upper left", fontsize=11, framealpha=0.95)
plt.tight_layout()
out = "figures/cross_family_fold1.png"
plt.savefig(out, dpi=140)
print(f"Saved {out}")

# Also print the table the post should include
print("\n=== Cross-family results table ===")
print(f"{'Model':18s}  K=1    K=4    K=16   K=64")
for label, (path, _, _) in sweeps.items():
    pts = load_qwen15() if path is None else load(path)
    row = {K: acc for K, acc in pts}
    cells = [f"{row.get(k, float('nan')):5.0f}%" for k in [1, 4, 16, 64]]
    print(f"{label:18s}  {'  '.join(cells)}")
