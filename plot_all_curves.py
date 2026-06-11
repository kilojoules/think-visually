"""Multi-curve comparison plot: model × task × scaffold."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path, k_col="K", scaffold_filter=None):
    """Load (K, acc) pairs from a results CSV."""
    by_K: dict[int, list[int]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if scaffold_filter and row.get("scaffold") != scaffold_filter:
                continue
            K = int(row[k_col])
            by_K.setdefault(K, []).append(int(row["correct"]))
    return [(K, sum(v)/len(v)*100, len(v)) for K, v in sorted(by_K.items())]


# Load all curves
sweeps = {
    "qwen1.5b · fold1 · verifier_guided": (
        load("results_qwen15_ksweep_fold1.csv") +
        # Append K=64 from the older run
        [(64, 11/20*100, 20)]
    ),
    "qwen3b · fold1 · verifier_guided":
        load("results_ksweep_qwen25_3b_fold1_verifier_guided.csv"),
    "qwen1.5b · fold1 · best_partial":
        load("results_ksweep_qwen25_15b_fold1_best_partial.csv"),
    "qwen1.5b · fold2 · verifier_guided":
        load("results_ksweep_qwen25_15b_fold2_verifier_guided.csv"),
    "qwen1.5b · maze · verifier_guided":
        load("results_ksweep_qwen25_15b_maze_verifier_guided.csv"),
}

# Dedupe Ks (sort + take unique)
for label, pts in sweeps.items():
    pts = sorted(set(pts), key=lambda p: p[0])
    # Group by K (avg duplicates)
    out: dict[int, tuple] = {}
    for K, acc, n in pts:
        if K in out:
            prev_acc, prev_n = out[K]
            total = prev_acc * prev_n + acc * n
            out[K] = (total / (prev_n + n), prev_n + n)
        else:
            out[K] = (acc, n)
    sweeps[label] = [(K, *out[K]) for K in sorted(out)]

print("Loaded curves:")
for label, pts in sweeps.items():
    print(f"  {label}: {[(p[0], round(p[1], 1)) for p in pts]}")

fig, ax = plt.subplots(figsize=(9, 5.5))

colors = ["C3", "C0", "C2", "C1", "C4"]
markers = ["o", "s", "^", "D", "v"]
for (label, pts), color, marker in zip(sweeps.items(), colors, markers):
    Ks = [p[0] for p in pts]
    accs = [p[1] for p in pts]
    ax.plot(Ks, accs, marker=marker, color=color, lw=2, markersize=9, label=label)

ax.set_xscale("log")
ax.set_xlabel("K (rejection-sampling / scoring budget, log scale)", fontsize=11)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_title("K-sweeps across (model × task × scaffold)\n"
             "All on a 16 GB Mac, local-only, $0 cash, qwen2.5 family",
             fontsize=12)
ax.grid(True, alpha=0.3, which="both")
ax.set_ylim(-3, 75)
ax.set_xticks([1, 4, 16, 64, 128, 256])
ax.set_xticklabels(["1", "4", "16", "64", "128", "256"])
ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

# Annotations for the key takeaways
ax.axhline(0, color="black", lw=0.5, alpha=0.5)
plt.tight_layout()
out = "all_ksweeps.png"
plt.savefig(out, dpi=140)
print(f"\nSaved {out}")
