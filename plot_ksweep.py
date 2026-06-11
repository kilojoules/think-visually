"""Plot the K-sweep accuracy curve, PTRM-style (Figure 6 analogue)."""
import csv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# Aggregate K-sweep data (includes K=64 from the earlier vg=64 run)
SWEEP_FILE = "results_qwen15_ksweep_fold1.csv"
K64_FILE = "results_qwen15_vg64_n20_fold1_txt.csv"


def acc_by_K(path, K_col="K"):
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if K_col == "K":
                k = int(row["K"])
            else:
                # the K64 file uses 'scaffold' = 'verifier_guided' instead
                if row["scaffold"] != "verifier_guided":
                    continue
                k = 64
            out.setdefault(k, [0, 0])
            out[k][0] += int(row["correct"])
            out[k][1] += 1
    return out


def main():
    a = acc_by_K(SWEEP_FILE)
    b = acc_by_K(K64_FILE, K_col="scaffold")
    combined = {**a, **b}
    Ks = sorted(combined.keys())
    accs = [combined[k][0] / combined[k][1] * 100 for k in Ks]
    ns = [combined[k][1] for k in Ks]
    print("K-sweep accuracy:")
    for k, acc, n in zip(Ks, accs, ns):
        print(f"  K={k:4d}  acc={acc:5.1f}%   n={n}")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(Ks, accs, "o-", color="C3", lw=2.5, markersize=10,
            label="verifier_guided @ qwen2.5:1.5b")
    # Independent-sample upper bound: P(at least one correct) = 1 - (1-p)^K with p=0.05
    import math
    p = 0.05
    ax.plot(Ks, [(1 - (1-p)**k) * 100 for k in Ks], "--",
            color="gray", lw=1.5, alpha=0.7,
            label=f"independent-sample upper bound (p={p})")
    ax.set_xscale("log")
    ax.set_xlabel("K (rejection-sampling budget, log scale)", fontsize=11)
    ax.set_ylabel("Accuracy on fold1_txt (%)", fontsize=11)
    ax.set_title("Verifier-guided rejection sampling on qwen2.5:1.5b (n=20 per K)\n"
                 "PTRM-style width-scaling curve, paper-folding (1 fold)",
                 fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(k) for k in Ks])
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", fontsize=10)
    # Annotate points
    for k, acc in zip(Ks, accs):
        ax.annotate(f"{acc:.0f}%", (k, acc), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10, color="C3")
    plt.tight_layout()
    out_path = "ksweep_fold1.png"
    plt.savefig(out_path, dpi=140)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
