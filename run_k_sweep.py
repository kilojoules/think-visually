"""K-sweep for verifier_guided (default) or best_partial on one task.

Mirrors PTRM Figure 6: accuracy vs K on a log scale, single scaffold,
fixed model, fixed task. Memory-safe (one model loaded, same call profile
as the previous successful runs).

Usage:
    python run_k_sweep.py --model qwen2.5:1.5b --task fold1
    python run_k_sweep.py --model qwen2.5:3b   --task fold1 --ks 1,4,16,64
    python run_k_sweep.py --model qwen2.5:1.5b --task fold2 --ks 1,4,16,64
    python run_k_sweep.py --model qwen2.5:1.5b --task maze  --ks 1,4,16,64
    python run_k_sweep.py --model qwen2.5:1.5b --task fold1 --scaffold best_partial --ks 4,8,16
"""
from __future__ import annotations
import argparse
import csv
import time
import traceback

from models import Moondream
from tasks import folding, maze
import scaffolds
import memsafe


TASKS = {
    "fold1": dict(
        gen=lambda i: folding.generate(seed=i, n_folds=1),
        prompt=folding.prompt_text,
        verify=folding.verify,
        is_parseable=folding.is_parseable,
        partial=folding.partial_score,
    ),
    "fold2": dict(
        gen=lambda i: folding.generate(seed=i, n_folds=2),
        prompt=folding.prompt_text,
        verify=folding.verify,
        is_parseable=folding.is_parseable,
        partial=folding.partial_score,
    ),
    "maze": dict(
        gen=lambda i: maze.generate(seed=i, size=2),
        prompt=maze.prompt_text,
        verify=maze.verify,
        is_parseable=maze.is_parseable,
        partial=None,  # no partial-credit defined for maze
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:1.5b")
    ap.add_argument("--task", choices=list(TASKS.keys()), default="fold1")
    ap.add_argument("--scaffold", choices=["verifier_guided", "best_partial"],
                    default="verifier_guided")
    ap.add_argument("--n", type=int, default=20, help="instances per K")
    ap.add_argument("--start-seed", type=int, default=0,
                    help="instance seed offset — for adding instances to existing run")
    ap.add_argument("--ks", default="1,4,16,64",
                    help="comma-separated K values to test")
    ap.add_argument("--min-available-gb", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    Ks = [int(k) for k in args.ks.split(",")]

    tag = f"{args.model.replace(':','_').replace('.','')}_{args.task}_{args.scaffold}"
    out = args.out or f"results_ksweep_{tag}.csv"

    task = TASKS[args.task]
    if args.scaffold == "best_partial" and task["partial"] is None:
        raise ValueError(f"best_partial scaffold requires partial-score; {args.task} has none")

    print(memsafe.status("preflight"))
    print(f"Config: model={args.model}  task={args.task}  scaffold={args.scaffold}")
    print(f"        Ks={Ks}  n={args.n}  start_seed={args.start_seed}")

    model = Moondream(model_name=args.model, num_ctx=1024, keep_alive="60s")

    rows: list[list] = []
    try:
        for K in Ks:
            print(f"\n=== K={K} ===")
            for j in range(args.n):
                i = args.start_seed + j  # use absolute seed
                inst = task["gen"](i)
                p = task["prompt"](inst)
                local_verify = lambda r, inst=inst: task["verify"](inst, r)

                ag = memsafe.available_gb()
                if ag < args.min_available_gb:
                    raise MemoryError(f"available {ag:.2f} GB < floor {args.min_available_gb}")

                t0 = time.time()
                try:
                    if args.scaffold == "verifier_guided":
                        resp = scaffolds.verifier_guided(
                            model, p, None, verifier=local_verify, n_max=K,
                        )
                    else:  # best_partial
                        local_score = lambda r, inst=inst: task["partial"](inst, r)
                        resp = scaffolds.best_partial(
                            model, p, None, partial_score=local_score, n=K,
                        )
                    err = ""
                except Exception:
                    resp, err = "", traceback.format_exc(limit=2)
                elapsed = time.time() - t0

                if err:
                    correct, valid = False, False
                    head = f"<ERROR> {err.splitlines()[-1][:120]}"
                else:
                    correct = bool(local_verify(resp))
                    valid = bool(task["is_parseable"](resp))
                    head = resp[:200].replace("\n", " ⏎ ")

                rows.append([args.task, args.scaffold, K, i,
                             int(correct), int(valid), round(elapsed, 2), head])
                with open(out, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["task", "scaffold", "K", "instance",
                                "correct", "valid", "seconds", "response_head"])
                    w.writerows(rows)

                print(f"K={K:3d} #{i:02d}  corr={str(correct):5s} valid={str(valid):5s} "
                      f"{elapsed:6.1f}s  avail={memsafe.available_gb():.1f}GB")

            ks_rows = [r for r in rows if r[2] == K]
            kc = sum(r[4] for r in ks_rows)
            print(f"  >> K={K}: {kc}/{len(ks_rows)} = {kc/len(ks_rows):.0%}")
    finally:
        print("\n=== K-SWEEP SUMMARY ===")
        print(f"{'K':>6s}  {'correct':>10s}  {'pct':>6s}  {'valid':>6s}  {'mean_s':>7s}")
        by_K: dict[int, list[list]] = {}
        for r in rows:
            by_K.setdefault(r[2], []).append(r)
        for K in Ks:
            if K in by_K:
                rs = by_K[K]
                c = sum(r[4] for r in rs)
                v = sum(r[5] for r in rs)
                t = sum(r[6] for r in rs) / len(rs)
                print(f"{K:6d}  {c}/{len(rs):<8d} {c/len(rs):6.0%}  {v}/{len(rs):<5d} {t:7.1f}")

        memsafe.cooldown_and_verify(args.model, min_free_gb=3.0)
        print(memsafe.status("after cooldown"))
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
