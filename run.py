"""MVP eval loop — image + text variants of each task to isolate the
perception ceiling from the reasoning ceiling.

Usage:
    python run.py                    # default model = moondream
    python run.py --model qwen2.5vl:3b
    python run.py --model llava-phi3 --tag phi3
"""
from __future__ import annotations
import argparse
import csv
import subprocess
import time
import traceback
from collections import defaultdict

from models import Moondream
from tasks import maze, folding
import scaffolds

# --- knobs ---
MAZE_SIZE = 3
N_INSTANCES_MAZE = 5
FOLD_CASES = [(1, 5), (2, 5)]   # (n_folds, n_instances)
SC_N = 4
VG_N = 8
WB_ROUNDS = 2

SCAFFOLDS = [
    ("bare",            scaffolds.bare,             {}),
    ("self_consist",    scaffolds.self_consistency, {"n": SC_N}),
    ("verifier_guided", scaffolds.verifier_guided,  {"n_max": VG_N}),
    ("whiteboard",      scaffolds.whiteboard,       {"max_rounds": WB_ROUNDS}),
]


def _safe(fn, *a, **kw) -> tuple[str, str]:
    try:
        return fn(*a, **kw), ""
    except Exception:
        return "", traceback.format_exc(limit=2)


def run_case(case_label, gen_fn, render_fn, prompt_fn, verify_fn, parseable_fn,
             use_image, n_instances, model, writer, log):
    for i in range(n_instances):
        inst = gen_fn(i)
        img = render_fn(inst) if use_image else None
        p = prompt_fn(inst)
        local_verify = (lambda inst=inst: lambda r: verify_fn(inst, r))()

        for scaff_name, scaff_fn, kwargs in SCAFFOLDS:
            kw = dict(kwargs)
            if scaff_name == "verifier_guided":
                kw["verifier"] = local_verify
            t0 = time.time()
            resp, err = _safe(scaff_fn, model, p, img, **kw)
            elapsed = time.time() - t0
            if err:
                correct = False
                valid = False
                head = f"<ERROR> {err.splitlines()[-1][:120]}"
            else:
                correct = bool(local_verify(resp))
                valid = bool(parseable_fn(resp))
                head = resp[:200].replace("\n", " ⏎ ")
            writer.writerow([case_label, i, scaff_name, int(correct),
                             int(valid), round(elapsed, 2), head])
            line = (f"{case_label:14s} #{i:02d} [{scaff_name:16s}] "
                    f"correct={str(correct):5s} valid={str(valid):5s} "
                    f"{elapsed:6.1f}s")
            print(line)
            log.write(line + "\n")
            log.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="moondream",
                    help="Ollama model name (e.g. moondream, qwen2.5vl:3b)")
    ap.add_argument("--tag", default=None,
                    help="Short tag for output filenames; defaults to a safe form of --model")
    ap.add_argument("--no-image-cases", action="store_true",
                    help="Skip image-mode cases (saves time on models w/ poor vision)")
    args = ap.parse_args()
    tag = args.tag or args.model.replace(":", "_").replace("/", "_").replace(".", "")
    out_csv = f"results_{tag}.csv"
    log_path = f"run_{tag}.log"
    print(f"Model: {args.model}  →  {out_csv}, {log_path}")

    model = Moondream(model_name=args.model)

    with open(out_csv, "w", newline="") as f_csv, open(log_path, "w") as f_log:
        writer = csv.writer(f_csv)
        writer.writerow(["case", "instance", "scaffold", "correct",
                         "valid", "seconds", "response_head"])

        # Maze
        run_case("maze_img",
                 lambda i: maze.generate(seed=i, size=MAZE_SIZE),
                 maze.render, maze.prompt, maze.verify, maze.is_parseable,
                 use_image=not args.no_image_cases,
                 n_instances=N_INSTANCES_MAZE if not args.no_image_cases else 0,
                 model=model, writer=writer, log=f_log)
        run_case("maze_txt",
                 lambda i: maze.generate(seed=i, size=MAZE_SIZE),
                 maze.render, maze.prompt_text, maze.verify, maze.is_parseable,
                 use_image=False,
                 n_instances=N_INSTANCES_MAZE,
                 model=model, writer=writer, log=f_log)

        # Folding (img + txt at each fold count)
        for n_folds, n_inst in FOLD_CASES:
            run_case(
                f"fold{n_folds}_img",
                (lambda nf=n_folds: lambda i: folding.generate(seed=i, n_folds=nf))(),
                folding.render, folding.prompt, folding.verify, folding.is_parseable,
                use_image=not args.no_image_cases,
                n_instances=n_inst if not args.no_image_cases else 0,
                model=model, writer=writer, log=f_log,
            )
            run_case(
                f"fold{n_folds}_txt",
                (lambda nf=n_folds: lambda i: folding.generate(seed=i, n_folds=nf))(),
                folding.render, folding.prompt_text, folding.verify, folding.is_parseable,
                use_image=False, n_instances=n_inst,
                model=model, writer=writer, log=f_log,
            )

    # --- Summary ---
    agg_correct: dict[tuple[str, str], int] = defaultdict(int)
    agg_valid:   dict[tuple[str, str], int] = defaultdict(int)
    agg_n:       dict[tuple[str, str], int] = defaultdict(int)
    agg_time:    dict[tuple[str, str], float] = defaultdict(float)
    with open(out_csv) as f:
        for row in csv.DictReader(f):
            k = (row["case"], row["scaffold"])
            agg_correct[k] += int(row["correct"])
            agg_valid[k]   += int(row["valid"])
            agg_n[k]       += 1
            agg_time[k]    += float(row["seconds"])

    print(f"\n=== SUMMARY (model={args.model}) ===")
    print(f"{'case':14s} {'scaffold':18s} {'acc':>6s} {'valid':>6s} {'mean_s':>7s}  n")
    for (case, scaff) in sorted(agg_correct.keys()):
        n = agg_n[(case, scaff)]
        acc = agg_correct[(case, scaff)] / n if n else 0.0
        val = agg_valid[(case, scaff)] / n if n else 0.0
        mean_s = agg_time[(case, scaff)] / n if n else 0.0
        print(f"{case:14s} {scaff:18s} {acc:6.2f} {val:6.2f} {mean_s:7.1f}  {n}")

    # Free memory: unload the model
    print(f"\nUnloading {args.model}…")
    subprocess.run(["ollama", "stop", args.model], capture_output=True)
    print(f"Wrote {out_csv} and {log_path}")


if __name__ == "__main__":
    main()
