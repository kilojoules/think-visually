"""Run a SINGLE case (one task × one mode) with explicit memory guards.

Designed for a 16 GB Mac: aborts before each call if free RAM drops below
floor; stops the model and waits for memory reclaim after the case.

Usage:
    python run_one_case.py --model qwen2.5vl:3b --case maze_txt --n 3
    python run_one_case.py --model qwen2.5vl:3b --case fold1_txt --n 3
"""
from __future__ import annotations
import argparse
import csv
import time
import traceback

from models import Moondream
from tasks import maze, folding
import scaffolds
import memsafe

MAZE_SIZE = int(__import__("os").environ.get("MAZE_SIZE", "2"))  # set via env var

def build_scaffolds(sc_n: int, vg_n_max: int, wb_rounds: int):
    return [
        ("bare",            scaffolds.bare,             {}),
        ("self_consist",    scaffolds.self_consistency, {"n": sc_n}),
        ("verifier_guided", scaffolds.verifier_guided,  {"n_max": vg_n_max}),
        ("whiteboard",      scaffolds.whiteboard,       {"max_rounds": wb_rounds}),
    ]

CASES = {
    "maze_img":  ("maze",    True,  lambda i: maze.generate(seed=i, size=MAZE_SIZE),
                  maze.render, maze.prompt, maze.verify, maze.is_parseable),
    "maze_txt":  ("maze",    False, lambda i: maze.generate(seed=i, size=MAZE_SIZE),
                  maze.render, maze.prompt_text, maze.verify, maze.is_parseable),
    "fold1_img": ("folding", True,  lambda i: folding.generate(seed=i, n_folds=1),
                  folding.render, folding.prompt, folding.verify, folding.is_parseable),
    "fold1_txt": ("folding", False, lambda i: folding.generate(seed=i, n_folds=1),
                  folding.render, folding.prompt_text, folding.verify, folding.is_parseable),
    "fold2_img": ("folding", True,  lambda i: folding.generate(seed=i, n_folds=2),
                  folding.render, folding.prompt, folding.verify, folding.is_parseable),
    "fold2_txt": ("folding", False, lambda i: folding.generate(seed=i, n_folds=2),
                  folding.render, folding.prompt_text, folding.verify, folding.is_parseable),
}


def _safe(fn, *a, **kw):
    try:
        return fn(*a, **kw), ""
    except MemoryError:
        raise
    except Exception:
        return "", traceback.format_exc(limit=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--case", required=True, choices=list(CASES.keys()))
    ap.add_argument("--n", type=int, default=3, help="instances")
    ap.add_argument("--min-free-gb", type=float, default=3.5,
                    help="abort if free RAM drops below this")
    ap.add_argument("--sc-n", type=int, default=4)
    ap.add_argument("--vg-n-max", type=int, default=16)
    ap.add_argument("--wb-rounds", type=int, default=3)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    SCAFFOLDS = build_scaffolds(args.sc_n, args.vg_n_max, args.wb_rounds)
    tag = args.tag or args.model.replace(":", "_").replace(".", "")
    out_csv = f"results_{tag}_{args.case}.csv"

    print(memsafe.status(f"start {args.case}"))
    print(f"Config: sc_n={args.sc_n}  vg_n_max={args.vg_n_max}  wb_rounds={args.wb_rounds}  n={args.n}")
    memsafe.require_free(args.min_free_gb, label=f"start {args.case}")

    _, use_image, gen_fn, render_fn, prompt_fn, verify_fn, parseable_fn = CASES[args.case]
    model = Moondream(model_name=args.model)

    rows = []
    try:
        for i in range(args.n):
            inst = gen_fn(i)
            img = render_fn(inst) if use_image else None
            p = prompt_fn(inst)
            local_verify = (lambda inst=inst: lambda r: verify_fn(inst, r))()

            for scaff_name, scaff_fn, kwargs in SCAFFOLDS:
                # Pre-call memory check
                try:
                    memsafe.require_free(args.min_free_gb,
                                         label=f"{args.case} #{i} {scaff_name}")
                except MemoryError as e:
                    print(f"!! ABORT: {e}")
                    raise

                kw = dict(kwargs)
                if scaff_name == "verifier_guided":
                    kw["verifier"] = local_verify

                t0 = time.time()
                resp, err = _safe(scaff_fn, model, p, img, **kw)
                elapsed = time.time() - t0
                if err:
                    correct, valid = False, False
                    head = f"<ERROR> {err.splitlines()[-1][:120]}"
                else:
                    correct = bool(local_verify(resp))
                    valid = bool(parseable_fn(resp))
                    head = resp[:200].replace("\n", " ⏎ ")

                rows.append([args.case, i, scaff_name, int(correct),
                             int(valid), round(elapsed, 2), head])
                # Flush incrementally so we have data even on abort
                with open(out_csv, "w", newline="") as _f:
                    _w = csv.writer(_f)
                    _w.writerow(["case", "instance", "scaffold", "correct",
                                 "valid", "seconds", "response_head"])
                    _w.writerows(rows)
                print(f"{args.case:10s} #{i:02d} [{scaff_name:16s}] "
                      f"corr={str(correct):5s} valid={str(valid):5s} {elapsed:6.1f}s")
                # Light mid-case check (use available — inactive is reclaimable cache)
                if memsafe.available_gb() < args.min_free_gb:
                    print(f"!! mid-case mem warning: {memsafe.status()}")
                # Hard abort: genuine pressure (available < 1 GB means cache reclaim can't save us)
                if memsafe.available_gb() < 1.0:
                    print(f"!! CRITICAL: available RAM dropped to "
                          f"{memsafe.available_gb():.2f} GB — aborting case")
                    raise MemoryError("available RAM critical")
    finally:
        # Always write what we have and unload
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case", "instance", "scaffold", "correct",
                        "valid", "seconds", "response_head"])
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows → {out_csv}")
        print(memsafe.status("before stop"))
        memsafe.cooldown_and_verify(args.model, min_free_gb=5.0)
        print(memsafe.status("after stop"))


if __name__ == "__main__":
    main()
