"""Collect per-attempt data for the README animation.

Re-runs ONE maze instance under the exact experiment conditions
(llama3.2:1b, 5x5 maze, temperature 0.8, num_ctx 1024, verifier-guided
rejection sampling) and logs EVERY attempt: full response text, parsed
moves, verifier verdict. The K-sweep CSVs only keep the final response,
so the animation needs this fresh trace.

Tries candidate seeds in order; keeps the first run whose accept lands
between attempts MIN_DRAMA and MAX_DRAMA (enough rejections to show the
scaffold working, not so many the GIF drags). Writes
data/animation_attempts.json.
"""
from __future__ import annotations
import json

from models import Moondream
from tasks import maze
import memsafe

MODEL = "llama3.2:1b"
SEEDS = [17, 5, 12, 19, 25]   # historically correct at K=64 with 40-115s wall-clock
K_MAX = 64
MIN_DRAMA = 4
MAX_DRAMA = 45
MIN_AVAILABLE_GB = 2.0


def run_instance(model: Moondream, seed: int) -> dict:
    inst = maze.generate(seed=seed, size=2)
    prompt = maze.prompt_text(inst)
    attempts = []
    for k in range(1, K_MAX + 1):
        ag = memsafe.available_gb()
        if ag < MIN_AVAILABLE_GB:
            raise MemoryError(f"available {ag:.2f} GB < floor {MIN_AVAILABLE_GB}")
        resp = model.query(prompt, image=None, temperature=0.8)
        moves = maze._parse_moves(resp)
        ok = maze.verify(inst, resp)
        attempts.append({"k": k, "response": resp, "moves": moves, "accepted": ok})
        print(f"seed={seed} attempt {k:2d}: {'ACCEPT' if ok else 'reject'} "
              f"moves={''.join(moves)[:20]} avail={ag:.1f}GB")
        if ok:
            break
    return {
        "model": MODEL,
        "seed": seed,
        "grid": inst.grid,
        "start": list(inst.start),
        "goal": list(inst.goal),
        "ascii": maze.ascii_grid(inst),
        "prompt": prompt,
        "attempts": attempts,
    }


def main() -> None:
    print(memsafe.status("preflight"))
    model = Moondream(model_name=MODEL, num_ctx=1024, keep_alive="60s")
    kept = None
    try:
        for seed in SEEDS:
            trace = run_instance(model, seed)
            n = len(trace["attempts"])
            accepted = trace["attempts"][-1]["accepted"]
            print(f"--- seed={seed}: {n} attempts, accepted={accepted}")
            if accepted and MIN_DRAMA <= n <= MAX_DRAMA:
                kept = trace
                break
            if kept is None and accepted:
                kept = trace  # fallback: any success beats none
    finally:
        memsafe.cooldown_and_verify(MODEL, min_free_gb=3.0)
        print(memsafe.status("after cooldown"))

    if kept is None:
        raise SystemExit("No accepted run collected — nothing written.")
    with open("data/animation_attempts.json", "w") as f:
        json.dump(kept, f, indent=1)
    print(f"Wrote data/animation_attempts.json "
          f"(seed={kept['seed']}, {len(kept['attempts'])} attempts)")


if __name__ == "__main__":
    main()
