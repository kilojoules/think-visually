"""Four inference-time scaffolds.

Each returns ONE string response (the final answer to score). The Whiteboard
scaffold also returns auxiliary info via a closure if needed; the simple
interface keeps run.py uniform.
"""
from __future__ import annotations
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from typing import Callable


# ---------- 0. Bare ----------

def bare(model, prompt: str, image, **_) -> str:
    return model.query(prompt, image=image, temperature=0.0)


# ---------- 1. Self-consistency ----------

def self_consistency(model, prompt: str, image, n: int = 4, **_) -> str:
    samples = [model.query(prompt, image=image, temperature=0.7) for _ in range(n)]
    return Counter(samples).most_common(1)[0][0]


# ---------- 2. Verifier-guided rejection ----------

def verifier_guided(
    model,
    prompt: str,
    image,
    verifier: Callable[[str], bool],
    n_max: int = 8,
    **_,
) -> str:
    last = ""
    for _ in range(n_max):
        resp = model.query(prompt, image=image, temperature=0.8)
        if verifier(resp):
            return resp
        last = resp
    return last


# ---------- 2b. Best-partial: PTRM-style soft scoring ----------

def best_partial(
    model,
    prompt: str,
    image,
    partial_score: Callable[[str], float],
    n: int = 16,
    **_,
) -> str:
    """Generate n samples; return the one with the highest partial-credit score.

    This is the soft-verifier analog of verifier_guided. Instead of accepting
    the first all-correct response, score every sample by a continuous quality
    signal and return the argmax. Mirrors PTRM's best-Q@K selection — except
    our 'Q' is a rule-based partial-correctness count rather than a learned
    head.
    """
    best_resp = ""
    best_score = float("-inf")
    for _ in range(n):
        resp = model.query(prompt, image=image, temperature=0.8)
        s = partial_score(resp)
        if s > best_score:
            best_score, best_resp = s, resp
    return best_resp


# ---------- 3. Whiteboard-of-Thought ----------

_CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str | None:
    m = _CODE_RE.search(text)
    return m.group(1).strip() if m else None


def _exec_python(code: str, timeout: float = 5.0) -> str:
    """Run code in a subprocess; return stdout (+ stderr if any). Truncated to 1500 chars."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = result.stdout
        if result.stderr:
            out += "\n[stderr]\n" + result.stderr
    except subprocess.TimeoutExpired:
        out = "[TIMEOUT after 5s]"
    finally:
        os.unlink(path)
    return out[:1500]


_WB_PREAMBLE = (
    "\n\nYou may write a complete Python program to solve this. "
    "Wrap it in a single ```python ... ``` block. "
    "IMPORTANT: embed any input data (the maze grid, the fold sequence, the "
    "punch position) directly in the code as Python literals — the program "
    "runs in a fresh interpreter with no globals. "
    "Use `print(...)` to output your answer in the exact required format. "
    "Available modules: math, itertools, collections, re, json. "
    "After the code's stdout is returned, your next message should be ONLY "
    "the final answer in the required format, no code, no prose."
)


def whiteboard(model, prompt: str, image, max_rounds: int = 3, **_) -> str:
    """Model writes Python; we execute; loop up to max_rounds; then force final answer."""
    transcript: list[str] = [prompt + _WB_PREAMBLE]

    for round_i in range(max_rounds):
        resp = model.query(
            "\n\n".join(transcript), image=image,
            temperature=0.0 if round_i == 0 else 0.4,
            max_tokens=1024,
        )
        code = _extract_code(resp)
        if not code:
            # No code → treat as the answer
            return resp
        stdout = _exec_python(code)
        # If stdout already looks like a clean answer, accept it
        if stdout.strip() and "[stderr]" not in stdout and "[TIMEOUT" not in stdout:
            # Heuristic: if the printed output is short and parseable-looking, return it
            if len(stdout) < 400:
                return stdout.strip()
        transcript.append(
            f"[Round {round_i + 1} your code's stdout]:\n{stdout}\n\n"
            f"If the output is correct, repeat it as your FINAL ANSWER. "
            f"If incorrect, write a new improved ```python``` block."
        )

    transcript.append(
        "Final round — give your FINAL ANSWER in the required format, no code, "
        "no prose."
    )
    return model.query("\n\n".join(transcript), image=image, temperature=0.0, max_tokens=256)
