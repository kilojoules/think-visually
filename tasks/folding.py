"""Multi-fold paper-folding instance with a fair physics-based verifier.

Paper is a 4x4 grid of cells. A fold is one of:
  - 'H': fold the top half (rows 0..1) DOWN onto the bottom half (rows 2..3)
  - 'V': fold the left half (cols 0..1) RIGHT onto the right half (cols 2..3)
After a sequence of folds, a single hole is punched on the folded paper, going
through all stacked layers. The task: predict where the holes appear when fully
unfolded.

The verifier independently re-computes the physics from the inputs
(fold_seq, punch_pos) and compares the predicted set to the simulated set.
No leakage of a pre-computed answer — the simulation IS the verifier.
"""
from __future__ import annotations
import random
import re
from dataclasses import dataclass, field
from PIL import Image, ImageDraw

GRID = 4  # 4x4 cells


@dataclass
class FoldInstance:
    fold_seq: list[str]               # e.g. ['H'] or ['H', 'V']
    punch_pos: tuple[int, int]        # (row, col) on the FOLDED paper (must be in visible region)
    seed: int
    n_folds: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_folds = len(self.fold_seq)


# ---------- Physics ----------

def _apply_fold_to_pos(pos: tuple[int, int], axis: str, grid: int = GRID) -> tuple[int, int]:
    """Map an original cell position through ONE fold."""
    r, c = pos
    if axis == "H":
        return (grid - 1 - r, c) if r < grid / 2 else (r, c)
    else:  # 'V'
        return (r, grid - 1 - c) if c < grid / 2 else (r, c)


def map_through_folds(pos: tuple[int, int], fold_seq: list[str], grid: int = GRID) -> tuple[int, int]:
    """Compose all folds in order."""
    p = pos
    for axis in fold_seq:
        p = _apply_fold_to_pos(p, axis, grid)
    return p


def visible_region(fold_seq: list[str], grid: int = GRID) -> set[tuple[int, int]]:
    """Cells that remain on top after the sequence of folds."""
    cells = {(r, c) for r in range(grid) for c in range(grid)}
    for axis in fold_seq:
        if axis == "H":
            cells = {(r, c) for (r, c) in cells if r >= grid / 2}
        else:
            cells = {(r, c) for (r, c) in cells if c >= grid / 2}
    return cells


def simulated_holes(inst: FoldInstance, grid: int = GRID) -> set[tuple[int, int]]:
    """Independently simulate: which original cells end up stacked under the punched position?"""
    return {
        (r, c)
        for r in range(grid)
        for c in range(grid)
        if map_through_folds((r, c), inst.fold_seq, grid) == inst.punch_pos
    }


# ---------- Instance generation ----------

def generate(seed: int | None = None, n_folds: int = 1) -> FoldInstance:
    """Generate an instance. For n_folds>=2 we force alternating axes — folding
    twice on the same axis is a no-op on a 4x4 grid and produces degenerate
    cases."""
    if seed is None:
        seed = random.randint(0, 10**9)
    rng = random.Random(seed)
    if n_folds == 1:
        fold_seq = [rng.choice(["H", "V"])]
    else:
        # Alternate axes: start with either H or V and alternate strictly.
        start = rng.choice(["H", "V"])
        fold_seq = [start if i % 2 == 0 else ("V" if start == "H" else "H")
                    for i in range(n_folds)]
    visible = sorted(visible_region(fold_seq))
    punch_pos = rng.choice(visible)
    return FoldInstance(fold_seq=fold_seq, punch_pos=punch_pos, seed=seed)


# ---------- Rendering ----------

def render(inst: FoldInstance, cell_px: int = 60) -> Image.Image:
    """Render the visible (folded) region with the punched hole as a black dot."""
    visible = visible_region(inst.fold_seq)
    rs = sorted({r for (r, _) in visible})
    cs = sorted({c for (_, c) in visible})
    n_rows, n_cols = len(rs), len(cs)
    img = Image.new("RGB", (n_cols * cell_px, n_rows * cell_px), "white")
    draw = ImageDraw.Draw(img)
    # Grid lines
    for i in range(n_rows + 1):
        draw.line([(0, i * cell_px), (n_cols * cell_px, i * cell_px)], fill="black", width=2)
    for j in range(n_cols + 1):
        draw.line([(j * cell_px, 0), (j * cell_px, n_rows * cell_px)], fill="black", width=2)
    # Hole
    pr, pc = inst.punch_pos
    local_r = rs.index(pr)
    local_c = cs.index(pc)
    cx = local_c * cell_px + cell_px // 2
    cy = local_r * cell_px + cell_px // 2
    rad = cell_px // 4
    draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill="black")
    return img


# ---------- Prompt ----------

_FOLD_DESC = {
    "H": "fold the top half (rows 0–1) DOWN onto the bottom half (rows 2–3)",
    "V": "fold the left half (cols 0–1) RIGHT onto the right half (cols 2–3)",
}


def _folds_text(inst: FoldInstance) -> str:
    if inst.n_folds == 1:
        return f"Step 1: {_FOLD_DESC[inst.fold_seq[0]]}."
    return "\n".join(
        f"Step {i+1}: {_FOLD_DESC[ax]}." for i, ax in enumerate(inst.fold_seq)
    )


def prompt(inst: FoldInstance) -> str:
    """Prompt that pairs with an image of the folded paper."""
    pr, pc = inst.punch_pos
    return (
        f"A 4x4 square paper is folded in {inst.n_folds} step(s):\n{_folds_text(inst)}\n"
        f"Then a single hole is punched at (row={pr}, col={pc}) on the folded paper, "
        f"going through ALL stacked layers. The image shows the folded paper with the hole.\n"
        f"When the paper is fully unfolded back to 4x4, where are the holes? "
        f"Rows are 0..3 top-to-bottom, columns are 0..3 left-to-right.\n"
        f"Respond with ONLY the (row, col) coordinates separated by semicolons, "
        f"nothing else. Example: (1,2);(2,2)"
    )


def prompt_text(inst: FoldInstance) -> str:
    """Text-only prompt (no image needed — punch position is symbolic)."""
    pr, pc = inst.punch_pos
    return (
        f"A 4x4 square paper is folded in {inst.n_folds} step(s):\n{_folds_text(inst)}\n"
        f"Then a single hole is punched at (row={pr}, col={pc}) on the folded paper, "
        f"going through ALL stacked layers.\n"
        f"When the paper is fully unfolded back to 4x4, where are the holes? "
        f"Rows are 0..3 top-to-bottom, columns are 0..3 left-to-right.\n"
        f"Respond with ONLY the (row, col) coordinates separated by semicolons, "
        f"nothing else. Example: (1,2);(2,2)"
    )


# ---------- Verification ----------

def parse_holes(response: str) -> set[tuple[int, int]]:
    """Pull (r,c) coordinate pairs from the response. Accepts (r,c), [r,c],
    and r,c forms. Filters to valid 0..GRID-1 range."""
    pred: set[tuple[int, int]] = set()
    # Bracketed form first (most reliable)
    for m in re.findall(r"[\(\[]\s*(\d+)\s*,\s*(\d+)\s*[\)\]]", response):
        r, c = int(m[0]), int(m[1])
        if 0 <= r < GRID and 0 <= c < GRID:
            pred.add((r, c))
    if pred:
        return pred
    # Fallback: bare 'd,d' pairs separated by semicolons
    for chunk in response.split(";"):
        m = re.search(r"(\d+)\s*,\s*(\d+)", chunk)
        if m:
            r, c = int(m.group(1)), int(m.group(2))
            if 0 <= r < GRID and 0 <= c < GRID:
                pred.add((r, c))
    return pred


def is_parseable(response: str) -> bool:
    """Did the model produce *any* in-range (r,c) coordinate?"""
    return len(parse_holes(response)) > 0


def verify(inst: FoldInstance, response: str) -> bool:
    """Fair physics verifier: independently simulates the fold sequence
    starting from the proposed unfolded holes, and checks that they all
    stack at the punched position with no missing/extra cells."""
    pred = parse_holes(response)
    expected = simulated_holes(inst)
    return pred == expected


def partial_score(inst: FoldInstance, response: str) -> float:
    """Soft-verifier score: +1 per correctly-identified hole, -0.5 per spurious
    extra hole. Used by the best_partial scaffold (PTRM-style soft selection)."""
    pred = parse_holes(response)
    expected = simulated_holes(inst)
    hits = len(pred & expected)
    extras = len(pred - expected)
    return float(hits) - 0.5 * float(extras)
