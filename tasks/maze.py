"""Procedurally generated small maze.

Render: white open cells, black walls, green Start, red Goal.
Verifier: walk the proposed moves; ensure each step lands on an open cell
and reaches the Goal. Pure physics — no ground-truth leakage.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from PIL import Image, ImageDraw


@dataclass
class MazeInstance:
    grid: list[list[int]]   # 1 = wall, 0 = open
    start: tuple[int, int]  # (row, col)
    goal: tuple[int, int]
    seed: int


def generate(size: int = 5, seed: int | None = None) -> MazeInstance:
    """Randomized DFS maze on a (2*size+1) x (2*size+1) grid."""
    if seed is None:
        seed = random.randint(0, 10**9)
    rng = random.Random(seed)
    n = size * 2 + 1
    grid = [[1] * n for _ in range(n)]

    def carve(r: int, c: int) -> None:
        grid[r][c] = 0
        dirs = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        rng.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 < nr < n and 0 < nc < n and grid[nr][nc] == 1:
                grid[r + dr // 2][c + dc // 2] = 0
                carve(nr, nc)

    carve(1, 1)
    return MazeInstance(grid=grid, start=(1, 1), goal=(n - 2, n - 2), seed=seed)


def render(maze: MazeInstance, cell_px: int = 36) -> Image.Image:
    n = len(maze.grid)
    img = Image.new("RGB", (n * cell_px, n * cell_px), "white")
    draw = ImageDraw.Draw(img)
    for r in range(n):
        for c in range(n):
            if maze.grid[r][c] == 1:
                draw.rectangle(
                    [c * cell_px, r * cell_px, (c + 1) * cell_px, (r + 1) * cell_px],
                    fill="black",
                )
    sr, sc = maze.start
    gr, gc = maze.goal
    draw.rectangle(
        [sc * cell_px, sr * cell_px, (sc + 1) * cell_px, (sr + 1) * cell_px],
        fill="green",
    )
    draw.rectangle(
        [gc * cell_px, gr * cell_px, (gc + 1) * cell_px, (gr + 1) * cell_px],
        fill="red",
    )
    return img


def prompt(maze: MazeInstance) -> str:
    n = len(maze.grid)
    return (
        f"This image shows a {n}x{n} maze. White cells are open, black cells are walls, "
        f"the green cell is the Start, the red cell is the Goal. "
        f"Output a path from Start to Goal as a comma-separated list of single-letter moves: "
        f"U (up), D (down), L (left), R (right). "
        f"Respond with ONLY the move list on one line. Example: D,D,R,R,U,R"
    )


def ascii_grid(maze: MazeInstance) -> str:
    """ASCII rendering: '#'=wall, '.'=open, 'S'=start, 'G'=goal."""
    n = len(maze.grid)
    lines = []
    for r in range(n):
        chars = []
        for c in range(n):
            if (r, c) == maze.start:
                chars.append("S")
            elif (r, c) == maze.goal:
                chars.append("G")
            elif maze.grid[r][c] == 1:
                chars.append("#")
            else:
                chars.append(".")
        lines.append("".join(chars))
    return "\n".join(lines)


def prompt_text(maze: MazeInstance) -> str:
    """Text-only prompt (no image)."""
    n = len(maze.grid)
    return (
        f"Below is a {n}x{n} maze. '#' is a wall, '.' is open, 'S' is the Start, "
        f"'G' is the Goal.\n\n"
        f"{ascii_grid(maze)}\n\n"
        f"Output a path from S to G as a comma-separated list of single-letter moves: "
        f"U (up), D (down), L (left), R (right). "
        f"Respond with ONLY the move list on one line. Example: D,D,R,R,U,R"
    )


def _parse_moves(response: str) -> list[str]:
    """Robust parse: pull standalone U/D/L/R tokens. Try comma/whitespace split
    first; if none found, fall back to per-character scan to recover paths like
    'UDDRR' written without separators."""
    moves: list[str] = []
    for tok in response.replace("\n", ",").replace(" ", ",").split(","):
        tok = tok.strip().upper()
        if tok in ("U", "D", "L", "R"):
            moves.append(tok)
    if not moves:
        # Char-level fallback for compact answers
        for ch in response.upper():
            if ch in ("U", "D", "L", "R"):
                moves.append(ch)
    return moves


def is_parseable(response: str) -> bool:
    """Did the model produce *any* move-like token? Distinguishes 'no answer'
    from 'wrong answer'."""
    return len(_parse_moves(response)) > 0


def verify(maze: MazeInstance, response: str) -> bool:
    """Pure physics check: do these moves traverse open cells and reach the goal?"""
    moves = _parse_moves(response)
    if not moves:
        return False
    r, c = maze.start
    n = len(maze.grid)
    deltas = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
    for m in moves:
        dr, dc = deltas[m]
        r, c = r + dr, c + dc
        if not (0 <= r < n and 0 <= c < n):
            return False
        if maze.grid[r][c] == 1:
            return False
        if (r, c) == maze.goal:
            return True
    return (r, c) == maze.goal
