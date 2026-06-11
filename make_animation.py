"""Render figures/maze_thinking.gif from data/animation_attempts.json.

Split-pane animation of one real verifier-guided run:
  left  — what the model actually sees and emits (ASCII maze in, text out)
  right — the verifier's rendering of each proposed path (the model never
          sees this picture; that asymmetry is the point)

Pure Pillow, no model calls, no matplotlib. Peak memory ~30 MB.
"""
from __future__ import annotations
import json
import textwrap
from PIL import Image, ImageDraw, ImageFont

W, H = 880, 540
LEFT_X, LEFT_W = 20, 490
RIGHT_X = 540
CELL = 56
MAZE_XY = (RIGHT_X + 10, 120)

INK = "#1a1a1a"
MUTED = "#6b6b6b"
PENDING = "#e8920c"
REJECT = "#c62828"
ACCEPT = "#2e7d32"
WALL = "#222222"
S_FILL = "#b3e5fc"
G_FILL = "#ffe082"
BG = "#fafafa"
PANE = "#ffffff"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", size,
                                  index=1 if bold else 0)
    except OSError:
        return ImageFont.load_default()


def walk(grid, start, goal, moves):
    """Walk moves exactly as the verifier does. Returns (visited cells,
    status, detail). Stops at goal arrival or first violation."""
    deltas = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
    n = len(grid)
    r, c = start
    visited = [(r, c)]
    if not moves:
        return visited, "reject", "no parseable moves"
    for m in moves:
        dr, dc = deltas[m]
        r, c = r + dr, c + dc
        if not (0 <= r < n and 0 <= c < n):
            return visited, "reject", "walks off the grid"
        visited.append((r, c))
        if grid[r][c] == 1:
            return visited, "reject", "walks into a wall"
        if (r, c) == tuple(goal):
            return visited, "accept", "reaches G on open cells"
    return visited, "reject", "stops short of G"


def draw_maze(d: ImageDraw.ImageDraw, grid, start, goal):
    x0, y0 = MAZE_XY
    n = len(grid)
    for r in range(n):
        for c in range(n):
            box = [x0 + c * CELL, y0 + r * CELL,
                   x0 + (c + 1) * CELL, y0 + (r + 1) * CELL]
            if grid[r][c] == 1:
                d.rectangle(box, fill=WALL)
            else:
                d.rectangle(box, fill="white", outline="#dddddd")
    for (rr, cc), fill, label in [(start, S_FILL, "S"), (goal, G_FILL, "G")]:
        box = [x0 + cc * CELL, y0 + rr * CELL,
               x0 + (cc + 1) * CELL, y0 + (rr + 1) * CELL]
        d.rectangle(box, fill=fill, outline="#dddddd")
        d.text((box[0] + CELL // 2, box[1] + CELL // 2), label,
               font=font(22, bold=True), fill=INK, anchor="mm")


def draw_path(d: ImageDraw.ImageDraw, visited, color, violated: bool):
    x0, y0 = MAZE_XY
    centers = [(x0 + c * CELL + CELL // 2, y0 + r * CELL + CELL // 2)
               for r, c in visited]
    if len(centers) >= 2:
        d.line(centers, fill=color, width=6, joint="curve")
    cx, cy = centers[0]
    d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=color)
    ex, ey = centers[-1]
    if violated:
        d.line([ex - 11, ey - 11, ex + 11, ey + 11], fill=color, width=6)
        d.line([ex - 11, ey + 11, ex + 11, ey - 11], fill=color, width=6)
    else:
        d.ellipse([ex - 9, ey - 9, ex + 9, ey + 9], outline=color, width=4)


def wrap_response(resp: str, max_lines: int = 9) -> list[str]:
    lines: list[str] = []
    for para in resp.split("\n"):
        wrapped = textwrap.wrap(para, width=56) or [""]
        lines.extend(wrapped)
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"… ({len(resp)} chars total)"]
    return lines


def frame(data, attempt, show_verdict: bool, k_max: int = 64) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([10, 52, LEFT_X + LEFT_W, H - 44], fill=PANE, outline="#e0e0e0")
    d.rectangle([RIGHT_X - 6, 52, W - 10, H - 44], fill=PANE, outline="#e0e0e0")

    d.text((20, 14), "llama3.2:1b — verifier-guided rejection sampling",
           font=font(17, bold=True), fill=INK)
    d.text((W - 20, 14), f"attempt {attempt['k']}/{k_max}",
           font=font(17, bold=True), fill=MUTED, anchor="ra")

    y = 64
    d.text((LEFT_X, y), "MODEL INPUT (text only)", font=font(13, bold=True),
           fill=MUTED)
    y += 22
    for line in data["ascii"].split("\n"):
        d.text((LEFT_X + 8, y), line, font=font(16), fill=INK)
        y += 19
    d.text((LEFT_X + 8, y + 2),
           '"Output a path from S to G as moves: U, D, L, R."',
           font=font(12), fill=MUTED)
    y += 30

    d.text((LEFT_X, y), f"MODEL OUTPUT — attempt {attempt['k']}",
           font=font(13, bold=True), fill=MUTED)
    y += 22
    for line in wrap_response(attempt["response"]):
        d.text((LEFT_X + 8, y), line, font=font(12), fill=INK)
        y += 17

    d.text((RIGHT_X + 10, 64), "VERIFIER VIEW", font=font(13, bold=True),
           fill=MUTED)
    d.text((RIGHT_X + 10, 84), "(the model never sees this picture)",
           font=font(11), fill=MUTED)
    draw_maze(d, data["grid"], data["start"], data["goal"])

    visited, status, detail = walk(data["grid"], data["start"], data["goal"],
                                   attempt["moves"])
    if show_verdict:
        color = ACCEPT if status == "accept" else REJECT
    else:
        color = PENDING
    draw_path(d, visited, color, violated=(status == "reject" and len(visited) > 1
                                           and detail != "stops short of G"))

    # Redraw S/G labels on top so the path never hides them
    x0, y0 = MAZE_XY
    for (rr, cc), label in [(data["start"], "S"), (data["goal"], "G")]:
        d.text((x0 + cc * CELL + CELL // 2, y0 + rr * CELL + CELL // 2),
               label, font=font(22, bold=True), fill=INK, anchor="mm")

    sy = MAZE_XY[1] + len(data["grid"]) * CELL + 18
    if show_verdict:
        label = "ACCEPTED" if status == "accept" else "REJECTED"
        tw = d.textlength(label, font=font(18, bold=True))
        d.rectangle([RIGHT_X + 10, sy, RIGHT_X + 30 + tw, sy + 30],
                    fill=color)
        d.text((RIGHT_X + 20, sy + 15), label, font=font(18, bold=True),
               fill="white", anchor="lm")
        d.text((RIGHT_X + 10, sy + 38), detail, font=font(12), fill=color)
    else:
        d.text((RIGHT_X + 10, sy + 8), "verifying…", font=font(14), fill=PENDING)

    d.text((W // 2, H - 22),
           "The model thinks in text tokens only — the right pane is our "
           "rendering of its proposed path.",
           font=font(12), fill=MUTED, anchor="mm")
    return img


def main() -> None:
    with open("data/animation_attempts.json") as f:
        data = json.load(f)

    frames: list[Image.Image] = []
    durations: list[int] = []
    for attempt in data["attempts"]:
        frames.append(frame(data, attempt, show_verdict=False))
        durations.append(800)
        frames.append(frame(data, attempt, show_verdict=True))
        durations.append(1100 if not attempt["accepted"] else 4000)

    frames = [f.quantize(colors=64, dither=Image.Dither.NONE) for f in frames]
    out = "figures/maze_thinking.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    print(f"Wrote {out}: {len(frames)} frames")

    frame(data, data["attempts"][-1], show_verdict=True).save(
        "/tmp/anim_preview_accept.png")
    frame(data, data["attempts"][0], show_verdict=True).save(
        "/tmp/anim_preview_reject.png")
    print("QA previews in /tmp/")


if __name__ == "__main__":
    main()
