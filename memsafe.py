"""Memory-safety helpers for running models on a 16 GB Mac.

Two checks:
  - free_gb()    : current free RAM (pages_free * page_size)
  - require_free(min_gb): assert at least `min_gb` of headroom; raise if not

Plus a small helper to unload a model and wait for ollama to release memory.
"""
from __future__ import annotations
import subprocess
import time

PAGE_SIZE = 16384  # Apple Silicon page size

# Hard floor — never start a new model call if free < this
DEFAULT_MIN_FREE_GB = 4.0


def _vm_stat() -> dict[str, int]:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    pages: dict[str, int] = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        v = v.strip().rstrip(".")
        if v.isdigit():
            pages[k.strip()] = int(v)
    return pages


def free_gb() -> float:
    pages = _vm_stat()
    free = pages.get("Pages free", 0) + pages.get("Pages speculative", 0)
    return free * PAGE_SIZE / (1024**3)


def available_gb() -> float:
    """Free + inactive (inactive can be reclaimed under pressure)."""
    pages = _vm_stat()
    avail = (
        pages.get("Pages free", 0)
        + pages.get("Pages speculative", 0)
        + pages.get("Pages inactive", 0)
    )
    return avail * PAGE_SIZE / (1024**3)


def loaded_models() -> list[tuple[str, str]]:
    """List currently-loaded ollama models as (name, size_str)."""
    out = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 3 and not parts[0].startswith("\x1b"):
            rows.append((parts[0], parts[2] + " " + parts[3] if len(parts) > 3 else parts[2]))
    return rows


def stop_model(name: str) -> None:
    subprocess.run(["ollama", "stop", name], capture_output=True)


def require_free(min_gb: float = DEFAULT_MIN_FREE_GB, label: str = "") -> None:
    fg = free_gb()
    ag = available_gb()
    if ag < min_gb:
        raise MemoryError(
            f"Memory safety abort [{label}]: free={fg:.2f}GB available={ag:.2f}GB "
            f"< floor={min_gb:.1f}GB"
        )


def status(label: str = "") -> str:
    fg = free_gb()
    ag = available_gb()
    loaded = loaded_models()
    loaded_str = ", ".join(f"{n}({s})" for n, s in loaded) if loaded else "none"
    return f"[mem {label}] free={fg:.2f}GB avail={ag:.2f}GB loaded=[{loaded_str}]"


def cooldown_and_verify(model_name: str, min_free_gb: float = 6.0,
                        max_wait_s: float = 30.0) -> None:
    """Stop the model and wait for the system to reclaim memory."""
    stop_model(model_name)
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        if free_gb() >= min_free_gb:
            return
        time.sleep(2)
