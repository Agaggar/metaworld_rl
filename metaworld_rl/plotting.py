"""Local matplotlib plots from training history."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def plot_history(history: list[dict[str, float]], out_dir: Path, prefix: str = "") -> None:
    """Save reward/loss-style curves from list of metric dicts (must include 'step')."""
    if not history:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = [k for k in history[0].keys() if k != "step"]
    if not keys:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    steps = [h["step"] for h in history]
    for k in keys:
        vals = [h.get(k, float("nan")) for h in history]
        ax.plot(steps, vals, label=k, alpha=0.85)
    ax.set_xlabel("step")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / f"{prefix}_curves.png" if prefix else out_dir / "curves.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)


def plot_from_csv(csv_path: str | Path, out_path: str | Path | None = None) -> None:
    """Convenience: load CSV history and plot."""
    import csv

    csv_path = Path(csv_path)
    rows: list[dict[str, Any]] = []
    with csv_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: float(v) for k, v in row.items() if v not in ("",)})
    if not rows:
        return
    out = Path(out_path) if out_path else csv_path.parent / "eval_curves.png"
    plot_history(rows, out.parent, prefix=out.stem)
