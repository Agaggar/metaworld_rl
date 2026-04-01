#!/usr/bin/env python3
"""
Create a 1x5 matplotlib heatmap grid for SAC ablations from W&B runs.

Each subplot corresponds to one environment:
- x-axis: sample_every
- y-axis: action_scale
- color: final reward score (last value of eval metric)

Uses only matplotlib for plotting.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb
from tqdm import tqdm

SAMPLE_EVERY_VALUES = [1, 2, 5, 10]
ACTION_SCALE_VALUES = [0.1, 0.25, 1.0, 5.0]

# Env ids as stored in metaworld_rl.config.name_to_env_name(...)
ENV_COLS = [
    ("button-press-v3", "button"),
    ("door-open-v3", "door"),
    ("drawer-open-v3", "drawer"),
    ("coffee-button-v3", "coffee"),
    ("faucet-open-v3", "faucet"),
]


def _float_eq(a: float, b: float, tol: float = 1e-8) -> bool:
    return abs(float(a) - float(b)) <= tol


def _snap_action_scale(x: Any) -> float | None:
    try:
        xf = float(x)
    except Exception:
        return None
    for a in ACTION_SCALE_VALUES:
        if _float_eq(xf, a, tol=1e-6):
            return a
    return None


def _get_nested(config: dict[str, Any], path: str) -> Any:
    """
    Best-effort nested getter for W&B config.

    Supports both:
    - top-level config["sample_every"]
    - nested dict: config["env"]["frame_skip"] (legacy fallback)
    - flattened keys: config["env.frame_skip"] (legacy fallback)
    """
    if "." in path:
        first, rest = path.split(".", 1)
        if first in config and isinstance(config[first], dict):
            cur: Any = config[first]
            for part in rest.split("."):
                if not isinstance(cur, dict) or part not in cur:
                    return None
                cur = cur[part]
            return cur
        return config.get(path)
    return config.get(path)


def _get_cfg(run: wandb.apis.public.Run, key: str) -> Any:
    try:
        cfg = dict(run.config)
    except Exception:
        return None
    return _get_nested(cfg, key)


def _scan_last_metric(run: wandb.apis.public.Run, metric_key: str) -> float | None:
    """Return the latest available metric value from run history."""
    rows = run.history(keys=[metric_key], samples=500)
    last_step = None
    last_val = None

    for _, row in rows.iterrows():
        val = row.get(metric_key)
        step = row.get("_step")
        if val is None or step is None:
            continue
        try:
            step_int = int(step)
            val_f = float(val)
        except Exception:
            continue
        if last_step is None or step_int >= last_step:
            last_step = step_int
            last_val = val_f

    return last_val


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--entity", type=str, default=None, help="W&B entity/user/org (required if not inferrable).")
    p.add_argument("--project", type=str, default="se_as_sweep")
    p.add_argument("--metric", type=str, default="eval/episode_return_mean")
    p.add_argument("--out", type=str, default=str("visualizations/heatmap_se_as_sweep.png"))
    p.add_argument(
        "--max-runs",
        type=int,
        default=200,
        help="Safety cap for scan size (will still only include SAC runs in the expected ablation grid).",
    )
    args = p.parse_args()

    api = wandb.Api()
    if args.entity:
        runs = api.runs(f"{args.entity}/{args.project}")
    else:
        runs = api.runs(args.project)

    env_names = {e for e, _ in ENV_COLS}

    # Group runs by (env_id, sample_every, action_scale)
    run_groups: dict[tuple[str, int, float], list[wandb.apis.public.Run]] = defaultdict(list)
    n = 0
    for run in tqdm(runs):
        n += 1
        if n > args.max_runs:
            break

        cfg_algo = _get_cfg(run, "algorithm")
        if cfg_algo != "sac":
            continue

        env_id = _get_cfg(run, "env.benchmark")
        if env_id not in env_names:
            continue

        fs = _get_cfg(run, "sample_every")
        if fs is None:
            fs = _get_cfg(run, "env.frame_skip")
        ascale = _get_cfg(run, "env.action_scale")

        try:
            fs_int = int(fs)
        except Exception:
            continue

        asnap = _snap_action_scale(ascale)
        if asnap is None:
            continue
        if fs_int not in SAMPLE_EVERY_VALUES or asnap not in ACTION_SCALE_VALUES:
            continue

        run_groups[(env_id, fs_int, asnap)].append(run)

    # Mean final reward per cell across matching runs.
    final_scores: dict[tuple[str, int, float], float] = {}
    for key, grouped_runs in run_groups.items():
        vals = []
        for run in grouped_runs:
            last_val = _scan_last_metric(run, args.metric)
            if last_val is not None:
                vals.append(last_val)
        if vals:
            final_scores[key] = float(sum(vals) / len(vals))

    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(24, 4.8), constrained_layout=True)

    cmap = plt.get_cmap("coolwarm").copy()
    # Design choice: show missing (sample_every, action_scale) combinations in light gray.
    cmap.set_bad(color="#e9e9e9")

    for i, (env_id, env_short) in tqdm(enumerate(ENV_COLS), desc="Columns"):
        ax = axes[i]
        grid = np.full((len(ACTION_SCALE_VALUES), len(SAMPLE_EVERY_VALUES)), np.nan, dtype=float)

        for y, action_scale in enumerate(ACTION_SCALE_VALUES):
            for x, sample_every in enumerate(SAMPLE_EVERY_VALUES):
                key = (env_id, sample_every, action_scale)
                if key in final_scores:
                    grid[y, x] = final_scores[key]

        valid_vals = grid[~np.isnan(grid)]
        if valid_vals.size > 0:
            vmin = float(np.min(valid_vals))
            vmax = float(np.max(valid_vals))
            if _float_eq(vmin, vmax):
                vmax = vmin + 1e-6

            best_y, best_x = np.unravel_index(np.nanargmax(grid), grid.shape)
            best_se = SAMPLE_EVERY_VALUES[best_x]
            best_as = ACTION_SCALE_VALUES[best_y]
            title = f"{env_short}: best se={best_se}, as={best_as}"
        else:
            vmin, vmax = 0.0, 1.0
            title = f"{env_short}: no data"

        image_handle = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(range(len(SAMPLE_EVERY_VALUES)))
        ax.set_xticklabels([str(v) for v in SAMPLE_EVERY_VALUES])
        ax.set_yticks(range(len(ACTION_SCALE_VALUES)))
        ax.set_yticklabels([str(v) for v in ACTION_SCALE_VALUES])
        ax.set_xlabel("sample_every")
        if i == 0:
            ax.set_ylabel("action_scale")

        cbar = fig.colorbar(image_handle, ax=ax, fraction=0.05, pad=0.04)
        cbar.set_label(f"Final reward ({args.metric})")

    fig.suptitle("SAC final reward heatmaps by environment", fontsize=14)
    plt.savefig(args.out, dpi=160)
    plt.close(fig)
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()
