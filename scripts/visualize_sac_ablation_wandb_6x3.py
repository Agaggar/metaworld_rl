#!/usr/bin/env python3
"""
Create a 6x3 matplotlib grid comparing SAC ablations from W&B runs.

Grid layout (columns x rows):
- Col 0: average across envs in that row (cols 1-5)
- Col 1-5: individual benchmark envs
- Row 0: action_scale fixed at 1.0, vary frame_skip
- Row 1: frame_skip fixed at 1, vary action_scale
- Row 2: each subplot title shows the best (frame_skip, action_scale)
        combo by last eval value (eval/episode_return_mean)

Uses only matplotlib.pyplot for visualization.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

import matplotlib
import math

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb


FRAME_SKIP_VALUES = [1, 2, 5, 10]
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
    - nested dict: config["env"]["frame_skip"]
    - flattened keys: config["env.frame_skip"]
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
    # wandb sometimes exposes config via run.config (dict-like).
    try:
        cfg = dict(run.config)
    except Exception:
        return None
    return _get_nested(cfg, key)


def _scan_curve(run: wandb.apis.public.Run, metric_key: str) -> dict[int, float]:
    """Return {step: value} for a single W&B run."""
    rows = run.history(keys=[metric_key], samples=500)
    step_to_vals = defaultdict(list)

    for _, row in rows.iterrows():
        val = row.get(metric_key)
        step = row.get("_step")
        if val is None or step is None:
            continue
        step_to_vals[int(step)].append(float(val))

    return {k: sum(v)/len(v) for k, v in step_to_vals.items()}
    # step_to_vals: dict[int, list[float]] = defaultdict(list)
    # for row in run.history(keys=[metric_key], samples=500):
    #     if metric_key not in row or row[metric_key] is None:
    #         continue
    #     step = row.get("_step")
    #     if step is None:
    #         continue
    #     try:
    #         step_int = int(step)
    #         val = float(row[metric_key])
    #     except Exception:
    #         continue
    #     step_to_vals[step_int].append(val)

    # return {k: (sum(v) / len(v)) for k, v in step_to_vals.items() if v}


def _average_step_dicts(step_dicts: list[dict[int, list[float]]]):
    combined: dict[int, list[float]] = defaultdict(list)

    for d in step_dicts:
        for step, vals in d.items():
            combined[step].extend(vals)

    return _compute_mean_std(combined)

def _compute_mean_std(step_dict: dict[int, list[float]]):
    steps = sorted(step_dict.keys())
    means = []
    stds = []

    for s in steps:
        vals = step_dict[s]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var)

        means.append(mean)
        stds.append(std)

    return steps, means, stds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--entity", type=str, default=None, help="W&B entity/user/org (required if not inferrable).")
    p.add_argument("--project", type=str, default="metaworld_rl")
    p.add_argument("--metric", type=str, default="eval/episode_return_mean")
    p.add_argument("--out", type=str, default=str("visualizations/sac_ablation_6x3.png"))
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
        # Best-effort: wandb may still resolve the project if you're authenticated
        # and your default entity is set.
        runs = api.runs(args.project)

    env_names = {e for e, _ in ENV_COLS}

    # Group runs by (env_id, frame_skip, action_scale).
    run_groups: dict[tuple[str, int, float], list[wandb.apis.public.Run]] = defaultdict(list)
    n = 0
    for run in runs:
        n += 1
        if n > args.max_runs:
            break

        cfg_algo = _get_cfg(run, "algorithm")
        if cfg_algo != "sac":
            continue

        env_id = _get_cfg(run, "env.benchmark")
        if env_id not in env_names:
            continue

        fs = _get_cfg(run, "env.frame_skip")
        ascale = _get_cfg(run, "env.action_scale")
        try:
            fs_int = int(fs)
        except Exception:
            continue

        asnap = _snap_action_scale(ascale)
        if asnap is None:
            continue
        if fs_int not in FRAME_SKIP_VALUES or asnap not in ACTION_SCALE_VALUES:
            continue

        run_groups[(env_id, fs_int, asnap)].append(run)

    # Pre-fetch all curves we need.
    # Each (env_id, frame_skip, action_scale) maps to {step: [values...]},
    # so we can compute mean/std at each step across runs.
    curves: dict[tuple[str, int, float], dict[int, list[float]]] = {}
    for (env_id, fs, ascale), rlist in run_groups.items():
        if not rlist:
            continue

        # Use summary to pick top runs (faster)
        combined: dict[int, list[float]] = defaultdict(list)

        for run in rlist:
            one = _scan_curve(run, args.metric)
            for step, val in one.items():
                combined[step].append(val)

        curves[(env_id, fs, ascale)] = combined

    # Pastel colors (slightly darker than earlier wandb-default-ish pastels).
    frame_colors = {
        1: "#6ea3c9",   # muted blue
        2: "#ee7f9c",   # muted pink
        5: "#eea765",   # muted peach
        10: "#76c88b",  # muted green
    }

    action_colors = {
        0.1: "#6faee0",
        0.25: "#eea765",
        1.0: "#ea86a7",
        5.0: "#86d894",
    }

    fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(22, 12))
    env_short_titles = ["average"] + [short for _, short in ENV_COLS]

    for c in range(6):
        axes[0, c].set_title(env_short_titles[c])

    # --- plotting helpers ---
    def plot_step_dict(ax, step_dict: dict[int, list[float]], color: str, label: str | None = None, alpha: float = 0.2) -> None:
        steps, means, stds = _compute_mean_std(step_dict)
        if not steps:
            return
        ax.plot(steps, means, color=color, linewidth=2, label=label)
        ax.fill_between(
            steps,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            color=color,
            alpha=alpha,
        )

    def last_mean(step_dict: dict[int, list[float]]) -> float | None:
        if not step_dict:
            return None
        last_step = max(step_dict.keys())
        vals = step_dict.get(last_step, [])
        if not vals:
            return None
        return sum(vals) / len(vals)

    eval_as_fixed = 1.0

    # Row 0: action_scale fixed at 1.0; vary frame_skip.
    for col_idx, (env_id, env_short) in enumerate(ENV_COLS, start=1):
        ax = axes[0, col_idx]
        for fs in FRAME_SKIP_VALUES:
            step_dict = curves.get((env_id, fs, eval_as_fixed), {})
            if not step_dict:
                continue
            plot_step_dict(ax, step_dict, frame_colors[fs], label=f"fs={fs}")
        ax.grid(True, alpha=0.3)
        if len(FRAME_SKIP_VALUES) > 1:
            ax.legend(loc="best", fontsize=8)

    # Row 0, col 0: average across envs for each frame_skip.
    ax_avg = axes[0, 0]
    for fs in FRAME_SKIP_VALUES:
        per_env: list[dict[int, list[float]]] = []
        for env_id, _ in ENV_COLS:
            step_dict = curves.get((env_id, fs, eval_as_fixed), {})
            if step_dict:
                per_env.append(step_dict)
        if not per_env:
            continue
        steps, means, stds = _average_step_dicts(per_env)
        if not steps:
            continue
        ax_avg.plot(steps, means, color=frame_colors[fs], linewidth=2, label=f"fs={fs}")
        ax_avg.fill_between(
            steps,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            color=frame_colors[fs],
            alpha=0.2,
        )
    ax_avg.grid(True, alpha=0.3)
    ax_avg.legend(loc="best", fontsize=9)

    # Row 1: frame_skip fixed at 1; vary action_scale.
    for col_idx, (env_id, _env_short) in enumerate(ENV_COLS, start=1):
        ax = axes[1, col_idx]
        for ascale in ACTION_SCALE_VALUES:
            step_dict = curves.get((env_id, 1, ascale), {})
            if not step_dict:
                continue
            plot_step_dict(ax, step_dict, action_colors[ascale], label=f"as={ascale}")
        ax.grid(True, alpha=0.3)
        if len(ACTION_SCALE_VALUES) > 1:
            ax.legend(loc="best", fontsize=8)

    # Row 1, col 0: average across envs for each action_scale.
    ax_avg = axes[1, 0]
    for ascale in ACTION_SCALE_VALUES:
        per_env: list[dict[int, list[float]]] = []
        for env_id, _ in ENV_COLS:
            step_dict = curves.get((env_id, 1, ascale), {})
            if step_dict:
                per_env.append(step_dict)
        if not per_env:
            continue
        steps, means, stds = _average_step_dicts(per_env)
        if not steps:
            continue
        ax_avg.plot(steps, means, color=action_colors[ascale], linewidth=2, label=f"as={ascale}")
        ax_avg.fill_between(
            steps,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            color=action_colors[ascale],
            alpha=0.2,
        )
    ax_avg.grid(True, alpha=0.3)
    ax_avg.legend(loc="best", fontsize=9)

    # Row 2: best (frame_skip, action_scale) by last eval mean.
    # Col 0: best across env-average.
    best_fs_as_avg: tuple[int, float] | None = None
    best_last_avg = float("-inf")
    for fs in FRAME_SKIP_VALUES:
        for ascale in ACTION_SCALE_VALUES:
            per_env: list[dict[int, list[float]]] = []
            for env_id, _ in ENV_COLS:
                step_dict = curves.get((env_id, fs, ascale), {})
                if step_dict:
                    per_env.append(step_dict)
            if not per_env:
                continue
            steps, means, stds = _average_step_dicts(per_env)
            if not steps:
                continue
            candidate_last = means[-1]
            if candidate_last > best_last_avg:
                best_last_avg = candidate_last
                best_fs_as_avg = (fs, ascale)
                # We'll replot using per_env pooled values below.

    if best_fs_as_avg is not None:
        best_fs, best_as = best_fs_as_avg
        per_env: list[dict[int, list[float]]] = []
        for env_id, _ in ENV_COLS:
            step_dict = curves.get((env_id, best_fs, best_as), {})
            if step_dict:
                per_env.append(step_dict)
        steps, means, stds = _average_step_dicts(per_env)
        ax = axes[2, 0]
        color = "#bda1d9"  # pastel-ish purple
        ax.plot(steps, means, color=color, linewidth=2)
        ax.fill_between(
            steps,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            color=color,
            alpha=0.2,
        )
        ax.grid(True, alpha=0.3)
        ax.set_title(f"best avg fs={best_fs}, as={best_as}")

    # Per-env best.
    for col_idx, (env_id, env_short) in enumerate(ENV_COLS, start=1):
        ax = axes[2, col_idx]
        best_pair: tuple[int, float] | None = None
        best_last = float("-inf")
        for fs in FRAME_SKIP_VALUES:
            for ascale in ACTION_SCALE_VALUES:
                step_dict = curves.get((env_id, fs, ascale), {})
                lm = last_mean(step_dict)
                if lm is None:
                    continue
                if lm > best_last:
                    best_last = lm
                    best_pair = (fs, ascale)

        if best_pair is not None:
            best_fs, best_as = best_pair
            step_dict = curves.get((env_id, best_fs, best_as), {})
            if step_dict:
                plot_step_dict(ax, step_dict, color=frame_colors[best_fs], label=None, alpha=0.2)
            ax.set_title(f"best fs={best_fs}, as={best_as}")
        ax.grid(True, alpha=0.3)

    for c in range(6):
        axes[2, c].set_xlabel("step")
        axes[0, c].set_ylabel("reward, as fixed at 1.0")
        axes[1, c].set_ylabel("reward, fs fixed at 1")
        axes[2, c].set_ylabel("reward, best fs, as")

    fig.suptitle("Reward for SAC: frame_skip, action_scale, and best", fontsize=16)
    fig.tight_layout()
    out_path = args.out
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()

