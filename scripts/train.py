#!/usr/bin/env python3
"""Train SAC or PPO on MetaWorld (MT10 or single task)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metaworld_rl.config import TrainConfig, load_train_config, save_train_config, name_to_env_name
from metaworld_rl.trainer import Trainer


def main() -> None:
    p = argparse.ArgumentParser(description="MetaWorld RL training")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config path (optional; defaults to built-in TrainConfig)",
    )
    p.add_argument("--algorithm", choices=["sac", "ppo"], default="sac")
    p.add_argument("--benchmark", choices=["MT10", "shelf", "sweep", "assembly", "plate", "button", "door", "drawer", "window", "lever", "coffee", "faucet"], default="MT10", help="MT10 or e.g. reach-v3")
    p.add_argument("--total-timesteps", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda:2")
    p.add_argument("--num-envs", type=int, default=None, help="Parallel envs (ignored for MT10)")
    p.add_argument("--warmup-steps", type=int, default=5000, help="SAC random-action steps before learning")
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--project", type=str, default="tester", help="Project name for Weights & Biases")
    p.add_argument("--frame-skip", type=int, default=1, help="Number of frames to skip (only for single-task envs)")
    p.add_argument("--action-scale", type=float, default=1.0, help="Scaling factor for actions (only for single-task envs)")
    args = p.parse_args()

    if args.config:
        cfg = load_train_config(args.config)
    else:
        cfg = TrainConfig()

    if args.algorithm is not None:
        cfg.algorithm = args.algorithm
    if args.benchmark is not None:
        cfg.env.benchmark = name_to_env_name(args.benchmark)
    if args.total_timesteps is not None:
        cfg.total_timesteps = args.total_timesteps
    if args.device is not None:
        cfg.device = args.device
    if args.num_envs is not None:
        cfg.env.num_envs = args.num_envs
    if args.warmup_steps is not None:
        cfg.sac.warmup_steps = args.warmup_steps
    if args.wandb:
        cfg.logging.use_wandb = True
    if args.project is not None:
        cfg.logging.wandb_project = args.project
        cfg.logging.wandb_run_name = args.benchmark + "_" + cfg.algorithm + "_" + str(args.frame_skip) + "_" + str(args.action_scale)
        # Ensure checkpoints don't get overwritten across ablations.
        def _float_tag(x: float) -> str:
            # Keep tags filename-friendly and stable across common float values like 0.5, 1.0, 2.0.
            return str(x).replace("-", "m").replace(".", "p")

        run_dir = (
            ROOT
            / "runs"
            / args.project
            / args.benchmark
            / f"{cfg.algorithm}_fs{args.frame_skip}_as{_float_tag(args.action_scale)}_seed{cfg.seed}"
        )
        cfg.checkpoint_dir = str(run_dir / "checkpoints")
        cfg.video_dir = str(run_dir / "videos")
        cfg.logging.plot_dir = str(run_dir / "plots")
        cfg.logging.history_csv = str(run_dir / "history.csv")
    if args.frame_skip is not None:
        cfg.env.frame_skip = int(args.frame_skip)
    if args.action_scale is not None:
        cfg.env.action_scale = float(args.action_scale)
    out = ROOT / "runs" / args.project / args.benchmark / "last_config.yaml"
    if args.project is not None:
        # Mirror the checkpoint directory uniqueness to the config snapshot path too.
        def _float_tag(x: float) -> str:
            return str(x).replace("-", "m").replace(".", "p")

        out = (
            ROOT
            / "runs"
            / args.project
            / args.benchmark
            / f"{cfg.algorithm}_fs{args.frame_skip}_as{_float_tag(args.action_scale)}_seed{cfg.seed}"
            / "last_config.yaml"
        )
    save_train_config(cfg, out)
    print(f"Wrote resolved config to {out}")

    Trainer(cfg).train()


if __name__ == "__main__":
    main()
