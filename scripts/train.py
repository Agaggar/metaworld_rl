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
        cfg.logging.wandb_run_name = args.benchmark
        cfg.checkpoint_dir = str(ROOT / "runs" / args.project / args.benchmark / "checkpoints")
        cfg.video_dir = str(ROOT / "runs" / args.project / args.benchmark / "videos")
        cfg.logging.plot_dir = str(ROOT / "runs" / args.project / args.benchmark / "plots")
        cfg.logging.history_csv = str(ROOT / "runs" / args.project / args.benchmark / "history.csv")
    
    out = ROOT / "runs" / args.project / args.benchmark / "last_config.yaml"
    save_train_config(cfg, out)
    print(f"Wrote resolved config to {out}")

    Trainer(cfg).train()


if __name__ == "__main__":
    main()
