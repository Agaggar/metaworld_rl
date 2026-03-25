"""Training loop: SAC or PPO on vectorized MetaWorld."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb

from metaworld_rl.agents.ppo import PpoAgent
from metaworld_rl.agents.sac import SacAgent
from metaworld_rl.buffers.replay import ReplayBuffer
from metaworld_rl.buffers.rollout import RolloutBuffer
from metaworld_rl.config import TrainConfig
from metaworld_rl.env.factory import make_vec_env
from metaworld_rl.evaluation import evaluate_vector_env, record_video
from metaworld_rl.plotting import plot_history


class Trainer:
    """Orchestrates env, agent, logging, evaluation, and checkpoints."""

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(
            cfg.device if torch.cuda.is_available() else "cpu"
        )
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.env = make_vec_env(cfg.env)
        self.num_envs = self.env.num_envs
        obs_dim = int(np.prod(self.env.single_observation_space.shape))
        act_dim = int(np.prod(self.env.single_action_space.shape))

        self.algorithm = cfg.algorithm
        if self.algorithm == "sac":
            self.agent: SacAgent | PpoAgent = SacAgent(
                obs_dim, act_dim, cfg.model, cfg.sac, self.device
            )
            self.replay = ReplayBuffer(
                obs_dim,
                act_dim,
                cfg.sac.buffer_capacity,
                self.device,
            )
        else:
            self.agent = PpoAgent(obs_dim, act_dim, cfg.model, cfg.ppo, self.device)
            self.rollout = RolloutBuffer(
                cfg.ppo.n_steps,
                self.num_envs,
                obs_dim,
                act_dim,
                self.device,
            )

        self.global_step = 0
        self.history: list[dict[str, float]] = []

        if cfg.logging.use_wandb:
            wandb.init(
                project=cfg.logging.wandb_project,
                name=cfg.logging.wandb_run_name,
                entity=cfg.logging.wandb_entity,
                config=self._wandb_config_dict(),
            )

    def _wandb_config_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self.cfg)

    def train(self) -> None:
        cfg = self.cfg
        obs, _ = self.env.reset(seed=cfg.seed)

        while self.global_step < cfg.total_timesteps:
            if self.algorithm == "sac":
                self._train_step_sac(obs)
                obs = self._last_obs_sac
            else:
                obs = self._train_phase_ppo(obs)

            if self.history:
                last = self.history[-1]
                if self.algorithm == "ppo" or int(last["step"]) % cfg.logging.log_interval == 0:
                    self._log_console(last)

            if self.global_step % cfg.logging.eval_interval == 0 and self.global_step > 0:
                metrics = evaluate_vector_env(
                    self.env,
                    self.agent,
                    self.device,
                    self.algorithm,
                    self.num_envs,
                )
                self._log_eval(metrics)
                if cfg.logging.save_plots:
                    plot_history(
                        self.history,
                        Path(cfg.logging.plot_dir),
                        prefix=f"step_{self.global_step}",
                    )

            if self.global_step % cfg.logging.checkpoint_interval == 0 and self.global_step > 0:
                self._save_checkpoint()

        self._save_checkpoint(final=True)
        if cfg.logging.use_wandb:
            wandb.finish()

    def _train_step_sac(self, obs: np.ndarray) -> None:
        cfg = self.cfg
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if self.global_step < cfg.sac.warmup_steps:
            actions = np.stack(
                [self.env.single_action_space.sample() for _ in range(self.num_envs)]
            )
        else:
            with torch.no_grad():
                actions = self.agent.act(obs_t, deterministic=False).cpu().numpy()

        next_obs, rewards, term, trunc, _ = self.env.step(actions)
        rewards = rewards * cfg.reward_scale
        dones = np.logical_or(term, trunc).astype(np.float32)

        self.replay.add_batch(obs, actions, rewards, next_obs, dones)
        self.global_step += self.num_envs
        self._last_obs_sac = next_obs

        metrics: dict[str, float] = {}
        if self.replay.size >= cfg.sac.warmup_steps:
            acc: dict[str, float] = {}
            for _ in range(cfg.sac.updates_per_step):
                batch = self.replay.sample(cfg.sac.batch_size)
                m = self.agent.update(batch)
                for k, v in m.items():
                    acc[k] = acc.get(k, 0.0) + v
            n_u = float(cfg.sac.updates_per_step)
            metrics = {k: v / n_u for k, v in acc.items()}
        if metrics:
            row = {"step": float(self.global_step), **metrics}
            self.history.append(row)
            self._log_wandb_train(row)

    def _train_phase_ppo(self, obs: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        self.rollout.reset_storage()
        for _ in range(cfg.ppo.n_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                actions, log_probs, _, values = self.agent.act(
                    obs_t, deterministic=False
                )
            actions_np = actions.cpu().numpy()
            lp_np = log_probs.cpu().numpy()
            val_np = values.cpu().numpy()

            next_obs, rewards, term, trunc, _ = self.env.step(actions_np)
            rewards = rewards * cfg.reward_scale
            dones = np.logical_or(term, trunc).astype(np.float32)

            self.rollout.add(
                obs,
                actions_np,
                lp_np,
                rewards,
                val_np,
                dones,
            )
            obs = next_obs
            self.global_step += self.num_envs

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            _, _, _, last_values = self.agent.act(obs_t, deterministic=False)
            last_v = last_values.cpu().numpy()

        adv, ret = self.rollout.compute_returns(
            last_v, cfg.ppo.gamma, cfg.ppo.gae_lambda
        )

        obs_t = torch.as_tensor(self.rollout.obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(self.rollout.actions, dtype=torch.float32, device=self.device)
        old_lp = torch.as_tensor(
            self.rollout.log_probs, dtype=torch.float32, device=self.device
        )
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=self.device)

        metrics = self.agent.update(obs_t, act_t, old_lp, adv_t, ret_t)
        row = {"step": float(self.global_step), **metrics}
        self.history.append(row)
        self._log_wandb_train(row)
        return obs

    def _log_console(self, row: dict[str, float]) -> None:
        parts = [f"{k}={v:.4g}" for k, v in row.items() if k != "step"]
        print(f"step {int(row['step'])} | " + " ".join(parts))

    def _log_wandb_train(self, row: dict[str, float]) -> None:
        if not self.cfg.logging.use_wandb:
            return
        m = {f"train/{k}": v for k, v in row.items() if k != "step"}
        m["train/step"] = int(row["step"])
        wandb.log(m)

    def _log_eval(self, metrics: dict[str, float]) -> None:
        print(f"\n[eval] step {self.global_step} {metrics}\n")
        if self.cfg.logging.use_wandb:
            wandb.log({f"eval/{k}": v for k, v in metrics.items()})
        self._append_csv(self.cfg.logging.history_csv, metrics)

    def _append_csv(self, path: str, metrics: dict[str, float]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        row = {"step": self.global_step, **metrics}
        file_exists = p.exists()
        with p.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                w.writeheader()
            w.writerow(row)

    def _save_checkpoint(self, final: bool = False) -> None:
        cfg = self.cfg
        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        tag = "final" if final else str(self.global_step)
        path = Path(cfg.checkpoint_dir) / f"{tag}.pt"
        payload = {
            "global_step": self.global_step,
            "agent": self.agent.state_dict(),
            "cfg": self._wandb_config_dict(),
        }
        torch.save(payload, path)
        print(f"Saved checkpoint {path}")
