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
        self.act_dim = act_dim
        self.sample_every = max(1, int(cfg.sample_every))
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
        self.global_sim_step = 0
        self.global_sample_step = 0
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

        # Use "next_*_step" scheduling instead of modulo checks so PPO doesn't skip
        # eval/checkpoint triggers due to larger step jumps per update.
        next_eval_step = cfg.logging.eval_interval
        next_ckpt_step = cfg.logging.checkpoint_interval

        while self.global_sim_step < cfg.total_timesteps:
            if self.algorithm == "sac":
                self._train_step_sac(obs)
                obs = self._last_obs_sac
            else:
                obs = self._train_phase_ppo(obs)

            if self.history:
                last = self.history[-1]
                if self.algorithm == "ppo" or int(last["step"]) % cfg.logging.log_interval == 0:
                    self._log_console(last)

            while (
                self.global_sim_step >= next_eval_step
                and next_eval_step <= cfg.total_timesteps
                and next_eval_step > 0
            ):
                metrics = evaluate_vector_env(
                    self.env,
                    self.agent,
                    self.device,
                    self.num_envs,
                )
                self._log_eval(metrics)
                if cfg.logging.save_plots:
                    plot_history(
                        self.history,
                        Path(cfg.logging.plot_dir),
                        prefix=f"step_{self.global_sim_step}",
                    )
                next_eval_step += cfg.logging.eval_interval

            while (
                self.global_sim_step >= next_ckpt_step
                and next_ckpt_step <= cfg.total_timesteps
                and next_ckpt_step > 0
            ):
                self._save_checkpoint()
                next_ckpt_step += cfg.logging.checkpoint_interval

        self._save_checkpoint(final=True)
        if cfg.logging.use_wandb:
            wandb.finish()

    def _train_step_sac(self, obs: np.ndarray) -> None:
        cfg = self.cfg
        start_obs = np.zeros_like(obs)
        start_actions = np.zeros((self.num_envs, self.act_dim), dtype=np.float32)
        reward_sums = np.zeros((self.num_envs,), dtype=np.float32)
        span_steps = np.zeros((self.num_envs,), dtype=np.int32)
        segment_active = np.zeros((self.num_envs,), dtype=bool)
        samples_before = self.global_sample_step

        while True:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            if self.global_sim_step < cfg.sac.warmup_steps:
                actions = np.stack(
                    [self.env.single_action_space.sample() for _ in range(self.num_envs)]
                ).astype(np.float32)
            else:
                with torch.no_grad():
                    actions = self.agent.act(obs_t, deterministic=False).cpu().numpy()

            new_segment = ~segment_active
            if np.any(new_segment):
                start_obs[new_segment] = obs[new_segment]
                start_actions[new_segment] = actions[new_segment]
                reward_sums[new_segment] = 0.0
                span_steps[new_segment] = 0
                segment_active[new_segment] = True

            next_obs, rewards, term, trunc, _ = self.env.step(actions)
            rewards = rewards * cfg.reward_scale
            dones = np.logical_or(term, trunc)

            reward_sums += rewards
            span_steps += 1
            self.global_sim_step += self.num_envs

            commit_mask = (span_steps >= self.sample_every) | dones
            if np.any(commit_mask):
                idx = np.where(commit_mask)[0]
                discounts = np.power(cfg.sac.gamma, span_steps[idx]).astype(np.float32)
                self.replay.add_batch(
                    start_obs[idx],
                    start_actions[idx],
                    reward_sums[idx],
                    next_obs[idx],
                    dones[idx].astype(np.float32),
                    discounts,
                )
                self.global_sample_step += int(idx.size)
                segment_active[idx] = False

            obs = next_obs
            self._last_obs_sac = next_obs

            if self.global_sim_step >= cfg.total_timesteps:
                break
            if self.global_sample_step > samples_before:
                break

        metrics: dict[str, float] = {}
        if self.global_sim_step >= cfg.sac.warmup_steps and self.replay.size >= cfg.sac.batch_size:
            acc: dict[str, float] = {}
            for _ in range(cfg.sac.updates_per_step):
                batch = self.replay.sample(cfg.sac.batch_size)
                m = self.agent.update(batch)
                for k, v in m.items():
                    acc[k] = acc.get(k, 0.0) + v
            n_u = float(cfg.sac.updates_per_step)
            metrics = {k: v / n_u for k, v in acc.items()}
        if metrics:
            row = {
                "step": float(self.global_sim_step),
                "sim_step": float(self.global_sim_step),
                "sample_step": float(self.global_sample_step),
                **metrics,
            }
            self.history.append(row)
            self._log_wandb_train(row)

    def _train_phase_ppo(self, obs: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        self.rollout.reset_storage()
        per_env_commits = np.zeros((self.num_envs,), dtype=np.int32)
        target_commits = int(cfg.ppo.n_steps)

        start_obs = np.zeros_like(obs)
        start_actions = np.zeros((self.num_envs, self.act_dim), dtype=np.float32)
        start_log_probs = np.zeros((self.num_envs,), dtype=np.float32)
        start_values = np.zeros((self.num_envs,), dtype=np.float32)
        reward_sums = np.zeros((self.num_envs,), dtype=np.float32)
        span_steps = np.zeros((self.num_envs,), dtype=np.int32)
        segment_active = np.zeros((self.num_envs,), dtype=bool)

        while np.any(per_env_commits < target_commits) and self.global_sim_step < cfg.total_timesteps:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                actions, log_probs, _, values = self.agent.act(obs_t, deterministic=False)
            actions_np = actions.cpu().numpy()
            lp_np = log_probs.cpu().numpy()
            val_np = values.cpu().numpy()

            new_segment = ~segment_active
            if np.any(new_segment):
                start_obs[new_segment] = obs[new_segment]
                start_actions[new_segment] = actions_np[new_segment]
                start_log_probs[new_segment] = lp_np[new_segment]
                start_values[new_segment] = val_np[new_segment]
                reward_sums[new_segment] = 0.0
                span_steps[new_segment] = 0
                segment_active[new_segment] = True

            next_obs, rewards, term, trunc, _ = self.env.step(actions_np)
            rewards = rewards * cfg.reward_scale
            dones = np.logical_or(term, trunc)

            with torch.no_grad():
                next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
                _, _, _, next_values_t = self.agent.act(next_obs_t, deterministic=False)
            next_vals_np = next_values_t.cpu().numpy()

            reward_sums += rewards
            span_steps += 1
            self.global_sim_step += self.num_envs

            ready = (span_steps >= self.sample_every) | dones
            under_limit = per_env_commits < target_commits
            commit_mask = ready & under_limit & segment_active
            if np.any(commit_mask):
                idx = np.where(commit_mask)[0]
                self.rollout.add(
                    start_obs[idx],
                    start_actions[idx],
                    start_log_probs[idx],
                    reward_sums[idx],
                    start_values[idx],
                    dones[idx].astype(np.float32),
                    next_vals_np[idx],
                    span_steps[idx].astype(np.float32),
                    idx.astype(np.int32),
                )
                per_env_commits[idx] += 1
                self.global_sample_step += int(idx.size)
                segment_active[idx] = False

            obs = next_obs

        adv, ret = self.rollout.compute_returns(cfg.ppo.gamma, cfg.ppo.gae_lambda)

        n = self.rollout.pos
        if n == 0:
            return obs
        obs_t = torch.as_tensor(self.rollout.obs[:n], dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(self.rollout.actions[:n], dtype=torch.float32, device=self.device)
        old_lp = torch.as_tensor(
            self.rollout.log_probs[:n], dtype=torch.float32, device=self.device
        )
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=self.device)

        metrics = self.agent.update(obs_t, act_t, old_lp, adv_t, ret_t)
        row = {
            "step": float(self.global_sim_step),
            "sim_step": float(self.global_sim_step),
            "sample_step": float(self.global_sample_step),
            **metrics,
        }
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
        wandb.log(m, step=int(row["step"]))

    def _log_eval(self, metrics: dict[str, float]) -> None:
        print(f"\n[eval] step {self.global_sim_step} {metrics}\n")
        if self.cfg.logging.use_wandb:
            wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=self.global_sim_step)
        self._append_csv(self.cfg.logging.history_csv, metrics)

        # Keep eval metrics in-memory too, so local plot_history() can render them.
        step_val = float(self.global_sim_step)
        if self.history and self.history[-1].get("step") == step_val:
            # Merge into the most recent training row (evaluation happens right after training).
            self.history[-1].update(metrics)
        else:
            self.history.append({"step": step_val, **metrics})

    def _append_csv(self, path: str, metrics: dict[str, float]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        row = {"step": self.global_sim_step, **metrics}
        file_exists = p.exists()
        with p.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                w.writeheader()
            w.writerow(row)

    def _save_checkpoint(self, final: bool = False) -> None:
        cfg = self.cfg
        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        tag = "final" if final else str(self.global_sim_step)
        path = Path(cfg.checkpoint_dir) / f"{tag}.pt"
        payload = {
            "global_step": self.global_sim_step,
            "agent": self.agent.state_dict(),
            "cfg": self._wandb_config_dict(),
        }
        torch.save(payload, path)
        print(f"Saved checkpoint {path}")
