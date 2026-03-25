"""Training and environment configuration (dataclasses + YAML load/save)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class EnvConfig:
    """Environment construction; supports MT10 or a single v3 task name."""

    benchmark: str = "MT10"
    """Use 'MT10' for the 10-task benchmark, or a task id like 'reach-v3'."""

    num_envs: int = 10
    """Parallel actors. For MT10 this must match the benchmark (10) unless you use a single-task name."""

    seed: int = 0
    frame_skip: int = 1
    """Repeat the same action this many env steps; rewards are summed."""

    action_scale: float = 1.0
    """Multiply policy actions before clip to [-1, 1]. Values <1 reduce effective motion / 'speed'."""

    normalize_observations: bool = False
    """Online running mean/var normalization of observations (training only)."""

    use_one_hot_task_id: bool = False
    """Append MT10 task one-hot to observations (requires MetaWorld OneHotWrapper)."""

    max_episode_steps: int | None = None
    """Override MetaWorld TimeLimit; None uses env default."""

    render_mode: Literal["rgb_array", "human", None] = None
    """Set for evaluation videos (rgb_array). Training leaves None for speed."""


@dataclass
class SacConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    auto_alpha: bool = True
    target_entropy: float | None = None
    """If None, defaults to -dim(A)."""

    buffer_capacity: int = 1_000_000
    batch_size: int = 256
    warmup_steps: int = 5_000
    updates_per_step: int = 1


@dataclass
class PpoConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_steps: int = 2048
    n_epochs: int = 10
    minibatch_size: int = 256


@dataclass
class ModelConfig:
    hidden_dims: tuple[int, ...] = (256, 256)
    activation: Literal["relu", "tanh"] = "relu"
    """Policy / value MLP trunk."""


@dataclass
class LoggingConfig:
    """Single place to tune what is logged and how often (wandb + local history)."""

    use_wandb: bool = False
    wandb_project: str = "metaworld_rl"
    wandb_run_name: str | None = None
    wandb_entity: str | None = None

    log_interval: int = 50
    """Steps between console / wandb training metrics."""

    eval_interval: int = 10_000
    checkpoint_interval: int = 50_000

    log_train_metrics: tuple[str, ...] = ("loss_critic", "loss_actor", "alpha", "q_gap")
    log_eval_metrics: tuple[str, ...] = ("success_rate", "episode_return_mean")

    save_plots: bool = True
    plot_dir: str = "runs/plots"

    history_csv: str = "runs/history.csv"


@dataclass
class TrainConfig:
    algorithm: Literal["sac", "ppo"] = "sac"
    total_timesteps: int = 1_000_000
    device: str = "cuda"
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sac: SacConfig = field(default_factory=SacConfig)
    ppo: PpoConfig = field(default_factory=PpoConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    eval_episodes: int = 10
    """Per-task eval rollouts for MT10 (total episodes = eval_episodes * num_tasks)."""

    reward_scale: float = 1.0
    """Multiply rewards before learning (ablations)."""

    seed: int = 0
    checkpoint_dir: str = "runs/checkpoints"
    video_dir: str = "runs/videos"


def config_from_dict(d: dict[str, Any]) -> TrainConfig:
    """Build TrainConfig from a nested dict (e.g. parsed YAML)."""

    def sub(cls: type, key: str, default: Any) -> Any:
        raw = d.get(key, {})
        if not isinstance(raw, dict):
            return default
        fields = {f.name for f in getattr(cls, "__dataclass_fields__", {}).values()}
        kwargs = {k: v for k, v in raw.items() if k in fields}
        return cls(**kwargs)  # type: ignore[arg-type]

    b = TrainConfig()
    return TrainConfig(
        algorithm=d.get("algorithm", b.algorithm),
        total_timesteps=d.get("total_timesteps", b.total_timesteps),
        device=d.get("device", b.device),
        env=sub(EnvConfig, "env", b.env),
        model=sub(ModelConfig, "model", b.model),
        sac=sub(SacConfig, "sac", b.sac),
        ppo=sub(PpoConfig, "ppo", b.ppo),
        logging=sub(LoggingConfig, "logging", b.logging),
        eval_episodes=d.get("eval_episodes", b.eval_episodes),
        reward_scale=d.get("reward_scale", b.reward_scale),
        seed=d.get("seed", b.seed),
        checkpoint_dir=d.get("checkpoint_dir", b.checkpoint_dir),
        video_dir=d.get("video_dir", b.video_dir),
    )


def load_train_config(path: str | Path) -> TrainConfig:
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return config_from_dict(data)


def save_train_config(cfg: TrainConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return {k: convert(v) for k, v in asdict(obj).items()}
        if isinstance(obj, tuple):
            return list(obj)
        return obj

    with path.open("w") as f:
        yaml.safe_dump(convert(cfg), f, sort_keys=False)


def default_train_config() -> TrainConfig:
    return TrainConfig()

def name_to_env_name(name: str):
    """Convert a task name to a MetaWorld environment name."""
    name_to_env_name = {
        'shelf': 'shelf-place-v3', 
        'sweep': 'sweep-into-v3', 
        'assembly': 'assembly-v3', 
        'plate': 'plate-slide-v3',
        'button': 'button-press-v3',
        'door': 'door-open-v3',
        'drawer': 'drawer-open-v3',
        'window': 'window-open-v3',
        'lever': 'lever-pull-v3',
        'coffee': 'coffee-button-v3',
        'faucet': 'faucet-open-v3',
        'MT10': 'MT10'
    }
    return name_to_env_name[name]
