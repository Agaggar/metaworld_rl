# NOTE
I am testing out some "vibe coding" functionality using Cursor. The vast majority of this code is generated using "Agentic AI" (a mixture of Claude, ChatGPT, Cursor, etc etc). Take it with a grain of salt! I've also included some of the prompts that I used in making this repository.

# MetaWorld RL

Research-oriented code to benchmark **SAC** and **PPO** (PyTorch, from scratch) on [MetaWorld](https://github.com/Farama-Foundation/Metaworld), starting with **state observations** (full MuJoCo state, including end-effector information). The layout is modular so you can swap algorithms, MLP sizes, and environment knobs without touching core learning code.

The control loop always runs at full simulator frequency. The experiment knob is `sample_every`, which keeps action frequency unchanged while reducing how often transitions are committed to learning. This isolates the effect of data sampling rate without reducing control fidelity.

## Requirements

- **Python 3.10+** (MetaWorld 3.x)
- Linux with MuJoCo / OpenGL for rendering (optional; training uses state only)
- GPU recommended for long MT10 runs

```bash
pip install -r requirements.txt
```
(^ this didn't work for me, but just create a virtual environment and install necessary packages)

## Quick start

Single-task sanity check (parallel `reach-v3` envs):

```bash
python scripts/train.py --algorithm sac --benchmark reach-v3 --total-timesteps 500000 --device cuda
```

Full **MT10** (10 parallel tasks, fixed 10 sub-envs):

```bash
python scripts/train.py --config configs/default.yaml --algorithm sac --device cuda
```

YAML config (`configs/default.yaml`) controls training, logging intervals, and environment options. CLI flags override the loaded file. The resolved config is written to a run-specific `last_config.yaml` path each run.

## Modular environment options

`TrainConfig` / `EnvConfig` (see `metaworld_rl/config.py`) include:

| Option | Role |
|--------|------|
| `benchmark` | `"MT10"` or a single task id (e.g. `reach-v3`) |
| `num_envs` | Parallel actors for single-task mode (ignored for MT10, which is always 10) |
| `sample_every` | Commit one learner transition every N control steps (action still computed every step) |
| `action_scale` | Multiply actions before clipping to `[-1, 1]` (lower ⇒ gentler motion) |
| `normalize_observations` | Online running mean/variance normalization |
| `use_one_hot_task_id` | Append task one-hot (via MetaWorld) for future multi-task conditioning |
| `render_mode` | Set to `rgb_array` when you need evaluation videos |

Wrappers live in `metaworld_rl/env/wrappers.py`; environment construction is in `metaworld_rl/env/factory.py`.

## Algorithms and layout

- **SAC** — `metaworld_rl/agents/sac.py` (twin Q, soft targets, optional auto-α)
- **PPO** — `metaworld_rl/agents/ppo.py` (GAE, clipped objective, shared actor–critic)
- **Models** — `metaworld_rl/models/mlp.py` (Gaussian policy, Q-networks, shared trunk for PPO)
- **Trainer** — `metaworld_rl/trainer.py` (vector env loop, checkpoints, optional W&B)
- **Logging** — `LoggingConfig` defines intervals and metric groups; W&B is optional (`logging.use_wandb`)
- **Plots** — `metaworld_rl/plotting.py` writes matplotlib curves under `runs/plots/`

## Weights & Biases

```bash
wandb login
python scripts/train.py --config configs/default.yaml --wandb
```

## Vision later

State-based training uses the default observation vector. For pixels, you will add a CNN encoder and point `EnvConfig.render_mode` / camera kwargs at the MetaWorld layer; the agent API (`act` / `update`) stays the same.

## License

See `LICENSE`.
