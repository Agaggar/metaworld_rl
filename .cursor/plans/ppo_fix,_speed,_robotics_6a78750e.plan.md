---
name: PPO fix, speed, robotics
overview: Fix the PPO crash for single-env runs (double value squeeze), document and optionally implement runtime levers given low GPU utilization, and integrate Gymnasium-Robotics Shadow Hand continuous-touch environments via a dict-flattening vector wrapper plus config/CLI hooks—keeping sampling-frequency experiments scientifically meaningful.
todos:
  - id: fix-ppo-squeeze
    content: Remove redundant v.squeeze(-1) in SharedActorCritic.get_action_and_value and evaluate_actions (mlp.py)
    status: in_progress
  - id: ppo-smoke
    content: Smoke-test PPO and SAC single-env (button) with small total_timesteps
    status: pending
  - id: robotics-deps-config
    content: Add gymnasium-robotics to requirements; EnvConfig suite + robotics_env_id; YAML example
    status: pending
  - id: robotics-factory-wrapper
    content: "factory.py: register_envs + SyncVectorEnv; VectorFlattenDictObs wrapper for (167,) Box obs"
    status: pending
  - id: eval-is-success
    content: "evaluation.py: read is_success from vector infos for success_rate"
    status: pending
  - id: train-cli
    content: "scripts/train.py: --suite robotics --robotics-env-id (or equivalent)"
    status: pending
  - id: speed-followup
    content: "Optional: document AsyncVectorEnv / num_envs / torch.compile in README or config comments"
    status: pending
isProject: false
---

# PPO bugfix, training speed, and Shadow Hand experiments

## Debate / methodology (short)

- **Sampling frequency vs wall clock**: Faster runs help iteration, but changing `num_envs`, vectorization strategy, or GPU batching changes **wall-clock** and **gradient noise**, not the physical **simulator step count** per sample. For the core research question (“does the rate we *commit* transitions matter?”), keep `sample_every`, `total_timesteps`, and env physics fixed when comparing policies; treat pure speed hacks (e.g. `torch.compile`, more subprocess envs) as infrastructure, not experimental conditions.
- **PPO + `sample_every`**: Your trainer already implements variable-length segments until `span_steps >= sample_every` (`[trainer.py](metaworld_rl/trainer.py)` ~253–271). That is coherent; the failure you saw is an implementation bug, not a conceptual one.
- **Shadow Hand flattening**: Concatenating `observation`, `desired_goal`, and `achieved_goal` into one vector (167 dims for all three continuous-touch envs we verified) matches your current MLP agents but **is not HER**; sparse rewards may be hard without replay + goal relabeling or reward shaping. That is acceptable for “set up experiments,” with the caveat that you may later want HER or asymmetric AC if learning stalls.

---

## 1. PPO `IndexError` (root cause and fix)

**Cause:** In `[metaworld_rl/models/mlp.py](metaworld_rl/models/mlp.py)`, `forward()` already returns `v.squeeze(-1)` (value shape `(B,)`). `get_action_and_value()` returns `v.squeeze(-1)` **again** (line 139). For `B=1`, the first squeeze yields shape `(1,)`; the second removes the last axis and produces a **0-d** tensor, so `.numpy()` is scalar and `val_np[new_segment]` fails in `[trainer.py](metaworld_rl/trainer.py)` line 235.

**Fix (minimal):** In `get_action_and_value`, return `v` without a second squeeze. Same for `evaluate_actions` line 154 (it also double-squeezes the value returned from `forward`).

**Confidence:** High (>90%). After the change, smoke test:

```bash
python3.10 scripts/train.py --config configs/default.yaml --benchmark button --sample-every 1 --action-scale 0.1 --algorithm ppo --device cuda:2 --total-timesteps 5000
```

---

## 2. Speeding up SAC / PPO (~30 min, ~20% GPU)

Interpretation: **low GPU use usually means the bottleneck is env stepping (MuJoCo / Python)**, not the MLP.


| Lever                                                   | Effect                                                                         | Caveat                                                                                                                                    |
| ------------------------------------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `**num_envs` > 1** (single-task MetaWorld)              | More parallel rollouts per wall-clock second; PPO collects more data per phase | More RAM; PPO batch grows; for SAC, more transitions per outer iteration—tune if you care about exact “steps per replay sample” semantics |
| `**gym.vector.AsyncVectorEnv`** (or subprocess workers) | Overlap CPU sim with GPU when env is CPU-bound                                 | Slightly more complexity; must be safe with MuJoCo (often one env per process is fine)                                                    |
| `**torch.compile` / TF32 / `matmul` precision**         | Faster forward/backward                                                        | `compile` first-epoch overhead; verify numerics for your paper runs                                                                       |
| **Fewer `cpu().numpy()` syncs**                         | Moderate gain                                                                  | Larger refactor (keep rollout tensors on GPU); optional follow-up                                                                         |
| **Shorter smoke runs**                                  | Faster iteration                                                               | `total_timesteps`, `eval_interval`, `logging.log_interval`                                                                                |


Recommendation: prioritize `**num_envs` and/or AsyncVectorEnv** for MetaWorld single-task; add optional `torch.compile` behind a config flag if you want more GPU headroom without changing simulator semantics.

---

## 3. Gymnasium-Robotics Shadow Hand (continuous touch)

**Verified locally (python3.10):**

- Env IDs: `HandManipulateBlock_ContinuousTouchSensors-v1`, `HandManipulateEgg_ContinuousTouchSensors-v1`, `HandManipulatePen_ContinuousTouchSensors-v1`
- Dict obs: `observation` (153,), `desired_goal` (7,), `achieved_goal` (7,) → **concatenated dim 167**; action dim **20**
- Vector `infos`: `is_success` as float array (not `success`)

**Integration plan:**

1. **Dependencies:** Add `gymnasium-robotics` (and ensure MuJoCo stack) to `[requirements.txt](requirements.txt)` for reproducibility on fresh clones.
2. **Env factory:** Extend `[metaworld_rl/env/factory.py](metaworld_rl/env/factory.py)` (and `[EnvConfig](metaworld_rl/config.py)`) with something like `suite: metaworld | robotics` and `robotics_env_id: str | None`. When `suite == robotics`, `import gymnasium_robotics`, `gym.register_envs(gymnasium_robotics)`, build `SyncVectorEnv` from partials of `gym.make(robotics_env_id)`, then apply existing `VectorActionScale` / `VectorObservationNormalize` as today.
3. **Dict → vector batch:** Add a small `**VectorWrapper`** (same file as `[metaworld_rl/env/wrappers.py](metaworld_rl/env/wrappers.py)`) that on `reset`/`step` maps `obs` from `dict[str, ndarray]` to `np.concatenate([observation, desired_goal, achieved_goal], axis=-1)` with dtype `float32`, and sets `single_observation_space` / `observation_space` to `Box` shape `(167,)`. This restores the assumption in `[trainer.py](metaworld_rl/trainer.py)` line 36 (`np.prod(single_observation_space.shape)`).
4. **Evaluation:** In `[metaworld_rl/evaluation.py](metaworld_rl/evaluation.py)`, treat vector `infos` like today but also accept `**is_success`** (and optionally `_is_success`) so `success_rate` is meaningful for Hand tasks.
5. **CLI / configs:** Extend `[scripts/train.py](scripts/train.py)` with flags such as `--suite robotics` and `--robotics-env-id ...` (or a YAML-only path). Add an example YAML (e.g. `configs/robotics_shadow_touch_block.yaml`) documenting the three env IDs.

**Example command (after implementation):**

```bash
python3.10 scripts/train.py --config configs/robotics_shadow_touch_block.yaml --wandb --project shadow_touch --algorithm sac --device cuda:2
```

---

## Implementation order

1. Fix double `squeeze` in `[mlp.py](metaworld_rl/models/mlp.py)` and confirm PPO smoke run.
2. Add robotics deps + factory branch + dict-flatten vector wrapper + eval `is_success`.
3. (Optional follow-up) AsyncVectorEnv / `num_envs` tuning and optional `torch.compile` flag.

