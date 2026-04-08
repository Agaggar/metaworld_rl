---
name: PPO SAC parity and env integration
overview: Audit and align PPO/SAC with their clean variants, add dual action-space support (continuous + discrete), and introduce Gymnasium classic-control/Box2D integration and config presets without changing the existing config layout.
todos:
  - id: audit-parity-ppo-sac
    content: Document concrete PPO/SAC mismatches vs clean variants and define exact parity fixes
    status: pending
  - id: add-dual-action-support
    content: Implement continuous+discrete action support paths for PPO and SAC models/agents/trainer
    status: pending
  - id: extend-env-factory
    content: Add Gymnasium classic-control and Box2D suite integration with env-id configuration
    status: pending
  - id: add-config-presets
    content: Create new classic-control and Box2D config presets using requested PPO hyperparameters
    status: pending
  - id: verify-and-checklist
    content: Run smoke validation and produce PPO 13-details status checklist
    status: pending
isProject: false
---

# PPO/SAC Alignment And Gymnasium Integration Plan

## Scope and success criteria
- Validate and reconcile logic mismatches between:
  - [`/home/ayush/Desktop/tutorials/metaworld_rl/metaworld_rl/agents/ppo.py`](/home/ayush/Desktop/tutorials/metaworld_rl/metaworld_rl/agents/ppo.py) and [`/home/ayush/Desktop/tutorials/metaworld_rl/metaworld_rl/agents/ppo_clean.py`](/home/ayush/Desktop/tutorials/metaworld_rl/metaworld_rl/agents/ppo_clean.py)
  - [`/home/ayush/Desktop/tutorials/metaworld_rl/metaworld_rl/agents/sac.py`](/home/ayush/Desktop/tutorials/metaworld_rl/metaworld_rl/agents/sac.py) and [`/home/ayush/Desktop/tutorials/metaworld_rl/metaworld_rl/agents/sac_clean.py`](/home/ayush/Desktop/tutorials/metaworld_rl/metaworld_rl/agents/sac_clean.py)
- Ensure both PPO and SAC support continuous and discrete action spaces in the project implementations.
- Add Gymnasium env integration for classic control (`Acrobot-v1`, `CartPole-v1`, `MountainCar-v0`) and Box2D (`BipedalWalker-v3`, `LunarLanderContinuous-v3`).
- Add new config preset(s) using the provided PPO hyperparameters mapped to existing project naming and flow.

## Findings that drive implementation
- Current `ppo.py` misses several CleanRL behaviors (LR anneal, value clipping, target-KL stop, optional advantage norm) and is effectively continuous-only through shared Gaussian policy in [`/home/ayush/Desktop/tutorials/metaworld_rl/metaworld_rl/models/mlp.py`](/home/ayush/Desktop/tutorials/metaworld_rl/metaworld_rl/models/mlp.py).
- Current `sac.py` is also continuous-only by architecture; discrete SAC needs a dedicated categorical branch (cannot be done by minor casting).
- Env factory currently supports MetaWorld/robotics only, so Gymnasium classic/Box2D requires extending [`/home/ayush/Desktop/tutorials/metaworld_rl/metaworld_rl/env/factory.py`](/home/ayush/Desktop/tutorials/metaworld_rl/metaworld_rl/env/factory.py) and config schema wiring.

## Implementation plan
- Extend policy/model abstractions in [`/home/ayush/Desktop/tutorials/metaworld_rl/metaworld_rl/models/mlp.py`](/home/ayush/Desktop/tutorials/metaworld_rl/metaworld_rl/models/mlp.py) to branch on action-space type:
  - PPO: Gaussian+tanh for `Box`, Categorical logits for `Discrete`.
  - SAC: keep Gaussian continuous path and add discrete SAC network heads (twin Q over actions + categorical policy expectation backup).
- Update PPO agent in [`/home/ayush/Desktop/tutorials/metaworld_rl/metaworld_rl/agents/ppo.py`](/home/ayush/Desktop/tutorials/metaworld_rl/metaworld_rl/agents/ppo.py) for CleanRL parity options:
  - LR annealing hook, clipped value loss toggle, optional adv normalization, target KL early stop, 0.5 scaling on value loss.
- Update SAC agent in [`/home/ayush/Desktop/tutorials/metaworld_rl/metaworld_rl/agents/sac.py`](/home/ayush/Desktop/tutorials/metaworld_rl/metaworld_rl/agents/sac.py):
  - Add discrete update equations and entropy target handling by space type.
  - Keep continuous behavior compatible with current flow and expose cadence options when needed.
- Adjust trainer data handling in [`/home/ayush/Desktop/tutorials/metaworld_rl/metaworld_rl/trainer.py`](/home/ayush/Desktop/tutorials/metaworld_rl/metaworld_rl/trainer.py):
  - Action tensor dtype/shape by space type.
  - Correct env stepping payload for discrete actions.
  - Ensure rollout/replay paths remain compatible with existing `sample_every` behavior.
- Extend environment integration:
  - Add Gymnasium suite support and env-id based construction in [`/home/ayush/Desktop/tutorials/metaworld_rl/metaworld_rl/env/factory.py`](/home/ayush/Desktop/tutorials/metaworld_rl/metaworld_rl/env/factory.py).
  - Wire config/dataclass and CLI mapping in [`/home/ayush/Desktop/tutorials/metaworld_rl/metaworld_rl/config.py`](/home/ayush/Desktop/tutorials/metaworld_rl/metaworld_rl/config.py) and [`/home/ayush/Desktop/tutorials/metaworld_rl/scripts/train.py`](/home/ayush/Desktop/tutorials/metaworld_rl/scripts/train.py) without changing overall layout.
- Add config file(s) in [`/home/ayush/Desktop/tutorials/metaworld_rl/configs`](/home/ayush/Desktop/tutorials/metaworld_rl/configs):
  - New classic-control PPO config using your provided hyperparameters with project-native argument names.
  - Include totals/iterations/env-count alignment with current trainer semantics.
  - Add Box2D config preset using `LunarLanderContinuous-v3` and `BipedalWalker-v3`.
- Verify with targeted smoke tests:
  - PPO discrete on `CartPole-v1` and continuous on `BipedalWalker-v3`.
  - SAC discrete on `Acrobot-v1` (or `CartPole-v1`) and continuous on `LunarLanderContinuous-v3`.
  - Confirm no regressions for existing MetaWorld training path.

## PPO 13-details audit deliverable
- Produce explicit checklist mapping each of the 13 PPO details to current implementation status (`implemented`, `partial`, `missing`) and where implemented/fixed in code after changes.

## Notes
- Keep existing config structure and logging flow; only add minimal fields required for action-space/env selection.
- Do not add HuggingFace upload flags/logic.