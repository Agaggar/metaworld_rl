# MetaWorld MT10 RL Experiment Plan
The goal of this repository is to test the performance of several RL algroithms on the MetaWorld environment.

## 1. Objectives

* [ ] Use only **MT10 benchmark tasks**
* [ ] Implement RL algorithms **from scratch (PyTorch)**:

  * [ ] PPO
  * [ ] SAC
* [ ] Design system to allow future:

  * [ ] MaxDiff RL
  * [ ] Custom policy architectures
* [ ] Use **Weights & Biases (wandb)** for logging

  * [ ] Single config/python file controls what gets logged
* [ ] Generate **local plots (matplotlib)**:

  * [ ] Reward curves
  * [ ] Loss curves
  * [ ] Evaluation rollouts (videos)

---

## 2. Design Philosophy

* [ ] Keep code **minimal but clean and well-documented**
* [ ] Avoid over-engineering; prioritize **research velocity**
* [ ] Modular enough to swap:

  * [ ] Algorithms
  * [ ] Architectures
  * [ ] Tasks

---

## 3. Core Components (No Overbuilt Structure)

### Environment Wrapper

* [ ] Wrap MetaWorld MT10 into Gymnasium-style API
* [ ] Handle:

  * [ ] Task sampling
  * [ ] Observation/action normalization (toggleable)
  * [ ] Task IDs (for multi-task learning later)

---

### Agent (Algorithms)

#### PPO

* [ ] Implement clipped objective
* [ ] GAE advantage estimation
* [ ] On-policy rollout buffer

#### SAC

* [ ] Twin Q-networks
* [ ] Stochastic policy (Gaussian)
* [ ] Entropy regularization (auto-tuning optional)

#### References (for guidance only)

* [ ] CleanRL
* [ ] Spinning Up (OpenAI)

---

### Models

* [ ] Shared MLP backbone (default)
* [ ] Actor network:

  * [ ] Gaussian policy
* [ ] Critic network:

  * [ ] Q-functions (SAC)
  * [ ] Value / advantage (PPO)

---

### Trainer (Single File / Class)

* [ ] One central `Trainer` class that handles:

  * [ ] Environment initialization
  * [ ] Agent initialization
  * [ ] Replay buffer / rollout storage
  * [ ] Training loop
  * [ ] Evaluation
  * [ ] Logging

#### Training Loop (Unified Concept)

* [ ] Collect experience
* [ ] Update agent
* [ ] Log metrics
* [ ] Periodically evaluate

---

### Replay / Rollout Buffers

* [ ] PPO: on-policy rollout buffer
* [ ] SAC: replay buffer
* [ ] Optional:

  * [ ] Task-aware sampling (future MT improvements)

---

## 4. Evaluation

* [ ] Metrics:

  * [ ] Success rate (primary)
  * [ ] Episode return
* [ ] Evaluate across all MT10 tasks
* [ ] Log evaluation metrics to wandb
* [ ] Save rollout videos locally

---

## 5. Logging & Visualization

### WandB

* [ ] Log:

  * [ ] Rewards
  * [ ] Losses
  * [ ] Success rates
* [ ] Use **single config file** to define:

  * [ ] What metrics to track
  * [ ] Logging frequency

### Local Plotting (matplotlib)

* [ ] Generate:

  * [ ] Training curves
  * [ ] Evaluation curves
* [ ] Save plots to disk

---

## 6. Experiment Plan

### Phase 1: Sanity Check

* [ ] Train SAC on a **single MT10 task**
* [ ] Verify:

  * [ ] Learning curve improves
  * [ ] Evaluation pipeline works

### Phase 2: Full MT10

* [ ] Train shared policy across all 10 tasks
* [ ] Track:

  * [ ] Per-task success rate
  * [ ] Aggregate performance

### Phase 3: PPO Baseline

* [ ] Run PPO on MT10
* [ ] Compare vs SAC

### Phase 4: Ablations

* [ ] Network size
* [ ] Normalization on/off
* [ ] Reward scaling

---

## 7. Compute Strategy

* [ ] Use 1–2 NVIDIA GPUs
* [ ] Keep batching efficient (especially SAC updates)
* [ ] Save checkpoints periodically

---

## 8. Extensibility Hooks (Important)

* [ ] Clean interface for:

  * [ ] Adding new algorithms (e.g., MaxDiff RL)
  * [ ] Swapping policy architectures
* [ ] Keep agent API consistent:

  * [ ] `act()`
  * [ ] `update()`

---

## 9. Implementation Order (for Cursor)

* [ ] Step 1: Environment wrapper (MT10)
* [ ] Step 2: Basic MLP models (actor + critic)
* [ ] Step 3: SAC (end-to-end, trainable)
* [ ] Step 4: Trainer class (single-file orchestration)
* [ ] Step 5: WandB logging + config system
* [ ] Step 6: Evaluation + video saving
* [ ] Step 7: PPO implementation
* [ ] Step 8: Multi-task training support

---

## 10. Notes / Constraints

* [ ] Algorithms must be **implemented from scratch**
* [ ] Code should be:

  * [ ] Readable
  * [ ] Well-commented
  * [ ] Easy to modify for research
* [ ] Avoid large frameworks unless absolutely necessary

---

## 11. (Optional Future Work)

* [ ] MaxDiff RL integration
* [ ] Task embeddings / conditioning
* [ ] Vision-based policies
* [ ] Offline RL experiments

---
