# RL Visualization Guide

## TL;DR

**Quick start examples:**

```bash
# Record policy video for a single trained model
python3.10 scripts/visualize.py --checkpoint runs/metaworld_rl/coffee/checkpoints/final.pt

# Just frame-skip analysis (no video recording)
python3.10 scripts/visualize.py --checkpoint runs/metaworld_rl/coffee/checkpoints/final.pt --mode frame_skip

# Just policy video (no frame-skip analysis)
python3.10 scripts/visualize.py --checkpoint runs/metaworld_rl/coffee/checkpoints/final.pt --mode video

# Visualize all trained models with batch script
python3.10 scripts/batch_visualize.py

# Custom output and parameters
python3.10 scripts/visualize.py --checkpoint runs/metaworld_rl/coffee/checkpoints/final.pt \
    --output-dir my_visualizations --max-steps 500 --fps 30
```

---

## Overview

The visualization suite provides three complementary tools:

1. **Policy Videos** - MP4 videos showing trained policies executing in the environment
2. **Frame-Skip Analysis** - Tests that validate action repetition behavior  
3. **Trajectory Plots** - 3D/2D visualization of end-effector movement with color gradients

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint` | (required) | Path to `.pt` checkpoint file |
| `--output-dir` | visualizations | Where to save outputs |
| `--mode` | both | `video`, `frame_skip`, or `both` |
| `--max-steps` | 200 | Steps in policy video |
| `--fps` | 20 | Video framerate |
| `--device` | cuda:2 | GPU device (`cpu` for CPU-only) |
| `--seed` | 0 | Random seed |

## Outputs

### Policy Video
- **File**: `{task}_policy_video.mp4`
- **Content**: Policy executing for configured timesteps
- **Info**: Total episode reward displayed

### Frame-Skip Analysis
- **Plot**: `frame_skip_tests/frame_skip_analysis.png` (4-panel figure)
  - Top-left: 3D end-effector trajectories with Z color gradient
  - Top-right: XY projection (top-down view)
  - Bottom-left: Z displacement over time
  - Bottom-right: Total displacement comparison
- **Videos**: `frame_skip_tests/frame_skip_N.mp4` for each frame-skip value

## How Frame-Skip Testing Works

Frame-skip testing validates that actions repeat correctly by:
1. Executing a predefined control sequence ([0, 0, 0.11, 0] by default - move +Z)
2. Testing with different frame-skip values (1, 2, 5)
3. Comparing end-effector trajectories

**Expected result**: Higher frame-skip → larger total displacement (action held longer)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "X11: DISPLAY environment missing" | Script auto-enables headless mode (MUJOCO_GL=egl) |
| "FFmpeg not found" | `apt-get install ffmpeg` (Linux) or `brew install ffmpeg` (Mac) |
| "CUDA out of memory" | Use `--device cpu` or reduce `--max-steps` |
| "Failed to render" | This is normal for headless rendering; frames still capture |
| "AttributeError: config_from_dict" | Update script - uses `TrainConfig(**cfg_dict)` instead |

## Integration Architecture

**Checkpoint Loading:**
- Loads agent weights (actor, q-networks, temperature)
- Loads and reconstructs TrainConfig from checkpoint
- Disables wandb logging (`WANDB_DISABLED=true`)
- Sets headless rendering (`MUJOCO_GL=egl`)

**Environment Setup:**
- Creates vectorized MetaWorld environment
- Sets `render_mode="rgb_array"` for video capture
- For frame-skip tests, creates separate environments for each skip value

## Examples

**Visualize coffee task with extended video:**
```bash
python3.10 scripts/visualize.py \
    --checkpoint runs/metaworld_rl/coffee/checkpoints/final.pt \
    --max-steps 500 --fps 30
```

**Batch visualize all tasks:**
```bash
python3.10 scripts/batch_visualize.py
```

**Frame-skip testing only:**
```bash
python3.10 scripts/visualize.py \
    --checkpoint runs/metaworld_rl/button/checkpoints/final.pt \
    --mode frame_skip --output-dir frame_skip_analysis
```

**CPU-only inference:**
```bash
python3.10 scripts/visualize.py \
    --checkpoint runs/metaworld_rl/door/checkpoints/final.pt \
    --device cpu
```

## Implementation Details

**Headless Rendering:**
- Sets `os.environ['MUJOCO_GL'] = 'egl'` before imports
- Uses matplotlib `Agg` backend (non-interactive)
- Continues gracefully if frames fail to render

**Wandb Integration:**
- Automatically disabled (`WANDB_DISABLED=true`)
- Sets `use_wandb=False` in config
- No wandb initialization for visualization-only workflows

**End-Effector Position:**
- Extracted from first 3 observation dimensions (X, Y, Z)
- Used for trajectory tracking and analysis
- Color gradient represents Z position (viridis colormap)

## Files

- `scripts/visualize.py` - Main visualization script (500+ lines)
- `scripts/batch_visualize.py` - Batch processing utility
- `VISUALIZATION.md` - This file
