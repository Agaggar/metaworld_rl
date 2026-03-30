#!/usr/bin/env python3
"""
Visualization module for trained RL policies:
1. Record videos of policy rollouts
2. Test frame-skip behavior with predefined control sequences
3. Visualize end-effector trajectories
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Set headless rendering BEFORE importing anything else
os.environ['MUJOCO_GL'] = 'egl'

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm

# Repo root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metaworld_rl.config import TrainConfig, config_from_dict
from metaworld_rl.trainer import Trainer
from metaworld_rl.env.factory import make_vec_env
from metaworld_rl.agents.sac import SacAgent
from metaworld_rl.agents.ppo import PpoAgent


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[SacAgent | PpoAgent, Any, TrainConfig]:
    """Load agent and config from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct config from checkpoint
    cfg_dict = checkpoint['cfg']
    cfg = config_from_dict(cfg_dict) if isinstance(cfg_dict, dict) else cfg_dict
    cfg.env.render_mode = "rgb_array"
    cfg.env.camera_name = "corner3"
    cfg.sample_every = 1
    print("Sample-every value", cfg.sample_every)
    cfg.device = str(device)
    cfg.logging.use_wandb = False  # Disable wandb
    
    # Create trainer to get agent and env
    trainer = Trainer(cfg)
    agent = trainer.agent
    env = trainer.env
    
    # Load agent state dict
    agent_state = checkpoint['agent']
    if isinstance(agent, SacAgent):
        agent.actor.load_state_dict(agent_state['actor'])
        agent.q1.load_state_dict(agent_state['q1'])
        agent.q2.load_state_dict(agent_state['q2'])
        agent.q1_t.load_state_dict(agent_state['q1_t'])
        agent.q2_t.load_state_dict(agent_state['q2_t'])
        agent.log_alpha.data = agent_state['log_alpha']
    else:  # PPO
        agent.net.load_state_dict(agent_state.get('net', agent_state))
    
    return agent, env, cfg


def record_policy_video(
    env: Any,
    agent: SacAgent | PpoAgent,
    device: torch.device,
    algorithm: str,
    output_path: Path,
    max_steps: int = 500,
    fps: int = 20,
    env_reset_seed: int = 0,
) -> None:
    """Record a video of the policy executing in the environment."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    obs, _ = env.reset(seed=env_reset_seed)
    frames: list[np.ndarray] = []
    rewards = []
    
    for step in tqdm(range(max_steps), desc="Recording policy video"):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            if algorithm == "sac":
                actions = agent.act(obs_t, deterministic=True).cpu().numpy()
            else:
                actions, _, _, _ = agent.act(obs_t, deterministic=True)
                actions = actions.cpu().numpy()
        
        try:
            frame = env.render()
            frame = np.rot90(frame, k=2) # Metaworld corner images are upside down
            if frame is not None:
                if isinstance(frame, (tuple, list)):
                    frame = frame[0]
                if hasattr(frame, "shape") and len(frame.shape) == 4:
                    frame = frame[0]
                frames.append(np.asarray(frame))
        except Exception as e:
            print(f"Render warning at step {step}: {type(e).__name__}")
            # Continue without this frame
            pass
        
        for _ in range(1): # TODO: Expose ability to frame skip based on config
            obs, rew, term, trunc, _ = env.step(actions)
        rewards.append(rew[0] if isinstance(rew, np.ndarray) else rew)
        
        if term[0] or trunc[0]:
            print(f"Episode finished at step {step}")
            break
    
    if frames:
        imageio.mimsave(str(output_path), frames, fps=fps)
        print(f"Video saved to {output_path}")
        print(f"Total reward: {sum(rewards):.2f}")
    else:
        print("Warning: No frames captured for video")


def test_frame_skip_control_sequence(
    base_env: Any,
    device: torch.device,
    output_dir: Path,
    frame_skip_values: list[int] = [1, 2, 5],
    action_sequence: np.ndarray | None = None,
    n_repeats: int = 1,
) -> dict[str, Any]:
    """
    Test frame-skip behavior with predefined control sequences.
    Records end-effector trajectories for different frame-skip values.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if action_sequence is None:
        n_steps = 5
        action_dim = base_env.single_action_space.shape[0]
        action_sequence = np.zeros((n_steps, action_dim))
        action_sequence[:, 2] = 0.11  # Move in +z direction
    
    results = {}
    
    try:
        benchmark = base_env.unwrapped.spec.id if hasattr(base_env.unwrapped, 'spec') else "reach-v3"
    except:
        benchmark = "reach-v3"
    
    for frame_skip in frame_skip_values:
        print(f"\n=== Testing frame_skip={frame_skip} ===")
        
        from metaworld_rl.config import EnvConfig
        env_cfg = EnvConfig(
            benchmark=benchmark,
            num_envs=1,
            frame_skip=frame_skip,
            render_mode="rgb_array",
        )
        test_env = make_vec_env(env_cfg)
        
        trajectories = []
        frames_list = []
        
        for repeat_idx in range(n_repeats):
            obs, _ = test_env.reset(seed=42 + repeat_idx)
            trajectory = []
            frames = []
            
            if len(obs.shape) > 1:
                obs = obs[0]
            
            try:
                initial_pos = obs[:3].copy()
                trajectory.append(initial_pos)
            except:
                trajectory.append(obs[:3].copy())
            
            print(f"Initial position: {trajectory[0]}")
            
            for step, action in enumerate(action_sequence):
                action_batch = np.array([action])
                for _ in range(frame_skip):
                    obs, rewards, term, trunc, info = test_env.step(action_batch)
                
                if len(obs.shape) > 1:
                    obs = obs[0]
                
                try:
                    current_pos = obs[:3].copy()
                    trajectory.append(current_pos)
                    print(f"Step {step}, Action: {action}, Position: {current_pos}, Expected Z: {current_pos[2] + action[2] * 0.01 * (step + 1)}")
                except:
                    trajectory.append(obs[:3].copy())
                
                try:
                    frame = test_env.render()
                    if frame is not None:
                        if isinstance(frame, (tuple, list)):
                            frame = frame[0]
                        if hasattr(frame, "shape") and len(frame.shape) == 4:
                            frame = frame[0]
                        frames.append(np.asarray(frame))
                except:
                    pass
                
                if term[0] or trunc[0]:
                    break
            
            trajectories.append(np.array(trajectory))
            frames_list.append(frames)
        
        results[frame_skip] = {
            "trajectories": trajectories,
            "frames": frames_list,
            "actions": action_sequence,
        }
        
        test_env.close()
    
    return results


def visualize_frame_skip_results(
    results: dict[str, Any],
    output_dir: Path,
    action_description: str = "Test control sequence",
) -> None:
    """Create visualizations of frame-skip test results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_skips = sorted(results.keys())
    
    fig = plt.figure(figsize=(14, 10))
    ax_3d = fig.add_subplot(2, 2, 1, projection='3d')
    
    all_z_values = []
    for trajectories in results.values():
        for traj in trajectories["trajectories"]:
            all_z_values.extend(traj[:, 2])
    
    z_min, z_max = np.min(all_z_values), np.max(all_z_values)
    
    for idx, frame_skip in enumerate(frame_skips):
        data = results[frame_skip]
        for traj in data["trajectories"]:
            z_normalized = (traj[:, 2] - z_min) / (z_max - z_min + 1e-8)
            
            ax_3d.scatter(
                traj[:, 0], traj[:, 1], traj[:, 2],
                c=z_normalized, cmap="viridis", s=50, alpha=0.6,
                label=f"fs={frame_skip}",
            )
            ax_3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.3, linewidth=1)
    
    ax_3d.set_xlabel("X Position")
    ax_3d.set_ylabel("Y Position")
    ax_3d.set_zlabel("Z Position")
    ax_3d.set_title("End-Effector 3D Trajectories")
    if len(frame_skips) > 0:
        ax_3d.legend(loc='best', fontsize=8)
    
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=z_min, vmax=z_max))
    sm.set_array([])
    plt.colorbar(sm, ax=ax_3d, label="Z Value")
    
    # XY projection
    ax_xy = fig.add_subplot(2, 2, 2)
    for idx, frame_skip in enumerate(frame_skips):
        data = results[frame_skip]
        for traj in data["trajectories"]:
            z_normalized = (traj[:, 2] - z_min) / (z_max - z_min + 1e-8)
            ax_xy.scatter(traj[:, 0], traj[:, 1], c=z_normalized, cmap="viridis", s=50, alpha=0.6,
                         label=f"fs={frame_skip}" if idx < len(frame_skips) else "")
            ax_xy.plot(traj[:, 0], traj[:, 1], alpha=0.3, linewidth=1)
    
    ax_xy.set_xlabel("X Position")
    ax_xy.set_ylabel("Y Position")
    ax_xy.set_title("End-Effector XY Projection")
    if len(frame_skips) > 0:
        ax_xy.legend(loc='best', fontsize=8)
    ax_xy.grid(True, alpha=0.3)
    
    # Z displacement over time
    ax_z = fig.add_subplot(2, 2, 3)
    for idx, frame_skip in enumerate(frame_skips):
        data = results[frame_skip]
        for traj in data["trajectories"]:
            time_steps = np.arange(len(traj))
            z_displacements = traj[:, 2] - traj[0, 2]
            ax_z.plot(time_steps, z_displacements, marker='o', label=f"fs={frame_skip}", alpha=0.7)
    
    ax_z.set_xlabel("Time Step")
    ax_z.set_ylabel("Z Displacement (from start)")
    ax_z.set_title("End-Effector Z Displacement Over Time")
    ax_z.legend(loc='best', fontsize=8)
    ax_z.grid(True, alpha=0.3)
    
    # Total displacement
    ax_displacement = fig.add_subplot(2, 2, 4)
    displacements = []
    displacements_labels = []
    
    for frame_skip in frame_skips:
        data = results[frame_skip]
        for traj in data["trajectories"]:
            displacement = np.linalg.norm(traj[-1] - traj[0])
            displacements.append(displacement)
        displacements_labels.append(f"fs={frame_skip}")
    
    ax_displacement.bar(range(len(displacements)), displacements, alpha=0.7)
    ax_displacement.set_xlabel("Frame Skip Value")
    ax_displacement.set_ylabel("Total Displacement")
    ax_displacement.set_title("Total End-Effector Displacement")
    ax_displacement.set_xticks(range(len(displacements_labels)))
    ax_displacement.set_xticklabels(displacements_labels, rotation=45)
    ax_displacement.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f"Frame-Skip Analysis: {action_description}", fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    output_path = output_dir / "frame_skip_analysis.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Analysis plot saved to {output_path}")
    plt.close(fig)
    
    # Create animations for each frame-skip value
    for frame_skip in frame_skips:
        data = results[frame_skip]
        frames_list = data["frames"]
        
        if frames_list and len(frames_list[0]) > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            def update(frame_idx):
                if frame_idx < len(frames_list[0]):
                    frame = frames_list[0][frame_idx]
                    ax.clear()
                    ax.imshow(frame)
                    ax.set_title(f"Frame Skip: {frame_skip}, Step: {frame_idx}")
                    ax.axis('off')
            
            ani = FuncAnimation(fig, update, frames=min(len(frames_list[0]), 100), interval=50)
            writer = FFMpegWriter(fps=20)
            video_path = output_dir / f"frame_skip_{frame_skip}.mp4"
            
            try:
                ani.save(str(video_path), writer=writer)
                print(f"Animation saved to {video_path}")
            except Exception as e:
                print(f"Failed to save animation: {e}")
            finally:
                plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize trained MetaWorld RL policies")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pt file)")
    parser.add_argument("--output-dir", type=str, default="visualizations", help="Directory to save visualizations")
    parser.add_argument("--mode", choices=["video", "frame_skip", "both"], default="both", help="Type of visualization")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum steps for policy video")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second for videos")
    parser.add_argument("--device", type=str, default="cuda:2", help="Device to use for inference")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(ROOT / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading checkpoint from {args.checkpoint}")
    agent, env, cfg = load_checkpoint(Path(args.checkpoint), device)
    
    if args.mode in ["video", "both"]:
        task_name = Path(args.checkpoint).parent.parent.name
        video_path = output_dir / f"{task_name}_policy_video.mp4"
        print(f"\nRecording policy video for {task_name}...")
        record_policy_video(env, agent, device, cfg.algorithm, video_path, 
                          max_steps=args.max_steps, fps=args.fps, env_reset_seed=args.seed)
    
    if args.mode in ["frame_skip", "both"]:
        print("\nTesting frame-skip behavior...")
        results = test_frame_skip_control_sequence(env, device, output_dir / "frame_skip_tests", 
                                                   frame_skip_values=[1, 2, 5])
        visualize_frame_skip_results(results, output_dir / "frame_skip_tests", 
                                    action_description="Predefined control sequence test")
    
    print(f"\nVisualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
