#!/usr/bin/env python3
"""
Batch visualization script for all trained models.
Generates videos and frame-skip analysis for each trained checkpoint.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    # Find all checkpoint directories
    checkpoints_dir = ROOT / "runs" / "metaworld_rl"
    
    if not checkpoints_dir.exists():
        print(f"Error: Checkpoints directory not found: {checkpoints_dir}")
        sys.exit(1)
    
    tasks = [d.name for d in checkpoints_dir.iterdir() if d.is_dir() and (d / "checkpoints" / "final.pt").exists()]
    
    if not tasks:
        print(f"No trained models found in {checkpoints_dir}")
        sys.exit(1)
    
    print(f"Found {len(tasks)} trained models: {', '.join(tasks)}")
    print()
    
    output_base = ROOT / "visualizations"
    
    for task in sorted(tasks):
        checkpoint = checkpoints_dir / task / "checkpoints" / "final.pt"
        output_dir = output_base / task
        
        print(f"{'='*60}")
        print(f"Visualizing {task}...")
        print(f"{'='*60}")
        
        cmd = [
            "python3.10",
            str(ROOT / "scripts" / "visualize.py"),
            "--checkpoint", str(checkpoint),
            "--output-dir", str(output_dir),
            "--mode", "video",
            "--max-steps", "300",
            "--device", "cuda:2",
        ]
        
        result = subprocess.run(cmd, cwd=ROOT)
        
        if result.returncode != 0:
            print(f"Warning: Visualization for {task} failed with exit code {result.returncode}")
        
        print()
    
    print(f"{'='*60}")
    print(f"Batch visualization complete!")
    print(f"Results saved to {output_base}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
