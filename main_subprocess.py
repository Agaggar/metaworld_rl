import subprocess as sp

for frame_skip in [1, 2, 5, 10]:
    for action_scale in [0.1, 0.25, 1.0, 5.0]:
        for benchmark in ["button", "door", "drawer", "coffee", "faucet"]:
            sp.run(["python3.10", "scripts/train.py", "--config", "configs/default.yaml", "--project", "fs_as_sweep", "--benchmark", benchmark, "--frame-skip", str(frame_skip), "--action-scale", str(action_scale), "--algorithm", "sac", "--device", "cuda:2"])