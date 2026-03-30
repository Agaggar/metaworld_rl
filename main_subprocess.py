import subprocess as sp

DEVICE = "cuda:2"
for action_scale in [0.1, 0.25]: # [1.0, 5.0]
    for sample_every in [2, 5, 10]:
        print(f"Training with action_scale={action_scale} and sample_every={sample_every}")
        for benchmark in ["button", "door", "drawer", "coffee", "faucet"]:
            sp.run(["python3.10", "scripts/train.py", "--wandb", "--config", "configs/default.yaml", "--project", "se_as_sweep", "--benchmark", benchmark, "--sample-every", str(sample_every), "--action-scale", str(action_scale), "--algorithm", "sac", "--device", DEVICE])