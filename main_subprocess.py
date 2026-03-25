import subprocess as sp

for benchmark in ["button", "door", "drawer", "coffee", "faucet"]:
    sp.run(["python3.10", "scripts/train.py", "--config", "configs/default.yaml", "--project", "metaworld_rl", "--benchmark", benchmark])