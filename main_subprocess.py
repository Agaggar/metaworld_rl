import subprocess as sp

DEVICE = "cuda:2"
algorithm_to_use = "ppo" # or "sac"
env_to_use = "metaworld" # or "robotics"
project_name = "se_as_sweep" # metaworld standard 5 for sac and ppo: "se_as_sweep"
# for handed manipulation, use "handed_manip"

if env_to_use == "metaworld":
    benchmarks = ["button", "door", "drawer", "coffee", "faucet"]
    config_to_use = "configs/default.yaml"
elif env_to_use == "metaworld10":
    benchmarks = ["MT10"]
    config_to_use = "configs/default.yaml"
elif env_to_use == "robotics":
    benchmarks = ["HandManipulateBlock_ContinuousTouchSensors-v1", "HandManipulateEgg_ContinuousTouchSensors-v1", "HandManipulatePen_ContinuousTouchSensors-v1"]
    config_to_use = "configs/robotics_shadow_touch_block.yaml"
else:
    raise ValueError(f"Invalid environment to use: {env_to_use}")

for action_scale in reversed([0.1, 0.25, 1.0, 5.0]):
    for sample_every in reversed([1, 2, 5, 10]):
        print(f"Training with action_scale={action_scale} and sample_every={sample_every}")
        for benchmark in benchmarks:
            sp.run(["python3.10", "scripts/train.py", "--wandb", "--config", config_to_use, "--project", project_name, "--benchmark", benchmark, "--sample-every", str(sample_every), "--action-scale", str(action_scale), "--algorithm", algorithm_to_use, "--device", DEVICE])