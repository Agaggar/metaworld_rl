import subprocess as sp

DEVICE = "cuda:1"
env_to_use = "gymnasium_classic_control"
project_name = "classic_control_sweep" # metaworld standard 5 for sac and ppo: "se_as_sweep"
# for handed manipulation, use "handed_manip"

if env_to_use == "metaworld":
    benchmarks = ["button", "door", "drawer", "coffee", "faucet"]
    config_to_use = "configs/default.yaml"
    suite_arg = "metaworld"
    env_arg_name = "--benchmark"
elif env_to_use == "metaworld10":
    benchmarks = ["MT10"]
    config_to_use = "configs/default.yaml"
    suite_arg = "metaworld"
    env_arg_name = "--benchmark"
elif env_to_use == "robotics":
    benchmarks = ["HandManipulateBlock_ContinuousTouchSensors-v1", "HandManipulateEgg_ContinuousTouchSensors-v1", "HandManipulatePen_ContinuousTouchSensors-v1"]
    config_to_use = "configs/robotics_shadow_touch_block.yaml"
    suite_arg = "robotics"
    env_arg_name = "--robotics-env-id"
elif env_to_use == "gymnasium_classic_control":
    benchmarks = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
    config_to_use = "configs/gymnasium_classic_control.yaml"
    suite_arg = "gymnasium"
    env_arg_name = "--gym-env-id"
elif env_to_use == "gymnasium_box2d":
    benchmarks = ["LunarLanderContinuous-v3", "BipedalWalker-v3"]
    config_to_use = "configs/gymnasium_box2d.yaml"
    suite_arg = "gymnasium"
    env_arg_name = "--gym-env-id"
else:
    raise ValueError(f"Invalid environment to use: {env_to_use}")

# around 4 mins per run for classic control ppo
# so 3 seeds, 3 action scales, 3 sample everys, 3 benchmarks = 3*3*3*3 = 81 runs
# approximately 81*4 mins = 324 mins = 5.4 hours
algorithms_to_use = ["sac"] # or "ppo"
num_seeds = 3
action_scales = [1.0] # [0.25, 5.0] # 1.0, 0.1, 
sample_everys = [1, 2] # 1, 2, 5, 10
for seed in range(num_seeds):
    for algorithm_to_use in algorithms_to_use:
        for action_scale in reversed(action_scales):
            for sample_every in sample_everys:
                print(f"Training with action_scale={action_scale} and sample_every={sample_every}")
                for benchmark in benchmarks:
                    cmd = [
                        "python3.10",
                        "scripts/train.py",
                        "--wandb",
                        "--config",
                        config_to_use,
                        "--seed",
                        str(seed),
                        "--project",
                        project_name,
                        "--suite",
                        suite_arg,
                        env_arg_name,
                        benchmark,
                        "--sample-every",
                        str(sample_every),
                        "--action-scale",
                        str(action_scale),
                        "--algorithm",
                        algorithm_to_use,
                        "--device",
                        DEVICE,
                    ]
                    sp.run(cmd)