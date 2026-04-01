Here are the next steps for this repository. Again, keep in mind that the goal of this repository is to investigate the effect of _sampling frequency_ on reinforcement learning; in other words, does the _rate_ at which we collect data make an impact on the overall performance of the policy?

1. Is there a way to speed up runtime for either SAC or PPO implementations? (For reference, each run takes roughly 30 minutes and ~20% of a GPU for 500k iterations with the default configutation.)

2. I am running into errors when running RL with PPO:
`python3.10 scripts/train.py --wandb --config configs/default.yaml --project se_as_sweep --benchmark button --sample-every 1 --action-scale 0.1 --algorithm ppo --device cuda:2` Error described here:
Traceback (most recent call last):
  File "/home/ayush/Desktop/tutorials/metaworld_rl/scripts/train.py", line 105, in <module>
    main()
  File "/home/ayush/Desktop/tutorials/metaworld_rl/scripts/train.py", line 101, in main
    Trainer(cfg).train()
  File "/home/ayush/Desktop/tutorials/metaworld_rl/metaworld_rl/trainer.py", line 91, in train
    obs = self._train_phase_ppo(obs)
  File "/home/ayush/Desktop/tutorials/metaworld_rl/metaworld_rl/trainer.py", line 235, in _train_phase_ppo
    start_values[new_segment] = val_np[new_segment]

3. Set up experiments for in hand manipulation using environments from gymnasium-robotics, documented here: https://robotics.farama.org/envs/shadow_dexterous_hand/. The enviironments I am interested use the Continuous Touch Sensor, and are: Manipulate Block Touch Sensors (https://robotics.farama.org/envs/shadow_dexterous_hand/manipulate_block_touch_sensors/), Manipulate Egg Touch Sensors (https://robotics.farama.org/envs/shadow_dexterous_hand/manipulate_egg_touch_sensors/) and Manipulate Pen Touch Sensors (https://robotics.farama.org/envs/shadow_dexterous_hand/manipulate_pen_touch_sensors/). Make sure to specifically use python3.10 for all terminal commands. Making environments can be done as follows:
import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)
env = gym.make("FetchReach-v3") # Swap with appropriate env names.

/debate /implement 