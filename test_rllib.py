# __quick_start_begin__
import yaml
import json
import os
import time
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.dqn import DQNTrainer

# Profiling configs
NUM_ITER = 3

# Open config file
dir_path = "/home/ooffordable/ray_fork/ray-ray-2.0.0/rllib/tuned_examples/ppo/"
config_file = "halfcheetah-ppo.yaml"
with open(dir_path+config_file) as f:
    _conf = yaml.safe_load(f)

_conf = list(_conf.values())[0]
config = _conf["config"]
config.update({"env": _conf["env"]})
print(config)
exit(0)
# Create an RLlib Trainer instance.
trainer = PPOTrainer(config)
for i in range(NUM_ITER):
    results = trainer.train()
    print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")

ctr = 0
for worker in trainer.workers.remote_workers():
    worker.save_recorder.remote()
    ctr += 1

time.sleep(3)