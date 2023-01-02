import subprocess
import time
import yaml
import gym
from itertools import chain
from auturi.benchmarks.tasks.football_wrap import FootballScenarios


TOTAL_GPUS = 6
GPU_ID = 3
NUM_TRIAL = 1
NUM_ENVS = 64

TRIAL_ID = 0

import re
import os

AUTURI_PATH = "/home/ooffordable/docker_making/auturi"
LOG_PATH = "/home/ooffordable/docker_making/log_atc/football_series_0102.txt"

def check_env_exist(env_name):
    try:
        env = gym.make(env_name)
        env.close()
        return True
    except gym.error.UnregisteredEnv as e:
        return False

def _visible_gpus(given_gpu):
    global GPU_ID
    ret = ",".join([str(GPU_ID + i) for i in range(given_gpu)])
    GPU_ID = (GPU_ID + given_gpu) % TOTAL_GPUS
    
    print(ret)
    return ret

def incr_trial():
    global TRIAL_ID
    TRIAL_ID += 1
    return TRIAL_ID


def try_once(env_name):
    my_env = os.environ
    my_env["CUDA_VISIBLE_DEVICES"] = str(_visible_gpus(1))
    commands = ["python", f"{AUTURI_PATH}/auturi/benchmarks/collection_loop.py", 
                f"--num-iteration=5", f"--num-envs=64",  f"--env={env_name}", f"--num-collect=1280", 
                f"--tuner-log-path={LOG_PATH}"]
              

    print(f"Starting {' '.join(commands)}")

    subprocess.run(commands, env=my_env)

    time.sleep(5)


if __name__ == "__main__":    
        
    atari_envs = ["Adventure-v4", "AirRaid-v4", "Alien-v4", "Amidar-v4", "Assault-v4", "Asterix-v4", \
        "Asteroids-v4", "Atlantis-v4", "BankHeist-v4"]
    
    for env_name in FootballScenarios:
        try_once(env_name)
    
        
    # pathname = f"/home/ooffordable/auturi/rl-baselines3-zoo/hyperparams/ppo.yml"
    # with open(pathname) as f:
    #     params = yaml.load(f, Loader=yaml.FullLoader)
        
    # for env_name in params.keys():
    #     if env_name in ["atari", "football"]:
    #         continue
            
    #     elif check_env_exist(env_name):
    #         try_once(env_name)
    
