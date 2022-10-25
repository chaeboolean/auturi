import argparse
import time

import gym
import numpy as np

from auturi.tuner import AuturiTuner
from auturi.executor import AuturiExecutor
from auturi.vector.ray_backend import RayParallelEnv, RayVectorPolicies
import auturi.test.utils as utils

def main():
    min_num_env = 32
    max_num_env = 64

    # In real example, user should implement env_fns, model, policy_cls, policy_kwargs.
    env_fns = utils.create_env_fns(max_num_env)
    model, policy_cls, policy_kwargs = utils.create_vector_policy()

    tuner = AuturiTuner(min_num_env=min_num_env, max_num_env=max_num_env)
    vector_envs = RayParallelEnv(env_fns)
    vector_policy = RayVectorPolicies(policy_cls, policy_kwargs)

    executor = AuturiExecutor(vector_envs, vector_policy, tuner)   
    executor.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auturi Motivation Experiments, without Tuner implemented."
    )
    main()
