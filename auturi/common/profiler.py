"""
This file profiles tasks before actual runtime.

"""
import time
import numpy as np
from auturi.executor.environment import AuturiEnv
from auturi.executor.policy import AuturiPolicy

SEED_LIST = [1, 12, 123, 1234, 12345]
NUM_REPETITION = 1e4

def measure_step_time(env: AuturiEnv, num_repeat=NUM_REPETITION):
    ret = []
    for seed in SEED_LIST:
        env.seed(seed)
        env.reset()
        for _ in range(int(num_repeat)):
            action = env.sample_action()
            stime = time.perf_counter()
            env.step(action)
            etime = time.perf_counter()
            ret.append(etime - stime)


    return np.array(ret)


def measure_action_generation(policy: AuturiPolicy, device: str, bs: int):
    ret = []
    policy.load_model(None, device)
    WARMUPS = 10

    obs = policy.sample_observation(bs=bs)
    for idx in range(WARMUPS + NUM_REPETITION):
        stime = time.perf_counter()
        policy.compute_actions(obs, -1)
        etime = time.perf_counter()

        if idx >= WARMUPS:
            ret.append(etime - stime)

    return np.array(ret)


def compute_data_size(env: AuturiEnv, policy: AuturiPolicy):
    obs = policy.sample_observation(bs=1)
    action = env.sample_action()

    return obs.size, action.size
