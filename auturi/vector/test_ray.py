"""
"""

import time
import unittest
from dataclasses import dataclass

import gym
import numpy as np
import timeout_decorator
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from auturi.vector.ray_backend import RayParallelEnv


@dataclass
class DumbEnv(gym.Env):
    def __init__(self, sleep, init_):
        self.sleep = sleep
        self.init_ = init_
        self.reset()
        self.action_space = gym.spaces.Discrete(10)
        self.observation_space = gym.spaces.Discrete(10)

    def step(self, action):
        time.sleep(self.sleep)
        self.returns += action
        return self.returns, -2.3, True, {}

    def reset(self):
        self.returns = self.init_
        return self.returns

    def seed(self, seed):
        pass


class TestRayParallelEnv(unittest.TestCase):
    def setup_gym(self, num_envs, env_name=None):
        if env_name is None:
            env_name = "HalfCheetah-v3"

        def env_fn():
            return gym.make(env_name)

        # return RayParallelEnv([env_fn for _ in range(num_envs)])

    def setUp(self):
        pass

    def test_correctness(self):
        self._test_correctness(lambda: gym.make("HalfCheetah-v3"), 3)

    def _test_correctness(self, env_fn, num_envs):
        SEED = 10
        ray_env = RayParallelEnv([env_fn for _ in range(num_envs)])
        dummy_env = DummyVecEnv([env_fn for _ in range(num_envs)])

        ray_env.seed(seed_dict={i: SEED + i for i in range(num_envs)})
        dummy_env.seed(SEED)

        ray_env.reset()
        dummy_env.reset()

        actions = np.stack([ray_env.action_space.sample() for _ in range(num_envs)])

        def compare_two():
            obs_1, _, _, _ = ray_env.step(actions)
            obs_2, _, _, _ = dummy_env.step(actions)
            self.assertTrue(np.array_equal(obs_1, obs_2))

        for idx in range(10):
            compare_two(actions)

    # def test_poll_all(self):
    #     num_envs = 10
    #     def env_fn(idx):
    #         return DumbEnv(idx * 0.5, idx)

    #     venv = RayParallelEnv([env_fn(i+1) for i in range(num_envs)])

    #     for _ in range(10):
    #         pass
    #     refs = venv.poll(num_envs)
    #     venv

    # def test_user_points_update(self):
    #     pass

    # def test_user_level_change(self):
    #     pass


from contextlib import contextmanager


@contextmanager
def timeoutcontext(timeout=None):
    stime = time.perf_counter()
    yield
    etime = time.perf_counter()
    print(f"Take {round(1000 * (etime-stime), 2)} ms seconds!")
    if timeout is not None:
        assert etime - stime <= timeout


if __name__ == "__main__":
    # unittest.main()

    num_envs = 64
    SEED = 64
    # env_fn = lambda : gym.make("HalfCheetah-v3")
    env_fn = lambda: DumbEnv(sleep=0, init_=21)

    ray_env = RayParallelEnv([env_fn for _ in range(num_envs)])
    # dummy_env = SubprocVecEnv([env_fn for _ in range(num_envs)])
    dummy_env = DummyVecEnv([env_fn for _ in range(num_envs)])

    ray_env.seed(seed_dict={i: SEED + i for i in range(num_envs)})
    dummy_env.seed(SEED)

    ray_env.reset()
    dummy_env.reset()

    actions = np.stack([ray_env.action_space.sample() for _ in range(num_envs)])

    TEST_ENV = dummy_env

    for idx in range(10):
        TEST_ENV.step(actions)

    print(type(TEST_ENV))
    for idx in range(10):
        with timeoutcontext(None):
            TEST_ENV.step(actions)