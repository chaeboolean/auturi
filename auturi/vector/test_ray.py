"""
"""

from multiprocessing import dummy
import time
import unittest
from dataclasses import dataclass

import gym
import numpy as np
import timeout_decorator
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from auturi.vector.ray_backend import RayParallelEnv
from auturi.vector.shm_backend import SHMParallelEnv


@dataclass
class DumbEnv(gym.Env):
    def __init__(self, sleep, init_):
        self.sleep = sleep
        self.init_ = init_
        self.action_space = gym.spaces.Discrete(10)
        self.observation_space = gym.spaces.Box(low=-10, high=30, shape=(14,2), dtype=np.float32)
        self.returns = self.observation_space.sample()

    def step(self, action):
        #Atime.sleep(self.sleep)
        
        self.returns += action
        #obs = np.copy(self.returns)
        #print(f"Env({self.init_})", self.returns[0], action)

        return self.returns, -2.3, False, {}

    def reset(self):
        self.returns.fill(self.init_)
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

    # def _test_correctness(self, env_fn, num_envs):
    #     SEED = 10
    #     ray_env = SHMParallelEnv([env_fn for _ in range(num_envs)])
    #     dummy_env = DummyVecEnv([env_fn for _ in range(num_envs)])

    #     # ray_env.seed(seed_dict={i: SEED + i for i in range(num_envs)})
    #     ray_env.seed(SEED)
    #     dummy_env.seed(SEED)

    #     ray_env.reset()
    #     dummy_env.reset()

    #     #actions = np.stack([ray_env.action_space.sample() for _ in range(num_envs)])
    #     actions = np.stack([2 for _ in range(num_envs)])

    #     def compare_two(actions):
    #         obs_1 = ray_env.step(actions)
    #         # obs_1, _, _, _ = ray_env.step(actions)
    #         obs_2, _, _, _ = dummy_env.step(actions)
    #         print(np.array_equal(obs_1, obs_2), "((((((")

    #         self.assertTrue(np.array_equal(obs_1, obs_2))
    #         assert False

    #     for idx in range(10):
    #         compare_two(actions)

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


def _test_correctness(env_fn, num_envs):
    SEED = 10
    ray_env = SHMParallelEnv([env_fn for i in range(num_envs)])
    dummy_env = DummyVecEnv([env_fn for i in range(num_envs)])

    ray_env.seed(SEED)
    dummy_env.seed(SEED)
    
    print(111111)

    a = ray_env.reset()
    
    b = dummy_env.reset()
    
    print(2222)


    print(f"RESET!!! SHM={a}\n\n DURMMY={b} \n\n")


    def compare_two(actions):
        #obs_1 = ray_env.step(actions)
        obs_1, _, _, _ = ray_env.step(actions)
        obs_2, _, _, _ = dummy_env.step(actions)
        #print(f"SHM={obs_1}\n\n DURMMY={obs_2} \n\n")
        # print(obs_1.dtype, obs_2.dtype)
        if not np.array_equal(obs_1, obs_2):
            print(obs_1 - obs_2)
        #    pass

        # self.assertTrue(np.array_equal(obs_1, obs_2))
        # assert False

    for _ in range(3):
        ray_env.start_loop()
        for idx in range(20):
            actions = np.stack([dummy_env.action_space.sample() for _ in range(num_envs)])
            compare_two(actions)
            print(idx, " idx\n\n\n")
            
        ray_env.finish_loop()
        
    
        
    ray_env.close()


if __name__ == "__main__":
    # unittest.main()
    
    env_fn = lambda: gym.make("HalfCheetah-v3")
    #env_fn = lambda: gym.make("CartPole-v1")
    def wrap_dumb(init):
        return lambda: DumbEnv(init * 0.1, init)
    
    
    #env = SubprocVecEnv([wrap_dumb(3), wrap_dumb(10)])
    #env.reset()
    
    # for step in range(10):
    #     obs, _, _, _ = env.step([2, 2])
    #     print(f"{step}: => {obs}")
    
    _test_correctness(env_fn, 2)
    exit(0)
    num_envs = 2
    SEED = 64
    # env_fn = lambda: DumbEnv(sleep=0, init_=21)

    ray_env = RayParallelEnv([env_fn for _ in range(num_envs)])
    subproc_env = SubprocVecEnv([env_fn for _ in range(num_envs)])
    dummy_env = DummyVecEnv([env_fn for _ in range(num_envs)])
    shm_env = SHMParallelEnv([env_fn for _ in range(num_envs)])

    # shm_env.seed(seed_dict={i: SEED + i for i in range(num_envs)})
    shm_env.seed(SEED)
    dummy_env.seed(SEED)

    shm_env.reset()
    shm_env.start_loop()
    dummy_env.reset()

    actions = np.stack([dummy_env.action_space.sample() for _ in range(num_envs)])

    TEST_ENV = subproc_env

    for TEST_ENV in [ray_env, subproc_env, dummy_env, shm_env]:
        TEST_ENV.reset()

        if isinstance(TEST_ENV, SHMParallelEnv):
            TEST_ENV.start_loop()

        for idx in range(10):
            TEST_ENV.step(actions)

        print(type(TEST_ENV))
        for idx in range(10):
            with timeoutcontext(None):
                TEST_ENV.step(actions)
