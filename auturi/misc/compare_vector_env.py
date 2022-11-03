"""
Compare the performance of VectorEnv implementations
RayVectorEnv and SHMVectorEnv
"""
import time

import gym
import numpy as np
from auturi.executor.environment import AuturiEnv, AuturiVectorEnv
from auturi.executor.ray import RayParallelEnv
from auturi.executor.shm import create_shm_actor_args
from auturi.test.test_env import mock_reconfigure
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv


class GymAuturiWrapper(AuturiEnv):
    def __init__(self, task_id):
        self.env = gym.make(task_id)
        self.setup_with_dummy(self.env)

    def step(self, action, _):
        observation, reward, done, info = self.env.step(action)
        if done:
            observation = self.env.reset()
        return observation

    def reset(self):
        return self.env.reset()

    def seed(self, seed):
        return self.env.seed(seed)

    def aggregate_rollouts(self):
        return dict()

    def terminate(self):
        self.env.close()


def create_vector_env(num_envs, backend, task_id="HalfCheetah-v3"):

    if backend == "subproc":
        env_fns = [lambda: gym.make(task_id) for _ in range(num_envs)]
        return SubprocVecEnv(env_fns)

    env_fns = [lambda: GymAuturiWrapper(task_id) for _ in range(num_envs)]
    if backend == "ray":
        vector_env = RayParallelEnv(env_fns)

    elif backend == "shm":
        action_ = GymAuturiWrapper(task_id).action_space.sample()

        class DummyPolicy:
            def compute_actions(self, obs, n_steps):
                return action_, np.array([[1.1]])

        create_shm_env, _ = create_shm_actor_args(env_fns, DummyPolicy, {})
        vector_env = create_shm_env()

    mock_reconfigure(vector_env, num_envs, num_envs)  # num_parallel = num_envs
    return vector_env


def run(num_envs, num_steps, backend):
    vector_env = create_vector_env(num_envs, backend)
    actions = np.stack([vector_env.action_space.sample()] * num_envs)
    artifacts = np.random.randn(num_envs, 1)

    if isinstance(vector_env, AuturiVectorEnv):
        vector_env.start_loop()
        actions = [actions, [artifacts]]
    else:
        vector_env.reset()

    stime = time.perf_counter()
    for step in range(num_steps):
        obs = vector_env.step(actions)

    etime = time.perf_counter()

    if isinstance(vector_env, AuturiVectorEnv):
        vector_env.stop_loop()
        vector_env.terminate()
    else:
        vector_env.close()

    return etime - stime


if __name__ == "__main__":
    for num_env in [4, 16, 32, 64, 70, 110, 128, 512]:
        for backend in ["ray", "subproc", "shm"]:

            time_ = run(num_env, 100, backend=backend)

            print(f"{backend}, num_env={num_env}, time={time_}")
