import signal
import time
from collections import defaultdict

import gym
import numpy as np
import ray
import torch

from auturi.typing.environment import AuturiEnv
from auturi.typing.policy import AuturiPolicy


class Timeout:
    """Assert that time for executing code is in range [min_sec, max_sec]."""

    def __init__(self, min_sec, max_sec):
        self.min_sec = min_sec
        self.max_sec = max_sec

    def handle_timeout(self, signum, frame):
        if self.elapsed < self.min_sec:
            raise TimeoutError(f"Timeout: Took Only {round(self.elapsed, 2)} seconds.")
        if self.elapsed > self.max_sec:
            error_msg = f"Timeout: Over {round(self.max_sec, 2)} seconds."
            raise TimeoutError(error_msg)

        print(f"Timeout: Took Only {round(self.elapsed, 2)} seconds.")

    def __enter__(self):
        self.start_time = time.time()
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(int(self.max_sec + 1))

    def __exit__(self, type, value, traceback):
        self.elapsed = time.time() - self.start_time
        signal.alarm(0)
        self.handle_timeout(None, None)


class DumbEnv(AuturiEnv):
    def __init__(self, idx, sleep):
        self.sleep = sleep
        self.init_ = idx * 10
        self.action_space = gym.spaces.Discrete(10)
        self.observation_space = gym.spaces.Box(
            low=-10, high=30, shape=(5, 2), dtype=np.float32
        )
        self.returns = self.observation_space.sample()
        self.metadata = None

        self.storage = defaultdict(list)

    def step(self, action, _):
        time.sleep(self.sleep)
        self.returns += action
        self.storage["obs"].append(np.copy(self.returns))
        self.storage["action"].append(np.copy(action))
        return np.copy(self.returns)  # , -2.3, False, {}

    def reset(self):
        time.sleep(self.sleep)
        self.storage.clear()
        self.returns.fill(self.init_)
        return self.returns

    def seed(self, seed):
        pass

    def aggregate_rollouts(self):
        return self.storage

    def close(self):
        pass


class DumbPolicy(AuturiPolicy):
    def __init__(self, idx):
        self.idx = idx
        self.sleep = 1
        self.cnt = 0
        self.device = None
        self.observation_space = gym.spaces.Box(
            low=-10, high=30, shape=(10,), dtype=np.float32
        )

    def load_model(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.cnt = 0  # reset

    def compute_actions(self, obs, n_steps):
        obs = self.np_to_tensor(obs)
        time.sleep(self.sleep)

        self.cnt += 1
        out = self.model(obs) + self.idx * 10 + self.cnt
        out = self.tensor_to_np(out)

        return out

    def np_to_tensor(self, np_array):
        return torch.Tensor(np_array).to(self.device)

    def tensor_to_np(self, tensor):
        return tensor.detach().cpu().numpy()


def create_env_fns(num_envs):
    def env_fn(idx, sleep):
        def _wrap():
            return DumbEnv(idx, sleep)

        return _wrap

    return [env_fn(idx=idx, sleep=1 + idx) for idx in range(num_envs)]


@ray.remote
def mock_ray(obj):
    return obj
