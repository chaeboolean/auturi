import math
import signal
import time
from collections import defaultdict
from functools import partial

import gym
import numpy as np
import ray
import torch

from auturi.executor.environment import AuturiEnv
from auturi.executor.gym_utils import get_action_sample
from auturi.executor.policy import AuturiPolicy


class Timeout:
    """Testing helper class that emits TimeoutError.

    Assert that time for executing code is in range [min_sec, max_sec].

    """

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


def make_space(is_action_discrete: bool):
    if is_action_discrete:
        action_space = gym.spaces.Discrete(10)
    else:
        action_space = gym.spaces.Box(low=-10, high=30, shape=(3,), dtype=np.float32)
    obs_space = gym.spaces.Box(low=-10, high=30, shape=(5, 2), dtype=np.float32)

    return obs_space, action_space


def make_dummy_action(is_action_discrete: bool, num: int):
    action_sample = make_space(is_action_discrete)[1].sample()
    if is_action_discrete:
        return np.stack([np.array([action_sample])] * num)
    else:
        return np.stack([action_sample] * num)


class DumbEnv(AuturiEnv):
    """Dumb environment class for testing.

    Initialized with self.value = 1000 * (idx+1).
    When step function called, 1) sleeps for 0.5 seconds,
    and 2) adds action to self.value, 3) return self.value

    """

    def __init__(self, idx, is_action_discrete):
        self.idx = idx
        self.init_value = 1000 * (idx + 1)
        self.sleep = 0.5

        self.observation_space, self.action_space = make_space(is_action_discrete)
        self.metadata = None

        self.value = np.expand_dims(self.observation_space.sample(), 0)
        self.artifacts_samples = [get_action_sample(self.action_space, 1)]

        self.storage = defaultdict(list)

        self._validate(self.observation_space, self.action_space)

    def step(self, action, _):
        time.sleep(self.sleep)
        self.value += action.flatten()[0]
        self.storage["obs"].append(np.copy(self.value))
        self.storage["action"].append(np.copy(action))

        return np.copy(self.value)

    def reset(self):
        self.storage.clear()

        self.value.fill(self.init_value)
        return np.copy(self.value)

    def seed(self, seed):
        pass

    def aggregate_rollouts(self):
        return self.storage

    def terminate(self):
        pass


def check_timeout(elapsed, timeout):
    assert timeout <= elapsed and timeout + 0.5 >= elapsed


def check_array(given, answer, keep_order=True):
    if keep_order:
        assert np.all(given == np.array(answer))
    else:
        assert set(given) == set(answer)


class DumbPolicy(AuturiPolicy):
    """Dumb policy class for testing.

    Initialized with self.value = 10 ^ idx
    Internally manages cnt variable which counts the number of compute_actions called.
    When compute_actions function called, 1) sleeps for 1 second,
    and 2) increase self.value by 1, 3) return self.value as action.

    """

    def __init__(self, idx, is_action_discrete):
        self.idx = idx
        self.init_value = 0 if idx == 0 else math.pow(10, idx)
        self.sleep = 1
        self.device = None
        self.observation_space, self.action_space = make_space(is_action_discrete)

        self.action_sample = get_action_sample(self.action_space, 1)
        self._validate(self.observation_space, self.action_space)

    def load_model(self, model, device):
        self.device = device
        self.action_sample.fill(self.init_value)

    # TODO: consider obs dimension.
    def compute_actions(self, obs, n_steps):
        time.sleep(self.sleep)
        self.action_sample += 1
        actions = np.concatenate([self.action_sample] * obs.shape[0])

        return actions, [actions]

    def terminate(self):
        pass


class DummyPolicyHandler:
    def __init__(self, out: np.ndarray):
        self.out = out

    def start_loop(self):
        pass

    def stop_loop(self):
        pass

    def compute_actions(self, *args, **kwargs):
        return self.out

    def terminate(self):
        pass


def _create_fn(idx, is_dicrete_action):
    return DumbEnv(idx, is_dicrete_action)


def create_env_fns(num_envs, is_dicrete_action):
    return [partial(_create_fn, idx, is_dicrete_action) for idx in range(num_envs)]


def create_policy_args(is_dicrete_action):
    model = torch.nn.Linear(10, 10, bias=False)
    model.eval()
    torch.nn.init.uniform_(model.weight, 1, 1)
    return model, DumbPolicy, {"is_action_discrete": is_dicrete_action}


@ray.remote
def mock_ray(obj):
    return obj
