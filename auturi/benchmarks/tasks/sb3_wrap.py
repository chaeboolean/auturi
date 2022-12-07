from collections import defaultdict

import gym
import numpy as np
import torch
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

from auturi.benchmarks.tasks.finrl_wrap import make_finrl_env
from auturi.executor.environment import AuturiEnv
from auturi.executor.policy import AuturiPolicy


def make_env(task_id: str, is_atari_: bool):
    if is_atari_:
        env = make_atari_env(task_id, n_envs=1, vec_env_cls=DummyVecEnv)
        env = VecFrameStack(env, 4)
        return VecTransposeImage(env)

    elif task_id in ["stock", "portfolio"]:
        return make_finrl_env(task_id)

    else:
        env_fn = lambda: gym.make(task_id)
        return DummyVecEnv([env_fn])


def _to_cpu_numpy(tensor):
    return tensor.detach().to("cpu").numpy()


def is_atari(env_id: str) -> bool:
    try:
        entry_point = gym.envs.registry.env_specs[env_id].entry_point
        return "AtariEnv" in str(entry_point)
    except KeyError as e:
        return False


class SB3EnvWrapper(AuturiEnv):
    def __init__(self, task_id: str, rank: int):
        self.rank = rank
        is_atari_ = is_atari(task_id)
        self.env = make_env(task_id, is_atari_)

        # self.env = gym.make("HalfCheetah-v3")
        self.setup_dummy_env(self.env)
        self.storage = defaultdict(list)
        self.artifacts_samples = [np.array([1.1, 1.4])]

    def step(self, action, artifacts):
        # if action.ndim == self.action_space.sample().ndim:
        #     action = np.expand_dims(action, -1)
        if not isinstance(action, np.ndarray):
            action = np.array([action])

        obs, reward, done, _ = self.env.step(action)
        obs, reward, done = obs[0], reward[0], done[0]

        self.storage["obs"].append(obs)
        self.storage["action"].append(action)
        self.storage["action_value"].append(artifacts[0])
        self.storage["reward"].append(reward)
        self.storage["done"].append(done)

        return obs

    def reset(self):
        self.storage.clear()
        return self.env.reset()[0]

    def seed(self, seed):
        self.env.seed(seed + self.rank)

    def aggregate_rollouts(self):
        return self.storage

    def terminate(self):
        self.env.close()


class SB3PolicyWrapper(AuturiPolicy):
    def __init__(self, task_id: str, idx: int):
        self.device = "cpu"
        self.is_atari_ = is_atari(task_id)
        dummy_env = make_env(task_id, self.is_atari_)
        policy_cls = ActorCriticCnnPolicy if self.is_atari_ else ActorCriticPolicy
        self.policy = policy_cls(
            dummy_env.observation_space, dummy_env.action_space, lambda _: 0.001
        )
        self.policy.set_training_mode(False)
        dummy_env.close()

    def load_model(self, model, device):
        self.device = device
        self.policy.to(device)

    def compute_actions(self, obs, n_steps):
        obs = torch.from_numpy(obs).to(self.device)
        actions, values, log_probs = self.policy(obs)
        actions = _to_cpu_numpy(actions)
        artifacts = np.array(
            [_to_cpu_numpy(values).flatten()[0], _to_cpu_numpy(log_probs)[0]]
        )
        if self.is_atari_:
            actions = np.expand_dims(actions, -1)

        return actions, [artifacts]

    def terminate(self):
        pass
