from collections import defaultdict
import os
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
from sb3_contrib.ppo_recurrent.policies import CnnLstmPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates

import pybullet_envs
from traci.exceptions import FatalTraCIError

import auturi.benchmarks.tasks.finrl_wrap as finrl
import auturi.benchmarks.tasks.flow_wrap as flow 
import auturi.benchmarks.tasks.football_kaggle as football


from auturi.executor.environment import AuturiEnv
from auturi.executor.policy import AuturiPolicy
from functools import partial

def make_env(task_id: str, is_atari_: bool):
    if is_atari_:
        env = make_atari_env(task_id, n_envs=1, vec_env_cls=DummyVecEnv)
        env = VecFrameStack(env, 4)
        return VecTransposeImage(env)

    elif task_id in football.scenarios:
        return VecTransposeImage(DummyVecEnv([football.make_env(task_id)]))

    elif task_id in flow.scenarios:
        return DummyVecEnv([flow.make_env(task_id)])

    elif task_id in ["stock", "portfolio"]:
        return DummyVecEnv([partial(finrl.make_finrl_env, task_id) ])

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

def is_football(env_id: str) -> bool:
    return env_id in football.scenarios

class SB3EnvWrapper(AuturiEnv):
    def __init__(self, task_id: str, rank: int):
        self.rank = rank
        is_atari_ = is_atari(task_id)
        self.is_flow_ = task_id in flow.scenarios
        self.env_fn = lambda: make_env(task_id, is_atari_)
        self.env = self.env_fn()

        # self.env = gym.make("HalfCheetah-v3")
        self.setup_dummy_env(self.env)
        self.storage = defaultdict(list)
        self.artifacts_samples = [np.array([[1.1, 1.4]])]

        #self._validate(self.observation_space, self.action_space)

        if is_atari_ or is_football(task_id):
            os.environ["ENV_SKIP_OBS_COPY"] = "1"

    def step(self, actions, artifacts):
        # if action.ndim == self.action_space.sample().ndim:
        #     action = np.expand_dims(action, -1)

        action = actions
        if isinstance(self.action_space, gym.spaces.Discrete):
            action = actions[0]

        clipped_actions = action
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = np.clip(
                action, self.action_space.low, self.action_space.high
            )
        obs, reward, done, _ = self.env.step(clipped_actions)  # all list


        self.storage["obs"].append(obs)
        self.storage["action"].append(actions)
        self.storage["action_value"].append(artifacts[0])
        self.storage["reward"].append(reward)
        self.storage["done"].append(done)

        return obs

    def reset(self):
        self.storage.clear()
        try:
            return self.env.reset()
        except Exception as e:
            self.env.close()
            print("CIError. Close and Restart.")
            self.env = self.env_fn()
            return self.env.reset()

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
        self.is_football_ = task_id in football.scenarios
        dummy_env = make_env(task_id, self.is_atari_)
        policy_cls = ActorCriticCnnPolicy if (self.is_atari_ or self.is_football_) else ActorCriticPolicy
        policy_kwargs = dict(
            observation_space=dummy_env.observation_space, 
            action_space=dummy_env.action_space,
            lr_schedule=lambda _: 0.001
        )
        if self.is_football_:
            policy_kwargs.update(dict(features_extractor_class=football.FootballCNN, features_extractor_kwargs=dict(features_dim=256)))
        self.policy = policy_cls(**policy_kwargs)
        self.policy.set_training_mode(False)
        #self._validate(dummy_env.observation_space, dummy_env.action_space)

        dummy_env.close()

    def load_model(self, model, device):
        self.device = device
        self.policy.to(device)

    def compute_actions(self, obs, n_steps):
        obs = torch.from_numpy(obs).to(self.device)
        actions, values, log_probs = self.policy(obs)
        actions = _to_cpu_numpy(actions)
        artifacts = np.stack(
            [_to_cpu_numpy(values).flatten(), _to_cpu_numpy(log_probs)], 1
        )
        if self.is_atari_ or self.is_football_:
            actions = np.expand_dims(actions, -1)
        return actions, [artifacts]

    def terminate(self):
        del self.policy
        torch.cuda.empty_cache()


class SB3LSTMPolicyWrapper(SB3PolicyWrapper):
    def __init__(self, task_id: str, idx: int):
        self.device = "cpu"
        assert is_atari(task_id)
        dummy_env = make_env(task_id, True)
        policy_kwargs = dict(
            observation_space=dummy_env.observation_space, 
            action_space=dummy_env.action_space,
            lr_schedule=lambda _: 0.001
        )
        self.policy = CnnLstmPolicy(**policy_kwargs)
        self.policy.set_training_mode(False)
        #self._validate(dummy_env.observation_space, dummy_env.action_space)

        dummy_env.close()


    def compute_actions(self, obs, n_steps):
        obs = torch.from_numpy(obs).to(self.device)
        batch_size = len(obs)
        lstm = self.policy.lstm_actor
        single_hidden_state_shape = (lstm.num_layers, batch_size, lstm.hidden_size)
        lstm_states = RNNStates(
            (
                torch.zeros(single_hidden_state_shape).to(self.device),
                torch.zeros(single_hidden_state_shape).to(self.device),
            ),
            (
                torch.zeros(single_hidden_state_shape).to(self.device),
                torch.zeros(single_hidden_state_shape).to(self.device),
            ),
        )
        episode_starts = torch.tensor([False] * len(obs)).float().to(self.device)
        actions, values, log_probs, lstm_states = self.policy.forward(obs, lstm_states, episode_starts)

        actions = _to_cpu_numpy(actions)
        artifacts = np.stack(
            [_to_cpu_numpy(values).flatten(), _to_cpu_numpy(log_probs)], 1
        )
        actions = np.expand_dims(actions, -1)
        return actions, [artifacts]

