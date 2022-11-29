"""
Training code of Google Football environment.
Code implementation & hyperparameter setting are from Kaggle. 
https://www.kaggle.com/code/kwabenantim/gfootball-stable-baselines3
(Did not check model final reward personally.)

"""
from collections import deque

import gym
import numpy as np
import torch.nn as nn
from gfootball.env import create_environment, observation_preprocessing
from gym.spaces import Box, Discrete
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# Export Football env create function
def make_env(name, rank=0):
    def _init():
        env = FootballGym({"env_name": name})
        return env

    return _init


class FootballGym(gym.Env):
    spec = None
    metadata = None

    def __init__(self, config=None):
        super(FootballGym, self).__init__()
        env_name = "academy_empty_goal_close"
        rewards = "scoring,checkpoints"
        if config is not None:
            env_name = config.get("env_name", env_name)
            rewards = config.get("rewards", rewards)
        self.env = create_environment(
            env_name=env_name,
            stacked=False,
            representation="raw",
            rewards=rewards,
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            render=False,
            write_video=False,
            dump_frequency=1,
            logdir=".",
            extra_players=None,
            number_of_left_players_agent_controls=1,
            number_of_right_players_agent_controls=0,
        )
        self.action_space = Discrete(19)
        self.observation_space = Box(
            low=0, high=255, shape=(72, 96, 16), dtype=np.uint8
        )
        self.reward_range = (-1, 1)
        self.obs_stack = deque([], maxlen=4)

    def transform_obs(self, raw_obs):
        obs = raw_obs[0]
        obs = observation_preprocessing.generate_smm([obs])
        if not self.obs_stack:
            self.obs_stack.extend([obs] * 4)
        else:
            self.obs_stack.append(obs)
        obs = np.concatenate(list(self.obs_stack), axis=-1)
        obs = np.squeeze(obs)
        return obs

    def reset(self):
        self.obs_stack.clear()
        obs = self.env.reset()
        obs = self.transform_obs(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step([action])
        obs = self.transform_obs(obs)
        return obs, float(reward), done, info


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True
    )


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.conv2 = conv3x3(out_channels, out_channels, stride)

    def forward(self, x):
        residual = x
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out


class FootballCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        in_channels = observation_space.shape[0]  # channels x height x width
        self.cnn = nn.Sequential(
            conv3x3(in_channels=in_channels, out_channels=32),
            nn.MaxPool2d(kernel_size=3, stride=2, dilation=1, ceil_mode=False),
            ResidualBlock(in_channels=32, out_channels=32),
            ResidualBlock(in_channels=32, out_channels=32),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=52640, out_features=features_dim, bias=True),
            nn.ReLU(),
        )

    def forward(self, obs):
        return self.linear(self.cnn(obs))


# Export Football scenario names
_scenarios = {
    0: "academy_empty_goal_close",
    1: "academy_empty_goal",
    2: "academy_run_to_score",
    3: "academy_run_to_score_with_keeper",
    4: "academy_pass_and_shoot_with_keeper",
    5: "academy_run_pass_and_shoot_with_keeper",
    6: "academy_3_vs_1_with_keeper",
    7: "academy_corner",
    8: "academy_counterattack_easy",
    9: "academy_counterattack_hard",
    10: "academy_single_goal_versus_lazy",
    11: "11_vs_11_kaggle",
}
scenarios = list(_scenarios.values())
