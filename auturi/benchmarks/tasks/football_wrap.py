from collections import defaultdict

import numpy as np
import torch

from auturi.benchmarks.tasks.football_hts_rl import Policy as FootballPolicy
from auturi.benchmarks.tasks.football_hts_rl import make_env
from auturi.executor.environment import AuturiEnv
from auturi.executor.policy import AuturiPolicy


class FootballEnvWrapper(AuturiEnv):
    def __init__(self, task_id: str, rank: int):
        self.rank = rank
        self.env = make_env(task_id, rank)()
        self.setup_dummy_env(self.env)
        self.storage = defaultdict(list)
        self.artifacts_samples = [np.array([1.20])]
        self._validate(self.observation_space, self.action_space)

    def step(self, action, artifacts):
        if not isinstance(action, np.ndarray):
            action = np.array([action])

        obs, reward, done, info = self.env.step(action)
        if done:
            obs = self.env.reset()

        self.storage["obs"].append(obs)
        self.storage["action"].append(action)
        self.storage["action_value"].append(artifacts[0])
        self.storage["reward"].append(reward)
        self.storage["done"].append(done)
        self.storage["score_reward"].append(info["score_reward"])

        return obs

    def reset(self):
        self.storage.clear()
        return self.env.reset()

    def seed(self, seed):
        self.env.seed(seed + self.rank)

    def aggregate_rollouts(self):
        return self.storage

    def terminate(self):
        self.env.close()


class FootballPolicyWrapper(AuturiPolicy):
    def __init__(self, task_id: str, idx: int):
        dummy_env = make_env(task_id, rank=0)()
        raw_shape = dummy_env.observation_space.shape
        obs_shape = raw_shape[1:] if len(raw_shape) == 4 else raw_shape
        self._validate(dummy_env.observation_space, dummy_env.action_space)
        dummy_env.close()

        self.policy = FootballPolicy(
            obs_shape,
            1,
            base="CNNBaseGfootball",
            base_kwargs={"recurrent": False, "hidden_size": 512},
        )
        self.device = "cpu"

    def load_model(self, model, device):
        self.device = device
        self.policy.to(device)

    def compute_actions(self, obs, n_steps):
        obs = torch.from_numpy(obs).to(self.device)

        # to be
        value, _, _, action_logit = self.policy.act(obs, None, None)
        ret = torch.distributions.Categorical(logits=action_logit).sample()
        ret = ret.detach().cpu().numpy()
        ret = np.expand_dims(ret, -1)
        action_artifacts = [value.detach().cpu().numpy()[0]]

        return ret, action_artifacts
        # return np.expand_dims(ret, -1)

    def terminate(self):
        pass


# task_id = "academy_3_vs_1_with_keeper"
# env = FootballEnvWrapper(task_id, rank=0)
# policy = FootballPolicyWrapper(task_id, idx=0)
