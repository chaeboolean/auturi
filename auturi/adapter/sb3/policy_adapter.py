from typing import Callable

import gym
import numpy as np
import torch as th
from stable_baselines3.common.utils import obs_as_tensor

from auturi.executor.policy import AuturiPolicy


def _to_cpu_numpy(tensor):
    return tensor.to("cpu").numpy()


class SB3PolicyAdapter(AuturiPolicy):
    def __init__(
        self,
        idx: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_cls: Callable,
        use_sde: bool,
        sde_sample_freq: int,
        model_path: str,
    ):
        self.policy_idx = idx
        self.model_path = model_path

        self.policy_model_cls = model_cls

        self.observation_space = observation_space
        self.action_space = action_space
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq
        self.device = "cpu"

    # Called at the beginning of collection loop
    def load_model(self, model, device="cpu"):
        self.policy_model = self.policy_model_cls.load(self.model_path, device=device)
        self.policy_model.set_training_mode(False)
        self.device = device

    def _to_sample_noise(self, n_steps):
        return (
            self.use_sde
            and self.sde_sample_freq > 0
            and n_steps % self.sde_sample_freq == 0
        )

    def compute_actions(self, env_obs, n_steps=3):
        # Sample a new noise matrix

        if self._to_sample_noise(n_steps):
            self.policy_model.reset_noise(len(env_obs))

        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs = th.from_numpy(env_obs).to(self.device)
            actions, values, log_probs = self.policy_model(obs)

        actions = _to_cpu_numpy(actions)
        artifacts = np.stack(
            [_to_cpu_numpy(values).flatten(), _to_cpu_numpy(log_probs)], 1
        )
        if isinstance(self.action_space, gym.spaces.Discrete):
            actions = np.expand_dims(actions, -1)
            
        return actions, [artifacts]

    def terminate(self):
        del self.policy_model
        th.cuda.empty_cache()
