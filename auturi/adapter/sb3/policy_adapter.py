from typing import Callable

import gym
import time
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
        
        self.time_ms = []


    # Called at the beginning of collection loop
    def load_model(self, model, device="cpu"):
        if len(self.time_ms) > 0:
            with open("policy.txt", "w") as f:
                print(self.time_ms)
            self.time_ms.clear()
        
        
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
        start_time = time.perf_counter()

        # Sample a new noise matrix

        if self._to_sample_noise(n_steps):
            self.policy_model.reset_noise(len(env_obs))

        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(env_obs, self.device)
            actions, values, log_probs = self.policy_model(obs_tensor)

        ret = _to_cpu_numpy(actions), [
            _to_cpu_numpy(values).flatten(),
            _to_cpu_numpy(log_probs),
        ]
        end_time = time.perf_counter()
        self.time_ms.append(end_time-start_time)
        
        return ret
        

