from typing import Callable

import gym
import os
import torch as th
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor

from auturi.typing.policy import AuturiPolicy

""" 
Fit abstraction to Auturi Collection Loop Imple, as described below.

    # Get step-finished simulators
    ready_env_refs = self.remoteEnvs.poll(self.batch_size)

    # Find free server and assign ready envs to it
    free_server = -1 # pick free server        
    action_refs = free_server.service(ready_env_refs)

    # send action to remote simulators
    self.remoteEnvs.send_actions(action_refs)        

    return len(ready_env_refs)

"""



def _to_cpu_numpy(tensor):
    return tensor.to("cpu").numpy()

class SB3PolicyAdapter(AuturiPolicy):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_cls: Callable,
        use_sde: bool,
        sde_sample_freq: int,
        model_path: str,
    ):
        self.model_path = model_path
        
        self.policy_model_cls = model_cls
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq


    # Called at the beginning of collection loop
    def load_model(self, device="cpu"):
        self.policy_model = self.policy_model_cls.load(self.model_path, device=device)
        self.policy_model.set_training_mode(False)
        self.device=device


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
            obs_tensor = obs_as_tensor(env_obs, self.device)            
            actions, values, log_probs = self.policy_model(obs_tensor)
                    
        return _to_cpu_numpy(actions), [_to_cpu_numpy(values).flatten(), _to_cpu_numpy(log_probs)]
