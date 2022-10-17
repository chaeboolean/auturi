from typing import Any, Callable, Dict, Union

import gym
import numpy as np
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


class SB3PolicyAdapter(AuturiPolicy):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_fn: Callable,
        use_sde: bool,
        sde_sample_freq: int,
    ):
        
        self.device = "cpu"  # TODO ????
        policy_model = model_fn()
        self.policy_model = policy_model.to(self.device)
        
        self.policy_model.set_training_mode(False)
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq

        print(type(self.policy_model), " (((((((( ", model_fn)

    # Called at the beginning of collection loop
    def reset(self):
        pass

    
    def set_device(self, device: str):
        pass


    def _to_sample_noise(self, n_steps):
        return (
            self.use_sde
            and self.sde_sample_freq > 0
            and n_steps % self.sde_sample_freq == 0
        )

    def compute_actions(self, env_obs, n_steps=3):
        # Sample a new noise matrix

        print("\n^^^^ Compute actions!")

        if self._to_sample_noise(n_steps):
            self.policy_model.reset_noise(len(env_obs))

        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(env_obs, self.device)            
            actions, values, log_probs = self.policy_model(obs_tensor)
            
            print("output=>", actions, values, log_probs)
            print("output=>", actions.dtype, values.dtype, log_probs.dtype)
        
        return actions, values, log_probs
