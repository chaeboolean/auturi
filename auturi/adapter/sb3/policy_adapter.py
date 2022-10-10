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
        device: str,
        use_sde: bool,
        sde_sample_freq: int,
    ):

        policy_model = model_fn()
        self.policy_model = policy_model.to("cuda:0")
        print(type(self.policy_model), " (((((((( ", model_fn)
        

        self.policy_model.set_training_mode(False)
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq

        self.device = "cuda:0"  # TODO ????

    # Called at the beginning of collection loop
    def reset(self):
        self.rollout_buffer.reset()

    def create_buffer(self):
        buffer_cls = (
            DictRolloutBuffer
            if isinstance(self.observation_space, gym.spaces.Dict)
            else RolloutBuffer
        )

        # buffer size calculation:..
        self.rollout_buffer = buffer_cls(
            buffer_size=200000,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=1,
        )

    def _to_sample_noise(self, n_steps):
        return True
        return (
            self.use_sde
            and self.sde_sample_freq > 0
            and n_steps % self.sde_sample_freq == 0
        )

    def compute_actions(self, env_obs, env_rewards=None, env_dones=None, env_infos=None, n_steps=3):
        # Sample a new noise matrix
        
        if self._to_sample_noise(n_steps):
            self.policy_model.reset_noise(len(env_obs))

        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(env_obs, self.device)
            print("obs_tensor=>", obs_tensor.device)
            actions, values, log_probs = self.policy_model(obs_tensor)
        
        actions = actions.cpu().numpy()

        # Rescale and perform action
        clipped_actions = actions

        # Clip the actions to avoid out of bound error
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = np.clip(
                actions, self.action_space.low, self.action_space.high
            )

        # set as attribute, for insert_buffer operation
        self.last_env_obs = env_obs
        self.last_env_rewards = env_rewards
        self.last_env_dones = env_dones
        self.last_env_infos = env_infos
        self.last_actions = actions
        self.last_values = values
        self.last_log_probs = log_probs

        return clipped_actions

    def insert_buffer(self):
        if isinstance(self.action_space, gym.spaces.Discrete):
            # Reshape in case of discrete action
            self.last_actions = self.last_actions.reshape(-1, 1)

        # Handle timeout by bootstraping with value function
        # see GitHub issue #633
        for idx, done in enumerate(self.last_env_dones):
            if (
                done
                and self.last_env_infos[idx].get("terminal_observation") is not None
                and self.last_env_infos[idx].get("TimeLimit.truncated", False)
            ):
                terminal_obs = self.policy_model.obs_to_tensor(
                    self.last_env_infos[idx]["terminal_observation"]
                )[0]
                with th.no_grad():
                    terminal_value = self.policy_model.predict_values(terminal_obs)[0]
                self.last_env_rewards[idx] += self.gamma * terminal_value

        if not hasattr(self, "_last_episode_starts"):
            self._last_episode_starts = np.ones((len(self.last_env_dones)), dtype=bool)

        self.rollout_buffer.add(
            self.last_env_obs,
            self.last_actions,
            self.last_env_rewards,
            self._last_episode_starts,
            self.last_values,
            self.last_log_probs,
        )

        self._last_episode_starts = self.last_env_dones
