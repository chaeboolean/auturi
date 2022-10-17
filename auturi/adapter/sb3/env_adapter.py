from auturi.typing.simulator import AuturiEnv
from collections import defaultdict
import numpy as np
import gym
import torch as th

class SB3LocalRolloutBuffer:
    def __init__(self, shm_dict):
        pass
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: np.ndarray,
        log_prob: th.Tensor,
        terminal_obs: np.ndarray,
    ):
        pass
        

    def aggregate(self, start_idx):
        pass
    

class SB3EnvAdapter(AuturiEnv):
    def __init__(self, env_fn, normalize=False):
        self.env = env_fn()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.metadata = self.env.metadata
        self._last_obs = None
        self._last_episode_starts = None

        self.local_buffer = SB3LocalRolloutBuffer(None)

    # Should explicitly call reset() before data collection.    
    def reset(self):
        self._last_obs = self.env.reset()
        self._last_episode_starts = True
        return self._last_obs

    def _step_and_reset(self, actions):
        observation, reward, done, info = self.env.step(actions)
        if done:
            # save final observation where user can get it, then reset
            info["terminal_observation"] = observation
            observation = self.env.reset()
        return observation, reward, done, info
            
    
    def step(self, actions, values, log_probs):
        """Return only observation, which policy worker needs."""

        # Rescale and perform action
        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

        new_obs, reward, done, info = self._step_and_reset(clipped_actions)
        
        if isinstance(self.action_space, gym.spaces.Discrete):
            # Reshape in case of discrete action
            actions = actions.reshape(-1, 1)

        
        terminal_obs = None
        if (
            done
            and info.get("terminal_observation") is not None
            and info.get("TimeLimit.truncated", False)
        ):
            terminal_obs = info["terminal_observation"]
        
        self.local_buffer.add(self._last_obs, actions, reward, self._last_episode_starts, values, log_probs, terminal_obs)
        
        self._last_obs = new_obs
        self._last_episode_starts = done

        #return (new_obs, reward, done, info)
        return new_obs


    def close(self):
        self.env.close()
        
    def seed(self, seed):
        self.env.seed(seed)
        
