from auturi.typing.simulator import AuturiEnv
from collections import defaultdict
import numpy as np
import gym
import torch as th



def process_buffer(agg_buffer, policy, gamma):
    # process reward from terminal observations 
    terminal_indices = np.where(agg_buffer["has_terminal_obs"] == True)[0]
    
    if len(terminal_indices) > 0:
        terminal_obs = agg_buffer["terminal_obs"][terminal_indices]
        terminal_obs = policy.obs_to_tensor(terminal_obs)[0]

        with th.no_grad():
            terminal_value = policy.predict_values(terminal_obs)
        
        agg_buffer["reward"][terminal_indices] += gamma * (terminal_value.numpy().flatten())


def insert_as_buffer(rollout_buffer, agg_buffer, num_envs):

    # insert to rollout_buffer
    bsize = rollout_buffer.buffer_size
    total_length = bsize * num_envs
    
    def _truncate_and_reshape(buffer_, add_dim=False, dtype=np.float32):
        shape_ = (bsize, num_envs, -1) if add_dim else (bsize, num_envs)
        ret = buffer_[:total_length].reshape(*shape_)
        return ret.astype(dtype)

    # reshape to (k, self.n_envs, obs_size)
    rollout_buffer.observations = _truncate_and_reshape(agg_buffer["obs"], add_dim=True)
    rollout_buffer.actions = _truncate_and_reshape(agg_buffer["action"], add_dim=True)
    rollout_buffer.rewards = _truncate_and_reshape(agg_buffer["reward"], add_dim=False)
    rollout_buffer.episode_starts = _truncate_and_reshape(agg_buffer["episode_start"], add_dim=False)
    rollout_buffer.values = _truncate_and_reshape(agg_buffer["value"], add_dim=False)
    rollout_buffer.log_probs = _truncate_and_reshape(agg_buffer["log_prob"], add_dim=False)
    rollout_buffer.pos = rollout_buffer.buffer_size
    rollout_buffer.full = True

class SB3LocalRolloutBuffer:
    def __init__(self, shm_dict):
        self.storage = defaultdict(list)
        self.counter = 0
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: np.ndarray,
        log_prob: th.Tensor,
        terminal_obs: np.ndarray=None,
    ):
        self.storage["obs"].append(obs)
        self.storage["action"].append(action)
        self.storage["reward"].append(reward)

        self.storage["episode_start"].append(episode_start)
        self.storage["value"].append(value.flatten())
        self.storage["log_prob"].append(log_prob)
        
        self.storage["has_terminal_obs"].append(terminal_obs is not None)
        terminal_obs = np.zeros_like(obs) if terminal_obs is None else terminal_obs
        self.storage["terminal_obs"].append(terminal_obs)
        
        self.counter += 1
        
    def stack_to_np(self, out=None):
        if self.counter ==0: 
            return dict()
        
        return_dict = {
            "obs": np.stack(self.storage["obs"], out=out),
            "action": np.stack(self.storage["action"], out=out),
            "reward": np.stack(self.storage["reward"], out=out),
            "episode_start": np.stack(self.storage["episode_start"], out=out),
            "value": np.stack(self.storage["value"], out=out),
            "log_prob": np.stack(self.storage["log_prob"], out=out),
            "has_terminal_obs": np.stack(self.storage["has_terminal_obs"], out=out),
            "terminal_obs": np.stack(self.storage["terminal_obs"], out=out),
        }
        
        print("**** Sending ", self.counter, " ....")
    
        self.storage.clear()
        self.counter = 0
        return return_dict 
    

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

    def fetch_rollouts(self):
        return self.local_buffer.stack_to_np()

    def close(self):
        self.env.close()
        
    def seed(self, seed):
        self.env.seed(seed)
        
