import gym
import numpy as np
import torch as th

from auturi.executor.environment import AuturiEnv


class SB3LocalRolloutBuffer:
    def __init__(self):
        self.storage = {
            "obs": [],
            "action": [],
            "reward": [],
            "episode_start": [],
            "value": [],
            "log_prob": [],
            "has_terminal_obs": [],
            "terminal_obs": [],
        }
        self.counter = 0
        self.clear_signal = False

    def clear(self):
        self.counter = 0
        for k, v in self.storage.items():
            v.clear()

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: np.ndarray,
        log_prob: th.Tensor,
        terminal_obs: np.ndarray = None,
    ):
        if self.clear_signal:
            self.clear()
            self.clear_signal = False

        self.storage["obs"].append(obs)
        self.storage["action"].append(action)
        self.storage["reward"].append(np.array(reward))

        self.storage["episode_start"].append(np.array(episode_start))
        self.storage["value"].append(np.array(value))
        self.storage["log_prob"].append(np.array(log_prob))

        self.storage["has_terminal_obs"].append(np.array([terminal_obs is not None]))
        terminal_obs = np.zeros_like(obs) if terminal_obs is None else terminal_obs
        self.storage["terminal_obs"].append(terminal_obs)

        self.counter += 1

    def get_local_rollouts(self):
        if self.counter > 0:
            self.clear_signal = True
            return self.storage
        else:
            return dict()


class SB3EnvAdapter(AuturiEnv):
    def __init__(self, env_fn, normalize=False):
        self.env = env_fn()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.metadata = self.env.metadata
        self._last_obs = None
        self._last_episode_starts = None

        self.local_buffer = SB3LocalRolloutBuffer()

        self.rollout_samples = {
            "obs": self.observation_space.sample(),
            "action": self.action_space.sample(),
            "reward": 2.31,  # any float32,
            "episode_start": True,
            "value": 2.31,
            "log_prob": 2.31,
            "has_terminal_obs": True,
            "terminal_obs": self.observation_space.sample(),
        }

    def get_rollout_samples(self):
        return self.rollout_samples

    # Should explicitly call reset() before data collection.
    def reset(self):
        self._last_obs = self.env.reset()
        self._last_episode_starts = [True]
        return self._last_obs

    def step(self, actions, action_artifacts):
        """Return only observation, which policy worker needs."""

        assert actions.shape == self.action_space.shape
        values, log_probs = action_artifacts

        # Rescale and perform action
        actions = np.expand_dims(actions, axis=0)
        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = np.clip(
                actions, self.action_space.low, self.action_space.high
            )

        new_obs, reward, done, info = self.env.step(clipped_actions)  # all list

        if isinstance(self.action_space, gym.spaces.Discrete):
            # Reshape in case of discrete action
            actions = actions.reshape(-1, 1)

        terminal_obs = None
        if (
            done[0]
            and info[0].get("terminal_observation") is not None
            and info[0].get("TimeLimit.truncated", False)
        ):
            terminal_obs = info[0]["terminal_observation"]

        self.local_buffer.add(
            self._last_obs,
            actions,
            reward,
            self._last_episode_starts,
            [values],
            [log_probs],
            terminal_obs,
        )

        self._last_obs = new_obs
        self._last_episode_starts = done

        return new_obs

    def aggregate_rollouts(self):
        return self.local_buffer.get_local_rollouts()

    def terminate(self):
        self.env.close()

    def seed(self, seed):
        self.env.seed(seed)
