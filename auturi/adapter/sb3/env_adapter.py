import gym
import numpy as np

from auturi.executor.environment import AuturiEnv
import time

class SB3LocalRolloutBuffer:
    def __init__(self):
        self.storage = {
            "obs": [],
            "action": [],
            "reward": [],
            "episode_start": [],
            "action_artifacts": [],  # [value, log_probs]
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
        reward: float,
        episode_start: bool,
        value: float,
        log_prob: float,
        terminal_obs: np.ndarray = None,
    ):
        if self.clear_signal:
            self.clear()
            self.clear_signal = False

        self.storage["obs"].append(obs)
        self.storage["action"].append(action)
        self.storage["reward"].append(reward)

        self.storage["episode_start"].append(episode_start)
        self.storage["action_artifacts"].append(np.array([value, log_prob]))
        self.storage["has_terminal_obs"].append(terminal_obs is not None)

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
    def __init__(self, env_fn):
        self.env = env_fn()
        self.setup_with_dummy(self.env)

        self._last_obs = None
        self._last_episode_starts = False

        self.local_buffer = SB3LocalRolloutBuffer()
        self.time_ms = []

    # Should explicitly call reset() before data collection.
    def reset(self):
        self._last_obs = self.env.reset()[0]
        self._last_episode_starts = True

        return self._last_obs

    def step(self, actions, action_artifacts):
        """Return only observation, which policy worker needs."""

        start_time = time.perf_counter()
        # assert actions.shape == self.action_space.shape
        # process action-related values.
        if isinstance(self.action_space, gym.spaces.Discrete):
            # Reshape in case of discrete action
            actions = actions.reshape(-1)

        # Rescale and perform action
        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = np.clip(
                actions, self.action_space.low, self.action_space.high
            )
        new_obs, reward, done, info = self.env.step(clipped_actions)  # all list
        new_obs, reward, done, info = new_obs[0], reward[0], done[0], info[0]  # unpack

        # set terminal_obs
        terminal_obs = None
        if (
            done
            and info.get("terminal_observation") is not None
            and info.get("TimeLimit.truncated", False)
        ):
            terminal_obs = info["terminal_observation"]

        values, log_probs = action_artifacts

        self.local_buffer.add(
            self._last_obs,
            actions,
            reward,
            self._last_episode_starts,
            values,
            log_probs,
            terminal_obs,
        )

        self._last_obs = new_obs
        self._last_episode_starts = done

        end_time = time.perf_counter()
        self.time_ms.append(end_time-start_time)

        return new_obs

    def aggregate_rollouts(self):
        if len(self.time_ms) > 0:
            with open("env.txt", "w"):
                print(self.time_ms)
            self.time_ms.clear()

        return self.local_buffer.get_local_rollouts()

    def terminate(self):
        self.env.close()

    def seed(self, seed):
        self.env.seed(seed)
