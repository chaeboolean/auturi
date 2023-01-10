import gym
import numpy as np

from auturi.executor.environment import AuturiEnv


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
        self.storage["action_artifacts"].append(np.array([[value, log_prob]]))
        self.storage["has_terminal_obs"].append(np.array([terminal_obs is not None]))

        terminal_obs = np.zeros_like(obs) if terminal_obs is None else terminal_obs.reshape(obs.shape)
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
        self.setup_dummy_env(self.env)
        self.artifacts_samples = [np.array([[1.1, 1.4]])]

        self._last_obs = None
        self._last_episode_starts = np.array([False])

        self.local_buffer = SB3LocalRolloutBuffer()

    # Should explicitly call reset() before data collection.
    def reset(self):
        self._last_obs = self.env.reset()
        self._last_episode_starts = np.array([True])

        return self._last_obs

    def step(self, actions, action_artifacts):
        """Return only observation, which policy worker needs."""
        action = actions
        if isinstance(self.action_space, gym.spaces.Discrete):
            action = actions[0]

        # Rescale and perform action
        clipped_actions = action
        # Clip the actions to avoid out of bound error
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = np.clip(
                action, self.action_space.low, self.action_space.high
            )
        new_obs, reward, done, info = self.env.step(clipped_actions)  # all list
        #new_obs, reward, done, info = new_obs[0], reward[0], done[0], info[0]  # unpack

        # set terminal_obs
        terminal_obs = None
        if (
            done[0]
            and info[0].get("terminal_observation", None) is not None
            and info[0].get("TimeLimit.truncated", False)
        ):
            terminal_obs = info[0]["terminal_observation"]

        values, log_probs = action_artifacts[0][0]
        self.local_buffer.add(
            self._last_obs,
            action,
            reward,
            self._last_episode_starts,
            values,
            log_probs,
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
