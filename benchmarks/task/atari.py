import numpy as np
import torch
from auturi.executor.environment import AuturiEnv
from auturi.executor.policy import AuturiPolicy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

TASK_ID = "PongNoFrameskip-v4"


def _create_atari_env(task_id):
    env = make_atari_env(task_id, n_envs=1)
    env = VecFrameStack(env, 4)
    env = VecTransposeImage(env)
    return env


class AtariEnv(AuturiEnv):
    def __init__(self):

        self.env = _create_atari_env(TASK_ID)
        self.rollout_buffer = dict()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.metadata = self.env.metadata

    def step(self, action, _):
        action = np.expand_dims(action, 0)

        print("step! ", action.shape, action)

        obs, reward, done, info = self.env.step(action)
        print("step done!! ", action.shape, action)

        return np.squeeze(obs, axis=0)

    def reset(self):
        obs = self.env.reset()
        obs = np.squeeze(obs, axis=0)
        print("reset shape = ", obs.shape)
        return obs

    def seed(self, seed):
        return self.env.seed(seed)

    def close(self):
        self.env.close()

    def aggregate_rollouts(self):
        return self.rollout_buffer


class AtariPolicy(AuturiPolicy):
    def __init__(self):
        dummy_env = _create_atari_env(TASK_ID)
        self.observation_space = dummy_env.observation_space
        policy = ActorCriticCnnPolicy(
            dummy_env.observation_space, dummy_env.action_space, lambda _: 0.001
        )
        policy.set_training_mode(False)
        dummy_env.close()

        self.device = "cpu"
        self._policy = policy

    def compute_actions(self, obs, n_steps):
        """Compute action with policy network."""
        obs = _np_to_tensor(obs, self.device)
        actions, values, log_probs = self._policy(obs)
        return _tensor_to_np(actions), [_tensor_to_np(values), _tensor_to_np(log_probs)]

    def load_model(self, _, device):
        """Load policy network on specified device."""
        self._policy = self._policy.to(device)
        self.device = device


def _tensor_to_np(tensor):
    return tensor.detach().cpu().numpy()


def _np_to_tensor(arr, device):
    return torch.from_numpy(arr).to(device)
