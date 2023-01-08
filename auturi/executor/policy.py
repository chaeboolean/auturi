"""
Defines typings related to Policy Worker: AuturiPolicy, VectorPolicy

"""
import inspect
from abc import ABCMeta, abstractmethod
from typing import Any, Dict

import gym
import numpy as np

import auturi.executor.typing as types
from auturi.tuner.config import ParallelizationConfig


class AuturiPolicy(metaclass=ABCMeta):
    @abstractmethod
    def compute_actions(self, obs: np.ndarray, n_steps: int = -1) -> types.ActionTuple:
        """Compute action with the policy network.

        obs dimension always should be [num_envs, *observation_space.shape]
        """
        raise NotImplementedError

    @abstractmethod
    def load_model(self, model: types.PolicyModel, device: str) -> None:
        """Load policy network on the specified device."""
        raise NotImplementedError

    @abstractmethod
    def terminate(self) -> None:
        raise NotImplementedError

    def sample_observation(self, bs=1) -> np.ndarray:
        return np.stack([self.observation_space.sample()] * bs)

    def _validate(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

        obs_sample = self.sample_observation(bs=3)
        action_sample, _ = self.compute_actions(obs_sample, 0)

        assert action_sample.shape[0] == 3

        if isinstance(action_space, gym.spaces.Discrete):
            assert action_sample.shape == (3, 1)
        else:
            assert action_sample.shape == (3, *action_space.shape)


class AuturiPolicyHandler(metaclass=ABCMeta):
    def __init__(
        self, actor_id: int, policy_cls, policy_kwargs: Dict[str, Any] = dict()
    ):
        """Abstraction for handling single or multiple AuturiPolicy.

        Args:
            actor_id (int): Id of parent actor
            policy_cls (classVar): Adapter class that inherits AuturiPolicy.
            policy_kwargs (Dict[str, Any]): Keyword arguments used for instantiating the policy.
        """
        assert AuturiPolicy in inspect.getmro(policy_cls)

        self.actor_id = actor_id
        self.policy_cls = policy_cls
        self.policy_kwargs = policy_kwargs

    @abstractmethod
    def reconfigure(self, config: ParallelizationConfig, model: types.PolicyModel):
        raise NotImplementedError

    @abstractmethod
    def compute_actions(self, obs: np.ndarray, n_steps: int = -1) -> types.ActionRefs:
        raise NotImplementedError

    def start_loop(self) -> None:
        """Setup before running collection loop."""
        pass

    def stop_loop(self) -> None:
        """Stop loop, but not terminate entirely."""
        pass

    @abstractmethod
    def terminate(self) -> None:
        raise NotImplementedError


class AuturiLocalPolicy(AuturiPolicyHandler):
    def __init__(self, actor_id, policy_cls, policy_kwargs):
        super().__init__(actor_id, policy_cls, policy_kwargs)
        policy_kwargs.update({"idx": 0})
        self.policy = policy_cls(**policy_kwargs)

    def reconfigure(self, config: ParallelizationConfig, model: types.PolicyModel):
        device = config[self.actor_id].policy_device
        self.policy.load_model(model, device)

    def compute_actions(
        self, obs_refs: types.ObservationRefs, n_steps: int = -1
    ) -> types.ActionTuple:
        return self.policy.compute_actions(obs_refs, n_steps)

    def terminate(self) -> None:
        self.policy.terminate()
        del self.policy


class AuturiVectorPolicy(AuturiPolicyHandler, metaclass=ABCMeta):
    pass
