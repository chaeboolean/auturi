"""
Defines typings related to Policy Worker: AuturiPolicy, VectorPolicy

"""
import inspect
from abc import ABCMeta, abstractmethod
from typing import Any, Dict

import numpy as np
import torch.nn as nn

from auturi.executor.vector_utils import VectorMixin
from auturi.tuner.config import ActorConfig


class AuturiPolicy(metaclass=ABCMeta):
    @abstractmethod
    def compute_actions(self, obs: np.ndarray, n_steps: int = -1):
        """Compute action with the policy network.

        obs dimension always should be [num_envs, *observation_space.shape]
        """
        raise NotImplementedError

    @abstractmethod
    def load_model(self, model: nn.Module, device: str):
        """Load policy network on the specified device."""
        raise NotImplementedError


class AuturiVectorPolicy(VectorMixin, AuturiPolicy, metaclass=ABCMeta):
    def __init__(self, policy_cls, policy_kwargs: Dict[str, Any] = dict()):
        """Abstraction for handling multiple AuturiPolicy.

        Args:
            policy_cls (classVar): Adapter class that inherits AuturiPolicy.
            policy_kwargs (Dict[str, Any]): Keyword arguments used for instantiating the policy.
        """
        assert AuturiPolicy in inspect.getmro(policy_cls)

        self.policy_cls = policy_cls
        self.policy_kwargs = policy_kwargs

        self.set_vector_attrs()

    @property
    def num_policies(self):
        return self.num_worker

    def reconfigure(self, config: ActorConfig, model: nn.Module):
        """Add remote policy if needed."""

        # set number of currently working workers
        self.num_workers = config.num_policy

        # call load_model for each policy.
        for wid, policy_worker in self._working_workers():
            self._load_policy_model(wid, policy_worker, model, config.policy_device)

    @abstractmethod
    def _load_policy_model(
        self, idx: int, policy: AuturiPolicy, model: nn.Module, device: str
    ) -> None:
        """Load the latest trained model parameter on the specified device for the policy."""
        raise NotImplementedError

    # TODO: No need to imple.
    def load_model(self, model: nn.Module, device: str):
        pass
