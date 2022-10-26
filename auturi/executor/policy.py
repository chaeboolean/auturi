"""
Defines typings related to Policy Worker: AuturiPolicy, VectorPolicy

"""
import inspect
from abc import ABCMeta, abstractmethod
from typing import Any, Dict

import torch.nn as nn

from auturi.executor.config import ActorConfig
from auturi.executor.vector_utils import VectorMixin


class AuturiPolicy(metaclass=ABCMeta):
    @abstractmethod
    def compute_actions(self, obs: Any, n_steps: int):
        """Compute action with policy network."""
        raise NotImplementedError

    @abstractmethod
    def load_model(self, model: nn.Module, device: str):
        """Load policy network on specified device."""
        raise NotImplementedError


class AuturiVectorPolicy(VectorMixin, AuturiPolicy, metaclass=ABCMeta):
    def __init__(self, policy_cls, policy_kwargs: Dict[str, Any] = dict()):
        """AuturiVectorPolicy is handler that manages multipler AuturiVectors.

        Args:
            policy_cls (classVar): Adapter class that inherits AuturiPolicy.
            policy_model (torch.nn.Module): Model should be on shared memory.
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
    def _load_policy_model(self, idx: int, policy: AuturiPolicy, device: str) -> None:
        """Load policy model to device for working policy worker each."""
        raise NotImplementedError

    # TODO: No need to imple.
    def load_model(self, model: nn.Module, device: str):
        pass
