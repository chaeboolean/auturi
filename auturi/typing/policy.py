"""
Defines typings related to Policy Worker: AuturiPolicy, VectorPolicy

"""
import inspect
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, Tuple

import torch.nn as nn

from auturi.typing.vector import VectorMixin


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

        self.local_policy = self._create_policy_worker(0)

        self.remote_policies = OrderedDict()
        self.num_policies = 1

    def _working_workers(self) -> Tuple[int, AuturiPolicy]:
        """Iterates all current working workers."""
        yield 0, self.local_policy
        for wid in range(1, self.num_policies):
            yield wid, self.remote_policies[wid]

    def reconfigure(self, num_policies: int):
        """Add remote policy if needed."""

        # Create AuturiEnv if needed.
        current_num_remote_policies = len(self.remote_policies)
        num_workers_need = num_policies - current_num_remote_policies - 1
        new_worker_id = current_num_remote_policies + 1
        while num_workers_need > 0:
            self.remote_policies[new_worker_id] = self._create_policy_worker(
                new_worker_id
            )
            num_workers_need -= 1
            new_worker_id += 1

        # set number of currently working workers
        self.num_policies = num_policies

    def load_model(self, model: nn.Module, device: str) -> None:
        for wid, policy_worker in self._working_workers():
            self._load_policy_model(wid, policy_worker, model, device)

    @abstractmethod
    def _create_policy_worker(self, idx: int) -> AuturiPolicy:
        """Create worker. If idx is 0, create local worker."""
        raise NotImplementedError

    @abstractmethod
    def _load_policy_model(self, idx: int, policy: AuturiPolicy, device: str) -> None:
        """Load policy model to device for working policy worker each."""
        raise NotImplementedError

    def _get_policy_worker(self, idx: int) -> AuturiPolicy:
        if idx == 0:
            return self.local_policy
        else:
            return self.remote_policies[idx]

    def start_loop(self):
        """Setup before running collection loop."""
        pass

    def stop_loop(self):
        """Stop loop, but not terminate entirely."""
        pass

    def terminate(self):
        """Terminate."""
        pass
