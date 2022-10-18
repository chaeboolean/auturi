"""
Defines typings related to Policy Worker: AuturiPolicy, VectorPolicy
Users who would like to plug Auturi on other DRL frameworks 
should implement policy adapter.

"""
from abc import ABCMeta, abstractmethod
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

from auturi.typing.auxilary import ObjRef


class AuturiPolicy(metaclass=ABCMeta):
    @abstractmethod
    def compute_actions(self, obs: ObjRef, n_steps: int):
        pass

    @abstractmethod
    def load_model(self, device: str):
        pass


class AuturiVectorPolicy(metaclass=ABCMeta):
    def __init__(self, num_policies: int, policy_fn: Callable):
        self.num_policies = num_policies
        self.policy_fn = policy_fn

    @abstractmethod
    def assign_free_server(self, obs_refs: Dict[int, ObjRef], n_steps: int):
        raise NotImplementedError

    def start_loop(self):
        """Setup when start loop."""
        pass

    def finish_loop(self):
        """Teardown when finish loop, but not terminate entirely."""
        pass

    def terminate(self):
        """Teardown when finish loop, but not terminate entirely."""
        pass
