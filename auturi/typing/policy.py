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
    def compute_actions(self, obs:ObjRef, n_steps:int):
        pass

    @abstractmethod
    def set_device(self, device: str):
        pass

    @abstractmethod
    def load_model(self):
        pass

class AuturiVectorPolicy:
    def __init__(self, num_policies: int, policy_fn: Callable):
        self.num_policies = num_policies
        self.policy_fn = policy_fn


    def _create_policy(self, index, policy_fn):
        raise NotImplementedError

    def get_free_server(self):
        raise NotImplementedError

    def _setup(self):
        pass

    def start_loop(self):
        """ Setup when start loop."""
        pass 

    def finish_loop(self):
        """ Teardown when finish loop, but not terminate entirely."""
        pass 