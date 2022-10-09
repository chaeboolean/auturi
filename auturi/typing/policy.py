"""
Defines typings related to Policy Worker: AuturiPolicy, VectorPolicy
Users who would like to plug Auturi on other DRL frameworks 
should implement policy adapter.

"""
from abc import ABCMeta, abstractmethod
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

from auturi.typing.auxilary import ObjRef


class AuturiPolicy(metaclass=ABCMeta):
    pass
    # @abstractmethod
    # def service(self, step_refs: Dict[int, ObjRef]):
    #     pass


class AuturiVectorPolicy:
    def __init__(self, num_policies: int, policy_fn: Callable):
        self.remote_policies = {
            i: self._create_policy(i, policy_fn) for i in range(num_policies)
        }

        self._setup()

    def _create_policy(self, index, policy_fn):
        raise NotImplementedError

    def get_free_server(self):
        raise NotImplementedError

    def _setup(self):
        pass
