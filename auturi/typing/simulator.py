"""
Defines typings related to Policy Worker: AuturiEnv, AuturiVecEnv
Users who would like to plug Auturi on other DRL frameworks 
should implement Env adapter.
"""

from abc import ABCMeta, abstractmethod
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import gym

from auturi.typing.auxilary import ObjRef


class AuturiEnv(metaclass=ABCMeta):
    @abstractmethod
    def step(self, action):
        pass

    def reset(self, seed):
        pass

    def close(self):
        pass


class AuturiVectorEnv(metaclass=ABCMeta):
    def poll(self, bs: int) -> Dict[int, ObjRef]:
        """Return reference of `bs` fastest environment ids

        Args:
            bs (int): batch size. Vulenerable to change in tuning mode.

        Returns:
            Dict[int, ObjRef]: Maps env_id to step_ref

        """
        raise NotImplementedError

    def send_actions(self, action_dict: Dict[int, ObjRef]) -> None:
        """Register action reference to remote env.

        Args:
            action_dict (Dict[int, ObjRef]): Maps env_id to service_ref
        """
        raise NotImplementedError


class AuturiParallelEnv(AuturiVectorEnv):
    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        self.num_envs = len(env_fns)
        self.env_fns = env_fns

        dummy_env = env_fns[0]()
        self.remote_envs = {
            i: self._create_env(i, env_fn_) for i, env_fn_ in enumerate(env_fns)
        }

        self._setup(dummy_env)
        dummy_env.close()

    def _create_env(self, index, env_fn):
        raise NotImplementedError

    def _setup(self, dummy_env):
        """Set attributes with Dummy Env."""
        pass


class AuturiSerialEnv(AuturiVectorEnv):
    pass
