"""
Defines typings related to Policy Worker: AuturiEnv, AuturiVecEnv
Users who would like to plug Auturi on other DRL frameworks 
should implement Env adapter.
"""

from abc import ABCMeta, abstractmethod
from typing import Callable, Dict, List

import gym

from auturi.typing.auxilary import ObjRef


class AuturiEnv(metaclass=ABCMeta):
    """Adaptor inherits AuturiEnv class."""

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def seed(self, seed):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def fetch_rollouts(self):
        """Fetch locally observed trajectories to main loop."""
        pass


class AuturiVectorEnv(metaclass=ABCMeta):
    def poll(self, bs: int = -1) -> Dict[ObjRef, int]:
        """Return reference of `bs` fastest environment ids

        Args:
            bs (int): batch size. Vulenerable to change in tuning mode.

        Returns:
            Dict[int, ObjRef]: Maps env_id to step_ref

        """
        if bs < 0:
            bs = self.num_envs
        return self._poll(bs)

    @abstractmethod
    def _poll(self, bs: int = -1) -> Dict[ObjRef, int]:
        raise NotImplementedError

    @abstractmethod
    def send_actions(self, action_ref: ObjRef) -> None:
        """Register action reference to remote env.

        Args:
            action_dict (Dict[int, ObjRef]): Maps env_id to service_ref
        """
        raise NotImplementedError

    def start_loop(self):
        """Setup when start loop."""
        pass

    def finish_loop(self):
        """Teardown when finish loop, but not terminate entirely."""
        pass

    def close(self):
        """Terminate."""
        pass


class AuturiParallelEnv(AuturiVectorEnv):
    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        self.num_envs = len(env_fns)
        self.env_fns = env_fns

        dummy_env = env_fns[0]()

        self.observation_space = dummy_env.observation_space
        self.action_space = dummy_env.action_space
        self.metadata = dummy_env.metadata

        self.setup_with_dummy(dummy_env)

        dummy_env.close()

    def validate_initialization(self):
        assert len(self.remote_envs) == self.num_envs

    def setup_with_dummy(self, dummy):
        pass


class AuturiSerialEnv(AuturiVectorEnv):
    pass
