"""
Typings related to Environment: AuturiEnv, AuturiSerialEnv, AuturiVecEnv.

"""
import math
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from auturi.executor.vector_utils import VectorMixin, aggregate_partial
from auturi.tuner.config import ParallelizationConfig


class AuturiEnv(metaclass=ABCMeta):
    """Base class that defines APIs for environment used in Auturi System.


    Auturi is designed with usability and portability in mind.
    By implementing the AuturiEnv as an interface, users can combine other DRL frameworks (e.g. Ray) with Auturi.
    """

    @abstractmethod
    def step(
        self, action: np.ndarray, action_artifacts: List[np.ndarray]
    ) -> np.ndarray:
        """Same functionality with gym.Env.step().

        It also take action artifacts also for buffer storage.
        The shape of action should be equal to that of self.action_space
        """
        raise NotImplementedError

    @abstractmethod
    def terminate(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def seed(self, seed) -> None:
        raise NotImplementedError

    @abstractmethod
    def aggregate_rollouts(self, to=Optional[np.ndarray]) -> Dict[str, np.ndarray]:
        """Aggregates rollout results from remote environments."""
        raise NotImplementedError

    def setup_dummy_env(self, dummy_env) -> None:
        """Set basic attributes from dummy_env."""
        self.observation_space = dummy_env.observation_space
        self.action_space = dummy_env.action_space
        self.metadata = dummy_env.metadata


class AuturiSerialEnv(AuturiEnv):
    """Abstraction for handling sequential execution of multiple environments.

    Reference implementation: DummyVecEnv in OpenAI Baselines (https://github.com/DLR-RM/stable-baselines3).
    """

    def __init__(self, actor_id: int, serialenv_id: int, env_fns: List[Callable] = []):
        self.actor_id = actor_id
        self.serialenv_id = serialenv_id

        self.env_fns = env_fns

        dummy_env = env_fns[0]()
        assert isinstance(dummy_env, AuturiEnv)
        self.setup_dummy_env(dummy_env)

        dummy_obs = dummy_env.reset()
        assert dummy_obs.shape == self.observation_space.shape
        dummy_env.terminate()

        self.envs = dict()  # maps env_id and AuturiEnv instance
        self.start_idx, self.end_idx, self.num_envs = -1, -1, 0

    def set_working_env(self, start_idx: int, num_envs: int):
        """Initialize environments with env_ids."""
        self.start_idx = start_idx
        self.end_idx = start_idx + num_envs
        self.num_envs = num_envs

        for env_id in range(self.start_idx, self.end_idx):
            if env_id not in self.envs:
                self.envs[env_id] = self.env_fns[env_id]()

    def reset(self) -> np.ndarray:
        obs_list = [env.reset() for _, env in self._working_envs()]
        return np.stack(obs_list)

    def seed(self, seed) -> None:
        for eid, env in self._working_envs():
            env.seed(seed + eid)

    def terminate(self):
        for _, env in self.envs.items():
            env.terminate()

    def step(self, actions: np.ndarray, action_artifacts: List[np.ndarray]):
        """Broadcast actions to each env.

        Args:
            actions (np.ndarray): shape should be [self.num_envs, *self.action_space.shape]
            action_artifacts (List[np.ndarray]): Each element' first dim should be equal to self.num_envs

        Returns:
            np.npdarray: shape should be [self.num_envs, *self.observation_space.shape]
        """
        obs_list = []
        for eid, env in self._working_envs():
            artifacts_ = [elem[eid - self.start_idx] for elem in action_artifacts]
            obs = env.step(actions[eid - self.start_idx], artifacts_)
            obs_list += [obs]

        return np.stack(obs_list)

    def aggregate_rollouts(self) -> Dict[str, Any]:
        rollouts_from_each_env = [
            env.aggregate_rollouts() for _, env in self._working_envs()
        ]
        res = aggregate_partial(rollouts_from_each_env, to_stack=True, to_extend=True)
        return res

    def _working_envs(self) -> Tuple[int, AuturiEnv]:
        """Iterates all current working environments."""
        for env_id in range(self.start_idx, self.end_idx):
            yield env_id, self.envs[env_id]


class AuturiVectorEnv(VectorMixin, AuturiEnv, metaclass=ABCMeta):
    def __init__(self, actor_id: int, env_fns: List[Callable]):

        self.actor_id = actor_id
        self.env_fns = env_fns
        self.batch_size = -1  # should be initialized below
        self.num_env_serial = 0  # should be initialized be

        # Init with dummy env
        dummy_env = env_fns[0]()
        assert isinstance(dummy_env, AuturiEnv)
        self.setup_dummy_env(dummy_env)
        dummy_env.terminate()

        super().__init__()

    @property
    def num_envs(self) -> int:
        """Return the total number of environments that are currently running."""
        return self.num_env_serial * self.num_workers

    def reconfigure(self, config: ParallelizationConfig) -> None:
        """Reconfigure AuturiSerialEnv with the environment-level parallelism specified in the config."""

        actor_config = config[self.actor_id]
        assert actor_config.num_envs <= len(self.env_fns)

        # number of observations that a policy consumes for computing an action
        self.batch_size = actor_config.batch_size

        # Set the number of environments that are executed sequentially
        # = total number of environments // the degree of environment-level parallelism
        self.num_env_serial = actor_config.num_envs // actor_config.num_parallel
        self.num_worker_to_poll = math.ceil(self.batch_size / self.num_env_serial)

        assert self.num_worker_to_poll > 0

        # set number of currently working workers
        self.reconfigure_workers(
            new_num_workers=actor_config.num_parallel, config=config
        )

    @abstractmethod
    def poll(self) -> Any:
        """Wait until at least 'self.batch_size' environments finish their step.

        Returns:
            Any: IDs or references of 'self.batch_size' number of environments that finished their steps the fastest.

        """
        raise NotImplementedError

    @abstractmethod
    def send_actions(self, action_ref: Any) -> None:
        """Register action reference to remote env."""
        raise NotImplementedError

    @abstractmethod
    def start_loop(self):
        """Setup before running collection loop."""
        raise NotImplementedError

    @abstractmethod
    def stop_loop(self):
        """Stop loop, but not terminate entirely."""
        raise NotImplementedError
