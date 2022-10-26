"""
Defines typings related to Environment: AuturiEnv, AuturiSerialEnv, AuturiVecEnv.

"""
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from auturi.executor.config import ActorConfig
from auturi.executor.vector_utils import VectorMixin, aggregate_partial


class AuturiEnv(metaclass=ABCMeta):
    """Base API for environment used in Auturi System.

    Users who would like to plug Auturi on other DRL frameworks
    should implement AuturiEnv as adapter.
    """

    @abstractmethod
    def step(self, action, action_artifacts):
        """Take action artifacts also for buffer storage."""
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def seed(self, seed) -> None:
        raise NotImplementedError

    @abstractmethod
    def aggregate_rollouts(self) -> Dict[str, Any]:
        """Aggregates rollout results from remote environments."""
        raise NotImplementedError

    def setup_with_dummy(self, dummy_env):
        """Set basic attributes from dummy_env."""
        self.observation_space = dummy_env.observation_space
        self.action_space = dummy_env.action_space
        self.metadata = dummy_env.metadata


class AuturiSerialEnv(AuturiEnv):
    """Handle serial execution of multiple environments.

    Implementation is similar to DummyVecEnv in Gym Library.
    """

    def __init__(self, idx=0, env_fns: List[Callable] = []):
        self.idx = idx
        self.env_fns = env_fns

        dummy_env = env_fns[0]()
        assert isinstance(dummy_env, AuturiEnv)
        self.setup_with_dummy(dummy_env)
        dummy_env.close()

        self.envs = dict()  # maps env_id and AuturiEnv instance
        self.start_idx, self.num_envs = -1, 0

    def set_working_env(self, start_idx: int, num_envs: int):
        """Initialize environments with env_ids."""
        self.start_idx = start_idx
        self.num_envs = num_envs

        for env_id in range(self.start_idx, self.start_idx + self.num_envs):
            if env_id not in self.envs:
                self.envs[env_id] = self.env_fns[env_id]()

    def reset(self) -> np.ndarray:
        obs_list = [env.reset() for eid, env in self._working_envs()]
        return np.stack(obs_list)

    def seed(self, seed) -> None:
        for eid, env in self._working_envs():
            env.seed(seed + eid)

    def terminate(self):
        for eid, env in self.envs.items():  # terminate not only working envs.
            env.close()

    def step(self, action_ref):
        obs_list = []
        actions, action_artifacts = action_ref
        assert len(actions) == self.num_envs
        for eid, env in self._working_envs():
            artifacts_ = [elem[eid - self.start_idx] for elem in action_artifacts]
            obs = env.step(actions[eid - self.start_idx], artifacts_)
            obs_list += [obs]

        return np.stack(obs_list)

    def aggregate_rollouts(self) -> Dict[str, Any]:
        partial_rollouts = [env.aggregate_rollouts() for _, env in self._working_envs()]
        return aggregate_partial(partial_rollouts, already_agg=False)

    def _working_envs(self) -> Tuple[int, AuturiEnv]:
        """Iterates all current working environments."""
        for env_id in range(self.start_idx, self.start_idx + self.num_envs):
            yield env_id, self.envs[env_id]


class AuturiVectorEnv(VectorMixin, AuturiEnv, metaclass=ABCMeta):
    def __init__(self, env_fns: List[Callable]):
        self.env_fns = env_fns
        self.batch_size = -1  # should be initialized
        self.num_env_serial = 0

        # check dummy
        dummy_env = env_fns[0]()
        assert isinstance(dummy_env, AuturiEnv)
        self.setup_with_dummy(dummy_env)
        dummy_env.close()

        self.set_vector_attrs()

    @property
    def num_envs(self) -> int:
        """Return the total number of environments that are currently running."""
        return self.num_env_serial * self.num_workers

    def reconfigure(self, config: ActorConfig) -> None:
        """Reconfigure env_workers when given number of total envs and parallel degree."""

        assert config.num_envs <= len(self.env_fns)

        # set number of currently working workers
        self.num_workers = config.num_parallel

        # Set serial_degree to each env_worker
        self.num_env_serial = config.num_envs // config.num_parallel

        # Set batch size (used when polling)
        self.batch_size = config.batch_size

        for idx, env_worker in self._working_workers():
            self._set_working_env(
                idx, env_worker, self.num_env_serial * idx, self.num_env_serial
            )

    @abstractmethod
    def _set_working_env(
        self, env_id: int, env_worker: AuturiEnv, start_idx: int, num_envs: int
    ) -> None:
        """Create remote worker."""
        raise NotImplementedError

    @abstractmethod
    def poll(self) -> Any:
        """Wait until at least 'self.batch_size' environments finish step.

        Returns:
            Any: 'self.batch_size' fastest environment ids or their references.
        """
        raise NotImplementedError

    @abstractmethod
    def send_actions(self, action_ref: Any) -> None:
        """Register action reference to remote env."""
        raise NotImplementedError
