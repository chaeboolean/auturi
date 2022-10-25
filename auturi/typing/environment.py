"""
Defines typings related to Environment: AuturiEnv, AuturiVecEnv.

"""
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Any, Callable, Dict, List, OrderedDict, Tuple

import numpy as np

from auturi.typing.vector import VectorMixin


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
        self.start_idx = -1
        self.num_envs = 0

    def set_working_env(self, start_idx: int, num_envs: int):
        """Initialize environemtns with env_ids."""
        self.start_idx = start_idx
        self.num_envs = num_envs

        for env_id in range(self.start_idx, self.start_idx + self.num_envs):
            if env_id not in self.envs:
                self.envs[env_id] = self.env_fns[env_id]()

    def reset(self) -> np.ndarray:
        obs_list = []
        print(f"RESET: working_envs = [{self.start_idx}, {self.num_envs}]")
        for eid, env in self._working_envs():
            obs = env.reset()
            obs_list += [obs]

        return np.stack(obs_list)

    def seed(self, seed) -> None:
        for eid, env in self._working_envs():
            env.seed(seed + eid)

    def terminate(self):
        for eid, env in self.envs.items():
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

        dones = list(filter(lambda elem: len(elem) > 0, partial_rollouts))

        keys = list(dones[0].keys())
        buffer_dict = dict()
        for key in keys:
            li = []
            for done in dones:
                li += done[key]
            buffer_dict[key] = np.stack(li)

        return buffer_dict

    def _working_envs(self) -> Tuple[int, AuturiEnv]:
        """Iterates all current working environments."""
        for env_id in range(self.start_idx, self.start_idx + self.num_envs):
            yield env_id, self.envs[env_id]


class AuturiVectorEnv(VectorMixin, AuturiEnv, metaclass=ABCMeta):
    def __init__(self, env_fns: List[Callable]):
        self.env_fns = env_fns
        self.batch_size = -1  # should be initialized

        self.local_env_worker = self._create_env_worker(idx=0)
        self.remote_env_workers = OrderedDict()

        self.num_env_workers = 1  # different from len(self.remote_env_workers) + 1
        self.num_env_serial = 0

        # check dummy
        dummy_env = env_fns[0]()
        assert isinstance(dummy_env, AuturiEnv)
        self.setup_with_dummy(dummy_env)
        dummy_env.close()

    def _working_workers(self) -> Tuple[int, AuturiSerialEnv]:
        """Iterates all current working workers."""
        yield 0, self.local_env_worker
        for wid in range(1, self.num_env_workers):
            yield wid, self.remote_env_workers[wid]

    @property
    def num_envs(self) -> int:
        """Return the total number of environments that are currently running."""
        return self.num_env_serial * self.num_env_workers

    def reconfigure(self, num_envs: int, num_parallel: int) -> None:
        """Reconfigure env_workers when given number of total envs and parallel degree."""

        assert num_envs <= len(self.env_fns)

        # Create AuturiEnv if needed.
        current_num_remote_workers = len(self.remote_env_workers)
        num_workers_need = num_parallel - current_num_remote_workers - 1
        new_env_worker_id = current_num_remote_workers + 1
        while num_workers_need > 0:
            self.remote_env_workers[new_env_worker_id] = self._create_env_worker(
                new_env_worker_id
            )
            num_workers_need -= 1
            new_env_worker_id += 1

        # set number of currently working workers
        self.num_env_workers = num_parallel

        # Set serial_degree to each env_worker
        self.num_env_serial = num_envs // num_parallel

        for idx, env_worker in self._working_workers():
            self._set_working_env(
                idx, env_worker, self.num_env_serial * idx, self.num_env_serial
            )

    def set_batch_size(self, batch_size) -> None:
        self.batch_size = batch_size

    @abstractmethod
    def _create_env_worker(self, idx: int):
        """Create worker. If idx is 0, create local worker."""
        raise NotImplementedError

    @abstractmethod
    def _set_working_env(
        self, env_id: int, env_worker: AuturiEnv, start_idx: int, num_envs: int
    ) -> None:
        """Create remote worker."""
        raise NotImplementedError

    @abstractmethod
    def poll(self) -> Any:
        """Wait until at least `bs` environments finish step.

        Returns:
            Any: `bs` fastest environment ids or their references.
        """
        raise NotImplementedError

    @abstractmethod
    def send_actions(self, action_ref: Any) -> None:
        """Register action reference to remote env."""
        raise NotImplementedError

    def _get_env_worker(self, idx: int) -> AuturiEnv:
        if idx == 0:
            return self.local_env_worker
        else:
            return self.remote_env_workers[idx]

    def start_loop(self):
        """Setup before running collection loop."""
        pass

    def stop_loop(self):
        """Stop loop, but not terminate entirely."""
        pass

    def terminate(self):
        """Terminate."""
        pass