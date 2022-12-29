"""
Typings related to Environment: AuturiEnv, AuturiSerialEnv, AuturiVecEnv.

"""
from abc import ABCMeta, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import auturi.executor.typing as types
from auturi.executor.vector_utils import aggregate_partial
from auturi.tuner.config import ParallelizationConfig


class AuturiEnv(metaclass=ABCMeta):
    """Base class that defines APIs for environment used in Auturi System.


    Auturi is designed with usability and portability in mind.
    By implementing the AuturiEnv as an interface, users can combine other DRL frameworks (e.g. Ray) with Auturi.
    """

    @abstractmethod
    def step(
        self, action: np.ndarray, action_artifacts: types.ActionArtifacts
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
    def aggregate_rollouts(self, to=Optional[np.ndarray]) -> types.Rollouts:
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

    def __init__(self):
        self.envs = dict()  # maps env_id and AuturiEnv instance
        self.start_idx, self.end_idx = -1, -1

    def set_working_env(self, start_idx, end_idx, env_fns: List[Callable]) -> None:
        """Initialize environments with env_ids."""
        self.start_idx = start_idx
        self.end_idx = end_idx

        for env_id in range(self.start_idx, self.end_idx):
            if env_id not in self.envs:
                self.envs[env_id] = env_fns[env_id]()

    def reset(self) -> np.ndarray:
        obs_list = [env.reset() for _, env in self._working_envs()]
        return np.stack(obs_list)

    def seed(self, seed) -> None:
        for eid, env in self._working_envs():
            env.seed(seed + eid)

    def terminate(self) -> None:
        for _, env in self.envs.items():
            env.terminate()

    def step(
        self, actions: np.ndarray, action_artifacts: types.ActionArtifacts
    ) -> np.ndarray:
        """Broadcast actions to each env.

        Args:
            actions (np.ndarray): shape should be [self.num_envs, *self.action_space.shape]
            action_artifacts (types.ActionArtifacts): Each element' first dim should be equal to self.num_envs

        Returns:
            np.npdarray: shape should be [self.num_envs, *self.observation_space.shape]
        """
        obs_list = []
        for eid, env in self._working_envs():
            artifacts_ = [elem[eid - self.start_idx, :] for elem in action_artifacts]
            obs = env.step(actions[eid - self.start_idx, :], artifacts_)
            obs_list += [obs]

        return np.stack(obs_list)

    def aggregate_rollouts(self) -> types.Rollouts:
        rollouts_from_each_env = [
            env.aggregate_rollouts() for _, env in self._working_envs()
        ]
        res = aggregate_partial(rollouts_from_each_env, to_stack=True, to_extend=True)
        return res

    def _working_envs(self) -> Tuple[int, AuturiEnv]:
        """Iterates all current working environments."""
        for env_id in range(self.start_idx, self.end_idx):
            yield env_id, self.envs[env_id]

    def __getitem__(self, index):
        return self.envs[index]


class AuturiEnvHandler(metaclass=ABCMeta):
    def __init__(self, actor_id: int, env_fns: List[Callable]):
        """Abstraction for handling single or multiple AuturiPolicy.

        Args:
            actor_id (int): Id of parent actor
            env_fns (List[Callable]): Functions creating Adapter class that inherits AuturiEnv.
        """
        self.actor_id = actor_id
        self.env_fns = env_fns
        self.batch_size = -1  # should be initialized when reconfigure

        # set metadata with dummy environment
        dummy_env = env_fns[0]()
        assert isinstance(dummy_env, AuturiEnv)
        self.observation_space = dummy_env.observation_space
        self.action_space = dummy_env.action_space
        self.metadata = dummy_env.metadata
        dummy_env.terminate()

    @property
    @abstractmethod
    def num_envs(self):
        raise NotImplementedError

    @abstractmethod
    def reconfigure(self, config: ParallelizationConfig):
        raise NotImplementedError

    @abstractmethod
    def poll(self) -> types.ObservationRefs:
        """Wait until at least 'self.batch_size' environments finish their step.

        Returns:
            ObservationRefs: numpy array or references of 'self.batch_size'
                number of environments that finished their steps the fastest.

        """
        raise NotImplementedError

    @abstractmethod
    def send_actions(self, action_ref: types.ActionRefs) -> None:
        """Register action reference to remote env."""
        raise NotImplementedError

    @abstractmethod
    def aggregate_rollouts(self, to=Optional[np.ndarray]) -> Dict[str, np.ndarray]:
        """Aggregates rollout results from remote environments."""
        raise NotImplementedError

    def start_loop(self) -> None:
        """Setup before running collection loop."""
        pass

    def stop_loop(self) -> None:
        """Stop loop, but not terminate entirely."""
        pass

    @abstractmethod
    def terminate(self) -> None:
        raise NotImplementedError


class AuturiLocalEnv(AuturiSerialEnv, AuturiEnvHandler):
    def __init__(self, actor_id, env_fns: List[Callable[[], AuturiEnv]]):
        AuturiEnvHandler.__init__(self, actor_id, env_fns)
        AuturiSerialEnv.__init__(self)
        self.last_action = None
        self._num_envs = -1

    @property
    def num_envs(self):
        return self._num_envs

    def start_loop(self) -> None:
        self.last_action = None

    def reconfigure(self, config: ParallelizationConfig) -> None:
        self._num_envs = config[self.actor_id].num_envs
        self.batch_size = config[self.actor_id].batch_size
        assert self.batch_size == self._num_envs

        start_idx = config.compute_index_for_actor("num_envs", self.actor_id)
        self.set_working_env(start_idx, start_idx + self._num_envs, self.env_fns)

    def poll(self) -> np.ndarray:
        if self.last_action is None:
            return self.reset()
        else:
            return self.step(*self.last_action)

    def send_actions(self, action_refs: types.ActionTuple) -> None:
        self.last_action = action_refs


class AuturiVectorEnv(AuturiEnvHandler, metaclass=ABCMeta):
    def __init__(self, actor_id, env_fns: List[Callable[[], AuturiEnv]]):

        super().__init__(actor_id, env_fns)
        self._env_offset, self._rollout_offset, self._num_envs = -1, -1, -1
        self.num_env_serial = -1

    @property
    def num_envs(self):
        return self._num_envs

    def reconfigure(self, config: ParallelizationConfig) -> None:
        """Reconfigure AuturiSerialEnv with the environment-level parallelism specified in the config."""

        self._env_offset = config.compute_index_for_actor("num_envs", self.actor_id)
        self._rollout_offset = config.compute_index_for_actor(
            "num_collect", self.actor_id
        )

        actor_config = config[self.actor_id]

        assert actor_config.num_envs <= len(self.env_fns)
        self._num_envs = actor_config.num_envs

        # number of observations that a policy consumes for computing an action
        self.batch_size = actor_config.batch_size

        # Set the number of environments that are executed sequentially
        # = total number of environments // the degree of environment-level parallelism
        self.num_env_serial = actor_config.num_envs // actor_config.num_parallel

        # set number of currently working workers
        self.reconfigure_workers(new_num_workers=actor_config.num_parallel)
