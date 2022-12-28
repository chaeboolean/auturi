import time
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Tuple

import auturi.executor.typing as types
from auturi.executor.environment import (
    AuturiEnv,
    AuturiEnvHandler,
    AuturiLocalEnv,
    AuturiVectorEnv,
)
from auturi.executor.policy import (
    AuturiLocalPolicy,
    AuturiPolicyHandler,
    AuturiVectorPolicy,
)
from auturi.tuner import AuturiMetric, ParallelizationConfig


class AuturiLoopHandler(metaclass=ABCMeta):
    def __init__(self, env_fns, policy_cls, policy_kwargs):
        self.env_fns = env_fns
        self.policy_cls = policy_cls
        self.policy_kwargs = policy_kwargs

        self.env_handler = None
        self.policy_handler = None
        self.num_collect = -1

    @abstractmethod
    @property
    def num_actors(self):
        raise NotImplementedError

    @abstractmethod
    def _create_env_handler(self) -> AuturiEnvHandler:
        raise NotImplementedError

    @abstractmethod
    def _create_policy_handler(self) -> AuturiPolicyHandler:
        raise NotImplementedError

    @abstractmethod
    def _validate_config(self, config: ParallelizationConfig) -> None:
        raise NotImplementedError

    def reconfigure(self, config: ParallelizationConfig, model: types.PolicyModel):
        """Reconfigure the number of envs and policies according to a given config found by AuturiTuner."""
        self._validate_config(config)
        self.num_collect = None  # ??? TODO: SET NUM_COLLECT

        if self.env_handler is None:
            self.env_handler = self._create_env_handler()

        if self.policy_handler is None:
            self.policy_handler = self._create_policy_handler()

        self.env_handler.reconfigure(config)
        self.policy_handler.reconfigure(config, model)

    def run(self) -> Tuple[types.RolloutRefs, AuturiMetric]:
        """Run collection loop for num_collect iterations, and return experience trajectories."""

        self.policy_handler.start_loop()
        self.env_handler.start_loop()

        n_steps = 0
        start_time = time.perf_counter()
        while n_steps < self.num_collect:
            obs_refs: types.ObservationRefs = self.env_handler.poll()
            action_refs: types.ActionRefs = self.policy_handler.compute_actions(
                obs_refs, n_steps
            )
            self.env_handler.send_actions(action_refs)

            n_steps += self.env_handler.batch_size  # len(obs_refs)

        self.policy_handler.stop_loop()
        self.env_handler.stop_loop()
        end_time = time.perf_counter()

        return self.env_handler.aggregate_rollouts(), AuturiMetric(
            self.num_collect, end_time - start_time
        )

    def terminate(self):
        self.policy_handler.terminate()
        self.env_handler.terminate()


class SimpleLoopHandler(AuturiLoopHandler):
    @property
    def num_actors(self):
        return 1

    def _create_env_handler(self):
        return AuturiLocalEnv(actor_id=0, env_fns=self.env_fns)

    def _create_policy_handler(self):
        return AuturiLocalPolicy(actor_id=0, env_fns=self.env_fns)

    def _validate_config(self, config: ParallelizationConfig):
        assert config.num_actors == 1
        actor_config = config[0]

        assert (actor_config.num_parallel == 1) and (
            actor_config.batch_size == actor_config.num_envs
        )


class NestedLoopHandler(AuturiLoopHandler):
    @property
    def num_actors(self):
        return 1

    def _create_env_handler(self):
        return AuturiLocalEnv(actor_id=0, env_fns=self.env_fns)

    def _create_policy_handler(self):
        pass

    def _validate_config(self, config: ParallelizationConfig):
        assert config.num_actors == 1
        actor_config = config[0]

        assert (actor_config.num_parallel == 1) and (
            actor_config.batch_size == actor_config.num_envs
        )


class MultiLoopHandler(AuturiLoopHandler):
    pass
