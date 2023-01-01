import time
from abc import ABCMeta, abstractmethod
from typing import Tuple

import auturi.executor.typing as types
from auturi.executor.environment import AuturiEnvHandler, AuturiLocalEnv
from auturi.executor.policy import AuturiLocalPolicy, AuturiPolicyHandler
from auturi.tuner import AuturiMetric, ParallelizationConfig


class AuturiLoopHandler(metaclass=ABCMeta):
    def __init__(self, env_fns, policy_cls, policy_kwargs):
        self.env_fns = env_fns
        self.policy_cls = policy_cls
        self.policy_kwargs = policy_kwargs

        self.env_handler = None
        self.policy_handler = None
        self.num_collect = -1  # initialized when reconfiguration

    @abstractmethod
    def reconfigure(
        self, config: ParallelizationConfig, model: types.PolicyModel
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def run(self) -> Tuple[types.RolloutRefs, AuturiMetric]:
        raise NotImplementedError


class AuturiSingleLoopHandler(AuturiLoopHandler, metaclass=ABCMeta):
    def __init__(self, loop_id, env_fns, policy_cls, policy_kwargs):
        super().__init__(env_fns, policy_cls, policy_kwargs)
        self.loop_id = loop_id

    @abstractmethod
    def _create_env_handler(self) -> AuturiEnvHandler:
        raise NotImplementedError

    @abstractmethod
    def _create_policy_handler(self) -> AuturiPolicyHandler:
        raise NotImplementedError

    def reconfigure(
        self, config: ParallelizationConfig, model: types.PolicyModel
    ) -> None:
        """Reconfigure the number of envs and policies according to a given config found by AuturiTuner."""
        if self.env_handler is None:
            self.env_handler = self._create_env_handler()

        if self.policy_handler is None:
            self.policy_handler = self._create_policy_handler()

        self.num_collect = config[self.loop_id].num_collect
        self.env_handler.reconfigure(config)
        self.policy_handler.reconfigure(config, model)

    def _num_iteration(self) -> int:
        return self.num_collect

    def run(self) -> Tuple[types.RolloutRefs, AuturiMetric]:
        """Run collection loop for num_collect iterations, and return experience trajectories."""

        self.policy_handler.start_loop()
        self.env_handler.start_loop()

        n_steps = 0
        start_time = time.perf_counter()

        loop_finish_cond = self._num_iteration()
        while n_steps < loop_finish_cond:
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


class SimpleLoopHandler(AuturiSingleLoopHandler):
    def _create_env_handler(self) -> AuturiEnvHandler:
        return AuturiLocalEnv(self.loop_id, self.env_fns)

    def _create_policy_handler(self) -> AuturiPolicyHandler:
        return AuturiLocalPolicy(self.loop_id, self.policy_cls, self.policy_kwargs)

    # TODO: This is hotfix.
    # We artificially add num_envs to prevent env.reset() to be counted as trajectories.
    def _num_iteration(self):
        return self.num_collect + self.env_handler.num_envs


class NestedLoopHandler(AuturiSingleLoopHandler, metaclass=ABCMeta):
    pass


class MultiLoopHandler(AuturiLoopHandler, metaclass=ABCMeta):
    pass
