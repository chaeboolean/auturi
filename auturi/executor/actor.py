import time
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Tuple

import ray
import torch.nn as nn

from auturi.executor.environment import AuturiEnv, AuturiVectorEnv
from auturi.executor.policy import AuturiVectorPolicy
from auturi.tuner.config import AuturiMetric, ParallelizationConfig


class AuturiActor(metaclass=ABCMeta):
    """AuturiActor is an abstraction designed to support the auto-parallelization of collection loops in RL.

    AuturiActor is comprised of AuturiVectorEnv and AuturiVectorPolicy, where each can have a varying level of parallelism.
    """

    def __init__(
        self,
        actor_id: int,
        env_fns: List[Callable[[], AuturiEnv]],
        policy_cls: Any,
        policy_kwargs: Dict[str, Any],
    ):

        self.actor_id = actor_id
        self.vector_envs = self._create_vector_env(env_fns)
        self.vector_policy = self._create_vector_policy(policy_cls, policy_kwargs)

    @abstractmethod
    def _create_vector_env(
        self, env_fns: List[Callable[[], AuturiEnv]]
    ) -> AuturiVectorEnv:
        """Create function that create VectorEnv with specific backend."""
        raise NotImplementedError

    @abstractmethod
    def _create_vector_policy(
        self, policy_cls: Any, policy_kwargs: Dict[str, Any]
    ) -> AuturiVectorPolicy:
        """Create function that create VectorPolicy with specific backend."""
        raise NotImplementedError

    def reconfigure(self, config: ParallelizationConfig, model: nn.Module):
        """Reconfigure the number of envs and policies according to a given config found by AuturiTuner."""

        # Adjust Policy
        self.policy.reconfigure(config, model)

        # Adjust Environment
        self.envs.reconfigure(config)

    def run(self, num_collect: int) -> Tuple[Dict[str, Any], AuturiMetric]:
        """Run collection loop for num_collect iterations, and return experience trajectories."""

        self.policy.start_loop()
        self.envs.start_loop()

        n_steps = 0
        start_time = time.perf_counter()
        while n_steps < num_collect:
            obs_refs = self.envs.poll()
            # print("obs_res -> ", ray.get(list(obs_refs.values()))[0].shape)

            action_refs = self.policy.compute_actions(obs_refs, n_steps)
            # print("action_refs -> ", ray.get(action_refs)[0].shape)

            self.envs.send_actions(action_refs)

            n_steps += self.envs.batch_size  # len(obs_refs)

        self.policy.stop_loop()
        self.envs.stop_loop()
        end_time = time.perf_counter()

        return self.envs.aggregate_rollouts(), AuturiMetric(
            num_collect, end_time - start_time
        )

    def terminate(self):
        self.policy.terminate()
        self.envs.terminate()
