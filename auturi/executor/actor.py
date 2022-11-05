import time
from typing import Any, Callable, Dict, Tuple

import torch.nn as nn

from auturi.executor.config import ActorConfig, AuturiMetric
from auturi.executor.environment import AuturiEnv
from auturi.executor.policy import AuturiPolicy

import ray
class AuturiActor:
    """AuturiActor is an abstraction of collection loop.

    AuturiActor is comprised of AuturiVectorEnv and AuturiPolicy.

    """

    def __init__(
        self,
        env_fn: Callable[[], AuturiEnv],
        policy_fn: Callable[[], AuturiPolicy],
    ):
        self.envs = env_fn()
        self.policy = policy_fn()

        assert isinstance(self.envs, AuturiEnv)
        assert isinstance(self.policy, AuturiPolicy)

    def reconfigure(self, config: ActorConfig, model: nn.Module):
        """Adjust envs and policy by given configs."""

        # Adjust Policy
        self.policy.reconfigure(config, model)

        # Adjust Environment
        self.envs.reconfigure(config)

    def run(self, num_collect: int) -> Tuple[Dict[str, Any], AuturiMetric]:
        """Run collection loop with `num_collect` iterations, and return experience trajectories."""

        self.policy.start_loop()
        self.envs.start_loop()

        n_steps = 0
        start_time = time.perf_counter()
        while n_steps < num_collect:
            obs_refs = self.envs.poll()
            #print("obs_res -> ", ray.get(list(obs_refs.values()))[0].shape)
            action_refs = self.policy.compute_actions(obs_refs, n_steps)
            #print("action_refs -> ", ray.get(action_refs)[0].shape)
            
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
