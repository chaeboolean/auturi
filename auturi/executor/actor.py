import time
from typing import Any, Callable, Dict, Tuple

import ray
import torch.nn as nn

from auturi.executor.environment import AuturiEnv
from auturi.executor.policy import AuturiPolicy
from auturi.tuner.config import ActorConfig, AuturiMetric


class AuturiActor:
    """AuturiActor is an abstraction designed to support the auto-parallelization of collection loops in RL.

    AuturiActor is comprised of AuturiVectorEnv and AuturiPolicy, where each can have a varying level of parallelism.
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

    def reconfigure(self, config: ActorConfig, start_env_idx: int, model: nn.Module):
        """Reconfigure the number of envs and policies according to a given config found by AuturiTuner."""

        # Adjust Policy
        self.policy.reconfigure(config, model)

        # Adjust Environment
        self.envs.reconfigure(config, start_env_idx)

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
