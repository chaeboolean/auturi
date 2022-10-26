from typing import Any, Dict
import time

import ray
import torch.nn as nn
from auturi.typing.environment import AuturiVectorEnv
from auturi.typing.policy import AuturiPolicy
from auturi.typing.tuning import ActorConfig, AuturiMetic


class AuturiActor:
    """AuturiActor is an abstraction of collection loop.

    AuturiActor is comprised of AuturiVectorEnv and AuturiPolicy.

    """

    def __init__(self, envs: AuturiVectorEnv, policy: AuturiPolicy):
        assert isinstance(envs, AuturiVectorEnv)
        assert isinstance(policy, AuturiPolicy)

        self.envs = envs
        self.policy = policy
        

    def reconfigure(self, config: ActorConfig, model: nn.Module):
        """Adjust envs and policy by given configs."""

        # Adjust Policy
        self.policy.reconfigure(config.num_policy)
        self.policy.load_model(model, config.policy_device)

        # Adjust Environment
        self.envs.reconfigure(
            num_envs=config.num_envs,
            num_parallel=config.num_parallel,
        )
        self.envs.set_batch_size(config.batch_size)

    def run(self, num_collect: int) -> Dict[str, Any]:
        """Run collection loop with `num_collect` iterations, and return experience trajectories."""

        self.policy.start_loop()
        self.envs.start_loop()

        n_steps = 0
        start_time = time.perf_counter()
        while n_steps < num_collect:
            print("before poll... ", round(time.perf_counter()-start_time, 2))
            obs_refs = self.envs.poll()

            print("after poll... ", round(time.perf_counter()-start_time, 2))
            action_refs = self.policy.compute_actions(obs_refs, n_steps)
            print("after send actions... ", round(time.perf_counter()-start_time, 2))

            self.envs.send_actions(action_refs)

            n_steps += self.envs.batch_size # len(obs_refs)

        self.policy.stop_loop()
        self.envs.stop_loop()
        end_time = time.perf_counter()
        
        return self.envs.aggregate_rollouts(), AuturiMetic(num_collect, end_time-start_time)


@ray.remote
class RayAuturiActor(AuturiActor):
    """Auturi Actor as Ray Wrapper."""

    pass
