from typing import List

import numpy as np

from auturi.executor.actor import AuturiActor
from auturi.executor.shm.environment import SHMParallelEnv
from auturi.executor.shm.policy import SHMVectorPolicy


class SHMActor(AuturiActor):
    def __init__(
        self,
        actor_id,
        env_fns,
        policy_cls,
        policy_kwargs,
        base_buffer_attr,
        rollout_buffer_attr,
    ):
        self.base_buffer_attr = base_buffer_attr
        self.rollout_buffer_attr = rollout_buffer_attr
        super().__init__(actor_id, env_fns, policy_cls, policy_kwargs)

    def _create_vector_env(self, env_fns):
        return SHMParallelEnv(
            self.actor_id, env_fns, self.base_buffer_attr, self.rollout_buffer_attr
        )

    def _create_vector_policy(self, policy_cls, policy_kwargs):
        return SHMVectorPolicy(
            self.actor_id, policy_cls, policy_kwargs, self.base_buffer_attr
        )
