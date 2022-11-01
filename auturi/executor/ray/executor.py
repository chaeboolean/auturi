import math
from typing import Any, Callable, Dict, Tuple

import ray
import torch.nn as nn

import auturi.executor.ray.util as util
from auturi.executor.actor import AuturiActor
from auturi.executor.config import ActorConfig, AuturiMetric
from auturi.executor.environment import AuturiEnv
from auturi.executor.executor import AuturiExecutor
from auturi.executor.policy import AuturiPolicy
from auturi.executor.vector_utils import aggregate_partial
from auturi.tuner import AuturiTuner


@ray.remote
class RayActorWrapper(AuturiActor):
    """Wrappers run in separated Ray process."""

    pass


class RayExecutor(AuturiExecutor):
    def __init__(
        self,
        vector_env_fn: Callable[[], AuturiEnv],
        vector_policy_fn: Callable[[], AuturiPolicy],
        tuner: AuturiTuner,
    ):
        super().__init__(vector_env_fn, vector_policy_fn, tuner)
        self.pending_actors = dict()

    def _create_worker(self, idx: int) -> AuturiActor:
        if idx == 0:
            return AuturiActor(self.vector_env_fn, self.vector_policy_fn)
        else:
            return RayActorWrapper.remote(self.vector_env_fn, self.vector_policy_fn)

    def _reconfigure_actor(
        self, idx: int, actor: AuturiActor, config: ActorConfig, model: nn.Module
    ):
        """Reconfigure each actor."""
        if idx == 0:
            actor.reconfigure(config, model)
        else:
            ref = actor.reconfigure.remote(config, model)
            self.pending_actors[ref] = idx

    def _run(self, num_collect: int) -> Tuple[Dict[str, Any], AuturiMetric]:
        util.clear_pending_list(self.pending_actors)

        num_collect_per_actor = math.ceil(num_collect / self.num_workers)
        for actor_id, actor in self._working_workers():
            if actor_id == 0:
                continue
            else:
                ref = actor.run.remote(num_collect_per_actor)
                self.pending_actors[ref] = actor_id

        # TODO: Hack
        local_rollouts, local_metric = self.local_worker.run(num_collect_per_actor)

        # Aggregate
        remote_rollouts = ray.get(list(self.pending_actors.keys()))
        partial_rollouts = [local_rollouts] + [
            remote_[0] for remote_ in remote_rollouts
        ]
        agg_rollouts = aggregate_partial(partial_rollouts)
        return agg_rollouts, local_metric
