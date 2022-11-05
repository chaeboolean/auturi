import math
from typing import Any, Callable, Dict, List, Tuple

import ray
import torch.nn as nn

import auturi.executor.ray.util as util
from auturi.executor.actor import AuturiActor
from auturi.executor.environment import AuturiEnv
from auturi.executor.ray.environment import RayParallelEnv
from auturi.executor.ray.policy import RayVectorPolicy
from auturi.executor.vector_actor import AuturiVectorActor
from auturi.executor.vector_utils import aggregate_partial
from auturi.tuner import AuturiTuner
from auturi.tuner.config import ActorConfig, AuturiMetric


@ray.remote(num_gpus=0.01)
class RayActorWrapper(AuturiActor):
    """Wrappers run in separated Ray process."""

    pass


class RayVectorActor(AuturiVectorActor):
    def __init__(
        self,
        env_fns: List[Callable[[], AuturiEnv]],
        policy_cls: Any,
        policy_kwargs: Dict[str, Any],
        tuner: AuturiTuner,
    ):
        self.pending_actors = dict()
        super().__init__(env_fns, policy_cls, policy_kwargs, tuner)

    def _create_env(self, _env_fns):
        def _wrap():
            return RayParallelEnv(_env_fns)

        return _wrap

    def _create_policy(self, _policy_cls, _policy_kwargs):
        def _wrap():
            return RayVectorPolicy(_policy_cls, _policy_kwargs)

        return _wrap

    def _create_worker(self, idx: int) -> AuturiActor:
        if idx == 0:
            return AuturiActor(self.vector_env_fn, self.vector_policy_fn)
        else:
            return RayActorWrapper.remote(self.vector_env_fn, self.vector_policy_fn)

    def _reconfigure_actor(
        self,
        idx: int,
        actor: AuturiActor,
        config: ActorConfig,
        start_env_idx: int,
        model: nn.Module,
    ):
        """Reconfigure each actor."""
        if idx == 0:
            actor.reconfigure(config, start_env_idx, model)
        else:
            ref = actor.reconfigure.remote(config, start_env_idx, model)
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

        local_rollouts, local_metric = self.local_worker.run(num_collect_per_actor)

        # Aggregate
        remote_rollouts = ray.get(list(self.pending_actors.keys()))
        partial_rollouts = [local_rollouts] + [
            remote_[0] for remote_ in remote_rollouts
        ]
        agg_rollouts = aggregate_partial(partial_rollouts)

        # TODO: metric should not be local_metric.
        return agg_rollouts, local_metric
