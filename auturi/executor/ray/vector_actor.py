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
from auturi.tuner import AuturiMetric, AuturiTuner, ParallelizationConfig


class RayActor(AuturiActor):
    """Wrappers run in separated Ray process."""

    def _create_vector_env(self, env_fns: List[Callable[[], AuturiEnv]]):
        return RayParallelEnv(self.actor_id, env_fns)

    def _create_vector_policy(self, policy_cls: Any, policy_kwargs: Dict[str, Any]):
        return RayVectorPolicy(self.actor_id, policy_cls, policy_kwargs)

    @classmethod
    def as_remote(cls, num_gpus=0.01):
        return ray.remote(num_gpus=num_gpus)(cls)


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

    def _create_worker(self, worker_id: int) -> AuturiActor:
        cls = RayActor.as_remote().remote
        return cls(worker_id, self.env_fns, self.policy_cls, self.policy_kwargs)

    def _reconfigure_worker(
        self,
        worker_id: int,
        worker: RayActor,
        config: ParallelizationConfig,
        model: nn.Module,
    ):
        ref = worker.reconfigure.remote(config, model)
        self.pending_actors[ref] = worker_id

    def _terminate_worker(self, worker_id: int, worker: RayActor):
        del worker

    def _run(self) -> Tuple[Dict[str, Any], AuturiMetric]:
        util.clear_pending_list(self.pending_actors)

        for actor_id, actor in self.workers():
            ref = actor.run.remote()
            self.pending_actors[ref] = actor_id

        # Aggregate
        rollouts_for_each_actor = ray.get(list(self.pending_actors.keys()))
        partial_rollouts = [remote_[0] for remote_ in rollouts_for_each_actor]
        agg_rollouts = aggregate_partial(partial_rollouts)

        # TODO: metric should not be local_metric.
        return agg_rollouts, rollouts_for_each_actor[0][1]

    def terminate(self):
        util.clear_pending_list(self.pending_actors)
        for worker_id, worker in self.workers():
            self._terminate_worker(worker_id, worker)
