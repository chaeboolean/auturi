from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Tuple

from auturi.executor.actor import AuturiActor
from auturi.executor.config import ActorConfig, AuturiMetic, TunerConfig
from auturi.executor.environment import AuturiVectorEnv
from auturi.executor.policy import AuturiVectorPolicy
from auturi.executor.vector_utils import VectorMixin
from auturi.tuner import AuturiTuner


class AuturiExecutor(VectorMixin, metaclass=ABCMeta):
    """Interacts with Tuner.
    Get configuration from tuner, and change its execution plan.
    One of major components in Auturi System.
    Handles multiple Actors, similar with VectorActor.
    """

    def __init__(
        self,
        vector_env: AuturiVectorEnv,
        vector_policy: AuturiVectorPolicy,
        tuner: AuturiTuner,
    ):
        self.vector_env = vector_env
        self.vector_policy = vector_policy
        self.tuner = tuner

        self.set_vector_attrs()

    def reconfigure(self, next_config: TunerConfig):
        """Adjust executor's component according to tuner-given config.

        Args:
            next_config (TunerConfig): Configurations for tuning.
        """
        # set number of currently working actors
        self.num_workers = next_config.num_actors

        # Set configs for each actor.
        for actor_id, actor in self._working_workers():
            actor = self._get_actor(self, actor_id)
            self._reconfigure_actor(actor_id, actor, next_config[actor_id])

    def run(self, num_collect: int) -> Tuple[Dict[str, Any], AuturiMetic]:
        """Run collection loop with `num_collect` iterations, and return experience trajectories"""

        next_config = self.tuner.step()
        self.reconfigure(next_config)

        for actor_id, actor in self._working_workers():
            aggs = self._run(actor_id, actor, num_collect)

    @abstractmethod
    def _reconfigure_actor(self, idx: int, actor: AuturiActor, config: ActorConfig):
        """Reconfigure each actor."""
        raise NotImplementedError
