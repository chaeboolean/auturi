from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, Tuple

import torch.nn as nn

from auturi.executor.actor import AuturiActor
from auturi.executor.config import ActorConfig, AuturiMetric, TunerConfig
from auturi.executor.environment import AuturiEnv
from auturi.executor.policy import AuturiPolicy
from auturi.executor.vector_utils import VectorMixin
from auturi.tuner import AuturiTuner


# TODO: Get env_fns at the beginning, and hands out to each actor.
class AuturiExecutor(VectorMixin, metaclass=ABCMeta):
    """Interacts with Tuner.
    Get configuration from tuner, and change its execution plan.
    One of major components in Auturi System.
    Handles multiple Actors, similar with VectorActor.
    """

    def __init__(
        self,
        vector_env_fn: Callable[[], AuturiEnv],
        vector_policy_fn: Callable[[], AuturiPolicy],
        tuner: AuturiTuner,
    ):
        self.vector_env_fn = vector_env_fn
        self.vector_policy_fn = vector_policy_fn
        self.tuner = tuner

        self.set_vector_attrs()

    def reconfigure(self, config: TunerConfig, model: nn.Module):
        """Adjust executor's component according to tuner-given config.

        Args:
            next_config (TunerConfig): Configurations for tuning.
        """
        # set number of currently working actors

        self.num_workers = config.num_actors

        # Set configs for each actor.
        for actor_id, actor in self._working_workers():
            self._reconfigure_actor(actor_id, actor, config.get(actor_id), model)

    def run(
        self, model: nn.Module, num_collect: int
    ) -> Tuple[Dict[str, Any], AuturiMetric]:
        """Run collection loop with `num_collect` iterations, and return experience trajectories"""
        next_config = self.tuner.next()
        self.reconfigure(next_config, model)
        return self._run(num_collect)

    @abstractmethod
    def _reconfigure_actor(
        self, idx: int, actor: AuturiActor, config: ActorConfig, model: nn.Module
    ):
        """Reconfigure each actor."""
        raise NotImplementedError

    @abstractmethod
    def _run(self, num_collect: int) -> Tuple[Dict[str, Any], AuturiMetric]:
        """Run each actor."""
        raise NotImplementedError

    def terminate(self):
        pass
