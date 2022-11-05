from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Tuple

import torch.nn as nn

from auturi.executor.actor import AuturiActor
from auturi.executor.environment import AuturiEnv
from auturi.executor.vector_utils import VectorMixin
from auturi.tuner import AuturiTuner
from auturi.tuner.config import ActorConfig, AuturiMetric, TunerConfig


class AuturiVectorActor(VectorMixin, metaclass=ABCMeta):
    """Executes parallelization strategy given by AuturiTuner.

    As the highest level component, it handles multiple actors.
    """

    def __init__(
        self,
        env_fns: List[Callable[[], AuturiEnv]],
        policy_cls: Any,
        policy_kwargs: Dict[str, Any],
        tuner: AuturiTuner,
    ):
        """Initialize AuturiVectorActor.

        Args:
            env_fns (List[Callable[[], AuturiEnv]]): List of create env functions.
            policy_cls (Any): Class that inherits AuturiPolicy.
            policy_kwargs (Dict[str, Any]): Keyword arguments used for instantiating policy.
            tuner (AuturiTuner): AuturiTuner.
        """

        self.vector_env_fn = self._create_env(env_fns)
        self.vector_policy_fn = self._create_policy(policy_cls, policy_kwargs)
        self.tuner = tuner
        self.set_vector_attrs()

    @abstractmethod
    def _create_env(self, env_fns: List[Callable[[], AuturiEnv]]):
        """Create function that create VectorEnv with specific backend."""
        pass

    @abstractmethod
    def _create_policy(self, policy_cls: Any, policy_kwargs: Dict[str, Any]):
        """Create function that create VectorPolicy with specific backend."""
        pass

    def reconfigure(self, config: TunerConfig, model: nn.Module):
        """Adjust executor's component according to tuner-given config.

        Args:
            next_config (TunerConfig): Configurations for tuning.
        """
        # set number of currently working actors
        self.num_workers = config.num_actors

        # Set configs for each actor.
        start_env_idx = 0
        for actor_id, actor in self._working_workers():
            self._reconfigure_actor(
                actor_id, actor, config[actor_id], start_env_idx, model
            )
            start_env_idx += config[actor_id].num_envs

    def run(
        self, model: nn.Module, num_collect: int
    ) -> Tuple[Dict[str, Any], AuturiMetric]:
        """Run collection loop with `num_collect` iterations, and return experience trajectories"""
        next_config = self.tuner.next()
        self.reconfigure(next_config, model)
        return self._run(num_collect)

    @abstractmethod
    def _reconfigure_actor(
        self,
        idx: int,
        actor: AuturiActor,
        config: ActorConfig,
        start_env_idx: int,
        model: nn.Module,
    ):
        """Reconfigure each actor."""
        raise NotImplementedError

    @abstractmethod
    def _run(self, num_collect: int) -> Tuple[Dict[str, Any], AuturiMetric]:
        """Run each actor."""
        raise NotImplementedError

    def terminate(self):
        pass
