from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Tuple

import torch.nn as nn

from auturi.executor.actor import AuturiActor
from auturi.executor.environment import AuturiEnv
from auturi.executor.vector_utils import VectorMixin
from auturi.tuner import AuturiTuner
from auturi.tuner.config import AuturiMetric, ParallelizationConfig


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

        self.env_fns = env_fns
        self.policy_cls = policy_cls
        self.policy_kwargs = policy_kwargs
        self.tuner = tuner

    @property
    def num_actors(self):
        return self.num_workers

    @abstractmethod
    def reconfigure(self, config: ParallelizationConfig, model: nn.Module):
        """Adjust executor's component according to tuner-given config.

        Args:
            config (ParallelizationConfig): Configurations for tuning.
            model (nn.Module): Policy network for compute next actions.

        """
        self.reconfigure_workers(config, model)

    def run(self, model: nn.Module) -> Tuple[Dict[str, Any], AuturiMetric]:
        """Run collection loop with `num_collect` iterations, and return experience trajectories and AuturiMetric."""
        next_config = self.tuner.next()
        self.reconfigure(next_config, model)

        return self._run(self.tuner.num_collect)

    @abstractmethod
    def _run(self, num_collect: int) -> Tuple[Dict[str, Any], AuturiMetric]:
        """Run each actor."""
        raise NotImplementedError

    @abstractmethod
    def terminate(self):
        raise NotImplementedError
