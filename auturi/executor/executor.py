from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Tuple

from auturi.executor.environment import AuturiEnv
from auturi.executor.loop import (
    AuturiLoopHandler,
    MultiLoopHandler,
    NestedLoopHandler,
    SimpleLoopHandler,
)
from auturi.executor.typing import PolicyModel
from auturi.logger import get_logger
from auturi.tuner import ActorConfig, AuturiMetric, AuturiTuner, ParallelizationConfig

logger = get_logger("Exeuctor")


def _is_simple_loop(actor_config: ActorConfig):
    return (
        (actor_config.num_policy == 1)
        and (actor_config.batch_size == actor_config.num_envs)
        and (actor_config.num_parallel == 1)
    )


class AuturiExecutor(metaclass=ABCMeta):
    """Executes parallelization strategy given by AuturiTuner.

    As the highest level component, it handles single or multiple actors.
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
        self._loop_handler = None

        super().__init__()

    def _create_simple_loop_handler(self) -> SimpleLoopHandler:
        return SimpleLoopHandler(0, self.env_fns, self.policy_cls, self.policy_kwargs)

    @abstractmethod
    def _create_nested_loop_handler(self) -> NestedLoopHandler:
        """Create VectorActor with specific backend."""
        raise NotImplementedError

    @abstractmethod
    def _create_multiple_loop_handler(self) -> MultiLoopHandler:
        """Create LocalActor with specific backend."""
        raise NotImplementedError

    def _get_or_create_loop_handler(self, cls: AuturiLoopHandler, create_fn: Callable):
        prev_handler = self._loop_handler
        if isinstance(prev_handler, cls):
            return prev_handler

        else:
            if prev_handler is not None:
                prev_handler.terminate()

            logger.debug(f"Create {cls}")
            return create_fn()

    def get_loop_handler(self, config):
        if config.num_actors == 1:
            if _is_simple_loop(config[0]):
                return self._get_or_create_loop_handler(
                    SimpleLoopHandler, self._create_simple_loop_handler
                )

            else:
                return self._get_or_create_loop_handler(
                    NestedLoopHandler, self._create_nested_loop_handler
                )

        else:
            for _, actor_config in config.actor_map.items():
                assert _is_simple_loop(
                    actor_config
                ), f"Do not support such config for now."
            return self._get_or_create_loop_handler(
                MultiLoopHandler, self._create_multiple_loop_handler
            )

    def reconfigure(self, config: ParallelizationConfig, model: PolicyModel):
        """Adjust executor's component according to tuner-given config.

        Args:
            config (ParallelizationConfig): Configurations for tuning.
            model (PolicyModel): Policy network for compute next actions.

        """

        self._loop_handler = self.get_loop_handler(config)
        self._loop_handler.reconfigure(config, model)

    def run(self, model: PolicyModel) -> Tuple[Dict[str, Any], AuturiMetric]:
        """Run collection loop with `tuner.num_collect` iterations, and return experience trajectories and AuturiMetric."""
        next_config = self.tuner.next()
        self.reconfigure(next_config, model)

        rollouts, metric = self._loop_handler.run()

        # Give result to tuner.
        self.tuner.feedback(metric)
        return rollouts, metric

    def terminate(self):
        self._loop_handler.terminate()
