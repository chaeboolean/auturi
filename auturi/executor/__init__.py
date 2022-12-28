from typing import Any, Callable, Dict, List

from auturi.executor.environment import AuturiEnv

# from auturi.executor.ray import RayVectorActor
from auturi.executor.shm import SHMExecutor
from auturi.tuner import AuturiTuner


def create_executor(
    env_fns: List[Callable[[], AuturiEnv]],
    policy_cls: Any,
    policy_kwargs: Dict[str, Any],
    tuner: AuturiTuner,
    backend="shm",
):
    engine_cls = {
        #        "ray": RayVectorActor,
        "shm": SHMExecutor,
    }[backend]

    return engine_cls(env_fns, policy_cls, policy_kwargs, tuner)


__all__ = ["create_executor"]
