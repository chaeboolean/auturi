from typing import Any, Callable, Dict, List, Tuple

from auturi.executor.environment import AuturiEnv, AuturiVectorEnv
from auturi.executor.policy import AuturiVectorPolicy
from auturi.executor.ray.environment import RayParallelEnv
from auturi.executor.ray.executor import RayExecutor
from auturi.executor.ray.policy import RayVectorPolicy
from auturi.tuner import AuturiTuner


def create_ray_actor_args(
    env_fns: List[Callable[[], AuturiEnv]],
    policy_cls: Any,
    policy_kwargs: Dict[str, Any],
) -> Tuple[Callable[[], AuturiVectorEnv], Callable[[], AuturiVectorPolicy]]:
    def create_env(env_fns):
        def _wrap():
            return RayParallelEnv(env_fns)

        return _wrap

    def create_policy(policy_cls, policy_kwargs):
        def _wrap():
            return RayVectorPolicy(policy_cls, policy_kwargs)

        return _wrap

    return create_env(env_fns), create_policy(policy_cls, policy_kwargs)


def create_ray_executor(
    env_fns: List[Callable[[], AuturiEnv]],
    policy_cls: Any,
    policy_kwargs: Dict[str, Any],
    tuner: AuturiTuner,
) -> RayExecutor:
    create_env_fn, create_policy_fn = create_ray_actor_args(env_fns, policy_cls, policy_kwargs)
    return RayExecutor(create_env_fn, create_policy_fn, tuner)

__all__ = ["RayParallelEnv", "RayVectorPolicy", "RayExecutor", "create_ray_actor_args"]
