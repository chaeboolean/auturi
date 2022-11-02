"""
Defines typings for Auturi Systems. 
The hierarchy of all data structures is described below.


Example code is like below.
    
    '''
    class MyOwnEnv(AuturiEnv):
        pass
        
    class MyOwnPolicy(AuturiPolicy):
        def __init__(self, **policy_kwargs):
            pass

    env_fns = [lambda: MyOwnEnv() for _ in range(num_max_envs)]

    auturi_env_fn, auturi_policy_fn = create_actor_args(
        env_fns = env_fns, 
        policy_cls = MyOwnPolicy, 
        policy_kwargs = policy_kwargs, 
        backend="ray",
    )
    '''

"""

from typing import Any, Callable, Dict, List, Tuple

from auturi.executor.environment import AuturiEnv, AuturiVectorEnv
from auturi.executor.policy import AuturiVectorPolicy
from auturi.executor.ray import create_ray_actor_args
from auturi.executor.ray.executor import RayExecutor

# def create_actor_args(
#     env_fns: List[Callable[[], AuturiEnv]],
#     policy_cls: Any,
#     policy_kwargs: Dict[str, Any],
#     backend="ray",
# ) -> Tuple[Callable[[], AuturiVectorEnv], Callable[[], AuturiVectorPolicy]]:

#     create_fn = {
#         "ray": create_ray_actor_args,
#         "shm": create_shm_actor_args,
#     }[backend]

#     return create_fn(env_fns, policy_cls, policy_kwargs)


# __all__ = ["create_actor_args", "RayExecutor"]
