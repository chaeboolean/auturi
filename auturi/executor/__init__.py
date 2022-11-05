"""
Defines typings for Auturi Systems. 
The hierarchy of all data structures is described below.

AuturiEnv    --- AuturiSerialEnv --- AuturiParallelEnv  --- 
                                                         |
                                                         --- AuturiActor --- AuturiVectorActor 
                                                         |
AuturiPolicy ----------------------- AuturiVectorPolicy ---



Example code is like below.
    
    '''
    class MyOwnEnv(AuturiEnv):
        pass
        
    class MyOwnPolicy(AuturiPolicy):
        def __init__(self, **policy_kwargs):
            pass

    env_fns = [lambda: MyOwnEnv() for _ in range(num_max_envs)]

    auturi_engine = create_executor(
        env_fns = env_fns, 
        policy_cls = MyOwnPolicy, 
        policy_kwargs = policy_kwargs, 
        tuner = GridSearchTuner,
        backend="ray",
    )
    
    auturi_engine.run(num_collect=1e6)

    '''

"""

from typing import Any, Callable, Dict, List

from auturi.executor.environment import AuturiEnv
from auturi.executor.ray import RayVectorActor
from auturi.tuner import AuturiTuner


def create_executor(
    env_fns: List[Callable[[], AuturiEnv]],
    policy_cls: Any,
    policy_kwargs: Dict[str, Any],
    tuner: AuturiTuner,
    backend="ray",
):
    engine_cls = {
        "ray": RayVectorActor,
    }[backend]

    return engine_cls(env_fns, policy_cls, policy_kwargs, tuner)


__all__ = ["create_executor"]
