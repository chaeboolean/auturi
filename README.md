Auturi(AUTomatic and Unified parallelization strategy searching framework for deep ReInforcement learning) is composed of two main components.

- Tuner (TODO): auturi/tuner
- Executor: auturi/executor

The hierarchy of Executor's data structures is described below.

AuturiEnv    --- AuturiSerialEnv --- AuturiParallelEnv  --- 
                                                         |
                                                         --- AuturiActor --- AuturiVectorActor 
                                                         |
AuturiPolicy ----------------------- AuturiVectorPolicy ---



Example code is like below.
    
    '''python
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