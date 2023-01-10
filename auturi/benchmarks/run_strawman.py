import argparse
import functools
import os

#from auturi.benchmarks.tasks.circuit_wrap import CircuitEnvWrapper, CircuitPolicyWrapper
from auturi.benchmarks.tasks.sb3_wrap import SB3EnvWrapper, SB3PolicyWrapper, SB3LSTMPolicyWrapper
from auturi.benchmarks.tasks.sb3_wrap import is_atari

from auturi.executor import create_executor
from auturi.tuner import ActorConfig, ParallelizationConfig, create_tuner_with_config
from auturi.tuner.greedy_tuner import GreedyTuner

from auturi.common.chrome_profiler import merge_file

def create_envs(cls, task_id, rank, dummy=None):
    return cls(task_id, rank)

def create_greedy_tuner(args):
    return GreedyTuner(
        min_num_env=args.num_envs,
        max_num_env=args.num_envs,
        num_collect=args.num_collect, 
        max_policy_num=8,
        use_gpu=args.cuda, 
        num_iterate=args.num_iteration,
        task_name=args.env,
        num_core=args.num_core, 
        log_path=f"/workspace/auturi/tuner-log/{args.env}_{args.cuda}.log",
    ) 

def create_maximum_tuner(args):
    log_path = "/workspace/auturi/tuner-log/maximum.log"
    if args.cuda:
        subproc_config = ActorConfig(
            num_envs=args.num_envs,
            num_parallel=8, # ep 
            num_policy=8, # pp
            batch_size=2, #bs
            num_collect=args.num_collect,
            policy_device="cuda:0",
        )

    else:
        subproc_config = ActorConfig(
            num_envs=args.num_envs,
            num_parallel=8, # ep 
            num_policy=8, # pp
            batch_size=2, #bs
            num_collect=args.num_collect,
            policy_device="cpu",
        )

    tuner_config = ParallelizationConfig.create([subproc_config])
    device = "cuda" if args.cuda else "cpu"
    task_name = f"{args.env}_{device}"
    return create_tuner_with_config(args.num_envs, args.num_iteration, tuner_config, log_path=log_path, task_name=task_name )


def prepare_task(env_name, num_envs):
    validator = None
    if env_name == "circuit":
        task_id = None
        env_cls, policy_cls = CircuitEnvWrapper, CircuitPolicyWrapper
        validator = lambda x: x[0].policy_device != "cpu"

    elif is_atari(env_name):
        task_id = env_name
        env_cls, policy_cls = SB3EnvWrapper, SB3LSTMPolicyWrapper
        validator = lambda x: x[0].policy_device != "cpu"

    else:
        task_id = env_name
        env_cls, policy_cls = SB3EnvWrapper, SB3PolicyWrapper
        validator = lambda x: x[0].policy_device != "cpu"        

    env_fns = [
        functools.partial(create_envs, env_cls, task_id, rank)
        for rank in range(num_envs)
    ]
    policy_kwargs = {"task_id": task_id}

    return env_fns, policy_cls, policy_kwargs



def run(args):
    env_fns, policy_cls, policy_kwargs = prepare_task(args.env, args.num_envs)
    
    if args.tuner_mode == "greedy": 
        tuner = create_greedy_tuner(args)
    elif args.tuner_mode == "maximum": 
        tuner = create_maximum_tuner(args)

    executor = create_executor(env_fns, policy_cls, policy_kwargs, tuner, "shm")

    try:
        while True:
            executor.run(None)
    except Exception as e:
        tuner.terminate_tuner()
        executor.terminate()
        print("search finish....")
        

if __name__ == "__main__":
    import sys
    print(sys.version)

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str)

    parser.add_argument("--tuner-mode", help="Tuner Mode", type=str, default="greedy", choices=["greedy", "maximum"])

    parser.add_argument(
        "--num-iteration", type=int, default=5, help="number of trials for each config."
    )

    parser.add_argument(
        "--num-envs", type=int, default=16, help="number of environments."
    )

    parser.add_argument(
        "--num-core", type=int, default=16, help="number of total core."
    )

    parser.add_argument(
        "--num-collect", type=int, default=3200, help="number of trajectories to collect."
    )

    parser.add_argument(
        "--cuda", action="store_true", help="Use cuda"
    )
    
    args = parser.parse_args()
    print(args)
    run(args)

