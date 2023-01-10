import argparse
import functools
import os

#from auturi.benchmarks.tasks.circuit_wrap import CircuitEnvWrapper, CircuitPolicyWrapper
from auturi.benchmarks.tasks.football_wrap import (
    FootballEnvWrapper,
    FootballPolicyWrapper,
    FootballScenarios, 
)
from auturi.benchmarks.tasks.sb3_wrap import SB3EnvWrapper, SB3PolicyWrapper
from auturi.executor import create_executor
from auturi.tuner import ActorConfig, ParallelizationConfig, create_tuner_with_config
from auturi.tuner.greedy_tuner import GreedyTuner

from auturi.common.chrome_profiler import merge_file

def create_envs(cls, task_id, rank, dummy=None):
    return cls(task_id, rank)

def create_tuner(args):
    return GreedyTuner(
        min_num_env=args.num_envs,
        max_num_env=args.num_envs,
        num_collect=args.num_collect, 
        max_policy_num=8,
        use_gpu=args.cuda, 
        num_iterate=args.num_iteration,
        task_name=args.env,
        num_core=args.num_core, 
        log_path=f"/workspace/auturi/tuner-log/{args.env}.log",
    ) 


def prepare_task(env_name, num_envs):
    validator = None
    if env_name in []:
        task_id = env_name
        env_cls, policy_cls = FootballEnvWrapper, FootballPolicyWrapper
        validator = lambda x: x[0].policy_device != "cpu"

    elif env_name == "circuit":
        task_id = None
        env_cls, policy_cls = CircuitEnvWrapper, CircuitPolicyWrapper
        validator = lambda x: x[0].policy_device != "cpu"

    else:
        atari_name = "PongNoFrameskip-v4"
        task_id = atari_name if env_name == "atari" else env_name
        env_cls, policy_cls = SB3EnvWrapper, SB3PolicyWrapper
        validator = lambda x: x[0].policy_device != "cpu"

    env_fns = [
        functools.partial(create_envs, env_cls, task_id, rank)
        for rank in range(num_envs)
    ]
    policy_kwargs = {"task_id": task_id}

    return env_fns, policy_cls, policy_kwargs, validator



def run(args):
    env_fns, policy_cls, policy_kwargs, validator = prepare_task(args.env, args.num_envs)
    tuner = create_tuner(args)
    executor = create_executor(env_fns, policy_cls, policy_kwargs, tuner, "shm")

    try:
        while True:
            executor.run(None)
    except StopIteration:
        executor.terminate()
        print("search finish....")
        print(tuner.dict_bs)
        

if __name__ == "__main__":
    import sys
    print(sys.version)

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str)

    parser.add_argument(
        "--num-iteration", type=int, default=3, help="number of trials for each config."
    )

    parser.add_argument(
        "--num-envs", type=int, default=4, help="number of environments."
    )

    parser.add_argument(
        "--num-core", type=int, default=16, help="number of total core."
    )

    parser.add_argument(
        "--num-collect", type=int, default=40, help="number of trajectories to collect."
    )

    parser.add_argument(
        "--cuda", action="store_true", help="Use cuda"
    )
    
    args = parser.parse_args()
    print(args)
    run(args)

