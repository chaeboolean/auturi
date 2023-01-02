import argparse
import functools

# from auturi.benchmarks.tasks.circuit_wrap import CircuitEnvWrapper, CircuitPolicyWrapper
from auturi.benchmarks.tasks.football_wrap import (
    FootballEnvWrapper,
    FootballPolicyWrapper,
    FootballScenarios, 
)
from auturi.benchmarks.tasks.sb3_wrap import SB3EnvWrapper, SB3PolicyWrapper
from auturi.executor import create_executor
from auturi.tuner import ActorConfig, ParallelizationConfig, create_tuner_with_config
from auturi.tuner.grid_search import GridSearchTuner
from auturi.tuner.specific_parallelism import SpecificParallelismComparator


def create_envs(cls, task_id, rank, dummy=None):
    return cls(task_id, rank)


def make_naive_tuner(args):
    num_loop = 1

    subproc_config = ActorConfig(
        num_envs=args.num_envs // num_loop,
        num_policy=1,
        num_parallel=2,
        batch_size=args.num_envs // num_loop,
        num_collect=args.num_collect // num_loop,
        policy_device="cuda:0",
    )
    tuner_config = ParallelizationConfig.create([subproc_config] * num_loop)
    return create_tuner_with_config(args.num_envs, tuner_config)


def make_specfic_tuner(args, validator=None):
    return SpecificParallelismComparator(
        ["L", "E+P", "E"],
        args.num_envs,
        args.num_envs,
        max_policy_num=8,
        num_collect=args.num_collect,
        num_iterate=args.num_iteration,
        validator=validator,
        out_file=args.tuner_log_path,
    )


def prepare_task(env_name, num_envs):
    validator = None
    if env_name in FootballScenarios:
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
    tuner = make_specfic_tuner(args, validator)
    executor = create_executor(env_fns, policy_cls, policy_kwargs, tuner, "shm")

    try:
        while True:
            executor.run(None)
    except StopIteration:
        executor.terminate()
        print("search finish....")
        print(tuner.tuning_results)


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
        "--num-collect", type=int, default=4, help="number of trajectories to collect."
    )

    parser.add_argument(
        "--policy", type=str, default="", help="Type of policy network."
    )

    parser.add_argument(
        "--tuner-log-path", type=str, default=None, help="Log file path."
    )

    args = parser.parse_args()
    print("\n\n", "=" * 20)
    print(args)
    
    run(args)
