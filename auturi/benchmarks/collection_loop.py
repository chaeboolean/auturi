import argparse
import functools
import os

# from auturi.benchmarks.tasks.circuit_wrap import CircuitEnvWrapper, CircuitPolicyWrapper
# from auturi.benchmarks.tasks.football_wrap import (
#     FootballEnvWrapper,
#     FootballPolicyWrapper,
#     FootballScenarios, 
# )
from auturi.benchmarks.tasks.sb3_wrap import SB3EnvWrapper, SB3PolicyWrapper, SB3LSTMPolicyWrapper, is_atari
from auturi.executor import create_executor
from auturi.tuner import ActorConfig, ParallelizationConfig, create_tuner_with_config
from auturi.tuner.specific_parallelism import SpecificParallelismComparator

from auturi.common.chrome_profiler import merge_file

def create_envs(cls, task_id, rank, dummy=None):
    return cls(task_id, rank)


def make_naive_tuner(args, _):
    num_loop = 1

    subproc_config = ActorConfig(
        num_envs=args.num_envs,
        num_parallel=args.num_envs, # ep 
        num_policy=1, # pp
        batch_size=args.num_envs, #bs
        num_collect=args.num_collect // num_loop,
        policy_device="cuda:0",
    )
    tuner_config = ParallelizationConfig.create([subproc_config] * num_loop)
    return create_tuner_with_config(args.num_envs, args.num_iteration, tuner_config, "", "no")


def make_specfic_tuner(args, validator=None):
    return SpecificParallelismComparator(
        [args.tuner_mode],
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

    return env_fns, policy_cls, policy_kwargs, validator


def trace_out_name(args, config):
    config = config[0]
    config_str = f"ep={config.num_parallel}pp={config.num_policy}bs={config.batch_size}"
    return f"{args.env}_{args.num_envs}", config_str

def run(args):
    env_fns, policy_cls, policy_kwargs, validator = prepare_task(args.env, args.num_envs)
    tuner = make_naive_tuner(args, None)
    executor = create_executor(env_fns, policy_cls, policy_kwargs, tuner, "shm")

    try:
        while True:
            executor.run(None)
    except StopIteration:
        executor.terminate()
        print("search finish....")
        print(tuner.tuning_results)

    if args.trace:
        out_dir, output_name = trace_out_name(args, tuner.config)
        merge_file(out_dir, output_name)
        print(output_name)
        

if __name__ == "__main__":
    import sys
    print(sys.version)

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="academy_3_vs_1_with_keeper")

    parser.add_argument(
        "--num-iteration", type=int, default=2, help="number of trials for each config."
    )

    parser.add_argument(
        "--num-envs", type=int, default=4, help="number of environments."
    )

    parser.add_argument(
        "--num-collect", type=int, default=20, help="number of trajectories to collect."
    )

    parser.add_argument(
        "--policy", type=str, default="", help="Type of policy network."
    )

    parser.add_argument(
        "--tuner-log-path", type=str, default=None, help="Log file path."
    )

    parser.add_argument("--tuner-mode", help="Tuner Mode", type=str, default="dummy", choices=["L", "E+P"])
    parser.add_argument("--trace", action="store_true", help="skip backprop stage.")


    args = parser.parse_args()
    print("\n\n", "=" * 20)
    print(args)

    if args.tuner_log_path is not None:
        with open(args.tuner_log_path, "a") as f:
            f.write(str(args) + "\n")
    
    if args.trace:
        os.environ["AUTURI_TRACE"] = "1"

    run(args)
