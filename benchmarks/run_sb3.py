import argparse
import functools
import time

import numpy as np
from auturi.adapter.sb3.buffer_utils import wrap_sb3_OnPolicyAlgorithm
from auturi.tuner.config import ActorConfig, TunerConfig
from auturi.tuner import create_tuner_with_config

from rl_zoo3.exp_manager import ExperimentManager

FILENAME = "/home/ooffordable/auturi/auturi/benchmarks/grid_search_sb3_1103.txt"



def create_sb3_algorithm(args, vec_cls="dummy"):
    exp_manager = ExperimentManager(
        args=args,
        algo="ppo",
        env_id=args.env,
        log_folder="",
        vec_env_type=vec_cls,
        device="cuda",
        verbose=0,
    )

    # _wrap = create_envs(exp_manager, 3)
    _wrap = functools.partial(exp_manager.create_envs, 1)
    model, saved_hyperparams = exp_manager.setup_experiment()
    model.env_fns = [_wrap for _ in range(exp_manager.n_envs)]

    model._auturi_iteration = args.num_iteration
    model._auturi_train_skip = args.skip_update

    return exp_manager, model


def search(args):
    # create ExperimentManager with minimum argument.
    exp_manager, model = create_sb3_algorithm(args)
    n_envs = exp_manager.n_envs
    actor_config = ActorConfig(num_envs=n_envs, num_parallel=n_envs, batch_size=n_envs)
    tuner_config = TunerConfig(1, {0: actor_config})
    tuner = create_tuner_with_config(n_envs, tuner_config)
        
    wrap_sb3_OnPolicyAlgorithm(model, tuner=tuner)
    
    
    model._auturi_executor.reconfigure(next_config, model=None)

        print(print_config(next_config))
        exp_manager.learn(model)
        print(f"*** Result = {sum(model.collect_time[2:])}")

        with open(FILENAME, "a") as f:
            f.write(print_config(next_config) + "\n")

            collect_times = np.array(model.collect_time[2:]) * 1000
            f.write(f"*** Avg time = {collect_times.mean()}\n")
            f.write(f"*** Result = {collect_times}\n\n")

        model.collect_time.clear()

        del model._auturi_executor
        time.sleep(10)


def run_sb3(args):
    exp_manager, model = create_sb3_algorithm(args, vec_cls=args.running_mode)
    exp_manager.learn(model)
    collect_times = np.array(model.collect_time[2:]) * 1000

    print(f"*** Result = {sum(model.collect_time[2:])}")
    print(f"*** Avg time = {collect_times.mean()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--running-mode",
        type=str,
        default="search",
        choices=["dummy", "subproc", "search"],
    )
    parser.add_argument("--env", type=str, default="CartPole-v1", help="environment ID")
    parser.add_argument(
        "--skip-update", action="store_true", help="skip backprop stage."
    )
    parser.add_argument("--num-iteration", type=int, default=10)

    args = parser.parse_args()
    if args.running_mode == "search":
        search(args)
    else:
        run_sb3(args)
