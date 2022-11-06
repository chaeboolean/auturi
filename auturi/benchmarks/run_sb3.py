import argparse
import functools

import numpy as np
from rl_zoo3.exp_manager import ExperimentManager

from auturi.adapter.sb3 import wrap_sb3_OnPolicyAlgorithm
from auturi.tuner.grid_search import GridSearchTuner


def create_sb3_algorithm(args, num_iteration, vec_cls="dummy"):
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
    model, _ = exp_manager.setup_experiment()
    model.env_fns = [_wrap for _ in range(exp_manager.n_envs)]

    model._auturi_iteration = num_iteration
    model._auturi_train_skip = args.skip_update

    return exp_manager, model


def run(args):
    # create ExperimentManager with minimum argument.
    num_iteration = -1 if args.running_mode == "auturi" else args.num_iteration
    vec_cls = "dummy" if args.running_mode == "auturi" else args.running_mode
    exp_manager, model = create_sb3_algorithm(args, num_iteration, vec_cls)

    if args.running_mode == "auturi":
        n_envs = exp_manager.n_envs
        tuner = GridSearchTuner(
            n_envs, n_envs, max_policy_num=8, num_iterate=args.num_iteration
        )
        wrap_sb3_OnPolicyAlgorithm(model, tuner=tuner)
        try:
            exp_manager.learn(model)
        except StopIteration:
            print("search finish....")
            print(tuner.tuning_results)

    else:
        exp_manager.learn(model)
        print(model.collect_times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--running-mode",
        type=str,
        default="auturi",
        choices=["dummy", "subproc", "auturi"],
    )
    parser.add_argument("--env", type=str, default="CartPole-v1", help="environment ID")
    parser.add_argument(
        "--skip-update", action="store_true", help="skip backprop stage."
    )
    parser.add_argument(
        "--num-iteration", type=int, default=3, help="number of trials for each config."
    )

    args = parser.parse_args()
    run(args)
