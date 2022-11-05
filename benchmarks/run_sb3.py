import argparse
import functools

import numpy as np
from auturi.adapter.sb3 import wrap_sb3_OnPolicyAlgorithm
from auturi.tuner import create_tuner_with_config
from auturi.tuner.config import ActorConfig, TunerConfig
from rl_zoo3.exp_manager import ExperimentManager


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
    model, _ = exp_manager.setup_experiment()
    model.env_fns = [_wrap for _ in range(exp_manager.n_envs)]

    model._auturi_iteration = args.num_iteration
    model._auturi_train_skip = args.skip_update

    return exp_manager, model


def run(args):
    # create ExperimentManager with minimum argument.
    vec_cls = "dummy" if args.running_mode == "auturi" else args.running_mode
    exp_manager, model = create_sb3_algorithm(args, vec_cls)

    if args.running_mode == "auturi":
        n_envs = exp_manager.n_envs
        actor_config = ActorConfig(num_envs=4, num_parallel=2, batch_size=4)
        tuner_config = TunerConfig(
            2,
            {0: actor_config, 1: actor_config},
        )
        tuner = create_tuner_with_config(n_envs, tuner_config)

        wrap_sb3_OnPolicyAlgorithm(model, tuner=tuner)

    exp_manager.learn(model)
    collect_times = np.array(model.collect_time[2:]) * 1000

    print(f"*** Result = {sum(model.collect_time[2:])}")
    # print(f"*** Avg time = {collect_times.mean()}")


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
    parser.add_argument("--num-iteration", type=int, default=10)

    args = parser.parse_args()
    run(args)
