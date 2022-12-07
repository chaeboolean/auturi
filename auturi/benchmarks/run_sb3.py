import argparse
import functools

from rl_zoo3.exp_manager import ExperimentManager

from auturi.adapter.sb3 import wrap_sb3_OnPolicyAlgorithm
from auturi.tuner import ActorConfig, ParallelizationConfig, create_tuner_with_config
from auturi.tuner.grid_search import GridSearchTuner


def create_sb3_algorithm(args, num_iteration, vec_cls="dummy"):
    exp_manager = ExperimentManager(
        args=args,
        algo="ppo",
        env_id=args.env,
        log_folder="",
        vec_env_type=vec_cls,
        device="cpu",
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
    print(args)
    # create ExperimentManager with minimum argument.
    num_iteration = -1 if args.running_mode == "search" else args.num_iteration
    vec_cls = (
        "dummy" if args.running_mode in ["auturi", "search"] else args.running_mode
    )
    exp_manager, model = create_sb3_algorithm(args, num_iteration, vec_cls)

    n_envs = exp_manager.n_envs
    num_collect = model.n_steps * n_envs

    if args.running_mode == "search":
        tuner = GridSearchTuner(
            n_envs,
            n_envs,
            max_policy_num=8,
            num_collect=num_collect,
            num_iterate=args.num_iteration,
        )
        wrap_sb3_OnPolicyAlgorithm(model, tuner=tuner, backend="shm")

        try:
            exp_manager.learn(model)
            exit(0)
        except StopIteration:
            print("search finish....")
            print(tuner.tuning_results)
            model._auturi_executor.terminate()
            return

    if args.running_mode == "auturi":

        # make specific config
        subproc_config = ActorConfig(
            num_envs=n_envs,
            num_policy=1,
            num_parallel=n_envs,
            batch_size=n_envs,
            num_collect=num_collect,
            policy_device="cuda:0",
        )
        tuner = create_tuner_with_config(
            n_envs, ParallelizationConfig.create([subproc_config])
        )
        wrap_sb3_OnPolicyAlgorithm(model, tuner=tuner, backend="shm")

    exp_manager.learn(model)
    print(f"Base Suproc (env={args.env}). Collect time: {model.collect_time}")

    if args.running_mode == "auturi":
        model._auturi_executor.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--running-mode",
        type=str,
        default="auturi",
        choices=["dummy", "subproc", "auturi", "search"],
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
