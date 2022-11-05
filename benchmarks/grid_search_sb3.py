import argparse
import functools
import time

import numpy as np
from auturi.adapter.sb3.wrapper import wrap_sb3_OnPolicyAlgorithm
from auturi.executor.config import ActorConfig, TunerConfig
from rl_zoo3.exp_manager import ExperimentManager

FILENAME = "/home/ooffordable/ray_fork/auturi/benchmarks/grid_search_sb3.txt"


def assert_config(tuner_config, total_num_envs, gpu_limit):
    ctr = 0
    for actor_id, actor_config in tuner_config.actor_config_map.items():
        ctr += actor_config.num_policy

    return ctr <= gpu_limit


def print_config(config):
    single = config.get(0)
    print_str = f"*** My config: num_actor={config.num_actors}, "
    print_str += f"num_envs={single.num_envs}, "
    print_str += f"num_policy={single.num_policy}, num_parallel={single.num_parallel}, batch_size={single.batch_size}, device={single.policy_device}"
    return print_str


def generate_two_exp(limit):
    num = 1
    while num <= limit:
        yield num
        num *= 2


def generate_next_cuda():
    class _CUDA_Generator:
        def __init__(self):
            self.device_order = 2

        def next(self, tuner_config):
            self.device_order = (1 + self.device_order) % 4
            dev = f"cuda:{self.device_order}"
            for _, actor_config in tuner_config.actor_config_map.items():
                actor_config.policy_device = dev
            return tuner_config

    return _CUDA_Generator()


def generate_config(total_num_envs, gpu_limit):
    for num_actor in generate_two_exp(total_num_envs):
        env_per_actor = total_num_envs // num_actor

        for num_parallel in generate_two_exp(env_per_actor):
            for num_policy in range(1, gpu_limit + 1):
                for batch_size in range(1, env_per_actor + 1):
                    try:
                        actor_config = ActorConfig(
                            num_envs=env_per_actor,
                            num_policy=num_policy,
                            num_parallel=num_parallel,
                            batch_size=batch_size,
                            policy_device="cuda:0",  # "cuda",
                        )
                        next_config = TunerConfig(
                            num_actors=num_actor,
                            actor_config_map={
                                aid: actor_config for aid in range(num_actor)
                            },
                        )
                        if assert_config(next_config, total_num_envs, gpu_limit):
                            yield next_config
                            # yield cuda_gen.next(next_config)

                    except AssertionError:
                        continue


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
    for next_config in generate_config(exp_manager.n_envs, 8):
        wrap_sb3_OnPolicyAlgorithm(model, tuner=None)
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
