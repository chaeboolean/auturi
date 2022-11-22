import argparse
import functools
import time

import numpy as np
import ray
from rl_zoo3.exp_manager import ExperimentManager

from auturi.adapter.sb3 import wrap_sb3_OnPolicyAlgorithm
from auturi.tuner import ActorConfig, ParallelizationConfig, create_tuner_with_config

OUTPUT_FILENAME = "/home/ooffordable/auturi/log/compare_same_config.txt"


def get_config(args, architecture, device="cpu"):
    n_envs = args.num_envs

    if architecture == "subproc":
        actor_config = ActorConfig(
            num_envs=n_envs,
            num_policy=1,
            num_parallel=n_envs,
            batch_size=n_envs,
            policy_device=device,
            num_collect=args.num_collect,
        )

        return ParallelizationConfig.create([actor_config])

    elif architecture == "dummy":
        actor_config = ActorConfig(
            num_envs=n_envs,
            num_policy=1,
            num_parallel=1,
            batch_size=n_envs,
            policy_device=device,
            num_collect=args.num_collect,
        )

        return ParallelizationConfig.create([actor_config])

    elif architecture == "rllib":
        num_actors = args.num_actors
        envs_per_actor = args.num_envs // num_actors
        actor_config = ActorConfig(
            num_envs=envs_per_actor,
            num_policy=1,
            num_parallel=1,
            batch_size=envs_per_actor,
            policy_device=device,
            num_collect=args.num_collect // num_actors,
        )

        return ParallelizationConfig.create([actor_config] * num_actors)

    else:
        raise NotImplementedError


def print_result(args, collect_times):
    collect_times = sorted(collect_times)
    name = args.architecture
    if name == "rllib":
        name += f"(actors={args.num_actors})"

    if args.run_auturi:
        name = f"Auturi[{name}]"

    with open(OUTPUT_FILENAME, "a") as f:
        f.write(
            f"\n\n{name}: env={args.env}, n_envs={args.num_envs}, num_collect={args.num_collect}\n"
        )
        f.write(f"Result: {np.median(np.array(collect_times))}, {collect_times}\n")

    print(
        f"\n\n{name}: env={args.env}, n_envs={args.num_envs}, num_collect={args.num_collect}"
    )
    print(collect_times)


@ray.remote
class RayActor:
    def __init__(self, args):
        self._init = False
        num_actors = args.num_actors

        n_envs_per_actor = args.num_envs // num_actors
        n_steps_per_actor = args.num_collect // num_actors // n_envs_per_actor

        self.exp_manager, self.model = create_sb3_algorithm(
            args, n_envs_per_actor, n_steps_per_actor, 1, "dummy"
        )

        self._init = True

    def initialized(self):
        while True:
            if self._init:
                return

    def run(self):
        self.exp_manager.learn(self.model)
        return self.model.rollout_buffer


def run_rllib(args):
    actors = dict()
    pending = dict()

    for actor_id in range(args.num_actors):
        actors[actor_id] = RayActor.remote(args)

    for actor_id, actor in actors.items():
        ref = actor.initialized.remote()
        pending[ref] = actor_id

    ray.wait(list(pending.keys()), num_returns=args.num_actors)
    pending.clear()

    times = []
    for _ in range(args.num_iteration):
        pending.clear()
        start_time = time.perf_counter()

        for actor_id, actor in actors.items():
            ref = actor.run.remote()
            pending[ref] = actor_id

        ray.wait(list(pending.keys()), num_returns=args.num_actors)
        end_time = time.perf_counter()
        times += [end_time - start_time]

    return times


def create_sb3_algorithm(args, num_envs, n_steps, num_iteration, vec_cls="dummy"):
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

    model, _ = exp_manager.setup_experiment(num_envs, n_steps)
    exp_manager.n_envs = num_envs
    model.env_fns = [_wrap for _ in range(exp_manager.n_envs)]

    model._auturi_iteration = num_iteration
    model._auturi_train_skip = True

    return exp_manager, model


def run(args):
    print(args)
    # create ExperimentManager with minimum argument.

    if args.run_auturi:
        num_envs = args.num_envs
        n_steps = args.num_collect // args.num_envs
        exp_manager, model = create_sb3_algorithm(
            args, num_envs, n_steps, args.num_iteration, "dummy"
        )
        tuner = create_tuner_with_config(
            args.num_envs, get_config(args, args.architecture)
        )
        wrap_sb3_OnPolicyAlgorithm(model, tuner=tuner, backend="shm")

        exp_manager.learn(model)
        print_result(args, model.collect_time)

        model._auturi_executor.terminate()

    elif args.architecture in ["dummy", "subproc"]:
        num_envs = args.num_envs
        n_steps = args.num_collect // args.num_envs
        exp_manager, model = create_sb3_algorithm(
            args, num_envs, n_steps, args.num_iteration, args.architecture
        )
        exp_manager.learn(model)
        print_result(args, model.collect_time)

    elif args.architecture == "rllib":
        collect_times = run_rllib(args)
        print_result(args, collect_times)

    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture",
        type=str,
        default="subproc",
        choices=["dummy", "subproc", "rllib"],
    )
    parser.add_argument("--env", type=str, default="CartPole-v1", help="environment ID")

    parser.add_argument(
        "--run-auturi", action="store_true", help="Run with AuturiExecutor."
    )
    parser.add_argument(
        "--num-envs", type=int, default=4, help="number of environments."
    )

    parser.add_argument(
        "--num-collect", type=int, default=4, help="number of trajectories to collect."
    )

    parser.add_argument(
        "--num-actors",
        type=int,
        default=2,
        help="number of actors in RLlib architecture.",
    )

    parser.add_argument(
        "--num-iteration", type=int, default=5, help="number of trials for each config."
    )

    args = parser.parse_args()
    run(args)
