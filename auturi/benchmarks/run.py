import argparse
import time

import numpy as np

from auturi.benchmarks.task.atari import AtariEnv, AtariPolicy
from auturi.executor import RayExecutor, create_actor_args
from auturi.executor.config import ActorConfig, TunerConfig


def main():
    num_envs = 3
    env_fns = [lambda: AtariEnv() for _ in range(num_envs)]

    env_create_fn, policy_create_fn = create_actor_args(
        env_fns, AtariPolicy, dict(), backend="ray"
    )

    executor = RayExecutor(env_create_fn, policy_create_fn, None)

    # mock "executor.run()", since we do not have tuner yet.

    actor_config = ActorConfig(
        num_envs=num_envs,
        num_policy=1,
        num_parallel=1,
        batch_size=num_envs,
        policy_device="cuda:0",
    )
    tuner_config = TunerConfig(1, {0: actor_config})

    executor.reconfigure(tuner_config, model=None)
    rollouts, metric = executor._run(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auturi Motivation Experiments, without Tuner implemented."
    )
    main()
