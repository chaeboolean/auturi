import argparse
import functools

import gym
import pybullet_envs
import ray
from rl_zoo3.exp_manager import ExperimentManager

from auturi.adapter.sb3.wrapper import wrap_sb3_OnPolicyAlgorithm
from auturi.executor.config import ActorConfig, TunerConfig
from auturi.tuner import AuturiTuner


def assert_config(num_envs, tuner_config):
    ctr = 0
    for actor_id, actor_config in tuner_config.actor_config_map.items():
        ctr += actor_config.num_envs

    assert ctr == num_envs


def create_mock_tuner(config_list):
    class _MockTuner(AuturiTuner):
        """Just generate pre-defined TunerConfigs."""

        def __init__(self):
            self._ctr = 0
            self.configs = config_list

        def next(self):
            next_config = self.configs[self._ctr]
            self._ctr += 1
            return next_config

    return _MockTuner()


def generate_config(num_envs):
    actor_config = ActorConfig(
        num_envs=4,
        num_policy=1,
        num_parallel=4,
        batch_size=2,
        policy_device="cuda:0",  # "cuda",
    )

    num_actors = 2
    next_config = TunerConfig(
        num_actors=num_actors,
        actor_config_map={aid: actor_config for aid in range(num_actors)},
    )
    return next_config


def main(args):
    # create ExperimentManager with minimum argument.
    exp_manager = ExperimentManager(
        args=args,
        algo="ppo",
        env_id=args.env,
        log_folder="",
        vec_env_type="dummy",
        device="cuda",
    )

    # _wrap = create_envs(exp_manager, 3)
    _wrap = functools.partial(exp_manager.create_envs, 1)
    model, saved_hyperparams = exp_manager.setup_experiment()
    model.env_fns = [_wrap for _ in range(exp_manager.n_envs)]
    next_config = generate_config(exp_manager.n_envs)
    tuner = create_mock_tuner([next_config])

    model._auturi_iteration = args.num_iteration
    model._auturi_train_skip = args.skip_update

    wrap_sb3_OnPolicyAlgorithm(model, tuner)
    model._auturi_executor.reconfigure(tuner.next(), model=None)

    exp_manager.learn(model)

    round_fn = lambda x: round(x, 3)
    print(f"train_time=> {round_fn(model.training_time)}")
    print(f"collect_time=> {round_fn(model.collect_time)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--running-mode",
        type=str,
        default="search",
        choices=["dummy", "subproc", "search"],
    )
    parser.add_argument("--env", type=str, default="CartPole-v1", help="environment ID")
    parser.add_argument("--skip-update", action="store_true", help="skip backprop stage.")
    parser.add_argument("--num-iteration", type=int, default=50)

    args = parser.parse_args()
    main(args)
