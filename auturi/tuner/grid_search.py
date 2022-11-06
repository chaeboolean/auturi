from typing import Callable, Optional

from auturi.tuner.base_tuner import AuturiTuner
from auturi.tuner.config import ActorConfig, TunerConfig


class GridSearchTuner(AuturiTuner):
    """AuturiTuner for Grid Search.

    For now, it only yeilds the TunerConfig with same ActorConfig.

    Additional Args:
        max_policy_num (int): Upper bound for number of policies and actors. TO BE FIXED.
        validator (Callable[[ActorConfig], bool]): Additional user-defined function.

    """

    def __init__(
        self,
        min_num_env: int,
        max_num_env: int,
        max_policy_num: int,
        num_iterate: int = 10,
        validator: Optional[Callable[[ActorConfig], bool]] = None,
    ):
        self.generator = naive_grid_generator(min_num_env, max_policy_num, validator)
        self.tuning_results = dict()  # TODO: Write to file the result at the end.

        super().__init__(min_num_env, max_num_env, num_iterate)

    def _generate_next(self):
        return next(self.generator)

    def _update_tuner(self, config, mean_metric):
        self.tuning_results[hash(config)] = (config, mean_metric)

        print("=" * 20)
        print(config)
        print(f"Mean Result: {mean_metric.elapsed / self.num_iterate} sec")
        print("=" * 20)


def naive_grid_generator(num_envs, num_max_policy, validator):
    for num_actors in _iter_to_max(max_num=min(num_envs, num_max_policy), mode_="two"):
        num_env_per_actor = num_envs // num_actors
        num_policy_per_actor = num_max_policy // num_actors

        for actor_config in _possible_actors(num_env_per_actor, num_policy_per_actor):
            try:
                tuner_config = TunerConfig(
                    {_id: actor_config for _id in range(num_actors)}
                )
                tuner_config.validate(
                    num_envs, num_envs, num_max_policy, validator=validator
                )
                yield tuner_config

            except AssertionError:
                continue


def _possible_actors(num_envs, num_max_policy):
    for num_policy in _iter_to_max(max_num=num_max_policy):
        for num_parallel in _iter_to_max(max_num=num_envs, mode_="two"):
            for batch_size in _iter_to_max(max_num=num_envs, mode_="two"):
                for device in ["cpu", "cuda"]:
                    try:
                        yield ActorConfig(
                            num_envs=num_envs,
                            num_policy=num_policy,
                            num_parallel=num_parallel,
                            batch_size=batch_size,
                            policy_device=device,
                        )

                    except AssertionError:
                        continue


def _iter_to_max(max_num, min_num=1, mode_="incr"):
    num = min_num
    while num <= max_num:
        yield num
        if mode_ == "two":
            num *= 2
        else:
            num += 1
