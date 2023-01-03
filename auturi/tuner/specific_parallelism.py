from functools import partial
from itertools import chain
from typing import List
from typing import Callable, Optional

from auturi.tuner.base_tuner import AuturiTuner
from auturi.tuner.config import ActorConfig, ParallelizationConfig
from datetime import datetime


ENV_PARALLEL = "E"
LOOP_PARALLEL = "L"
ENV_POLICY_OVERLAP="E+P"

class SpecificParallelismComparator(AuturiTuner):
    """AuturiTuner for Comparing Specific Paralellism Strategies.

    We support those modes for now.
    'E': Environment parallelism
    'L': Loop parallelism
    'P': Policy parallelism
    'E+P': Environment + Policy parallelism
    'H': Heterogeneous loop parallelism

    """

    def __init__(
        self,
        names: List[str],
        min_num_env: int,
        max_num_env: int,
        max_policy_num: int,
        num_collect: int,
        num_iterate: int = 10,
        validator: Optional[Callable[[ActorConfig], bool]] = None,
        out_file: Optional[str] = None,
        is_cuda_only: bool = False, 
    ):

        gen_dict = {
            ENV_PARALLEL: partial(_e_generator, min_num_env, num_collect),
            LOOP_PARALLEL: partial(_l_generator, min_num_env, num_collect, max_policy_num),
            ENV_POLICY_OVERLAP: partial(_e_p_generator, min_num_env, num_collect, max_policy_num),
        }
        
        args = [gen_dict[name]() for name in names]
        self.generator = chain(*args)
        self.tuning_results = dict()  # TODO: Write to file the result at the end.
        self.validator = (lambda _: True) if validator is None else validator
        self.max_policy_num = max_policy_num
        self.out_file = out_file
        super().__init__(min_num_env, max_num_env, num_collect, num_iterate)

    def _generate_next(self):
        while True:
            parallelism, next_config = next(self.generator)
            if self.validator(next_config) or \
                parallelism == LOOP_PARALLEL and (next_config.num_actors > self.max_policy_num):
                return next_config

    def _update_tuner(self, config, res):
        mean_metric = res[1]
        self.tuning_results[hash(config)] = (config, mean_metric)

        out_log = f'\n\n{"=" * 20}\n{datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}\n'
        out_log += f"{config}\nMean Result: {mean_metric.elapsed} sec\n"
        out_log += f"{res[0]}\n"
        
        if self.out_file is not None:
            with open(self.out_file, "a") as f:
                f.write(out_log)
                
        print(out_log)

def _e_generator(num_envs, num_collect):
    def _gen_actor_config(device):
        return ActorConfig(
            num_envs=num_envs,
            num_policy=1,
            num_parallel=num_envs,
            batch_size=num_envs,
            policy_device=device,
            num_collect=num_collect,
        )

    yield ENV_PARALLEL, ParallelizationConfig.create([_gen_actor_config("cpu")])
    yield ENV_PARALLEL, ParallelizationConfig.create([_gen_actor_config("cuda:0")])


def _l_generator(num_envs, num_collect, max_num_policy):
    def _gen_actor_config(num_loop, device):
        num_env_per_loop = num_envs // num_loop
        num_collect_per_loop = num_collect // num_loop
        return ActorConfig(
            num_envs=num_env_per_loop,
            num_policy=1,
            num_parallel=1,
            batch_size=num_env_per_loop,
            policy_device=device,
            num_collect=num_collect_per_loop,
        )

    num_loop = 1
    while num_loop <= num_envs:
        yield LOOP_PARALLEL, ParallelizationConfig.create(
            [_gen_actor_config(num_loop, "cpu")] * num_loop
        )
        if num_loop <= max_num_policy:
            yield LOOP_PARALLEL, ParallelizationConfig.create(
                [_gen_actor_config(num_loop, "cuda:0")] * num_loop
            )
        num_loop = num_loop * 2


def _e_p_generator(num_envs, num_collect, max_num_policy):
    for num_policy in range(1, max_num_policy + 1):
        for num_parallel in _iter_to_max(max_num=num_envs, mode_="two"):
            for batch_size in _iter_to_max(max_num=num_envs // num_policy, mode_="two"):
                for device in ["cpu", "cuda"]:
                    try:
                        yield ENV_POLICY_OVERLAP, ParallelizationConfig.create(
                            [
                                ActorConfig(
                                    num_envs=num_envs,
                                    num_policy=num_policy,
                                    num_parallel=num_parallel,
                                    batch_size=batch_size,
                                    policy_device=device,
                                    num_collect=num_collect,
                                )
                            ]
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
