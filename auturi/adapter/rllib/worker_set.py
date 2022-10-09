from types import FunctionType
from typing import Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from ray.rllib.evaluation import WorkerSet
from ray.rllib.policy.policy import Policy, PolicySpec
from ray.rllib.utils.typing import (AlgorithmConfigDict, EnvCreator, EnvType,
                                    PolicyID, SampleBatchType, TensorType)

import auturi


class WorkerSetWrapper(WorkerSet):
    def __init__(
        self,
        *,
        env_creator: Optional[EnvCreator] = None,
        validate_env: Optional[Callable[[EnvType], None]] = None,
        policy_class: Optional[Type[Policy]] = None,
        trainer_config: Optional[AlgorithmConfigDict] = None,
        num_workers: int = 0,
        local_worker: bool = True,
        logdir: Optional[str] = None,
        _setup: bool = True,
    ):

        use_auturi = trainer_config.get("use_auturi", False)

        if use_auturi:

            # Step 1. create Tuner
            assert "auturi_config" in trainer_config
            auturi_config = trainer_config["auturi_config"]

            number_of_total_env = num_workers * trainer_config["num_envs_per_worker"]
            auturi_config.setdefault("num_sim", number_of_total_env)

            print(f"Auturi Tuner started with {auturi_config['num_sim']} Simulators. ")
            tuner = auturi.Tuner(auturi_config)

            # Step 2. Change Workerset config in order to create only local Rolloutworker
            super().__init__(
                env_creator,
                validate_env,
                policy_class,
                trainer_config,
                num_workers=0,
                local_worker=True,
                logdir=logdir,
                _setup=_setup,
            )

            self._local_worker.set_tuner(tuner)

        else:
            super().__init__(
                env_creator,
                validate_env,
                policy_class,
                trainer_config,
                num_workers=num_workers,
                local_worker=local_worker,
                logdir=logdir,
                _setup=_setup,
            )
