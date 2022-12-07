import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

import auturi.executor.shm.util as util
import auturi.executor.typing as types
from auturi.executor.environment import AuturiEnv
from auturi.executor.shm.actor import SHMActorProc
from auturi.executor.shm.constant import ActorCommand
from auturi.executor.shm.mp_mixin import SHMVectorMixin
from auturi.executor.vector_actor import AuturiVectorActor
from auturi.tuner import AuturiMetric, AuturiTuner, ParallelizationConfig

MAX_ACTOR = 128
MAX_ACTOR_DATA = 7


class SHMVectorActor(AuturiVectorActor, SHMVectorMixin):
    def __init__(
        self,
        env_fns: List[Callable[[], AuturiEnv]],
        policy_cls: Any,
        policy_kwargs: Dict[str, Any],
        tuner: AuturiTuner,
    ):

        self.rollout_size = tuner.num_collect
        (
            self.base_buffers,
            self.base_buffer_attr,
            self.rollout_buffers,
            self.rollout_buffer_attr,
        ) = util.create_shm_from_env(env_fns[0], len(env_fns), self.rollout_size)

        self.env_fns = env_fns
        self.policy_cls = policy_cls
        self.policy_kwargs = policy_kwargs

        AuturiVectorActor.__init__(self, env_fns, policy_cls, policy_kwargs, tuner)
        SHMVectorMixin.__init__(self, MAX_ACTOR, MAX_ACTOR_DATA)
        self._print_buffers()

    # for debugging message
    @property
    def proc_name(self) -> str:
        return "VectorActor"

    def reconfigure(
        self, config: ParallelizationConfig, model: types.PolicyModel
    ) -> None:
        self._logger.info(f"\n\n============================reconfigure {config}\n")
        util.copy_config_to_buffer(config, self._command_buffer[:, 1:])
        super().reconfigure(config, model)
        self.sync()

    def _create_worker(self, worker_id: int) -> SHMActorProc:
        kwargs = {
            "env_fns": self.env_fns,
            "policy_cls": self.policy_cls,
            "policy_kwargs": self.policy_kwargs,
            "base_buffer_attr": self.base_buffer_attr,
            "rollout_buffer_attr": self.rollout_buffer_attr,
        }
        return self.init_proc(worker_id, SHMActorProc, kwargs)

    def _reconfigure_worker(
        self,
        worker_id: int,
        worker: SHMActorProc,
        config: ParallelizationConfig,  # do not need
        model: types.PolicyModel,  # do not need
    ) -> None:
        self.request(
            ActorCommand.RECONFIGURE,
            worker_id=worker_id,
        )

        self._logger.info(f"RECONFIGURE({worker_id})")

    def _terminate_worker(self, worker_id: int, worker: SHMActorProc) -> None:
        SHMVectorMixin.terminate_single_worker(self, worker_id, worker)
        self._logger.info(f"Join worker={worker_id} pid={worker.pid}")

    def terminate(self) -> None:
        SHMVectorMixin.terminate_all_workers(self)

        # Responsible to unlink created shm buffer
        for _, tuple_ in self.base_buffers.items():
            tuple_[0].unlink()

        # Responsible to unlink created rollout shm buffer
        for _, tuple_ in self.rollout_buffers.items():
            tuple_[0].unlink()

    def _run(self) -> Tuple[Dict[str, Any], AuturiMetric]:
        self._logger.info(f"\n\n============================RUN\n")
        self.request(ActorCommand.RUN)
        self._logger.debug("Set command RUN")

        start_time = time.perf_counter()
        self.sync()
        end_time = time.perf_counter()

        # Aggregate
        agg_rollouts = self._aggregate_rollouts(self.rollout_size)
        return agg_rollouts, AuturiMetric(
            self.rollout_size, end_time - start_time
        )  # TODO: how to measure time of each worker?

    def _aggregate_rollouts(self, num_collect: int) -> Dict[str, np.ndarray]:
        ret_dict = dict()
        for key, tuple_ in self.rollout_buffers.items():
            # ret_dict[key] = tuple_[1][:num_collect, :]
            ret_dict[key] = tuple_[1]

        return ret_dict

    def _print_buffers(self) -> None:
        self._logger.debug("==================================")
        for key, val in self.base_buffers.items():
            self._logger.debug(f"{key}: shape={val[1].shape}")

        for key, val in self.rollout_buffers.items():
            self._logger.debug(f"Rollout_{key}: shape={val[1].shape}")

        self._logger.debug("==================================\n\n")
