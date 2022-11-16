import time
from typing import Any, Callable, Dict, List, Tuple

import torch.nn as nn

from auturi.executor.environment import AuturiEnv
from auturi.executor.shm.actor import SHMActorProc
from auturi.executor.shm.constant import ActorCommand
from auturi.executor.shm.mp_mixin import SHMVectorMixin
from auturi.executor.shm.util import create_shm_from_env
from auturi.executor.vector_actor import AuturiVectorActor
from auturi.logger import get_logger
from auturi.tuner import AuturiTuner, AuturiMetric, ParallelizationConfig

logger = get_logger()


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
        ) = create_shm_from_env(env_fns[0], len(env_fns), self.rollout_size)

        self.env_fns = env_fns
        self.policy_cls = policy_cls
        self.policy_kwargs = policy_kwargs

        AuturiVectorActor.__init__(self, env_fns, policy_cls, policy_kwargs, tuner)
        SHMVectorMixin.__init__(self)

    # for debugging message
    @property
    def identifier(self):
        return "VectorActor: "

    def reconfigure(self, config: ParallelizationConfig, model: nn.Module):
        logger.info(f"\n\n============================reconfigure {config}\n")

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
        config: ParallelizationConfig,
        model: nn.Module,  # do not need
    ):
        self.request(
            ActorCommand.RECONFIGURE,
            worker_id=worker_id,
            data=[config],
        )

        logger.info(self.identifier + f"RECONFIGURE({worker_id})")

    def _terminate_worker(self, worker_id: int, worker: SHMActorProc) -> None:
        super().teardown_handler(worker_id)
        worker.join()
        logger.info(self.identifier + f"Join worker={worker_id} pid={worker.pid}")

    def terminate(self):
        # self.request(EnvCommand.TERM)
        for wid, p in self.workers():
            self._terminate_worker(wid, p)

        # Responsible to unlink created shm buffer
        for _, tuple_ in self.base_buffers.items():
            tuple_[0].unlink()

        # Responsible to unlink created rollout shm buffer
        for _, tuple_ in self.rollout_buffers.items():
            tuple_[0].unlink()

    def _run(self) -> Tuple[Dict[str, Any], AuturiMetric]:
        logger.info(f"\n\n============================RUN\n")
        self.request(ActorCommand.RUN)
        logger.debug(self.identifier + "Set command RUN")

        start_time = time.perf_counter()
        self.sync()
        end_time = time.perf_counter()

        # Aggregate
        agg_rollouts = self._aggregate_rollouts(self.rollout_size)
        return agg_rollouts, AuturiMetric(
            self.rollout_size, end_time - start_time
        )  # TODO: how to measure time of each worker?

    def _aggregate_rollouts(self, num_collect: int):
        ret_dict = dict()
        for key, tuple_ in self.rollout_buffers.items():
            ret_dict[key] = tuple_[1][:num_collect, :]

        return ret_dict
