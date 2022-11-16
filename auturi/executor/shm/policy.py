from typing import Any, Dict, List

import numpy as np
import torch.nn as nn

from auturi.executor.policy import AuturiPolicy, AuturiVectorPolicy
from auturi.executor.shm.constant import EnvStateEnum, PolicyCommand, PolicyStateEnum
from auturi.executor.shm.mp_mixin import SHMVectorLoopMixin
from auturi.executor.shm.policy_proc import SHMPolicyProc
from auturi.executor.shm.util import _create_buffer_from_sample, set_shm_from_attr, wait
from auturi.logger import get_logger
from auturi.tuner.config import ParallelizationConfig

logger = get_logger()
MAX_POLICY = 16


class SHMVectorPolicy(AuturiVectorPolicy, SHMVectorLoopMixin):
    # shm_config, command_buffer
    def __init__(
        self,
        actor_id,
        policy_cls,
        policy_kwargs: Dict[str, Any],
        base_buffer_attr: Dict[str, Any],
    ):
        self.base_buffer_attr = base_buffer_attr

        # create policy buffer
        self.__policy, self.policy_buffer, policy_attr = _create_buffer_from_sample(
            sample_=1, max_num=MAX_POLICY
        )
        self.base_buffer_attr["policy"] = policy_attr

        self.__env, self.env_buffer = set_shm_from_attr(self.base_buffer_attr["env"])
        logger.debug(
            f"Env buffer shape = {self.env_buffer.shape}, polic={self.policy_buffer.shape}"
        )
        self._env_offset = -1  # should be intialized when reconfigure
        self._env_mask = None

        AuturiVectorPolicy.__init__(self, actor_id, policy_cls, policy_kwargs)
        SHMVectorLoopMixin.__init__(self)

    @property
    def identifier(self):
        return f"VectorPolicy(aid={self.actor_id}): "

    def reconfigure(self, config: ParallelizationConfig, model: nn.Module) -> None:
        self._env_offset = config.compute_index_for_actor("num_envs", self.actor_id)
        self._env_mask = slice(
            self._env_offset, self._env_offset + config[self.actor_id].num_envs
        )

        super().reconfigure(config, model)

    def _create_worker(self, worker_id: int):
        kwargs = {
            "actor_id": self.actor_id,
            "policy_cls": self.policy_cls,
            "policy_kwargs": self.policy_kwargs,
            "base_buffer_attr": self.base_buffer_attr,
        }
        return self.init_proc(worker_id, SHMPolicyProc, kwargs)

    def _reconfigure_worker(
        self, worker_id: int, worker: SHMPolicyProc, config: ParallelizationConfig
    ):
        self.request(
            PolicyCommand.SET_POLICY_ENV,
            worker_id=worker_id,
            data=[self._env_offset, config[self.actor_id].num_envs],
        )

    def _terminate_worker(self, worker_id: int, worker: SHMPolicyProc) -> None:
        super().teardown_handler(worker_id)
        worker.join()

    def terminate(self):
        # self.request(EnvCommand.TERM)
        for wid, p in self.workers():
            self._terminate_worker(wid, p)

        self.__policy.unlink()

    def _load_policy_model(
        self, worker_id: int, policy: AuturiPolicy, model: nn.Module, device: str
    ) -> None:
        self.request(PolicyCommand.LOAD_MODEL, worker_id=worker_id, data=[device])

    def compute_actions(self, env_ids: List[int], n_steps: int) -> object:
        while True:
            # assert np.all(self._get_env_state()[env_ids] == EnvStateEnum.QUEUED)

            if not np.all(self._get_env_state()[env_ids] == EnvStateEnum.QUEUED):
                logger.debug(
                    self.identifier
                    + f"Assertion False: env state= {self.env_buffer}, got id={env_ids}"
                )
                assert False
            ready_policies = np.where(self._get_state() == PolicyStateEnum.READY)[0]

            if len(ready_policies) > 0:
                policy_id = int(ready_policies[0])  # pick any

                logger.info(self.identifier + f"assigned {env_ids} to pol{policy_id}")
                self._get_env_state()[env_ids] = policy_id + EnvStateEnum.POLICY_OFFSET
                self._get_state()[policy_id] = PolicyStateEnum.ASSIGNED
                return None

    def _get_env_state(self):
        return self.env_buffer[self._env_mask]

    def _get_state(self):
        return self.policy_buffer[: self.num_workers]

    def start_loop(self):
        self.policy_buffer.fill(0)
        SHMVectorLoopMixin.start_loop(self)

    def stop_loop(self):
        wait(
            lambda: np.all(self._get_state() == PolicyStateEnum.READY),
            self.identifier + "Wait to stop loop..",
        )
        SHMVectorLoopMixin.stop_loop(self)
