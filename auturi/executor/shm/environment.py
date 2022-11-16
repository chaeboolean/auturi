import itertools
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from auturi.executor.environment import AuturiVectorEnv
from auturi.executor.shm.constant import EnvCommand
from auturi.executor.shm.env_proc import EnvStateEnum, SHMEnvProc
from auturi.executor.shm.mp_mixin import SHMVectorLoopMixin
from auturi.executor.shm.util import WaitingQueue, set_shm_from_attr, wait
from auturi.logger import get_logger
from auturi.tuner.config import ParallelizationConfig

logger = get_logger()


class SHMParallelEnv(AuturiVectorEnv, SHMVectorLoopMixin):
    """SHMParallelVectorEnv

    Uses Python Shared memory implementation as backend

    """

    def __init__(
        self,
        actor_id: int,
        env_fns: List[Callable],
        base_buffer_attr: Dict[str, Any],
        rollout_buffer_attr: Dict[str, Any],
    ):
        self.base_buffer_attr = base_buffer_attr
        self.rollout_buffer_attr = rollout_buffer_attr

        self.__obs, self.obs_buffer = set_shm_from_attr(base_buffer_attr["obs"])
        self.__action, self.action_buffer = set_shm_from_attr(
            base_buffer_attr["action"]
        )
        self.__env, self.env_buffer = set_shm_from_attr(self.base_buffer_attr["env"])

        self.queue = WaitingQueue(len(env_fns))
        self.env_counter = np.zeros(len(env_fns), dtype=np.int32)

        # should be intialized when reconfigure
        self._env_offset, self._rollout_offset = -1, -1

        AuturiVectorEnv.__init__(self, actor_id, env_fns)
        SHMVectorLoopMixin.__init__(self)

    @property
    def identifier(self):
        return f"VectorEnv(aid={self.actor_id}): "

    def reconfigure(self, config: ParallelizationConfig):
        # set offset
        self._env_offset = config.compute_index_for_actor("num_envs", self.actor_id)
        self._rollout_offset = config.compute_index_for_actor(
            "num_collect", self.actor_id
        )

        super().reconfigure(config)

    def _create_worker(self, worker_id: int):
        kwargs = {
            "actor_id": self.actor_id,
            "env_fns": self.env_fns,
            "base_buffer_attr": self.base_buffer_attr,
            "rollout_buffer_attr": self.rollout_buffer_attr,
        }
        return self.init_proc(worker_id, SHMEnvProc, kwargs)

    def _reconfigure_worker(
        self, worker_id: int, worker: SHMEnvProc, config: ParallelizationConfig
    ):
        pass

    def _terminate_worker(self, worker_id: int, worker: SHMEnvProc) -> None:
        super().teardown_handler(worker_id)
        worker.join()
        logger.info(self.identifier + f"Join worker={worker_id} pid={worker.pid}")

    def terminate(self):
        # self.request(EnvCommand.TERM)
        for wid, p in self.workers():
            self._terminate_worker(wid, p)

    # Internally call reset.
    def start_loop(self):
        self.env_counter.fill(0)
        self.queue.pop("all")
        assert np.all(self._get_env_state() == EnvStateEnum.STOPPED)
        SHMVectorLoopMixin.start_loop(self)

    def stop_loop(self):
        # Env states can be STEP_DONE or QUEUED
        SHMVectorLoopMixin.stop_loop(self)

    def reset(self):
        self.request(EnvCommand.RESET)

    def seed(self, seed):
        self.request(EnvCommand.SEED, data=[seed])

    def set_working_env(self, worker_id, worker, start_idx, num_env_serial):
        self.request(
            EnvCommand.SET_ENV, worker_id=worker_id, data=[start_idx, num_env_serial]
        )

    def poll(self) -> List[int]:
        while True:
            new_req = np.where(self._get_env_state() == EnvStateEnum.STEP_DONE)[0]

            self.queue.insert(new_req)
            self._set_env_state(EnvStateEnum.QUEUED, mask=new_req)

            if self.queue.cnt >= self.batch_size:
                ret = self.queue.pop(num=self.batch_size)

                self.env_counter[ret] += 1
                return ret

    def send_actions(self, action_ref) -> None:
        """SHM Implementation do not need send_actions."""
        pass

    def aggregate_rollouts(self):
        acc_ctr = list(itertools.accumulate(self.env_counter))

        prev_ctr = self._rollout_offset
        for wid, _ in self.workers():
            cur_ctr = (
                acc_ctr[(wid + 1) * self.num_env_serial - 1] + self._rollout_offset
            )
            self.request(EnvCommand.AGGREGATE, worker_id=wid, data=[prev_ctr, cur_ctr])
            prev_ctr = cur_ctr

        self.sync()
        return None

    def step(self, action: np.ndarray, action_artifacts: List[np.ndarray]):
        """For debugging Purpose. Synchronous step wrapper."""
        self.batch_size = self.num_envs

        assert len(action) == self.num_envs

        cond_ = lambda: np.all(self._get_env_state() != EnvStateEnum.STOPPED)
        wait(cond_, self.identifier + "wait for step")

        np.copyto(self.action_buffer[: self.num_envs, :], action)
        self._set_env_state(EnvStateEnum.POLICY_DONE)

        logger.debug(self.identifier + f"single_state={self._get_env_state()}")

        _ = self.poll()  # no need to output
        logger.debug(self.identifier + f"step poll finish")

        return np.copy(self.obs_buffer)

    def _set_env_state(self, state, mask: Optional[np.ndarray] = None):
        ptr = self._get_env_state()
        if mask is not None:
            ptr[mask] = state
        else:
            ptr.fill(state)

    def _get_env_state(self):
        return self.env_buffer[self._env_offset : self._env_offset + self.num_envs]
