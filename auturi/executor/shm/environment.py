import itertools
from typing import Any, Callable, Dict, List, Optional

import numpy as np

import auturi.executor.typing as types
from auturi.executor.environment import AuturiVectorEnv
from auturi.executor.shm.constant import EnvCommand
from auturi.executor.shm.env_proc import EnvStateEnum, SHMEnvProc
from auturi.executor.shm.mp_mixin import SHMVectorLoopMixin
from auturi.executor.shm.util import WaitingQueue, set_shm_from_attr, wait

MAX_ENV_NUM = 64


class SHMParallelEnv(AuturiVectorEnv, SHMVectorLoopMixin):
    """SHMParallelVectorEnv

    Uses Python Shared memory implementation as backend.

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

        AuturiVectorEnv.__init__(self, actor_id, env_fns)
        SHMVectorLoopMixin.__init__(self, MAX_ENV_NUM)

    @property
    def proc_name(self) -> str:
        return f"VectorEnv(aid={self.actor_id})"

    def _create_worker(self, worker_id: int) -> SHMEnvProc:
        kwargs = {
            "actor_id": self.actor_id,
            "env_fns": self.env_fns,
            "base_buffer_attr": self.base_buffer_attr,
            "rollout_buffer_attr": self.rollout_buffer_attr,
        }
        return self.init_proc(worker_id, SHMEnvProc, kwargs)

    def _reconfigure_worker(self, worker_id: int, worker: SHMEnvProc) -> None:
        start_idx = self._env_offset + self.num_env_serial * worker_id
        self.request(
            EnvCommand.SET_ENV,
            worker_id=worker_id,
            data=[start_idx, self.num_env_serial],
        )

    # Internally call reset.
    def start_loop(self) -> None:
        self.env_counter.fill(0)
        self.queue.pop("all")
        assert np.all(self._get_env_state() == EnvStateEnum.STOPPED)
        SHMVectorLoopMixin.start_loop(self)

    def stop_loop(self) -> None:
        # Env states can be STEP_DONE or QUEUED
        SHMVectorLoopMixin.stop_loop(self)

    def reset(self) -> None:
        self.request(EnvCommand.RESET)

    def seed(self, seed) -> None:
        self.request(EnvCommand.SEED, data=[seed])

    def poll(self) -> types.ObservationRefs:
        while True:
            new_done_ids = np.where(self._get_env_state() == EnvStateEnum.STEP_DONE)[0]

            self.queue.insert(new_done_ids)
            self._set_env_state(EnvStateEnum.QUEUED, mask=new_done_ids)

            if self.queue.cnt >= self.batch_size:
                ret = self.queue.pop(num=self.batch_size)

                self.env_counter[ret] += 1
                return ret

    def send_actions(self, action_ref: types.ActionRefs) -> None:
        """SHM Implementation do not need send_actions."""
        pass

    def aggregate_rollouts(self) -> None:
        acc_ctr = list(itertools.accumulate(self.env_counter))

        prev_ctr = self._rollout_offset
        for wid, _ in self.workers():
            cur_ctr = (
                acc_ctr[(wid + 1) * self.num_env_serial - 1] + self._rollout_offset
            )
            self.request(EnvCommand.AGGREGATE, worker_id=wid, data=[prev_ctr, cur_ctr])
            prev_ctr = cur_ctr

        self.sync()
        self._logger.info("Sync after Rollouts. ")

        return None

    def step(
        self, action: np.ndarray, action_artifacts: types.ActionArtifacts
    ) -> np.ndarray:
        """For debugging Purpose. Synchronous step wrapper."""
        self.batch_size = self.num_envs

        assert len(action) == self.num_envs

        cond_ = lambda: np.all(self._get_env_state() != EnvStateEnum.STOPPED)
        wait(cond_, lambda: self._logger.warning("wait for step"))

        np.copyto(self.action_buffer[: self.num_envs, :], action)
        self._set_env_state(EnvStateEnum.POLICY_DONE)

        self._logger.debug(f"single_state={self._get_env_state()}")

        _ = self.poll()  # no need to output
        self._logger.debug(f"step poll finish")

        return np.copy(self.obs_buffer)

    def _set_env_state(self, state, mask: Optional[np.ndarray] = None) -> None:
        ptr = self._get_env_state()
        if mask is not None:
            ptr[mask] = state
        else:
            ptr.fill(state)

    def _get_env_state(self) -> np.ndarray:
        return self.env_buffer[self._env_offset : self._env_offset + self.num_envs]

    def terminate(self):
        SHMVectorLoopMixin.terminate(self)
