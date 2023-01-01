from typing import List

import auturi.executor.typing as types
import numpy as np
from auturi.executor.environment import AuturiSerialEnv
from auturi.executor.shm.constant import EnvCommand, EnvStateEnum
from auturi.executor.shm.mp_mixin import SHMProcLoopMixin
from auturi.executor.shm.util import set_rollout_buffer_from_attr, set_shm_from_attr


class SHMEnvProc(SHMProcLoopMixin):
    def __init__(
        self,
        actor_id,
        worker_id,
        cmd_attr_dict,
        env_fns,
        base_buffer_attr,
        rollout_buffer_attr,
    ):
        self.actor_id = actor_id
        self.env_fns = env_fns
        self.base_buffer_attr = base_buffer_attr
        self.rollout_buffer_attr = rollout_buffer_attr
        super().__init__(worker_id, cmd_attr_dict=cmd_attr_dict)

    def initialize(self) -> None:
        self.env = AuturiSerialEnv()

        self.__env, self.env_buffer = set_shm_from_attr(self.base_buffer_attr["env"])

        self.__obs, self.obs_buffer = set_shm_from_attr(self.base_buffer_attr["obs"])
        self.__action, self.action_buffer = set_shm_from_attr(
            self.base_buffer_attr["action"]
        )
        self.__artifact, self.artifact_buffer = set_shm_from_attr(
            self.base_buffer_attr["artifact"]
        )

        self.rollout_buffers = set_rollout_buffer_from_attr(self.rollout_buffer_attr)

        SHMProcLoopMixin.initialize(self)

    @property
    def proc_name(self) -> str:
        return f"EnvProc(aid={self.actor_id}, wid={self.worker_id})"

    def set_command_handlers(self) -> None:
        self.cmd_handler[EnvCommand.RESET] = self.reset_handler
        self.cmd_handler[EnvCommand.SEED] = self.seed_handler
        self.cmd_handler[EnvCommand.SET_ENV] = self.set_visible_env_handler
        self.cmd_handler[EnvCommand.AGGREGATE] = self.aggregate_handler

    def reset_handler(self, cmd: int, data_list: List[int]) -> None:
        obs = self.env.reset()
        self.insert_obs_buffer(obs)
        self.reply(cmd)

    def seed_handler(self, cmd: int, data_list: List[int]) -> None:
        self.env.seed(data_list[0])
        self.reply(cmd)

    def set_visible_env_handler(self, cmd: int, data_list: List[int]) -> None:
        self.env.set_working_env(data_list[0], data_list[1], self.env_fns)
        self.reply(cmd)

    def aggregate_handler(self, cmd: int, data_list: List[int]) -> None:
        local_rollouts = self.env.aggregate_rollouts()

        for key_, trajectories in local_rollouts.items():
            roll_buffer = self.rollout_buffers[key_][1]

            end_idx = min(len(roll_buffer), data_list[1])
            num_to_copy = end_idx - data_list[0]
            np.copyto(roll_buffer[data_list[0] : end_idx], trajectories[:num_to_copy])

        self.reply(cmd)

    def _reset_curr_env_id(self):
        self._curr_env_id = self.env.start_idx - 1

    def _get_curr_env_id(self):
        self._curr_env_id += 1
        if self._curr_env_id >= self.env.end_idx:
            self._curr_env_id = self.env.start_idx

        return self._curr_env_id

    def _step_loop_once(self, is_first: bool) -> None:
        # call env.reset first
        if is_first:
            assert np.all(self._get_env_state() == EnvStateEnum.STOPPED)

            self._logger.debug("Entered the loop.")
            obs = self.env.reset()
            self.insert_obs_buffer(obs)
            self._set_env_state(EnvStateEnum.STEP_DONE)
            self._reset_curr_env_id()
            return

        # wait until POLICY_DONE, and then call env.step again
        curr_id = self._get_curr_env_id()
        if np.all(self._get_env_state(curr_id) == EnvStateEnum.POLICY_DONE):
            action, artifacts_list = self.get_actions(curr_id)
            self.curr_env_idx = self.env.start_idx

            obs = self.env[curr_id].step(action, artifacts_list)
            self._logger.debug(f"env_{curr_id}.step({action.flat[0]}) => {obs.flat[0]}")

            self.insert_obs_buffer(obs, curr_id)
            self._set_env_state(EnvStateEnum.STEP_DONE)

    def insert_obs_buffer(self, obs, curr_id=None) -> None:
        slice_ = (
            slice(curr_id, curr_id + 1)
            if curr_id is not None
            else slice(self.env.start_idx, self.env.end_idx)
        )
        self.obs_buffer[slice_, :] = obs

    def get_actions(self, curr_id=None) -> types.ActionTuple:
        # if curr_id is not None:
        #     action = self.action_buffer[curr_id]
        #     artifacts = self.artifact_buffer[curr_id]
        #     return action, [artifacts]

        slice_ = (
            slice(curr_id, curr_id + 1)
            if curr_id is not None
            else slice(self.env.start_idx, self.env.end_idx)
        )
        action = self.action_buffer[slice_, :]
        artifacts = self.artifact_buffer[slice_, :]
        return action, [artifacts]

    def _set_env_state(self, state, curr_id=None) -> None:
        slice_ = (
            slice(curr_id, curr_id + 1)
            if curr_id is not None
            else slice(self.env.start_idx, self.env.end_idx)
        )
        self.env_buffer[slice_] = state

    def _get_env_state(self, curr_id=None) -> np.ndarray:
        slice_ = (
            slice(curr_id, curr_id + 1)
            if curr_id is not None
            else slice(self.env.start_idx, self.env.end_idx)
        )
        return self.env_buffer[slice_]

    def _check_loop_done(self) -> bool:
        return np.all(
            np.ma.mask_or(
                (self._get_env_state() == EnvStateEnum.STEP_DONE),
                (self._get_env_state() == EnvStateEnum.QUEUED),
            )
        )

    def _stop_loop_handler(self) -> bool:
        self._set_env_state(EnvStateEnum.STOPPED)
        super()._stop_loop_handler()
