from typing import List

import numpy as np

from auturi.executor.environment import AuturiSerialEnv
from auturi.executor.shm.constant import EnvCommand, EnvStateEnum
from auturi.executor.shm.mp_mixin import SHMProcLoopMixin
from auturi.executor.shm.util import set_shm_from_attr, wait


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
        self.env = AuturiSerialEnv(self.actor_id, self.worker_id, self.env_fns)
        self.__env, self.env_buffer = set_shm_from_attr(self.base_buffer_attr["env"])

        self.__obs, self.obs_buffer = set_shm_from_attr(self.base_buffer_attr["obs"])
        self.__action, self.action_buffer = set_shm_from_attr(
            self.base_buffer_attr["action"]
        )
        self.__artifact, self.artifact_buffer = set_shm_from_attr(
            self.base_buffer_attr["artifact"]
        )

        self.rollout_buffers = dict()
        for key, attr_dict in self.rollout_buffer_attr.items():
            raw_buf, np_buffer = set_shm_from_attr(attr_dict)
            self.rollout_buffers[key] = (raw_buf, np_buffer)

        SHMProcLoopMixin.initialize(self)

    @property
    def proc_name(self):
        return f"EnvProc(aid={self.actor_id}, wid={self.worker_id})"

    def set_command_handlers(self):
        self.cmd_handler[EnvCommand.RESET] = self.reset_handler
        self.cmd_handler[EnvCommand.SEED] = self.seed_handler
        self.cmd_handler[EnvCommand.SET_ENV] = self.set_visible_env_handler
        self.cmd_handler[EnvCommand.AGGREGATE] = self.aggregate_handler

    def reset_handler(self, cmd: int, data_list: List[int]):
        obs = self.env.reset()
        self.insert_obs_buffer(obs)
        self.reply(cmd)

    def seed_handler(self, cmd: int, data_list: List[int]):
        self.env.seed(data_list[0])
        self.reply(cmd)

    def set_visible_env_handler(self, cmd: int, data_list: List[int]):
        self.env.set_working_env(data_list[0], data_list[1])
        self.reply(cmd)

    def aggregate_handler(self, cmd: int, data_list: List[int]):
        local_rollouts = self.env.aggregate_rollouts()

        for key_, trajectories in local_rollouts.items():
            roll_buffer = self.rollout_buffers[key_][1]

            # TODO: stack first
            np.copyto(roll_buffer[data_list[0] : data_list[1]], trajectories)

        self.reply(cmd)

    def _step_loop_once(self, is_first: bool) -> None:
        if is_first:
            assert np.all(self._get_env_state() == EnvStateEnum.STOPPED)

            self._logger.debug("Entered the loop.")
            obs = self.env.reset()
            self.insert_obs_buffer(obs)
            self._set_env_state(EnvStateEnum.STEP_DONE)

        elif np.all(self._get_env_state() == EnvStateEnum.POLICY_DONE):
            action, artifacts_list = self.get_actions()
            obs = self.env.step(action, artifacts_list)
            self._logger.debug(f"env.step({action.flat[0]}) => {obs.flat[0]}")

            self.insert_obs_buffer(obs)
            self._set_env_state(EnvStateEnum.STEP_DONE)

    def insert_obs_buffer(self, obs):
        self.obs_buffer[self.env.start_idx : self.env.end_idx, :] = obs

    def get_actions(self):
        action = self.action_buffer[self.env.start_idx : self.env.end_idx, :]
        artifacts = self.artifact_buffer[self.env.start_idx : self.env.end_idx, :]
        return action, [artifacts]

    def _set_env_state(self, state):
        self.env_buffer[self.env.start_idx : self.env.end_idx] = state

    def _get_env_state(self):
        return self.env_buffer[self.env.start_idx : self.env.end_idx]

    def _check_loop_done(self) -> bool:
        return np.all(
            np.ma.mask_or(
                (self._get_env_state() == EnvStateEnum.STEP_DONE),
                (self._get_env_state() == EnvStateEnum.QUEUED),
            )
        )

    def _stop_loop_handler(self):
        self._set_env_state(EnvStateEnum.STOPPED)
        self._logger.debug("Escaped the loop.")
        super()._stop_loop_handler()
