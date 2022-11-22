from typing import List

import numpy as np

import auturi.executor.shm.util as util
from auturi.executor.policy import AuturiPolicy
from auturi.executor.shm.constant import EnvStateEnum, PolicyCommand, PolicyStateEnum
from auturi.executor.shm.mp_mixin import SHMProcLoopMixin
from auturi.logger import get_logger

logger = get_logger()


class SHMPolicyProc(SHMProcLoopMixin):
    def __init__(
        self,
        actor_id,
        worker_id,
        policy_cls,
        policy_kwargs,
        cmd_attr_dict,
        base_buffer_attr,
    ):
        self.actor_id = actor_id
        self.policy_cls = policy_cls
        self.policy_kwargs = policy_kwargs

        self.base_buffer_attr = base_buffer_attr

        super().__init__(worker_id, cmd_attr_dict=cmd_attr_dict)

    def initialize(self) -> None:
        self.policy_kwargs["idx"] = self.worker_id
        self.policy = self.policy_cls(**self.policy_kwargs)

        assert isinstance(self.policy, AuturiPolicy)

        self.__env, self.env_buffer = util.set_shm_from_attr(
            self.base_buffer_attr["env"]
        )
        self.__policy, self.policy_buffer = util.set_shm_from_attr(
            self.base_buffer_attr["policy"]
        )
        self.__obs, self.obs_buffer = util.set_shm_from_attr(
            self.base_buffer_attr["obs"]
        )
        self.__action, self.action_buffer = util.set_shm_from_attr(
            self.base_buffer_attr["action"]
        )
        self.__artifact, self.artifact_buffer = util.set_shm_from_attr(
            self.base_buffer_attr["artifact"]
        )

        self._env_mask = None
        SHMProcLoopMixin.initialize(self)

    @property
    def identifier(self):
        return f"PolicyProc(aid={self.actor_id}, wid={self.worker_id}): "

    def set_handler_for_command(self):
        self.cmd_handler[PolicyCommand.LOAD_MODEL] = self.load_model_handler
        self.cmd_handler[PolicyCommand.SET_POLICY_ENV] = self.set_env_handler

    def load_model_handler(self, cmd: int, data_list: List[int]):
        # cannot receive model parameters via SHM for now.
        self.policy.load_model(None, util.int_to_device(data_list[0]))
        self.reply(cmd)

    def set_env_handler(self, cmd: int, data_list: List[int]):
        self._env_mask = slice(data_list[0], data_list[0] + data_list[1])
        logger.debug(self.identifier + f"set visible mask ({self._env_mask})")
        self.reply(cmd)

    def _set_state(self, state):
        self.policy_buffer[self.worker_id] = state

    def _get_state(self):
        return self.policy_buffer[self.worker_id]

    def get_visible_buffer(self, buffer: np.ndarray):
        assert self._env_mask is not None
        return buffer[self._env_mask]

    def _step_loop_once(self, is_first: bool) -> None:
        if is_first:
            logger.debug(self.identifier + f"Enter the loop")
            self._set_state(PolicyStateEnum.READY)

        elif self._get_state() == PolicyStateEnum.ASSIGNED:
            assigned_env_ids = np.where(
                self.get_visible_buffer(self.env_buffer)
                == self.worker_id + EnvStateEnum.POLICY_OFFSET
            )[0]

            assert len(assigned_env_ids) > 0
            logger.info(
                self.identifier
                + f"policy.step: given={assigned_env_ids}, env_buffer={self.env_buffer}, visible={(np.where(self.env_buffer == 30))[0]}, set fn={(np.where(self.env_buffer[self._env_mask] == 30))[0]},, mask={self._env_mask}"
            )

            obs = self.get_visible_buffer(self.obs_buffer)[assigned_env_ids]
            actions, artifacts = self.policy.compute_actions(obs, n_steps=1)

            # Write actions and the artifacts
            self.get_visible_buffer(self.action_buffer)[assigned_env_ids] = actions
            self.get_visible_buffer(self.artifact_buffer)[assigned_env_ids] = artifacts[
                0
            ]

            # state transition
            self.get_visible_buffer(self.env_buffer)[
                assigned_env_ids
            ] = EnvStateEnum.POLICY_DONE

            self._set_state(PolicyStateEnum.READY)
