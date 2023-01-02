from typing import List

import numpy as np

import auturi.executor.shm.util as util
from auturi.executor.policy import AuturiPolicy
from auturi.executor.shm.constant import EnvStateEnum, PolicyCommand, PolicyStateEnum
from auturi.executor.shm.mp_mixin import SHMProcLoopMixin


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
        self._env_mask_for_actor = None

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

        SHMProcLoopMixin.initialize(self)

    @property
    def proc_name(self) -> str:
        return f"PolicyProc(aid={self.actor_id}, wid={self.worker_id})"

    def set_command_handlers(self) -> None:
        self.cmd_handler[PolicyCommand.SET_POLICY] = self.set_policy_handler

    def set_policy_handler(self, cmd: int, data_list: List[int]) -> None:
        self._env_mask_for_actor = slice(data_list[0], data_list[0] + data_list[1])
        self.policy.load_model(None, util.int_to_device(data_list[2]))
        self._logger.debug(f"set visible mask ({self._env_mask_for_actor})")
        self.reply(cmd)

    def _set_state(self, state) -> None:
        """Set policy_buffer state."""
        self.policy_buffer[self.worker_id] = state

    def _get_state(self) -> int:
        """Return policy_buffer state."""
        return self.policy_buffer[self.worker_id]

    def get_visible_buffer(self, buffer: np.ndarray) -> np.ndarray:
        """Return visibil region of given buffer."""
        assert self._env_mask_for_actor is not None
        return buffer[self._env_mask_for_actor]

    def _step_loop_once(self, is_first: bool) -> None:
        # set state to READY
        if is_first:
            self._logger.debug("Enter the loop")
            self._set_state(PolicyStateEnum.READY)

        # if env.step() is done, call compute_actions
        elif self._get_state() == PolicyStateEnum.ASSIGNED:

            # get ids of assigned environments
            assigned_envs = np.where(
                self.get_visible_buffer(self.env_buffer)
                == self.worker_id + EnvStateEnum.POLICY_OFFSET
            )[0]

            assert len(assigned_envs) > 0

            # read observations from assigned env ids
            obs = self.get_visible_buffer(self.obs_buffer)[assigned_envs]
            actions, artifacts = self.policy.compute_actions(obs, n_steps=1)

            # Write actions and the artifacts
            self.get_visible_buffer(self.action_buffer)[assigned_envs] = actions

            self.get_visible_buffer(self.artifact_buffer)[assigned_envs] = artifacts[0]

            # state transition
            self.get_visible_buffer(self.env_buffer)[
                assigned_envs
            ] = EnvStateEnum.POLICY_DONE

            # set policy buffer ASSIGNED -> READY
            self._set_state(PolicyStateEnum.READY)
