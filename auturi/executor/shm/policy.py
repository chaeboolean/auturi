from typing import Any, Dict

import auturi.executor.shm.util as util
import auturi.executor.typing as types
import numpy as np
from auturi.executor.policy import AuturiVectorPolicy
from auturi.executor.shm.constant import EnvStateEnum, PolicyCommand, PolicyStateEnum
from auturi.executor.shm.mp_mixin import SHMVectorLoopMixin
from auturi.executor.shm.policy_proc import SHMPolicyProc
from auturi.tuner.config import ParallelizationConfig


class SHMVectorPolicy(AuturiVectorPolicy, SHMVectorLoopMixin):
    # shm_config, command_buffer
    def __init__(
        self,
        actor_id,
        policy_cls,
        policy_kwargs: Dict[str, Any],
        base_buffers: Dict[str, Any],
        base_buffer_attr: Dict[str, Any],
        max_num_policy: int,
    ):
        self.base_buffers = base_buffers
        self.base_buffer_attr = base_buffer_attr

        # create policy buffer
        (
            self.__policy,
            self.policy_buffer,
            policy_attr,
        ) = util._create_buffer_from_sample(
            sample_=np.array([[1]]), max_num=max_num_policy
        )
        self.base_buffer_attr["policy"] = policy_attr
        self.policy_buffer.fill(PolicyStateEnum.STOPPED)

        self.env_buffer = self.base_buffers["env"][1]

        # Env visibility is limited for each actor
        self._env_mask_for_actor = None

        AuturiVectorPolicy.__init__(self, actor_id, policy_cls, policy_kwargs)
        SHMVectorLoopMixin.__init__(self, max_num_policy, max_data=3)

    @property
    def proc_name(self) -> str:
        return f"VectorPolicy(aid={self.actor_id})"

    # override
    def reconfigure(
        self, config: ParallelizationConfig, model: types.PolicyModel
    ) -> None:
        # set the range of visible environment
        env_offset = config.compute_index_for_actor("num_envs", self.actor_id)
        num_visible_envs = config[self.actor_id].num_envs
        self._env_mask_for_actor = slice(env_offset, env_offset + num_visible_envs)

        new_num_workers = config[self.actor_id].num_policy
        device = config[self.actor_id].policy_device
        self.reconfigure_workers(
            new_num_workers=new_num_workers,
            env_offset=env_offset,
            num_visible_envs=num_visible_envs,
            device=device,
        )

    def _create_worker(self, worker_id: int) -> SHMPolicyProc:
        kwargs = {
            "actor_id": self.actor_id,
            "policy_cls": self.policy_cls,
            "policy_kwargs": self.policy_kwargs,
            "base_buffer_attr": self.base_buffer_attr,
        }
        return self.init_proc(worker_id, SHMPolicyProc, kwargs)

    def _reconfigure_worker(
        self,
        worker_id: int,
        worker: SHMPolicyProc,
        env_offset: int,
        num_visible_envs: int,
        device: str,
    ) -> None:
        device_num = util.device_to_int(device)
        self.request(
            PolicyCommand.SET_POLICY,
            worker_id=worker_id,
            data=[env_offset, num_visible_envs, device_num],
        )

    def terminate(self) -> None:
        SHMVectorLoopMixin.terminate(self)
        self.__policy.unlink()

    def compute_actions(
        self, env_ids: types.ObservationRefs, n_steps: int
    ) -> types.ActionRefs:
        while True:
            assert np.all(self._get_env_state()[env_ids] == EnvStateEnum.QUEUED)

            # poll until there is at least one idle policy process
            ready_policies = np.where(self._get_state() == PolicyStateEnum.READY)[0]

            if len(ready_policies) > 0:
                policy_id = int(ready_policies[0])  # pick any available policy

                self._logger.debug(f"assigned {env_ids} to pol{policy_id}")

                # change env_buffer state to id of the idle policy
                self._get_env_state()[env_ids] = policy_id + EnvStateEnum.POLICY_OFFSET

                # change policy_buffer state to ASSIGNED
                self._get_state()[policy_id] = PolicyStateEnum.ASSIGNED
                return None

    def _get_env_state(self) -> np.ndarray:
        """Return env_buffer state."""
        return self.env_buffer[self._env_mask_for_actor]

    def _get_state(self) -> np.ndarray:
        """Return policy_buffer state."""
        return self.policy_buffer[: self.num_workers]

    def start_loop(self) -> None:
        self.policy_buffer.fill(0)
        SHMVectorLoopMixin.start_loop(self)

    def stop_loop(self) -> None:
        # wait until env-assigned policy finish compute_actions()
        util.wait(
            lambda: np.all(self._get_state() == PolicyStateEnum.READY),
            lambda: self._logger.warning("Wait to stop loop.."),
        )
        self.policy_buffer.fill(PolicyStateEnum.STOPPED)
        SHMVectorLoopMixin.stop_loop(self)
