import multiprocessing as mp
import time
from typing import Any, Callable, Dict, List

import numpy as np
import torch.nn as nn

import auturi.executor.shm.env_proc as env_proc
from auturi.executor.policy import AuturiPolicy, AuturiVectorPolicy
from auturi.executor.shm.mixin import SHMVectorMixin
from auturi.executor.shm.policy_proc import POLICY_COMMAND, POLICY_STATE, SHMPolicyProc


class SHMVectorPolicy(AuturiVectorPolicy, SHMVectorMixin):
    # shm_config, command_buffer
    def __init__(
        self,
        policy_cls,
        policy_kwargs: Dict[str, Any],
        shm_buffer_dict: Dict[str, Any],
        shm_buffer_attr_dict: Dict[str, Any],
    ):
        self.shm_buffer_dict = shm_buffer_dict
        self.shm_buffer_attr_dict = shm_buffer_attr_dict

        self.env_buffer = self.shm_buffer_dict["env"][1]
        self.policy_buffer = self.shm_buffer_dict["policy"][1]
        self._set_command_buffer()

        self.events = dict()
        self.policy_buffer.fill(
            POLICY_COMMAND.STOP_LOOP
        )  # anything different from CMD_DONE

        super().__init__(policy_cls, policy_kwargs)

    def _set_command_buffer(self):
        """Should set attributes "command_buffer", "cmd_enum"."""
        self.command_buffer = self.policy_buffer
        self.cmd_enum = POLICY_COMMAND

    def _create_worker(self, idx: int):
        self.events[idx] = mp.Event()
        self.events[idx].clear()

        p = SHMPolicyProc(
            idx,
            self.policy_cls,
            self.policy_kwargs,
            self.shm_buffer_attr_dict,
            self.events[idx],
        )
        p.start()
        return p

    def _load_policy_model(
        self, worker_id: int, policy: AuturiPolicy, model: nn.Module, device: str
    ) -> None:
        device_num = -1 if device == "cpu" else int(device.split(":")[-1])
        self._wait_command_done(worker_id)
        self._set_command(
            POLICY_COMMAND.LOAD_MODEL, worker_id=worker_id, data1=device_num
        )

    def compute_actions(self, env_ids: List[int], n_steps: int) -> object:
        while True:
            ready_policies = np.where(self.policy_buffer[:, 1] == POLICY_STATE.READY)[0]
            if len(ready_policies) > 0:
                policy_id = ready_policies[0]
                print(f"\n\n Assign: {env_ids} => ", policy_id)
                self.policy_buffer[policy_id, 1] = POLICY_STATE.ASSIGNED

                self.env_buffer[env_ids, 1] = (
                    policy_id + env_proc.ENV_STATE.POLICY_OFFSET
                )
                return None
            time.sleep(1)
            print(ready_policies, self.policy_buffer[0, :])

    def start_loop(self):
        self._wait_command_done()
        self._set_command(POLICY_COMMAND.START_LOOP)

    def stop_loop(self):
        while not np.all(
            np.ma.mask_or(
                (self._get_state() == POLICY_STATE.READY),
                (self._get_state() == POLICY_STATE.STOPPED),
            )
        ):
            pass

        self._set_command(POLICY_COMMAND.STOP_LOOP, set_event=False)
        self._wait_command_done()

    def terminate(self):
        self._set_command(POLICY_COMMAND.TERMINATE)
        for idx, p in self.remote_workers.items():
            p.join()
