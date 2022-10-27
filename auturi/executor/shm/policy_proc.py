import multiprocessing as mp

import numpy as np

import auturi.executor.shm.env_proc as env_proc
import auturi.executor.shm.util as shm_util
from auturi.executor.policy import AuturiPolicy
from auturi.executor.shm.mixin import SHMProcMixin


class POLICY_COMMAND:
    STOP_LOOP = 9
    START_LOOP = 423
    LOAD_MODEL = 2
    TERMINATE = 4
    CMD_DONE = 5


class POLICY_STATE:
    STOPPED = 3
    READY = 4
    ASSIGNED = 10


class SHMPolicyProc(mp.Process, SHMProcMixin):
    def __init__(self, index, policy_cls, policy_kwargs, shm_buffer_attr_dict, event):
        self.worker_id = index
        self.policy_fn = lambda: policy_cls(**policy_kwargs)
        self.shm_buffer_attr_dict = shm_buffer_attr_dict
        self.event = event

        super().__init__()

    def initialize(self):
        self.command_buffer = self.policy_buffer
        self.cmd_enum = POLICY_COMMAND

    def _run_loop(self, state):
        # print(f"My name is POLICY")
        if state == POLICY_STATE.STOPPED:
            self._set_state(POLICY_STATE.READY)

        elif state == POLICY_STATE.ASSIGNED:
            assigned_env_ids = np.where(
                self.env_buffer[:, 4]
                == self.worker_id + env_proc.SINGLE_ENV_STATE.POLICY_OFFSET
            )[0]

            assert len(assigned_env_ids) > 0
            obs = self.obs_buffer[assigned_env_ids, :]
            actions, artifacts = self.policy.compute_actions(obs, n_steps=1)
            print(f"Pol{self.worker_id} got {obs[:, 0, 0]}")

            # artifacts = np.stack(artifacts, -1)

            self.action_buffer[assigned_env_ids] = actions
            self.artifacts_buffer[assigned_env_ids] = artifacts

            # convert state
            self.env_buffer[assigned_env_ids, 4] = env_proc.SINGLE_ENV_STATE.POLICY_DONE
            self._set_state(POLICY_STATE.READY)

    def _run(self) -> POLICY_COMMAND:
        while True:
            cmd, state, data = self.policy_buffer[self.worker_id]
            if cmd == POLICY_COMMAND.START_LOOP:
                self._run_loop(state)

            elif cmd == POLICY_COMMAND.TERMINATE or cmd == POLICY_COMMAND.STOP_LOOP:
                self._set_state(POLICY_STATE.STOPPED)
                self._set_cmd_done()
                return cmd

            elif cmd == POLICY_COMMAND.LOAD_MODEL:
                self._assert_state(POLICY_STATE.STOPPED)
                device = "cpu" if data < 0 else f"cuda:{int(data)}"

                # TODO: should give model as argument.
                self.policy.load_model(None, device)
                self._set_cmd_done()
                return cmd

    def run(self):
        self.policy = self.policy_fn()
        assert isinstance(self.policy, AuturiPolicy)

        shm_util.set_shm_buffer_from_attr(self, self.shm_buffer_attr_dict)
        self.initialize()

        self._set_state(POLICY_STATE.STOPPED)
        self._set_cmd_done()
        self.main()
