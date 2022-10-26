import copy
import time
from collections import OrderedDict
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple, Type,
                    Union)

import numpy as np

from auturi.executor.policy import AuturiPolicy, AuturiVectorPolicy
from auturi.vector.shm_util import *
from auturi.vector.shm_util import ENV_STATE


class SHMPolicyWrapper(SHMProcBase):
    def __init__(self, index, policy_fn, queue, shm_configs, event):
        self.policy_id = index
        self.policy_fn = policy_fn
        self.queue = queue
        self.shm_configs = shm_configs
        self.event = event

        super().__init__()

    def _set_cmd_done(self):
        self.pol_buffer[self.policy_id, 0] = POLICY_COMMAND.CMD_DONE

    def _set_device(self, device):
        self.policy.load_model(device=device)
        self.pol_buffer[self.policy_id, 0] = POLICY_COMMAND.START_LOOP

        # put itself to queue after ready
        self.queue.put(self.policy_id)

    def run_loop(self):
        cmd, data = self.pol_buffer[self.policy_id]
        while True:
            cmd, data = self.pol_buffer[self.policy_id]
            if cmd == POLICY_COMMAND.TERMINATE or cmd == POLICY_COMMAND.STOP_LOOP:
                self.event.clear()
                self._set_cmd_done()
                break

            elif cmd == POLICY_COMMAND.SET_CPU:
                self._set_device(device="cpu")

            elif cmd == POLICY_COMMAND.SET_GPU:
                self._set_device(device=f"cuda:{int(data)}")

            elif cmd == POLICY_COMMAND.START_LOOP:
                to_process = np.where(
                    self.command_buffer[:, 1]
                    == self.policy_id + ENV_STATE.POLICY_OFFSET
                )[0]
                if len(to_process) == 0:
                    continue

                obs = self.obs_buffer[to_process, :]
                actions, artifacts = self.policy.compute_actions(obs)
                self.action_buffer[to_process] = actions

                stacked_artifacts = np.stack(artifacts, -1)
                self.polartifacts_buffer[to_process] = stacked_artifacts
                self.command_buffer[to_process, 1] = ENV_STATE.POLICY_DONE

                self.queue.put(self.policy_id)

            return cmd

        def teardown(self):
            pass

    def run(self):
        self.set_shm_buffer(self.shm_configs)
        self.policy = self.policy_fn()
        assert isinstance(self.policy, AuturiPolicy)
        print(f"******* {self.policy_id}: POLICY CREATE!!")

        while True:
            last_cmd = self.run_loop()
            if last_cmd == POLICY_COMMAND.TERMINATE:
                self.teardown()
                break
            elif last_cmd == POLICY_COMMAND.STOP_LOOP:
                print("POLICY Event SLEEP ==========================")
                self.event.wait()
                print("POLICY Event Set ==========================")


class SHMVectorPolicies(AuturiVectorPolicy):
    # shm_config, command_buffer
    def __init__(self, shm_config: Dict, num_policies: int, policy_fn: Callable):
        self.pending_policies = dict()
        self.remote_policies = dict()
        self.pending_policies = mp.Queue()

        self.shm_config = copy.deepcopy(shm_config)

        # create policy command buffer
        self._pol, self.pol_buffer = create_shm_from_sample(
            np.array([1, 1], dtype=np.int32),
            "pol",
            self.shm_config,
            num_policies,
        )

        # set env-specific command buffer
        self._command, self.command_buffer = get_np_shm("command", self.shm_config)

        self.events = {pid: mp.Event() for pid in range(num_policies)}

        super().__init__(num_policies, policy_fn)

        for index in range(num_policies):
            p = SHMPolicyWrapper(
                index,
                policy_fn,
                self.pending_policies,
                self.shm_config,
                self.events[index],
            )
            p.start()
            self.remote_policies[index] = p

    def assign_free_server(self, env_ids: List[int], n_steps: int):
        server_id = self.pending_policies.get()

        # set state
        self.command_buffer[env_ids, 1] = server_id + ENV_STATE.POLICY_OFFSET
        return None, server_id

    def _set_command(self, command):
        for eid, event in self.events.items():
            if not event.is_set():
                event.set()
                print("event set for", eid)

        self.pol_buffer[:, 0].fill(command)

    def _wait_command_done(self):
        while not np.all(self.pol_buffer[:, 0] == POLICY_COMMAND.CMD_DONE):
            pass

    def start_loop(self, device="cpu"):
        cmd = POLICY_COMMAND.SET_CPU if device == "cpu" else POLICY_COMMAND.SET_GPU
        self._set_command(cmd)

    def finish_loop(self):
        self._set_command(POLICY_COMMAND.STOP_LOOP)
        self._wait_command_done()
        # clear pending policies
        while self.pending_policies.qsize() > 0:
            self.pending_policies.get()

    def terminate(self):
        self._set_command(POLICY_COMMAND.STOP_LOOP)
        for p in self.remote_policies:
            p.join()

        self._pol.unlink()
