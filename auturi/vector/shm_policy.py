import time
from collections import OrderedDict
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple, Type,
                    Union)

import numpy as np
import copy 

from auturi.typing.policy import AuturiVectorPolicy
from auturi.vector.common_util import _flatten_obs
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
    
    def read_from_buffer(self, env_ids: List[int]):
        return (
            self.obs_buffer[env_ids, :],
            None, 
            None, 
            None,
        )
    
    def _set_cmd_done(self):
        self.pol_buffer[self.policy_id, 0] = POLICY_COMMAND.CMD_DONE


    def run_loop(self):
        cmd, data = self.pol_buffer[self.policy_id]
        while True:
            cmd, data = self.pol_buffer[self.policy_id]
            if cmd == POLICY_COMMAND.TERMINATE or cmd == POLICY_COMMAND.STOP_LOOP:
                self.event.clear()
                self._set_cmd_done()
                break
            
            elif cmd == POLICY_COMMAND.SET_CPU:
                self.policy.set_device("cpu")
                self._set_cmd_done()

            elif cmd == POLICY_COMMAND.SET_GPU:
                self.policy.set_device(f"cuda:{int(data)}")
                self._set_cmd_done()
                        
            elif cmd == POLICY_COMMAND.RUN_LOOP:
                to_process = np.where(
                    self.command_buffer[:, 1] == self.policy_id + ENV_STATE.POLICY_OFFSET
                )[0]
                if len(to_process) == 0:
                    continue

                print(f" \n I am server! assigned = {to_process}")
                obs, reward, dones, infos = self.read_from_buffer(to_process)

                actions = self.policy.compute_actions(obs)
                self.action_buffer[to_process] = actions

                self.command_buffer[to_process, 1] = ENV_STATE.POLICY_DONE

                self.queue.put(self.policy_id)
                #self.policy.insert_buffer(obs, obs, reward, dones, infos)

            return cmd
    
        def teardown(self):
            pass
    
    def run(self):
        self.set_shm_buffer(self.shm_configs)
        self.policy = self.policy_fn()
        print(f"******* {self.policy_id}: POLICY CREATE!!")      

        # put itself to queue after ready
        self.queue.put(self.policy_id)

        while True:
            last_cmd = self.run_loop()
            if last_cmd == POLICY_COMMAND.TERMINATE:
                self.teardown()
                break
            elif last_cmd == POLICY_COMMAND.STOP_LOOP:
                self.event.wait()
                print("POLICY Event Set ==========================")


    def set_device(self, device):
        pass

class SHMVectorPolicies(AuturiVectorPolicy):
    # shm_config, command_buffer
    def __init__(self, shm_config: Dict, num_policies: int, policy_fn: Callable):
        self.shm_config = copy.deepcopy(shm_config)

        self._pol, self.pol_buffer = create_shm_from_space(
            np.array([1, 1], dtype=np.int32), "pol",  self.shm_config, num_policies,
        )

        # set command buffer
        self._command, self.command_buffer = get_np_shm("command", self.shm_config)
        self.events = {pid: mp.Event() for pid in range(num_policies)}

        super().__init__(num_policies, policy_fn)    

    def _create_policy(self, index, policy_fn):
        p = SHMPolicyWrapper(index, policy_fn, self.pending_policies, self.shm_config, self.events[index])
        p.start()
        return p

    def _setup(self):
        self.pending_policies = mp.Queue()
        self.reset()

    def reset(self):
        pass

    def assign_free_server(self, obs_refs: List[int], n_steps: int):
        server_id = self.pending_policies.get()
        
        # set state        
        self.command_buffer[obs_refs, 1] = server_id + ENV_STATE.POLICY_OFFSET
        return None, server_id

    def close(self):
        for p in self.remote_policies:
            p.join()


    def _set_command(self, command):
        for eid, event in self.events.items():
            if not event.is_set(): 
                event.set()
                print("event set for", eid )

        self.pol_buffer[:, 0].fill(command)

    def _wait_command_done(self):
        while not np.all(self.pol_buffer[:, 0] == POLICY_COMMAND.CMD_DONE):
            pass


    def start_loop(self):
        while self.pending_policies.qsize() != len(self.remote_policies):
            pass
        self._set_command(POLICY_COMMAND.RUN_LOOP)

    def finish_loop(self):
        self._set_command(POLICY_COMMAND.STOP_LOOP)
        self._wait_command_done()
