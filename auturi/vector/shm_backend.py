import functools
import time
from collections import OrderedDict
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple, Type,
                    Union)

import gym
import numpy as np
import ray

from auturi.typing.policy import AuturiVectorPolicy
from auturi.vector.common_util import _flatten_obs
from auturi.vector.shm_util import _get_np_shm, _create_shm_from_space

class POLICY_COMMAND:
    STOP_LOOP = 0
    RUN_LOOP = 1
    SET_CPU = 2
    SET_GPU = 3
    TERMINATE = 4
    CMD_DONE = 5


class ENV_COMMAND:
    STOP_LOOP = 0
    RUN_LOOP = 1
    RESET = 2  # start
    SEED = 3
    TERMINATE = 4
    CMD_DONE = 5


class ENV_STATE:
    """Indicates simulator state.
    Initialized to CLOSED
    """

    STOPPED = 23
    STEP_DONE = 1  # Newly arrived requests
    QUEUED = 2  # Inside Server side waiting queue
    POLICY_DONE = 3  # Processed requests
    POLICY_OFFSET = 40  # offset


import multiprocessing as mp
from multiprocessing import shared_memory as shm

from auturi.typing.simulator import AuturiParallelEnv
from auturi.vector.shm_util import *


class SHMEnvWrapper(SHMProcWrapper):
    def __init__(self, idx, env_fn, shm_configs, event):
        self.env_id = idx
        self.env_fn = env_fn
        self.shm_configs = shm_configs
        self.event = event
        super().__init__()

    def write_to_shm(self, obs, reward, done, info):
        #print(f"Env{self.env_id}[{self.cnt}] {obs}")
        # print(f"Env{self.env_id}: write_to_shm")
        #print(f"Env{self.env_id}: write_to_shm, obs={obs.shape}")
        self.obs_buffer[self.env_id, :] = obs
        # TODO: ...

    def _set_cmd_done(self):
        self.command_buffer[self.env_id, 0] = ENV_COMMAND.CMD_DONE

    def _set_state(self, state):
        self.command_buffer[self.env_id, 1] = state

    def _assert_state(self, state):
        assert self.command_buffer[self.env_id, 1] == state


    def step_wrap(self, action):
        observation, reward, done, info = self.env.step(action)
        if done:
            # save final observation where user can get it, then reset
            info["terminal_observation"] = observation
            observation = self.env.reset()
        return observation, reward, done, info

    def run_loop(self):
        cmd, state, seed_ = self.command_buffer[self.env_id]
        while True:
            cmd, state, seed_ = self.command_buffer[self.env_id]
            if cmd == ENV_COMMAND.TERMINATE or cmd == ENV_COMMAND.STOP_LOOP:
                self._set_state(ENV_STATE.STOPPED)
                self.event.clear()
                self._set_cmd_done()
                break

            elif cmd == ENV_COMMAND.SEED:
                self._assert_state(ENV_STATE.STOPPED)
                print(f"Env{self.env_id}: CMD=SEED, {seed_}")

                self.env.seed(int(seed_))
                self._set_cmd_done()

            elif cmd == ENV_COMMAND.RESET:
                print(f"Env{self.env_id}: CMD=RESET")

                self._assert_state(ENV_STATE.STOPPED)
                obs = self.env.reset()
                # print(f"RESET => obs {obs}", )
                self.write_to_shm(obs, None, None, None)
                self._set_cmd_done()

            elif cmd == ENV_COMMAND.RUN_LOOP and state == ENV_STATE.POLICY_DONE:
                action = self.action_buffer[self.env_id]
                print(f"SHMEnv{self.env_id}: CMD=RUN_STEP action={action}")

                obs, reward, done, info = self.step_wrap(action)
                print(obs, reward, done, info)
                
                # put observation to shared buffer
                self.write_to_shm(obs, reward, done, info)

                # change state to STEP_DONE
                self._set_state(ENV_STATE.STEP_DONE)
                
        return cmd # return last cmd

    def run(self):
        self.env = self.env_fn()

        self.set_shm_buffer(self.shm_configs)
        self._set_cmd_done()
        self._set_state(ENV_STATE.STOPPED)

        # run while loop
        while True:
            last_cmd = self.run_loop()
            if last_cmd == ENV_COMMAND.TERMINATE:
                self.teardown()
                break
            elif last_cmd == ENV_COMMAND.STOP_LOOP:
                self.event.wait()
                print("Enter Loop Event Set ==========================")


    def teardown(self):
        self.env.close()


class SHMParallelEnv(AuturiParallelEnv):
    """SHMParallelVectorEnv

    Uses Python Shared memory implementation as backend

    """

    def _setup(self, dummy_env):

        self.shm_configs = dict()

        dummy_env.reset()
        obs__, reward__, done__, info__ = dummy_env.step(
            dummy_env.action_space.sample()
        )

        self._obs, self.obs_buffer = _create_shm_from_space(
            dummy_env.observation_space.sample(), "obs", self.shm_configs, self.num_envs
        )
        self._action, self.action_buffer = _create_shm_from_space(
            dummy_env.action_space.sample(), "action",  self.shm_configs, self.num_envs
        )
        self._command, self.command_buffer = _create_shm_from_space(
            np.array([1, 1, 1], dtype=np.int64), "command",  self.shm_configs, self.num_envs
        )

        self.queue = WaitingQueue(self.num_envs)
        self.events = {eid: mp.Event() for eid in range(self.num_envs)}

    def _create_env(self, index, env_fn):
        p = SHMEnvWrapper(index, env_fn, self.shm_configs, self.events[index])
        p.start()
        return p

    def _wait_states(self, state):
        while not np.all(self.command_buffer[:, 1] == state):
            pass

    def _wait_command_done(self):
        while not np.all(self.command_buffer[:, 0] == ENV_COMMAND.CMD_DONE):
            pass

    def _set_command(self, command):
        for eid, event in self.events.items():
            if not event.is_set(): 
                event.set()
                print("event set for", eid )

        self.command_buffer[:, 0].fill(command)

    def _set_state(self, state):
        self.command_buffer[:, 1].fill(state)

    def reset(self):
        self._wait_states(ENV_STATE.STOPPED)
        self._set_command(ENV_COMMAND.RESET)
        self._wait_command_done()
        time.sleep(2)
        return np.copy(self.obs_buffer)

    def seed(self, seed):
        print(f"n\n &&&&&&&&&&&&&&& SHM SEED called!!!, seed= {seed}")
        self._wait_states(ENV_STATE.STOPPED)

        # set seed
        self.command_buffer[:, 2] = np.array(
            [seed + eid for eid in range(self.num_envs)]
        )

        self._set_command(ENV_COMMAND.SEED)
        self._wait_command_done()

    # Called after reset 
    def start_loop(self):
        self._set_command(ENV_COMMAND.RUN_LOOP)
        self._set_state(ENV_STATE.STEP_DONE)

    def finish_loop(self):
        self._set_command(ENV_COMMAND.STOP_LOOP)
        self._wait_command_done()


    def poll(self, bs: int = -1) -> List[int]:
        if bs < 0:
            bs = self.num_envs

        while True:
            new_requests_ = np.where(self.command_buffer[:, 1] == ENV_STATE.STEP_DONE)[
                0
            ]
            # if len(new_requests_) > 0: print(new_requests_)
            self.queue.insert(new_requests_)
            self.command_buffer[new_requests_, 1] = ENV_STATE.QUEUED
            if self.queue.cnt >= bs:
                return self.queue.pop(num=bs)

    def send_actions(self, action_dict):
        pass

    def step(self, actions: np.ndarray):
        """For debugging Purpose. Synchronous step wrapper."""
        assert len(actions) == self.num_envs
        np.copyto(self.action_buffer, actions)
        self.command_buffer[:, 1].fill(ENV_STATE.POLICY_DONE)

        poll_out = self.poll(bs=self.num_envs)  # no need to output
        print(poll_out)

        return np.copy(self.obs_buffer), None, None, None


    def close(self):
        self._wait_states(ENV_STATE.STOPPED)
        self._set_command(ENV_COMMAND.TERMINATE)
        
        print("Call TERMINATE!!!")
        for idx, p in self.remote_envs.items():
            p.join()

        self._obs.unlink()
        self._action.unlink()
        self._command.unlink()



class SHMPolicyWrapper(SHMProcWrapper):
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

import copy 
class SHMVectorPolicies(AuturiVectorPolicy):
    # shm_config, command_buffer
    def __init__(self, shm_config: Dict, num_policies: int, policy_fn: Callable):
        self.shm_config = copy.deepcopy(shm_config)

        self._pol, self.pol_buffer = _create_shm_from_space(
            np.array([1, 1], dtype=np.int32), "pol",  self.shm_config, num_policies,
        )

        # set command buffer
        self._command, self.command_buffer = _get_np_shm("command", self.shm_config)
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
        self._set_command(POLICY_COMMAND.RUN_LOOP)

    def finish_loop(self):
        self._set_command(POLICY_COMMAND.STOP_LOOP)
        self._wait_command_done()
