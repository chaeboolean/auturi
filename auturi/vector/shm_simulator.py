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
from auturi.vector.shm_util import *

import multiprocessing as mp
from multiprocessing import shared_memory as shm

from auturi.typing.simulator import AuturiParallelEnv
from auturi.vector.shm_util import *



class SHMEnvProc(SHMProcBase):
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

    def _setup(self, dummy_env, total_collect):

        self.shm_configs = dict()

        dummy_env.reset()
        obs__, reward__, done__, info__ = dummy_env.step(
            dummy_env.action_space.sample()
        )

        self._obs, self.obs_buffer = create_shm_from_sample(
            dummy_env.observation_space.sample(), "obs", self.shm_configs, self.num_envs
        )
        self._action, self.action_buffer = create_shm_from_sample(
            dummy_env.action_space.sample(), "action",  self.shm_configs, self.num_envs
        )
        self._command, self.command_buffer = create_shm_from_sample(
            np.array([1, 1, 1], dtype=np.int64), "command",  self.shm_configs, self.num_envs
        )
        
        self._rollobs, self.rollobs_buffer = create_shm_from_sample(
            dummy_env.observation_space.sample(), "obs", self.shm_configs, total_collect
        )


        
        self.queue = WaitingQueue(self.num_envs)
        self.events = {eid: mp.Event() for eid in range(self.num_envs)}

    def _create_env(self, index, env_fn):
        p = SHMEnvProc(index, env_fn, self.shm_configs, self.events[index])
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


