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


class ENV_COMMAND:
    RUN_LOOP = 0
    STOP_LOOP = 1
    RESET = 2  # start
    SEED = 3
    TERMINATE = 4
    CMD_DONE = 5


class ENV_STATE:
    """Indicates simulator state.
    Initialized to CLOSED
    """

    STOPPED = 0
    STEP_DONE = 1  # Newly arrived requests
    QUEUED = 2  # Inside Server side waiting queue
    POLICY_DONE = 3  # Processed requests
    INF_WORKER = 40  # offset


import multiprocessing as mp
from multiprocessing import shared_memory as shm

from auturi.typing.simulator import AuturiParallelEnv
from auturi.vector.shm_util import *


class SHMEnvWrapper(SHMProcWrapper):
    def __init__(self, idx, env_fn, shm_configs):
        self.env_id = idx
        self.env_fn = env_fn
        self.shm_configs = shm_configs
        super().__init__()
        self.cnt = 0

    def write_to_shm(self, obs, reward, done, info):
        print(f"Env{self.env_id}[{self.cnt}] {obs}")
        # print(f"Env{self.env_id}: write_to_shm")
        # print(f"Env{self.env_id}: write_to_shm, obs-buffer={self.obs_buffer.shape}, obs={obs.shape}")
        self.obs_buffer[self.env_id, :] = obs
        # TODO: ...
        self.cnt += 1

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

    def run(self):
        self.env = self.env_fn()

        self.set_shm_buffer(self.shm_configs)
        self._set_cmd_done()
        self._set_state(ENV_STATE.STOPPED)

        local_cnt = 0

        # run while loop
        while True:
            cmd, state, seed_ = self.command_buffer[self.env_id]
            if cmd == ENV_COMMAND.TERMINATE:
                self._set_cmd_done()
                self.teardown()
                return local_cnt

            elif cmd == ENV_COMMAND.STOP_LOOP:
                # TODO: implement sleep
                self._set_state(ENV_STATE.STOPPED)
                self._set_cmd_done()
                pass

            elif cmd == ENV_COMMAND.SEED:
                # print(f"Env{self.env_id}: CMD=SEED")
                self._assert_state(ENV_STATE.STOPPED)
                self.env.seed(int(seed_))
                self._set_cmd_done()

            elif cmd == ENV_COMMAND.RESET:
                # print(f"Env{self.env_id}: CMD=RESET")

                self._assert_state(ENV_STATE.STOPPED)
                obs = self.env.reset()
                # print(f"RESET => obs {obs}", )
                self.write_to_shm(obs, None, None, None)
                self._set_cmd_done()

            elif cmd == ENV_COMMAND.RUN_LOOP and state == ENV_STATE.POLICY_DONE:
                # print(f"Env{self.env_id}: CMD=RUN_STEP")

                action = self.action_buffer[self.env_id]
                obs, reward, done, info = self.step_wrap(action)
                # put observation to shared buffer
                self.write_to_shm(obs, reward, done, info)

                # change state to STEP_DONE
                self._set_state(ENV_STATE.STEP_DONE)
                local_cnt += 1

    def teardown(self):
        self._obs.unlink()
        self._action.unlink()
        self._command.unlink()
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

        def _create_shm_from_space(sample_, name):
            sample_ = sample_ if hasattr(sample_, "shape") else np.array(sample_)
            # shape_ = sample_.shape if hasattr(sample_, "shape") else ()
            shape_ = (self.num_envs,) + sample_.shape
            # dtype_= sample_.shape if hasattr(sample_, "shape") else ()
            buffer_ = shm.SharedMemory(create=True, size=align(self.num_envs, sample_))
            np_buffer_ = np.ndarray(shape_, dtype=sample_.dtype, buffer=buffer_.buf)

            self.shm_configs[f"{name}_shape"] = shape_
            self.shm_configs[f"{name}_dtype"] = sample_.dtype
            self.shm_configs[f"{name}_buffer"] = buffer_.name
            return buffer_, np_buffer_

        self._obs, self.obs_buffer = _create_shm_from_space(
            dummy_env.observation_space.sample(), "obs"
        )
        self._action, self.action_buffer = _create_shm_from_space(
            dummy_env.action_space.sample(), "action"
        )
        self._command, self.command_buffer = _create_shm_from_space(
            np.array([1, 1, 1], dtype=np.int32), "command"
        )

        self.queue = WaitingQueue(self.num_envs)

    def _create_env(self, index, env_fn):
        p = SHMEnvWrapper(index, env_fn, self.shm_configs)
        p.start()
        return p

    def _wait_states(self, state):
        while not np.all(self.command_buffer[:, 1] == state):
            pass

    def _wait_command_done(self):
        while not np.all(self.command_buffer[:, 0] == ENV_COMMAND.CMD_DONE):
            pass

    def _set_command(self, command):
        self.command_buffer[:, 0].fill(command)

    def reset(self):
        self._wait_states(ENV_STATE.STOPPED)
        self._set_command(ENV_COMMAND.RESET)
        self._wait_command_done()
        time.sleep(2)
        return np.copy(self.obs_buffer)

    def seed(self, seed):
        self._wait_states(ENV_STATE.STOPPED)

        # set seed
        self.command_buffer[:, 2] = np.array(
            [seed + eid for eid in range(self.num_envs)]
        )

        self._set_command(ENV_COMMAND.SEED)
        self._wait_command_done()

    def start_loop(self):
        self._set_command(ENV_COMMAND.RUN_LOOP)

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
        self.action_buffer = actions
        self.command_buffer[:, 1].fill(ENV_STATE.POLICY_DONE)

        poll_out = self.poll(bs=self.num_envs)  # no need to output
        print(poll_out)

        return np.copy(self.obs_buffer), None, None, None


class SHMPolicyWrapper(SHMProcWrapper):
    def run(self, index, policy_fn, queue, shm_configs):
        self.policy_id = index
        self.env = policy_fn()
        self.set_shm_buffer(shm_configs)
        self.queue = queue

        # put itself to queue after ready
        self.queue.put(self.policy_id)

        while True:
            to_process = np.where(
                self.command_states == self.policy_id + ENV_STATE.INF_WORKER
            )[0]
            if len(to_process) == 0:
                continue

            obs, reward, dones, infos = _flatten_obs(_obs_buffer[to_process, :])

            actions = self.policy.compute_actions(obs)
            self.action_buffer[to_process] = actions
            self.command_states[to_process, :] = STATE.POLICY_DONE
            self.queue.put(self.policy_id)

            self.policy.insert_buffer(obs, obs, reward, dones, infos)

    def set_device(self, device):
        pass


class SHMVectorPolicies(AuturiVectorPolicy):
    # shm_config, command_buffer s

    def _create_policy(self, index, policy_fn):
        p = SHMPolicyWrapper()
        p.start(index, policy_fn, self.pending_policies, shm_config)
        return p

    def _setup(self):
        self.pending_policies = mp.Queue()
        self.reset()

    def reset(self):
        pass

    def assign_free_server(self, obs_refs: List[int], n_steps: int):
        server_id = self.pending_policies.get()
        self.command_buffer[obs_refs] = server_id + ENV_STATE.INF_WORKER
        return None, server_id

    def close(self):
        for p in self.remote_policies:
            p.join()
