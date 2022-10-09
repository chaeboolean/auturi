import functools
import time
from collections import OrderedDict
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple, Type,
                    Union)

import gym
import numpy as np
import ray


class ENV_CMD:
    STEP = 0
    SEED = 1
    RESET = 2
    CLOSE = 3


class ENV_STATE:
    STEP_DONE = 0  # Newly arrived requests
    INF_DONE = 1  # Processed requests
    QUEUED = 2  # Inside Server side waiting queue
    CLOSE = 3
    INF_WORKER = 4  # offset


import multiprocessing as mp
from multiprocessing import shared_memory as shm

from auturi.typing.simulator import AuturiParallelEnv


class SHMEnvWrapper:
    def _setup(self, idx, env_fn, shm_configs):
        self.env_id = idx
        self.env = env_fn()
        self.configs = shm_configs

        self._obs, self.obs_buffer = self._get_np_shm("obs")
        self._action, self.action_buffer = self._get_np_shm("action")
        self._command, self.command_buffer = self._get_np_shm("command")

    def run(self, idx, env_fn, shm_configs):
        self._setup(idx, env_fn, shm_configs)

        cnt = 0
        while True:
            action = get_action()

            if _state_buffer[env_id][0] == STATE.CLOSE:
                break

            observation, reward, done, info = self.env.step(action)

            with prof_wrapper.em.timespan("write_obs"):
                put_obs(obs)

            _state_buffer[env_id][0] = STATE.STEP_DONE
            cnt += 1

        self._teardown()

    def _teardown(self):
        self._obs.unlink()
        self._action.unlink()
        self._command.unlink()
        self.env.close

    def _get_np_shm(self, ident_):
        _buffer = shm.SharedMemory(self.configs[f"{ident_}_buffer"])
        np_buffer = np.ndarray(
            self.configs[f"{ident_}_shape"],
            dtype=self.configs[f"{ident_}_dtype"],
            buffer=_buffer.buf,
        )
        return _buffer, np_buffer


def _run_shm_env():
    env_id = idx
    _get_np_shm(configs, id_)

    def put_obs(obs_):
        _obs_buffer[env_id, :] = obs_

    def get_action():
        while (
            _state_buffer[env_id][0] != STATE.INF_DONE
            and _state_buffer[env_id][0] != STATE.CLOSE
        ):
            continue

        return _action_buffer[[env_id]]

    env = env_fn()

    # reset simulator and put first observations
    obs = env.reset()
    put_obs(obs)
    _state_buffer[env_id][0] = STATE.STEP_DONE


class SHMParallelEnv(AuturiParallelEnv):
    def _setup(self, dummy_env):

        self.shm_configs = dict()

        dummy_env.reset()
        obs__, reward__, done__, info__ = dummy_env.step(
            dummy_env.action_space.sample()
        )

        def _create_shm_from_space(sample_, name):
            shape_ = sample_.shape if hasattr(sample_, "shape") else (1,)
            self.shm_configs[f"{name}_shape"] = shape_
            self.shm_configs[f"{name}_dtype"] = sample_.dtype
            buffer_ = shm.SharedMemory(create=True, size=align(self.num_envs, sample_))
            self.shm_configs[f"{name}_buffer"] = buffer_.name

        _create_shm_from_space(dummy_env.observation_space.sample(), "obs")
        _create_shm_from_space(dummy_env.action_space.sample(), "action")

        self.env_state_buffer = shm.SharedMemory(create=True, size=self.num_envs)
        self.env_states = np.ndarray(
            (self.num_envs,), dtype=np.byte, buffer=env_state_buffer.buf
        )
        self.shm_configs["command_buffer"] = self.env_state_buffer.name

    def _create_env(self, index, env_fn):
        p = mp.Process(
            target=_run_shm_env,
            name=f"env_{index}",
            args=(index, env_fn, self.shm_configs),
        )
        p.start()
        return p

    def poll(self, bs: int = -1):
        if bs < 0:
            bs = self.num_envs
        new_requests_ = np.where(env_states == STATE.STEP_DONE)[0]
        waiting_reqs.insert(new_requests_)
        env_states[new_requests_] = STATE.QUEUED

    def send_actions(self, action_dict):
        pass
