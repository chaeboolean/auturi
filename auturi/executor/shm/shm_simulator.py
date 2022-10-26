import itertools
import multiprocessing as mp
import time
from typing import Callable, List

import gym
import numpy as np

from auturi.executor.environment import AuturiEnv, AuturiParallelEnv
from auturi.vector.shm_util import *


class SHMEnvProc(SHMProcBase):
    def __init__(self, idx, env_fn, shm_configs, event):
        self.env_id = idx
        self.env_fn = env_fn
        self.shm_configs = shm_configs
        self.event = event

        super().__init__()

    def _set_cmd_done(self):
        self.command_buffer[self.env_id, 0] = ENV_COMMAND.CMD_DONE

    def _set_state(self, state):
        self.command_buffer[self.env_id, 1] = state

    def _assert_state(self, state):
        assert self.command_buffer[self.env_id, 1] == state

    def insert_buffer(self, end_idx):
        start_idx = 0
        if self.env_id > 0:
            start_idx = self.command_buffer[self.env_id - 1, 2]

        cnt = end_idx - start_idx

        local_rollouts = self.env.fetch_rollouts()

        for _key, trajectories in local_rollouts.items():
            roll_buffer = getattr(self, f"roll{_key}_buffer")
            try:
                np.stack(
                    trajectories[:cnt],
                    out=roll_buffer[
                        start_idx:end_idx,
                    ],
                )

            except Exception as e:

                print(
                    f"[{self.env_id}] {_key} Error!!!=> -- out={roll_buffer[start_idx: end_idx, ].shape}"
                )
                print(
                    f"[{self.env_id}] cnt= {cnt}, len={len(trajectories)} => {trajectories[0]}"
                )

                raise e

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
                self.env.seed(int(seed_))
                self._set_cmd_done()

            elif cmd == ENV_COMMAND.RESET:
                self._assert_state(ENV_STATE.STOPPED)
                obs = self.env.reset()
                self.obs_buffer[self.env_id, :] = obs
                self._set_cmd_done()

            elif cmd == ENV_COMMAND.AGGREGATE:
                self.insert_buffer(seed_)
                self._set_cmd_done()

            # First time to call run_loop. We should call reset
            elif cmd == ENV_COMMAND.START_LOOP and state == ENV_STATE.STOPPED:
                obs = self.env.reset()
                self.obs_buffer[self.env_id, :] = obs
                self._set_state(ENV_STATE.STEP_DONE)

            elif cmd == ENV_COMMAND.START_LOOP and state == ENV_STATE.POLICY_DONE:
                action = self.action_buffer[self.env_id]
                action_artifacts = self.polartifacts_buffer[self.env_id]

                # print(f"SHMEnv{self.env_id}: CMD=RUN_STEP action={action}")
                obs = self.env.step(action, action_artifacts)

                # put observation to shared buffer
                self.obs_buffer[self.env_id, :] = obs

                # change state to STEP_DONE
                self._set_state(ENV_STATE.STEP_DONE)

        return cmd  # return last cmd

    def run(self):
        self.env = self.env_fn()
        assert isinstance(self.env, AuturiEnv)

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
                # print("Enter Loop Event SLEEP ==========================")
                self.event.wait()
                # print("Enter Loop Wake up! ==========================")

    def teardown(self):
        self.env.close()


class SHMParallelEnv(AuturiParallelEnv):
    """SHMParallelVectorEnv

    Uses Python Shared memory implementation as backend

    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], rollout_size=256):
        self.rollout_size = rollout_size

        self.remote_envs = dict()
        self.shm_configs = dict()
        self.buffer_links = []

        super().__init__(env_fns)

        self.queue = WaitingQueue(self.num_envs)
        self.events = {eid: mp.Event() for eid in range(self.num_envs)}
        self.setup_comm_buffer()

        self.command_buffer.fill(ENV_COMMAND.TERMINATE)
        for index, env_fn in enumerate(env_fns):
            p = SHMEnvProc(index, env_fn, self.shm_configs, self.events[index])
            p.start()
            self.remote_envs[index] = p

        self._wait_command_done()
        self.command_buffer.fill(ENV_COMMAND.STOP_LOOP)
        self.env_counter = np.array([-1 for _ in range(self.num_envs)])

    def setup_with_dummy(self, dummy):
        super().setup_with_dummy(dummy)
        self.rollout_samples = dummy.get_rollout_samples()
        for key, sample in self.rollout_samples.items():
            rollkey = f"roll{key}"
            _raw_buffer, _np_buffer = create_shm_from_sample(
                sample, rollkey, self.shm_configs, self.rollout_size
            )
            setattr(self, f"_{rollkey}", _raw_buffer)
            setattr(self, f"{rollkey}_buffer", _np_buffer)
            self.buffer_links += [getattr(self, f"_{rollkey}")]

    def setup_comm_buffer(self):

        self._obs, self.obs_buffer = create_shm_from_sample(
            self.observation_space.sample(), "obs", self.shm_configs, self.num_envs
        )
        self._action, self.action_buffer = create_shm_from_sample(
            self.action_space.sample(), "action", self.shm_configs, self.num_envs
        )

        self._polartifacts, self.polartifacts_buffer = create_shm_from_sample(
            np.array([1, 1], dtype=np.float32),
            "polartifacts",
            self.shm_configs,
            self.num_envs,
        )

        self._command, self.command_buffer = create_shm_from_sample(
            np.array([1, 1, 1], dtype=np.int64),
            "command",
            self.shm_configs,
            self.num_envs,
        )

        self.buffer_links += [
            self._obs,
            self._action,
            self._polartifacts,
            self._command,
        ]

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

        self.command_buffer[:, 0].fill(command)

    def _set_state(self, state):
        self.command_buffer[:, 1].fill(state)

    def reset(self):
        self._wait_states(ENV_STATE.STOPPED)
        self._set_command(ENV_COMMAND.RESET)
        self._wait_command_done()

    def seed(self, seed):
        self._wait_states(ENV_STATE.STOPPED)

        # set seed
        self.command_buffer[:, 2] = np.array(
            [seed + eid for eid in range(self.num_envs)]
        )

        self._set_command(ENV_COMMAND.SEED)
        self._wait_command_done()

    def _poll(self, bs: int = -1) -> List[int]:
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
                ret = self.queue.pop(num=bs)
                self.env_counter[ret] += 1
                return ret

    def send_actions(self, action_ref):
        pass

    # Internally call reset.
    def start_loop(self):
        self.env_counter.fill(-1)
        self._set_command(ENV_COMMAND.START_LOOP)

    def stop_loop(self):
        # Env states can be STEP_DONE or QUEUED
        while np.all(
            (self.command_buffer[:, 1] == ENV_STATE.STEP_DONE)
            & (self.command_buffer[:, 1] == ENV_STATE.QUEUED)
        ):
            pass

        self._set_command(ENV_COMMAND.STOP_LOOP)
        self._wait_command_done()

    def terminate(self):
        self._wait_states(ENV_STATE.STOPPED)
        self._set_command(ENV_COMMAND.TERMINATE)

        print("Call TERMINATE!!!")
        for idx, p in self.remote_envs.items():
            p.join()

        for raw_buffer in self.buffer_links:
            raw_buffer.unlink()

    def step(self, actions: np.ndarray):
        """For debugging Purpose. Synchronous step wrapper."""
        assert len(actions) == self.num_envs

        np.copyto(self.action_buffer, actions)
        self.command_buffer[:, 1].fill(ENV_STATE.POLICY_DONE)

        _ = self.poll(bs=self.num_envs)  # no need to output

        return np.copy(self.obs_buffer), None, None, None

    # Should be called before STOP_LOOP
    def aggregate_rollouts(self):
        print(f"Before accumul =====> ", self.env_counter)
        accumulated_counter = list(itertools.accumulate(self.env_counter))
        accumulated_counter[-1] = self.rollout_size  # TODO: HACK

        self.command_buffer[:, 2] = np.array(accumulated_counter)
        self._set_command(ENV_COMMAND.AGGREGATE)
        self._wait_command_done()

        ret = dict()
        for key, _ in self.rollout_samples.items():
            rollkey = f"roll{key}_buffer"
            ret[key] = np.copy(getattr(self, rollkey))

        return ret
