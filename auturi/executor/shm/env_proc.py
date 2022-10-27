import multiprocessing as mp
import time
from typing import Callable, List

import numpy as np

import auturi.executor.shm.util as util
from auturi.executor.environment import AuturiEnv, AuturiSerialEnv


class ENV_COMMAND:
    STOP_LOOP = 123
    START_LOOP = 422
    RESET = 2  # start
    SEED = 3
    SET_ENV = 21
    TERMINATE = 4
    AGGREGATE = 19
    CMD_DONE = 5


class ENV_STATE:
    """Indicates simulator state.
    Initialized to STOPPED.
    """

    STOPPED = 23
    STEP_DONE = 1  # Newly arrived requests
    QUEUED = 2  # Inside Server side waiting queue
    POLICY_DONE = 3  # Processed requests
    POLICY_OFFSET = 40  # offset => ASSINGED


class SHMEnvProc(mp.Process):
    def __init__(self, idx, env_fns, shm_buffer_attr_dict, event):
        self.worker_id = idx
        self.env = AuturiSerialEnv(idx, env_fns)
        self.shm_buffer_attr_dict = shm_buffer_attr_dict
        self.event = event

        super().__init__()

    def _set_cmd_done(self):
        self.command_buffer[self.worker_id, 0] = ENV_COMMAND.CMD_DONE

    def _set_state(self, state):
        self.command_buffer[self.worker_id, 1] = state

    def _assert_state(self, state):
        assert self.command_buffer[self.worker_id, 1] == state

    # def aggregate(self, end_idx):
    #     start_idx = 0
    #     if self.env_id > 0:
    #         start_idx = self.command_buffer[self.env_id - 1, 2]

    #     cnt = end_idx - start_idx

    #     local_rollouts = self.env.fetch_rollouts()

    #     for _key, trajectories in local_rollouts.items():
    #         roll_buffer = getattr(self, f"roll{_key}_buffer")
    #         try:
    #             np.stack(
    #                 trajectories[:cnt],
    #                 out=roll_buffer[
    #                     start_idx:end_idx,
    #                 ],
    #             )

    #         except Exception as e:

    #             print(
    #                 f"[{self.env_id}] {_key} Error!!!=> -- out={roll_buffer[start_idx: end_idx, ].shape}"
    #             )
    #             print(
    #                 f"[{self.env_id}] cnt= {cnt}, len={len(trajectories)} => {trajectories[0]}"
    #             )

    #             raise e

    def _aggregate(self, start_idx, end_idx):
        pass

    def insert_obs_buffer(self, obs):
        self.obs_buffer[self.env.start_idx : self.env.end_idx, :] = obs

    def get_actions(self):
        action = self.action_buffer[self.env.start_idx : self.env.end_idx]
        action_artifacts = self.artifacts_buffer[self.env.start_idx : self.env.end_idx]
        return action, action_artifacts

    def _run_loop(self, state):
        if state == ENV_STATE.STOPPED:
            obs = self.env.reset()
            self.insert_obs_buffer(obs)
            self._set_state(ENV_STATE.STEP_DONE)

        elif state == ENV_STATE.POLICY_DONE:
            action, artifacts = self.get_actions()
            obs = self.env.step((action, artifacts))
            print(f"SHMEnv{self.worker_id}: CMD=RUN_STEP action={action} => obs={obs}")

            self.insert_obs_buffer(obs)
            self._set_state(ENV_STATE.STEP_DONE)

    def _run(self) -> ENV_COMMAND:
        while True:
            cmd, state, data1_, data2_ = self.command_buffer[self.worker_id]

            # First time to call run_loop. We should call reset
            if cmd == ENV_COMMAND.START_LOOP:
                self._run_loop(state)

            # should break if cmd is not START_LOOP
            elif cmd == ENV_COMMAND.TERMINATE or cmd == ENV_COMMAND.STOP_LOOP:
                self._set_state(ENV_STATE.STOPPED)
                self._set_cmd_done()
                return cmd

            elif cmd == ENV_COMMAND.SET_ENV:
                print(f"Env({self.worker_id}) SET ENV!!!")
                self._assert_state(ENV_STATE.STOPPED)
                self.env.set_working_env(int(data1_), int(data2_))
                self._set_cmd_done()
                print("gonna return~~~~ ")
                return cmd

            elif cmd == ENV_COMMAND.SEED:
                print(f"Env({self.worker_id}) SEED!!!")

                self._assert_state(ENV_STATE.STOPPED)
                self.env.seed(int(data1_))
                self._set_cmd_done()
                return cmd

            elif cmd == ENV_COMMAND.RESET:
                print(f"Env({self.worker_id}) RESET!!!")

                self._assert_state(ENV_STATE.STOPPED)
                obs = self.env.reset()
                self.insert_obs_buffer(obs)
                self._set_cmd_done()
                return cmd

            elif cmd == ENV_COMMAND.AGGREGATE:
                self._assert_state(ENV_STATE.STOPPED)
                self._aggregate(int(data1_), int(data2_))
                self._set_cmd_done()
                return cmd

            else:
                raise RuntimeError(f"Not allowed: {cmd}")

    def run(self):
        util.set_shm_buffer_from_attr(self, self.shm_buffer_attr_dict)
        assert hasattr(self, "command_buffer")
        print(self.command_buffer, "===> Inside envproc")

        self._set_state(ENV_STATE.STOPPED)
        self._set_cmd_done()

        # run while loop
        while True:
            print(self.command_buffer, "===> Inside envproc 2222")
            self.event.wait()
            print("\nENV Loop Wake up! ==========================")
            last_cmd = self._run()
            print("\nENV Loop LAST CMD = ", last_cmd)

            if last_cmd == ENV_COMMAND.TERMINATE:
                self.teardown()
                break
            else:
                print("ENV Loop Event SLEEP ==========================")
                self.event.clear()
                self.event.wait()

                print("Someone wake me uup ==========================")

    def teardown(self):
        self.env.terminate()
