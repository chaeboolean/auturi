import multiprocessing as mp
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

import auturi.executor.shm.util as util
from auturi.executor.environment import AuturiEnv, AuturiVectorEnv
from auturi.executor.shm.env_proc import ENV_COMMAND, ENV_STATE, SHMEnvProc
from auturi.executor.shm.mixin import SHMVectorMixin


class SHMParallelEnv(AuturiVectorEnv, SHMVectorMixin):
    """SHMParallelVectorEnv

    Uses Python Shared memory implementation as backend

    """

    def __init__(
        self,
        env_fns: List[Callable],
        shm_buffer_dict: Dict[str, Any],
        shm_buffer_attr_dict: Dict[str, Any],
    ):
        self.shm_buffer_dict = shm_buffer_dict
        self.shm_buffer_attr_dict = shm_buffer_attr_dict

        self.obs_buffer = self.shm_buffer_dict["obs"][1]
        self.env_buffer = self.shm_buffer_dict["env"][1]
        self.action_buffer = self.shm_buffer_dict["action"][1]
        self._set_command_buffer()
        assert hasattr(self, "command_buffer") and hasattr(self, "cmd_enum")

        self.env_buffer.fill(ENV_COMMAND.STOP_LOOP)  # anything different from CMD_DONE

        self.queue = util.WaitingQueue(len(env_fns))
        self.events = dict()
        self.env_counter = np.array([-1 for _ in range(len(env_fns))])

        super().__init__(env_fns)

    def _set_command_buffer(self):
        """Should set attributes "command_buffer", "cmd_enum"."""
        self.command_buffer = self.env_buffer
        self.cmd_enum = ENV_COMMAND

    def _create_worker(self, idx: int):
        self.events[idx] = mp.Event()
        self.events[idx].clear()

        p = SHMEnvProc(idx, self.env_fns, self.shm_buffer_attr_dict, self.events[idx])
        p.start()
        return p

    def _set_working_env(
        self, worker_id: int, env_worker: AuturiEnv, start_idx: int, num_envs: int
    ) -> None:
        self._wait_command_done(worker_id)
        self._set_command(
            ENV_COMMAND.SET_ENV, worker_id=worker_id, data1=start_idx, data2=num_envs
        )

    # Internally call reset.
    def start_loop(self):
        self._wait_command_done()
        self.env_counter.fill(-1)
        self._set_command(ENV_COMMAND.START_LOOP)

    def stop_loop(self):
        # Env states can be STEP_DONE or QUEUED
        while not np.all(
            np.ma.mask_or(
                (self._get_state() == ENV_STATE.STEP_DONE),
                (self._get_state() == ENV_STATE.QUEUED),
            )
        ):
            pass

        self._set_command(ENV_COMMAND.STOP_LOOP, set_event=False)
        self._wait_command_done()
        print("stop loop done!!!")

    def reset(self):
        self._wait_command_done()
        self._set_command(ENV_COMMAND.RESET)
        return

    def seed(self, seed):
        self._wait_command_done()
        self._set_command(ENV_COMMAND.SEED, data1=seed)

    def poll(self) -> List[int]:
        while True:
            new_req = np.where(self._get_state() == ENV_STATE.STEP_DONE)[0]
            # if len(new_req) > 0: print(new_req)
            self.queue.insert(new_req)
            self.env_buffer[new_req, 1] = ENV_STATE.QUEUED
            if self.queue.cnt >= self.batch_size:
                ret = self.queue.pop(num=self.batch_size)
                self.env_counter[ret] += 1
                return ret

    def send_actions(self, action_ref) -> None:
        """SHM Implementation do not need send_actions."""
        pass

    def terminate(self):
        self._set_command(ENV_COMMAND.TERMINATE)

        for idx, p in self.remote_workers.items():
            p.join()

        # Responsible to unlink created shm buffer
        for key, tuple_ in self.shm_buffer_dict.items():
            tuple_[0].unlink()

    def step(self, actions: Tuple[np.ndarray, List[np.ndarray]]):
        """For debugging Purpose. Synchronous step wrapper."""
        self.batch_size = self.num_envs

        action_, action_artifacts = actions  # Ignore action artifacts
        assert len(action_) == self.num_envs

        while not np.all(self._get_state() != ENV_STATE.STOPPED):
            pass

        np.copyto(self.action_buffer, action_)
        self._set_state(ENV_STATE.POLICY_DONE)
        _ = self.poll()  # no need to output

        return np.copy(self.obs_buffer)

    # Should be called before STOP_LOOP
    def aggregate_rollouts(self):
        # print(f"Before accumul =====> ", self.env_counter)
        # accumulated_counter = list(itertools.accumulate(self.env_counter))
        # accumulated_counter[-1] = self.rollout_size  # TODO: HACK

        # self.command_buffer[:, 2] = np.array(accumulated_counter)
        # self._set_command(ENV_COMMAND.AGGREGATE)
        # self._wait_command_done()

        # ret = dict()
        # for key, _ in self.rollout_samples.items():
        #     rollkey = f"roll{key}_buffer"
        #     ret[key] = np.copy(getattr(self, rollkey))

        # return ret
        pass
