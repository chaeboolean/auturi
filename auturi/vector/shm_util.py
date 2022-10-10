ALIGN_BYTES = 64
import multiprocessing as mp
from multiprocessing import shared_memory as shm

import numpy as np


def align(num_envs, dummy_arr):
    # if isinstance(dummy_arr, int):
    #     return 32 * num_envs
    # else:
    #     return ((num_envs * dummy_arr.nbytes + ALIGN_BYTES - 1) // ALIGN_BYTES) * ALIGN_BYTES

    return dummy_arr.nbytes * num_envs


class SHMProcWrapper(mp.Process):
    def _get_np_shm(self, ident_):
        _buffer = shm.SharedMemory(self.configs[f"{ident_}_buffer"])
        np_buffer = np.ndarray(
            self.configs[f"{ident_}_shape"],
            dtype=self.configs[f"{ident_}_dtype"],
            buffer=_buffer.buf,
        )
        return _buffer, np_buffer

    def set_shm_buffer(self, shm_configs):
        self.configs = shm_configs

        self._obs, self.obs_buffer = self._get_np_shm("obs")
        self._action, self.action_buffer = self._get_np_shm("action")
        self._command, self.command_buffer = self._get_np_shm("command")


class WaitingQueue:
    """Imitate circular queue, but minimizing redundant numpy copy or traverse array."""

    def __init__(self, num_envs):
        self.limit = 2 * num_envs
        self.q = np.array([0 for _ in range(self.limit)], dtype=np.int64)
        self.cnt = 0
        self.head = 0
        self.tail = 0

    def insert(self, requests):
        rsz = requests.size
        first_ = rsz if self.tail + rsz <= self.limit else self.limit - self.tail
        second_ = rsz - first_

        self.q[self.tail : self.tail + first_] = requests[:first_]
        self.q[:second_] = requests[first_:]

        self.tail = (self.tail + rsz) % self.limit
        self.cnt += rsz

    def pop(self, num="all"):
        npop = self.cnt if num == "all" else num
        assert npop <= self.cnt

        first_ = npop if self.head + npop <= self.limit else self.limit - self.head
        second_ = npop - first_

        ret = np.append(self.q[self.head : self.head + first_], self.q[:second_])
        self.head = (self.head + npop) % self.limit
        self.cnt -= npop

        return ret
