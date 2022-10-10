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


def _get_np_shm(ident_, configs):
    _buffer = shm.SharedMemory(configs[f"{ident_}_buffer"])
    np_buffer = np.ndarray(
        configs[f"{ident_}_shape"],
        dtype=configs[f"{ident_}_dtype"],
        buffer=_buffer.buf,
    )
    return _buffer, np_buffer



def _create_shm_from_space(sample_, name, shm_configs, num_envs):
    sample_ = sample_ if hasattr(sample_, "shape") else np.array(sample_)
    # shape_ = sample_.shape if hasattr(sample_, "shape") else ()
    shape_ = (num_envs,) + sample_.shape
    # dtype_= sample_.shape if hasattr(sample_, "shape") else ()
    buffer_ = shm.SharedMemory(create=True, size=align(num_envs, sample_))
    np_buffer_ = np.ndarray(shape_, dtype=sample_.dtype, buffer=buffer_.buf)

    shm_configs[f"{name}_shape"] = shape_
    shm_configs[f"{name}_dtype"] = sample_.dtype
    shm_configs[f"{name}_buffer"] = buffer_.name
    return buffer_, np_buffer_


class SHMProcWrapper(mp.Process):
    def set_shm_buffer(self, shm_configs):
        self.configs = shm_configs
        
        identifiers = set([key.split("_")[0] for key in shm_configs])
        print(identifiers)
        for ident in identifiers:
            raw_buf, np_buffer = _get_np_shm(ident, shm_configs)
            setattr(self, f"_{ident}", raw_buf)
            setattr(self, f"{ident}_buffer", np_buffer)
            
            
        # self._obs, self.obs_buffer = _get_np_shm("obs", shm_configs)
        # self._action, self.action_buffer = _get_np_shm("action", shm_configs)
        # self._command, self.command_buffer = _get_np_shm("command", shm_configs)
        

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
