ALIGN_BYTES = 64
import multiprocessing as mp
from multiprocessing import shared_memory as shm
from typing import Any, Dict, Tuple

import numpy as np


def align(num_envs, dummy_arr):
    # if isinstance(dummy_arr, int):
    #     return 32 * num_envs
    # else:
    #     return ((num_envs * dummy_arr.nbytes + ALIGN_BYTES - 1) // ALIGN_BYTES) * ALIGN_BYTES

    return dummy_arr.nbytes * num_envs


def create_shm_buffer_from_dict(
    sample_dict: Dict[str, Tuple[np.ndarray, int]]
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Create shm buffer dict from samples.

    Args:
        sample_dict (Dict[key, (sample, max_num)]): Indicates sample array being element.
    Returns:
        Dict[key, (raw_buffer, np_buffer)]: Contains raw buffer and usuable buffer
        Dict[key, Dict[str, Any]]: Includes attributes such as dtype, shape, buffer name for each key.

    """
    shm_buffer_dict = dict()
    shm_buffer_attr_dict = dict()

    for key, (sample, max_num) in sample_dict.items():
        buffer_, np_buffer_, attr_ = _create_shm_from_sample(sample, max_num)
        shm_buffer_dict[key] = (buffer_, np_buffer_)
        shm_buffer_attr_dict[key] = attr_

    return shm_buffer_dict, shm_buffer_attr_dict


def _create_shm_from_sample(sample_: np.ndarray, max_num: int):
    sample_ = sample_ if hasattr(sample_, "shape") else np.array(sample_)
    shape_ = (max_num,) + sample_.shape
    buffer_ = shm.SharedMemory(create=True, size=align(max_num, sample_))
    np_buffer_ = np.ndarray(shape_, dtype=sample_.dtype, buffer=buffer_.buf)
    attr_dict = {"shape": shape_, "dtype": sample_.dtype, "name": buffer_.name}
    return buffer_, np_buffer_, attr_dict


def _get_buffer_from_other_proc(attr_dict):
    _buffer = shm.SharedMemory(attr_dict["name"])
    np_buffer = np.ndarray(
        attr_dict["shape"],
        dtype=attr_dict["dtype"],
        buffer=_buffer.buf,
    )
    return _buffer, np_buffer


def set_shm_buffer_from_attr(obj, shm_buffer_attr_dict):
    for key, attr_dict in shm_buffer_attr_dict.items():
        raw_buf, np_buffer = _get_buffer_from_other_proc(attr_dict)
        setattr(obj, f"_{key}", raw_buf)
        setattr(obj, f"{key}_buffer", np_buffer)


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


# def create_obs_shm_from_dummy(dummy_env):
#     obs_offset = dict()
#     total_bytes = 0
#     dummy_env.reset()
#     info = dict()
#     for _ in range(3):
#         obs__, reward__, done__, info__ = dummy_env.step(
#             dummy_env.action_space.sample()
#         )
#         if done__:
#             dummy_env.reset()
#         info.update(info__)
#     info.update({"terminal_observation": obs__})

#     obs_offset["obs"] = {
#         "shape": obs__.shape,
#         "dtype": obs__.dtype,
#         "nbytes": obs__.nbytes,
#         "offset": total_bytes,
#     }
#     total_bytes += obs_offset["obs"]["nbytes"]

#     assert isinstance(reward__, float)
#     obs_offset["reward"] = {
#         "shape": (),
#         "dtype": np.float32,
#         "nbytes": 32,
#         "offset": total_bytes,
#     }
#     total_bytes += obs_offset["reward"]["nbytes"]

#     assert isinstance(done__, bool)
#     obs_offset["done"] = {
#         "shape": (),
#         "dtype": np.int8,
#         "nbytes": 8,
#         "offset": total_bytes,
#     }
#     total_bytes += obs_offset["done"]["nbytes"]
