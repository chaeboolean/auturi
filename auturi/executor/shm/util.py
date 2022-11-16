import time
from multiprocessing import shared_memory as shm
from typing import Any, Callable, Dict, Tuple

import numpy as np

from auturi.executor.environment import AuturiEnv
from auturi.logger import get_logger

logger = get_logger()

ALIGN_BYTES = 64


def wait(cond_, debug_msg, timeout=2):
    last_timeout = time.time()
    while not cond_():
        if time.time() - last_timeout > timeout:
            msg = debug_msg() if callable(debug_msg) else debug_msg
            logger.debug(msg)
            print(msg)
            last_timeout = time.time()


def align(num_envs, dummy_arr):
    # if isinstance(dummy_arr, int):
    #     return 32 * num_envs
    # else:
    #     return ((num_envs * dummy_arr.nbytes + ALIGN_BYTES - 1) // ALIGN_BYTES) * ALIGN_BYTES

    return dummy_arr.nbytes * num_envs


def create_buffer_from_sample(
    sample_dict: Dict[str, Tuple[np.ndarray, int]]
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Create shm buffer dict from samples.

    Args:
        sample_dict (Dict[key, (sample, max_num)]): Indicates sample array being element.
    Returns:
        Dict[key, (raw_buffer, np_buffer)]: Contains raw buffer and usuable buffer
        Dict[key, Dict[str, Any]]: Includes attributes such as dtype, shape, buffer name for each key.

    """
    buffer_dict = dict()
    attr_dict = dict()

    for key, (sample, max_num) in sample_dict.items():
        buffer_, np_buffer_, attr_ = _create_buffer_from_sample(sample, max_num)
        buffer_dict[key] = (buffer_, np_buffer_)
        attr_dict[key] = attr_

    return buffer_dict, attr_dict


def _create_buffer_from_sample(sample_: np.ndarray, max_num: int):
    """Create buffer from sample np.ndarray.

    Returns buffer with shape (max_num, *sample_.shape).
    """

    sample_ = sample_ if hasattr(sample_, "shape") else np.array(sample_)
    shape_ = (max_num,) + sample_.shape
    buffer_ = shm.SharedMemory(create=True, size=align(max_num, sample_))
    np_buffer_ = np.ndarray(shape_, dtype=sample_.dtype, buffer=buffer_.buf)
    attr_dict = {"shape": shape_, "dtype": sample_.dtype, "name": buffer_.name}

    return buffer_, np_buffer_, attr_dict


def set_shm_from_attr(attr_dict):
    _buffer = shm.SharedMemory(attr_dict["name"])
    np_buffer = np.ndarray(
        attr_dict["shape"],
        dtype=attr_dict["dtype"],
        buffer=_buffer.buf,
    )
    return _buffer, np_buffer


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


def create_shm_from_env(
    env_fn: Callable[[], AuturiEnv], max_num_envs: int, rollout_size: int
):
    # Collect sample data
    dummy_env = env_fn()  # Single environment (not serial)
    obs = dummy_env.reset()
    action = dummy_env.action_space.sample()
    action_artifacts_sample_list = dummy_env.artifacts_samples

    # TODO
    if len(action_artifacts_sample_list) > 1:
        raise NotImplementedError

    dummy_env.step(action, action_artifacts_sample_list)
    rollouts = dummy_env.aggregate_rollouts()

    # Create basic buffers
    buffer_sample_dict = {
        "obs": (obs, max_num_envs),
        "action": (action, max_num_envs),
        "artifact": (action_artifacts_sample_list[0], max_num_envs),
        # indicating env_state
        "env": (np.array([1], dtype=np.int8), max_num_envs),
    }

    # Create rollout sample dictionary
    rollout_sample_dict = dict()
    for key, rollout in rollouts.items():
        rollout_sample_dict[key] = (rollout[0], rollout_size)

    base_buffers, base_buffer_attr = create_buffer_from_sample(buffer_sample_dict)
    rollout_buffers, rollout_buffer_attr = create_buffer_from_sample(
        rollout_sample_dict
    )

    dummy_env.terminate()

    # initialize env buffers
    base_buffers["env"][1].fill(0)
    return base_buffers, base_buffer_attr, rollout_buffers, rollout_buffer_attr
