import time
from multiprocessing import shared_memory as shm
from typing import List

import numpy as np

from auturi.executor.shm.mp_mixin import SHMProcLoopMixin, SHMVectorLoopMixin

RESET = 11
INCR = 12
DECR = 13


class _TestProc(SHMProcLoopMixin):
    def __init__(self, worker_id, cmd_attr_dict, buf_name: str):
        self.buf_name = buf_name
        super().__init__(worker_id, cmd_attr_dict)

    @property
    def proc_name(self):
        return "TestProc"

    def initialize(self) -> None:
        self.__buffer = shm.SharedMemory(self.buf_name)
        self.buffer = np.ndarray((80,), np.int32, buffer=self.__buffer.buf)
        super().initialize()

    def set_command_handlers(self) -> None:
        self.cmd_handler[RESET] = self._reset_handler
        self.cmd_handler[INCR] = self._incr_handler
        self.cmd_handler[DECR] = self._decr_handler

    def _incr_handler(self, cmd: int, data_list: List[int]):
        self.buffer[self.worker_id] += data_list[0]
        self.reply(cmd)

    def _decr_handler(self, cmd: int, data_list: List[int]):
        self.buffer[self.worker_id] -= data_list[0]
        self.reply(cmd)

    def _reset_handler(self, cmd: int, data_list: List[int]):
        self.buffer[self.worker_id] = 0
        self.reply(cmd)

    def _step_loop_once(self, is_first: bool):
        self.buffer[self.worker_id] += 1
        time.sleep(0.5)


class _TestVector(SHMVectorLoopMixin):
    def __init__(self):
        shape = (80,)
        self.__buffer = shm.SharedMemory(create=True, size=4 * 80)
        self.buffer = np.ndarray(shape, dtype=np.int32, buffer=self.__buffer.buf)

        super().__init__(max_workers=100)

    @property
    def proc_name(self):
        return "TestVectorClass"

    def _create_worker(self, worker_id: int) -> _TestProc:
        kwargs = {"buf_name": self.__buffer.name}
        return self.init_proc(worker_id, _TestProc, kwargs)

    def _reconfigure_worker(self, worker_id: int, worker: _TestProc) -> None:
        pass

    def reconfigure(self, num_workers: int):
        self.reconfigure_workers(num_workers)

    def terminate(self):
        super().terminate()
        self.__buffer.unlink()


def test_basic():
    vector_manager = _TestVector()
    buffer = vector_manager.buffer
    vector_manager.reconfigure(2)

    vector_manager.request(cmd=RESET)
    vector_manager.request(cmd=INCR, data=[2])
    vector_manager.sync()
    assert buffer[0] == 2

    vector_manager.reconfigure(1)

    vector_manager.terminate()


def test_multiple_children():
    vector_manager = _TestVector()
    buffer = vector_manager.buffer
    vector_manager.reconfigure(50)

    vector_manager.request(cmd=RESET)
    vector_manager.request(cmd=INCR, data=[2])
    vector_manager.sync()
    assert np.all(buffer[:50] == 2)

    vector_manager.request(cmd=INCR, data=[2], worker_id=7)
    vector_manager.request(cmd=INCR, data=[2], worker_id=8)
    vector_manager.request(cmd=DECR, data=[10], worker_id=9)
    vector_manager.sync()
    assert np.all(buffer[7:10] == np.array([4, 4, -8]))

    vector_manager.terminate()


def test_reconfigure():
    vector_manager = _TestVector()
    buffer = vector_manager.buffer
    vector_manager.reconfigure(10)

    vector_manager.request(cmd=RESET)
    vector_manager.request(cmd=INCR, data=[1])
    vector_manager.request(cmd=INCR, data=[1])
    vector_manager.request(cmd=INCR, data=[1])
    vector_manager.sync()
    assert np.all(buffer[:10] == 3)

    vector_manager.reconfigure(60)
    vector_manager.request(cmd=RESET)
    vector_manager.request(cmd=INCR, data=[1])
    vector_manager.request(cmd=INCR, data=[1])
    vector_manager.sync()
    assert np.all(buffer[:60] == 2)

    vector_manager.reconfigure(20)
    vector_manager.request(cmd=RESET)
    vector_manager.request(cmd=DECR, data=[1])
    vector_manager.sync()
    assert np.all(buffer[:20] == -1)
    assert np.all(buffer[20:60] == 2)
    vector_manager.terminate()


def test_run_loop():
    vector_manager = _TestVector()
    buffer = vector_manager.buffer
    num_workers = 3
    vector_manager.reconfigure(num_workers)

    vector_manager.request(cmd=RESET)
    vector_manager.start_loop()
    time.sleep(0.9)
    vector_manager.stop_loop()
    vector_manager.sync()

    assert np.all(buffer[:num_workers] == 2)

    vector_manager.request(cmd=RESET)
    vector_manager.start_loop()
    time.sleep(1.6)
    vector_manager.stop_loop()
    vector_manager.sync()
    assert np.all(buffer[:num_workers] == 4)

    vector_manager.terminate()
