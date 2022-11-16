import time
from multiprocessing import shared_memory as shm

import numpy as np

from auturi.executor.shm.mp_mixin import Request, SHMProcMixin, SHMVectorMixin


class _TestVector(SHMVectorMixin):
    def __init__(self):
        shape = (80,)
        self.__buffer = shm.SharedMemory(create=True, size=4 * 80)
        self.buffer = np.ndarray(shape, dtype=np.int32, buffer=self.__buffer.buf)
        self.workers = dict()

        super().__init__()

    def reconfigure(self, num_workers: int):
        old_num_wokers = len(self.workers)
        for worker_id in range(old_num_wokers):
            if worker_id >= num_workers:
                self.terminate_worker(worker_id)
                self.workers[worker_id].join()
                del self.workers[worker_id]

        for worker_id in range(old_num_wokers, num_workers):
            p = self.init_proc(worker_id, _TestProc, {"buf_name": self.__buffer.name})
            self.workers[worker_id] = p

        for id, worker in self.workers.items():
            assert worker.is_alive(), f"{id} is not alive..."

        self.sync()

    def terminate(self):
        super().terminate()
        self.__buffer.unlink()


class _TestProc(SHMProcMixin):
    def __init__(self, worker_id, req_queue, rep_queue, buf_name: str):
        self.buf_name = buf_name
        super().__init__(worker_id, req_queue=req_queue, rep_queue=rep_queue)

    def initialize(self) -> None:
        self.__buffer = shm.SharedMemory(self.buf_name)
        self.buffer = np.ndarray((80,), np.int32, buffer=self.__buffer.buf)

    def set_handler_for_command(self) -> None:
        self.cmd_handler["RESET"] = self._reset_handler
        self.cmd_handler["INCR"] = self._incr_handler
        self.cmd_handler["DECR"] = self._decr_handler
        self.cmd_handler["RUN_LOOP"] = self._loop_handler

    def _incr_handler(self, req: Request):
        self.buffer[self.worker_id] += req.data[0]
        self.reply(req.cmd)

    def _decr_handler(self, req: Request):
        self.buffer[self.worker_id] -= req.data[0]
        self.reply(req.cmd)

    def _reset_handler(self, req: Request):
        self.buffer[self.worker_id] = 0
        self.reply(req.cmd)

    def _step_loop_once(self, is_first: bool):
        self.buffer[self.worker_id] += 1
        time.sleep(0.5)


def test_basic():
    vector_manager = _TestVector()
    buffer = vector_manager.buffer
    vector_manager.reconfigure(2)

    vector_manager.request(Request(cmd="RESET"))
    vector_manager.request(Request(cmd="INCR", data=[2]))
    vector_manager.sync()
    assert buffer[0] == 2

    vector_manager.reconfigure(1)

    vector_manager.terminate()


def test_multiple_children():
    vector_manager = _TestVector()
    buffer = vector_manager.buffer
    vector_manager.reconfigure(50)

    vector_manager.request(Request(cmd="RESET"))
    vector_manager.request(Request(cmd="INCR", data=[2]))
    vector_manager.sync()
    assert np.all(buffer[:50] == 2)

    vector_manager.request(Request(cmd="INCR", data=[2], worker_id=7))
    vector_manager.request(Request(cmd="INCR", data=[2], worker_id=8))
    vector_manager.request(Request(cmd="DECR", data=[10], worker_id=9))
    vector_manager.sync()
    assert np.all(buffer[7:10] == np.array([4, 4, -8]))

    vector_manager.terminate()


def test_reconfigure():
    vector_manager = _TestVector()
    buffer = vector_manager.buffer
    vector_manager.reconfigure(10)

    vector_manager.request(Request(cmd="RESET"))
    vector_manager.request(Request(cmd="INCR", data=[1]))
    vector_manager.request(Request(cmd="INCR", data=[1]))
    vector_manager.request(Request(cmd="INCR", data=[1]))
    vector_manager.sync()
    assert np.all(buffer[:10] == 3)

    vector_manager.reconfigure(60)
    vector_manager.request(Request(cmd="RESET"))
    vector_manager.request(Request(cmd="INCR", data=[1]))
    vector_manager.request(Request(cmd="INCR", data=[1]))
    vector_manager.sync()
    assert np.all(buffer[:60] == 2)

    vector_manager.reconfigure(20)
    vector_manager.request(Request(cmd="RESET"))
    vector_manager.request(Request(cmd="DECR", data=[1]))
    vector_manager.sync()
    assert np.all(buffer[:20] == -1)
    assert np.all(buffer[20:60] == 2)
    vector_manager.terminate()


def test_run_loop():
    vector_manager = _TestVector()
    buffer = vector_manager.buffer
    vector_manager.reconfigure(10)

    vector_manager.request(Request(cmd="RESET"))
    vector_manager.request(Request(cmd="RUN_LOOP"))
    time.sleep(0.9)
    vector_manager.request(Request(cmd="STOP_LOOP"))
    vector_manager.sync()
    assert np.all(buffer[:10] == 2)

    vector_manager.request(Request(cmd="RESET"))
    vector_manager.request(Request(cmd="RUN_LOOP"))
    time.sleep(1.6)
    vector_manager.request(Request(cmd="STOP_LOOP"))
    vector_manager.sync()
    assert np.all(buffer[:10] == 4)

    vector_manager.terminate()
