from multiprocessing import shared_memory as shm
import numpy as np

from auturi.executor.shm.mp_mixin import SHMVectorMixin, SHMProcMixin, Request



class _TestVector(SHMVectorMixin):
    def __init__(self):
        shape = (10, )
        self.__buffer = shm.SharedMemory(create=True, size=4 * 10)
        self.buffer = np.ndarray(shape, dtype=np.int32, buffer=self.__buffer.buf)
        self.workers = dict()
        
        super().__init__()
    
    
    def reconfigure(self, num_workers: int):
        old_num_wokers = len(self.workers)
        for worker_id in range(old_num_wokers):
            if worker_id >= num_workers:
                self.request_handler.send_request(Request("TERM"), worker_id=worker_id)
                self.workers[worker_id].join()
        
        for worker_id in range(old_num_wokers, num_workers):
            p = self.init_proc(worker_id, _TestProc, {"buf_name": self.__buffer.name})
            self.workers[worker_id] = p
    
    def request(self, req: Request):
        self.request_handler.send_request(req)
    
    
    def terminate(self):
        super().terminate()
        self.__buffer.unlink()

class _TestProc(SHMProcMixin):
    def __init__(
        self, worker_id, queue, master_queue, buf_name: str
    ):
        self.buf_name = buf_name
        super().__init__(worker_id, queue, master_queue)

    def initialize(self) -> None:
        self.__buffer = shm.SharedMemory(self.buf_name)
        self.buffer = np.ndarray((10, ), np.int32, buffer=self.__buffer.buf)
        
    def set_handler_for_command(self) -> None:
        self.cmd_handler["RESET"] = self._reset_handler
        self.cmd_handler["INCR"] = self._incr_handler
        self.cmd_handler["DECR"] = self._decr_handler
        

    def _incr_handler(self, req: Request):
        self.buffer[self.worker_id] += req.data[0]
        self.reply(req.cmd)

    def _decr_handler(self, req: Request):
        self.buffer[self.worker_id] -= req.data[0]
        self.reply(req.cmd)

    def _reset_handler(self, req: Request):
        self.buffer[self.worker_id] = 0
        self.reply(req.cmd)
    
    
def test_basic():
    vector_manager = _TestVector()
    buffer = vector_manager.buffer
    vector_manager.reconfigure(1)
    
    vector_manager.request(Request(cmd="RESET"))
    vector_manager.request(Request(cmd="INCR", data=[2]))
    vector_manager.sync()
    assert buffer[0] == 2
    
    vector_manager.terminate()

def test_multiple_children():
    vector_manager = _TestVector()
    buffer = vector_manager.buffer
    vector_manager.reconfigure(10)
    
    vector_manager.request(Request(cmd="RESET"))
    vector_manager.sync()

    vector_manager.request(Request(cmd="INCR", data=[2]))
    vector_manager.sync()

    vector_manager.sync()
    assert np.all(buffer == 2)


    # vector_manager.request(Request(cmd="INCR", data=[2], worker_id=7))
    # vector_manager.request(Request(cmd="INCR", data=[2], worker_id=8))
    # vector_manager.sync()

    # assert buffer[7] == buffer[8] == 4
    
    # vector_manager.terminate()
