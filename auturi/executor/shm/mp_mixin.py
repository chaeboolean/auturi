"""Define Multiprocessing Mixin class that supports for SHMVectorXXX and SHMXXXProc.

"""
import queue as naive_queue
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch.multiprocessing as mp

from auturi.executor.shm.constant import SHMCommand
from auturi.executor.shm.util import wait
from auturi.logger import get_logger

# mp = _mp.get_context('spawn')


logger = get_logger()


@dataclass
class Request:
    cmd: str
    data: List[Any] = field(default_factory=list)
    worker_id: Optional[int] = None


@dataclass
class Reply:
    worker_id: int
    cmd: str


class RequestHandler:
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.req_queue = mp.Queue()  # master -> worker
        self.rep_queue = mp.Queue()  # worker -> master
        self._called_reqs = naive_queue.Queue()
        self._called_reqs.put(SHMCommand.INIT)

    def send_req(self, req: Request):
        self.req_queue.put(req)
        self._called_reqs.put(req.cmd)
        logger.debug(f"Handler: Send {req} to {self.worker_id}")

    def sync(self):
        """Assert called reqs are all processed by worker."""
        while not self._called_reqs.empty():
            last_req = self._called_reqs.get()
            rep = self.rep_queue.get()
            assert rep.cmd == last_req
            logger.debug(f"Handler(wid={self.worker_id}): Check {rep}")


class SHMVectorMixin:
    def __init__(self):
        self._request_handlers: Dict[int, RequestHandler] = dict()

    def init_proc(self, worker_id: int, proc_cls: Any, kwargs: Dict[str, Any]):
        req_hander = RequestHandler(worker_id)
        kwargs["worker_id"] = worker_id
        kwargs["req_queue"] = req_hander.req_queue
        kwargs["rep_queue"] = req_hander.rep_queue
        self._request_handlers[worker_id] = req_hander

        p = proc_cls(**kwargs)
        p.start()
        return p

    def _working_ids(self, worker_id: Optional[int] = None):
        working_ids = list(self._request_handlers.keys())
        if worker_id is not None:
            working_ids = [worker_id]
        return working_ids

    def request(self, cmd: str, worker_id: Optional[int] = None, data: List[Any] = []):
        req = Request(cmd, data=data, worker_id=worker_id)
        working_ids = self._working_ids(req.worker_id)
        for _worker_id in working_ids:
            self._request_handlers[_worker_id].send_req(req)

    def sync(self, worker_id: Optional[int] = None):
        working_ids = self._working_ids(worker_id)
        for _worker_id in working_ids:
            self._request_handlers[_worker_id].sync()

    def teardown_handler(self, worker_id: int):
        self.request(SHMCommand.TERM, worker_id=worker_id)
        self.sync(worker_id)
        del self._request_handlers[worker_id]


class SHMProcMixin(mp.Process):
    """Mixin class for children processe of SHMVectorMixin.

    This class defines common utility functions about handling command and states via shm buffer.
    """

    def __init__(self, worker_id: int, req_queue: mp.Queue, rep_queue: mp.Queue):
        """Initialization of SHMProcMixin

        Args:
            worker_id (int): Local id inside actor, differentiating from its siblings.
            polling_ptr (int): Index where to poll of control buffer.
            event (mp.Event): Control wake up and sleep of the process.
        """
        # Does not change during runtime
        self.worker_id = worker_id
        self._req_queue = req_queue
        self._rep_queue = rep_queue

        self.cmd_handler = {SHMCommand.TERM: self._term_handler}

        super().__init__()

    def _term_handler(self, req: Request):
        self.reply(req.cmd)

    def initialize(self) -> None:
        """The entrypoint of child process."""
        raise NotImplementedError

    def set_handler_for_command(self) -> None:
        """Set handler function for all possible commands."""
        raise NotImplementedError

    def reply(self, cmd: str) -> None:
        self._rep_queue.put(Reply(worker_id=self.worker_id, cmd=cmd))

    def run(self):
        self.initialize()
        self.set_handler_for_command()
        self.reply(SHMCommand.INIT)

        while True:
            # req = self.queue.get()
            wait(
                lambda: not self._req_queue.empty(),
                f"{self.worker_id} Waitign for queue... ",
            )
            req = self._req_queue.get()

            logger.debug(f"Worker({self.worker_id}): Got {req}")
            self.cmd_handler[req.cmd](req)

            if req.cmd == SHMCommand.TERM:
                logger.debug(f"Worker({self.worker_id}): Termination")
                break


class SHMVectorLoopMixin(SHMVectorMixin):
    def start_loop(self):
        self.request(SHMCommand.INIT_LOOP)

    def stop_loop(self):
        self.request(SHMCommand.STOP_LOOP)
        self.sync()


class SHMProcLoopMixin(SHMProcMixin):
    def __init__(self, worker_id: int, req_queue: mp.Queue, rep_queue: mp.Queue):
        super().__init__(worker_id, req_queue, rep_queue)
        self.cmd_handler[SHMCommand.INIT_LOOP] = self._loop_handler

    def _loop_handler(self, req: Request):
        """Unlike other handler, it watches if STOP_LOOP request have come."""
        self.reply(req.cmd)
        self._step_loop_once(is_first=True)

        while True:
            if not self._req_queue.empty():
                req = self._req_queue.get()
                assert req.cmd == SHMCommand.STOP_LOOP
                self._wait_to_stop()
                break

            else:
                self._step_loop_once(is_first=False)

    def _wait_to_stop(self):
        self.reply(cmd=SHMCommand.STOP_LOOP)

    def _step_loop_once(self, is_first: bool):
        raise NotImplementedError
