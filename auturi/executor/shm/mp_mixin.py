"""Define Multiprocessing Mixin class that supports for SHMVectorXXX and SHMXXXProc.

"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch.multiprocessing as _mp

mp = _mp.get_context("spawn")

from auturi.executor.shm.constant import SHMCommand
from auturi.executor.shm.util import _create_buffer_from_sample, set_shm_from_attr, wait
from auturi.logger import get_logger
import time

logger = get_logger()

BUFFER_COMMAND_IDX = 0
BUFFER_DATA_OFFSET = 1


class SHMVectorMixin:
    def __init__(self, max_workers: int, max_data=2):
        self.max_workers = max_workers
        (
            self.__command,
            self._command_buffer,
            self.cmd_attr_dict,
        ) = _create_buffer_from_sample(
            np.array([1] * (1 + max_data), dtype=np.int32), max_workers
        )
        self._command_buffer.fill(SHMCommand.TERM)

    @property
    def num_workers(self):
        pass

    def init_proc(self, worker_id: int, proc_cls: Any, kwargs: Dict[str, Any]):
        assert self._command_buffer[worker_id, BUFFER_COMMAND_IDX] == SHMCommand.TERM
        kwargs["worker_id"] = worker_id
        kwargs["cmd_attr_dict"] = self.cmd_attr_dict
        self._command_buffer[worker_id, BUFFER_COMMAND_IDX] = SHMCommand.INIT
        p = proc_cls(**kwargs)
        logger.debug(self.identifier + f"Create worker={worker_id} pid={p.pid}")
        p.start()
        return p

    @property
    def identifier(self):
        raise NotImplementedError

    def _slice(self, worker_id: Optional[int] = None):
        if worker_id is None:
            return slice(0, self.num_workers)
        else:
            return slice(worker_id, worker_id + 1)

    def _wait_cmd_done(self, worker_id: Optional[int] = None):
        cond_ = lambda: np.all(
            self._command_buffer[self._slice(worker_id), BUFFER_COMMAND_IDX]
            == SHMCommand.CMD_DONE
        )
        msg_fn = (
            lambda: self.identifier
            + f" waiting those=> {np.where(self._command_buffer[self._slice(worker_id), BUFFER_COMMAND_IDX] != SHMCommand.CMD_DONE)[0]}\n"
            + f" they are=> {self._command_buffer[np.where(self._command_buffer[self._slice(worker_id), BUFFER_COMMAND_IDX] != SHMCommand.CMD_DONE)[0], 0]}\n"
        )
        wait(cond_, msg_fn)

    def request(
        self,
        cmd: str,
        worker_id: Optional[int] = None,
        data: List[Any] = [],
        to_wait=True,
    ):
        if to_wait:
            self._wait_cmd_done(worker_id)
        _slice = self._slice(worker_id)
        for idx, data_elem in enumerate(data):
            self._command_buffer[_slice, idx + 1] = data_elem

        self._command_buffer[_slice, BUFFER_COMMAND_IDX] = cmd

    def sync(self, worker_id: Optional[int] = None):
        self._wait_cmd_done(worker_id)

    def teardown_handler(self, worker_id: int, worker: mp.Process):
        self.request(SHMCommand.TERM, worker_id=worker_id)
        self._wait_cmd_done(worker_id)
        worker.join()
        self._command_buffer[worker_id, BUFFER_COMMAND_IDX] = SHMCommand.TERM

    def terminate_all_worker(self, workers: List[mp.Process]):
        self.request(SHMCommand.TERM)
        for worker in workers:
            worker.join()

        self.__command.unlink()


class SHMProcMixin(mp.Process):
    """Mixin class for children processe of SHMVectorMixin.

    This class defines common utility functions about handling command and states via shm buffer.
    """

    def __init__(self, worker_id: int, cmd_attr_dict: Dict[str, Any]):
        """Initialization of SHMProcMixin

        Args:
            worker_id (int): Local id inside actor, differentiating from its siblings.
            polling_ptr (int): Index where to poll of control buffer.
            event (mp.Event): Control wake up and sleep of the process.
        """
        # Does not change during runtime
        self.worker_id = worker_id
        self.cmd_attr_dict = cmd_attr_dict
        self.cmd_handler = {SHMCommand.TERM: self._term_handler}

        super().__init__()

    @property
    def identifier(self):
        raise NotImplementedError

    def _term_handler(self, cmd: int, data_list: List[int]):
        self.reply(cmd)

    def initialize(self) -> None:
        """The entrypoint of child process."""
        self.__command, self._command_buffer = set_shm_from_attr(self.cmd_attr_dict)

    def set_handler_for_command(self) -> None:
        """Set handler function for all possible commands."""
        raise NotImplementedError

    def _wait_cmd(self, cmd: int):
        cond_ = lambda: self._command_buffer[self.worker_id, BUFFER_COMMAND_IDX] == cmd
        wait(cond_, self.identifier + f" waiting for {cmd}")

    def reply(self, cmd: int) -> None:
        self._wait_cmd(cmd)
        self._command_buffer[self.worker_id, BUFFER_COMMAND_IDX] = SHMCommand.CMD_DONE

    def _get_command(self) -> Tuple[int, List[int]]:
        my_line = self._command_buffer[self.worker_id]
        return my_line[BUFFER_COMMAND_IDX], my_line[BUFFER_DATA_OFFSET:]

    def run(self):
        self.initialize()
        self.set_handler_for_command()
        self.reply(SHMCommand.INIT)

        logger.debug(self.identifier + f"Enter Loop")

        ts = time.perf_counter()
        while True:
            cmd, data_list = self._get_command()
            # map handler
            if cmd == SHMCommand.CMD_DONE:
                continue
            else:
                logger.debug(self.identifier + f"Got CMD={cmd}")
                self.cmd_handler[cmd](cmd, data_list)

            if cmd == SHMCommand.TERM:
                logger.debug(self.identifier + f"Terminate")
                break

            if time.perf_counter() - ts > 2:
                logger.info(self.identifier + f"Polling... last cmd={cmd}")
                ts = time.perf_counter()


class SHMVectorLoopMixin(SHMVectorMixin):
    def start_loop(self):
        self.request(SHMCommand.INIT_LOOP)

    def stop_loop(self):
        self.request(SHMCommand.STOP_LOOP, to_wait=False)
        self.sync()


class SHMProcLoopMixin(SHMProcMixin):
    def __init__(self, worker_id: int, cmd_attr_dict: Dict[str, Any]):
        super().__init__(worker_id, cmd_attr_dict)
        self.cmd_handler[SHMCommand.INIT_LOOP] = self._loop_handler

    def _loop_handler(self, cmd: int, data_list: List[int]):
        """Unlike other handler, it watches if STOP_LOOP request have come."""
        self._step_loop_once(is_first=True)
        while True:
            cmd, _ = self._get_command()
            if cmd == SHMCommand.STOP_LOOP:
                if self._check_loop_done():
                    self._stop_loop_handler()
                    break
                else:
                    self._step_loop_once(is_first=False)

            elif cmd == SHMCommand.INIT_LOOP:
                self._step_loop_once(is_first=False)

            else:
                raise RuntimeError(f"{cmd}: FORBIDDEN COMMAND inside RUN_LOOP")

    def _check_loop_done(self) -> bool:
        return True

    def _stop_loop_handler(self):
        self.reply(cmd=SHMCommand.STOP_LOOP)

    def _step_loop_once(self, is_first: bool):
        raise NotImplementedError
