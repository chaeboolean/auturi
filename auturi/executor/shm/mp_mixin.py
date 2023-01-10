"""Define Multiprocessing Mixin class that supports for SHMVectorXXX and SHMXXXProc.

"""
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch.multiprocessing as _mp

mp = _mp.get_context("spawn")

import time

from auturi.executor.shm.constant import SHMCommand
from auturi.executor.shm.util import _create_buffer_from_sample, set_shm_from_attr, wait
from auturi.executor.vector_utils import VectorMixin
from auturi.logger import get_logger
from auturi.common.chrome_profiler import create_tracer

BUFFER_COMMAND_IDX = 0
BUFFER_DATA_OFFSET = 1


class SHMVectorMixin(VectorMixin, metaclass=ABCMeta):
    """Mixin class that supports handling multiple children processes (SHMProcMixin)

    This class exploits python shared memory library to control its children.
    The control buffer is like a numpy array, formatted as [command, data1, data2, ...].

    """

    def __init__(self, max_workers: int, max_data=2):
        self.max_workers = max_workers
        (
            self.__command,
            self._command_buffer,
            self.cmd_attr_dict,
        ) = _create_buffer_from_sample(
            np.array([[1] * (1 + max_data)], dtype=np.int32), max_workers
        )
        self._command_buffer.fill(SHMCommand.TERM)
        self._logger = get_logger(self.proc_name)

        super().__init__()

    @property
    @abstractmethod
    def proc_name(self) -> str:
        """Identifier for logging."""
        raise NotImplementedError

    def init_proc(
        self, worker_id: int, proc_cls: Any, kwargs: Dict[str, Any]
    ) -> mp.Process:
        """Spawn child process and initiate it.

        Args:
            worker_id (int): Id of child process to spawn.
            proc_cls (Any): Class of child process to spawn
            kwargs (Dict[str, Any]): Kwargs used for spawning the child process.

        """
        assert self._command_buffer[worker_id, BUFFER_COMMAND_IDX] == SHMCommand.TERM

        kwargs["worker_id"] = worker_id
        kwargs["cmd_attr_dict"] = self.cmd_attr_dict
        p = proc_cls(**kwargs)
        self._logger.debug(f"Create worker={worker_id} pid={p.pid}")
        self._command_buffer[worker_id, BUFFER_COMMAND_IDX] = SHMCommand.INIT

        p.start()
        return p

    def _slice(self, worker_id: Optional[int] = None) -> slice:
        if worker_id is None:
            return slice(0, self.num_workers)
        else:
            return slice(worker_id, worker_id + 1)

    def _wait_cmd_done(self, worker_id: Optional[int] = None) -> None:
        """Wait until the child finishes executing given command."""
        cond_ = lambda: np.all(
            self._command_buffer[self._slice(worker_id), BUFFER_COMMAND_IDX]
            == SHMCommand.CMD_DONE
        )
        timeout_fn = lambda: self._logger.warning(
            f" waiting those=> {np.where(self._command_buffer[self._slice(worker_id), BUFFER_COMMAND_IDX] != SHMCommand.CMD_DONE)[0]}\n"
            + f" they are=> {self._command_buffer[np.where(self._command_buffer[self._slice(worker_id), BUFFER_COMMAND_IDX] != SHMCommand.CMD_DONE)[0], 0]}\n"
        )
        wait(cond_, timeout_fn)

    def request(
        self,
        cmd: str,
        worker_id: Optional[int] = None,
        data: List[Any] = [],
        to_wait=True,
    ) -> None:
        """Set command to its specific child or all children."""
        if to_wait:
            self._wait_cmd_done(worker_id)
        _slice = self._slice(worker_id)

        # write command data to command buffer
        for idx, data_elem in enumerate(data):
            self._command_buffer[_slice, idx + 1] = data_elem

        self._command_buffer[_slice, BUFFER_COMMAND_IDX] = cmd

    def sync(self, worker_id: Optional[int] = None) -> None:
        self._wait_cmd_done(worker_id)

    def _terminate_worker(self, worker_id: int, worker: mp.Process) -> None:
        self._logger.info(f"Join worker={worker_id} pid={worker.pid}")
        self.request(SHMCommand.TERM, worker_id=worker_id)
        self._wait_cmd_done(worker_id)
        worker.join()
        self._command_buffer[worker_id, BUFFER_COMMAND_IDX] = SHMCommand.TERM
        self._logger.info(f"Join worker={worker_id} pid={worker.pid}")

    def terminate(self) -> None:
        print("\n\nWHO CALL THIS???? ")
        self.request(SHMCommand.TERM)
        for _, worker in self.workers():
            worker.join()

        self.__command.unlink()


class SHMProcMixin(mp.Process, metaclass=ABCMeta):
    """Mixin class for child process of SHMVectorMixin.

    This class defines common utility functions about handling command and states via shm buffer.
    """

    def __init__(self, worker_id: int, cmd_attr_dict: Dict[str, Any]):
        """Initialization of SHMProcMixin

        Args:
            worker_id (int): Local id inside actor, differentiating from its siblings.
            cmd_attr_dict (Dict[str, Any]): Basic attributes of control buffer, created by its parent.
        """
        # Does not change during runtime
        self.worker_id = worker_id
        self.cmd_attr_dict = cmd_attr_dict
        self.cmd_handler = {SHMCommand.TERM: self._term_handler}
        self._logger = get_logger(self.proc_name)

        super().__init__()

    @property
    @abstractmethod
    def proc_name(self) -> str:
        """Identifier for logging."""
        raise NotImplementedError

    def _term_handler(self, cmd: int, data_list: List[int]):
        self.reply(cmd)

    def initialize(self) -> None:
        """The entrypoint of child process."""
        self.__command, self._command_buffer = set_shm_from_attr(self.cmd_attr_dict)

    @abstractmethod
    def set_command_handlers(self) -> None:
        """Set handler function for all possible commands."""
        raise NotImplementedError

    def reply(self, cmd: int) -> None:
        """Set CMD_DONE sign to command buffer to notify its parent."""
        # assert that currently written command equals to given command.
        cond_ = lambda: self._command_buffer[self.worker_id, BUFFER_COMMAND_IDX] == cmd
        wait(cond_, lambda: self._logger.warning(f" waiting for {cmd}"))

        # set CMD_DONE
        self._command_buffer[self.worker_id, BUFFER_COMMAND_IDX] = SHMCommand.CMD_DONE

    def _get_command(self) -> Tuple[int, List[int]]:
        """Return command and following data list, read from command buffer."""
        my_line = self._command_buffer[self.worker_id]
        return my_line[BUFFER_COMMAND_IDX], my_line[BUFFER_DATA_OFFSET:]

    def run(self):
        self.initialize()
        self.set_command_handlers()
        self.reply(SHMCommand.INIT)

        self._logger.debug("Enter Loop")
        ts = time.perf_counter()
        while True:
            cmd, data_list = self._get_command()
            # map handler
            if cmd == SHMCommand.CMD_DONE:
                continue
            else:
                self._logger.debug(f"Got CMD={cmd}")
                self.cmd_handler[cmd](cmd, data_list)

            # terminate
            if cmd == SHMCommand.TERM:
                self._logger.debug("Terminate")
                break

            # debugging
            if time.perf_counter() - ts > 2:
                self._logger.info(f"Polling... last cmd={cmd}")
                ts = time.perf_counter()


class SHMVectorLoopMixin(SHMVectorMixin, metaclass=ABCMeta):
    def start_loop(self):
        self.request(SHMCommand.INIT_LOOP)

    def stop_loop(self):
        self.request(SHMCommand.STOP_LOOP, to_wait=False)
        self.sync()


class SHMProcLoopMixin(SHMProcMixin, metaclass=ABCMeta):
    def __init__(self, worker_id: int, cmd_attr_dict: Dict[str, Any]):
        super().__init__(worker_id, cmd_attr_dict)
        self.cmd_handler[SHMCommand.INIT_LOOP] = self._loop_handler
        self._loop_start = -1

    def initialize(self):
        super().initialize()
        self._trace_wrapper = create_tracer(self.proc_name)

    def _term_handler(self, cmd: int, data_list: List[int]):
        self.reply(cmd)


    def _loop_handler(self, cmd: int, data_list: List[int]):
        """Execute this function from when INIT_LOOP is set until STOP_LOOP is set."""
        self._trace_wrapper.start_loop()
        self._step_loop_once(is_first=True)
        while True:
            cmd, _ = self._get_command()
            if cmd == SHMCommand.STOP_LOOP:
                # wait until _check_loop_done is True
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
        """Check whether the loop can be terminated or not."""
        return True

    def _stop_loop_handler(self) -> None:
        """Handler function called when SHMCommand.STOP_LOOP is set."""
        self._trace_wrapper.stop_loop()
        self._trace_wrapper.dump_stats()
        self.reply(cmd=SHMCommand.STOP_LOOP)

    @abstractmethod
    def _step_loop_once(self, is_first: bool) -> None:
        """Handler function called when SHMCommand.INIT_LOOP is set"""
        raise NotImplementedError
