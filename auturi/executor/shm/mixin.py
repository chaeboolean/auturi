import numpy as np


class SHMVectorMixin:
    """Mixin class that supports for SHMVectorEnv and SHMVectorPolicy.

    This class defines common utility functions about handling command and states via shm buffer.
    """

    def _set_command_buffer(self):
        """Should set attributes "command_buffer", "cmd_enum"."""
        raise NotImplementedError

    def _wait_command_done(self, worker_id: int = -1):
        slice_ = self._get_slice(worker_id)
        while not np.all(self.command_buffer[slice_, 0] == self.cmd_enum.CMD_DONE):
            pass

    def _get_state(self, worker_id: int = -1):
        slice_ = self._get_slice(worker_id)
        return self.command_buffer[slice_, 1]

    def _set_state(self, state, worker_id: int = -1):
        slice_ = self._get_slice(worker_id)
        self.command_buffer[slice_, 1] = state

    def _set_command(
        self, command, worker_id=-1, data1=None, data2=None, set_event=True
    ):
        slice_ = self._get_slice(worker_id)
        if data1 is not None:
            self.command_buffer[slice_, 2].fill(data1)
        if data2 is not None:
            self.command_buffer[slice_, 3].fill(data2)

        self.command_buffer[slice_, 0].fill(command)

        if set_event:
            self._set_event(worker_id)

    def _set_event(self, worker_id: int = -1):
        wids = (
            [worker_id]
            if worker_id >= 0
            else [wid for wid, _ in self._working_workers()]
        )

        for wid in wids:
            assert not self.events[wid].is_set()
            self.events[wid].set()

    def _get_slice(self, worker_id: int = -1):
        if worker_id < 0:
            return slice(0, self.num_workers)

        else:
            return slice(worker_id, worker_id + 1)


class SHMProcMixin:
    """Mixin class that supports inner process of SHMVectorEnv and SHMVectorPolicy.

    This class defines common utility functions about handling command and states via shm buffer.
    """

    def initialize(self):
        """Should set attributes "command_buffer", "cmd_enum", "state_enum"."""
        raise NotImplementedError

    def _set_cmd_done(self):
        self.command_buffer[self.worker_id, 0] = self.cmd_enum.CMD_DONE

    def _set_state(self, state):
        self.command_buffer[self.worker_id, 1] = state

    def _assert_state(self, state):
        assert self.command_buffer[self.worker_id, 1] == state

    def teardown(self):
        pass

    def main(self):
        """Main function."""
        # run while loop
        while True:
            self.event.wait()
            last_cmd = self._run()

            if last_cmd == self.cmd_enum.TERMINATE:
                self.teardown()
                break
            else:
                self.event.clear()
