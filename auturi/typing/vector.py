from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Tuple, TypeVar

T = TypeVar("T")


class VectorMixin(metaclass=ABCMeta):
    """Mixin used for VectorXX class, including AuturiExecutor.

    Provides utility functions to handle one local and multiple remote
    inner components.

    """

    def set_vector_attrs(self):
        self.local_worker = self._create_worker(0)
        self.remote_workers = OrderedDict()
        self.num_workers = 1

    @abstractmethod
    def _create_worker(self, idx: int) -> T:
        """Create worker. If idx is 0, create local worker."""
        raise NotImplementedError

    def _get_worker(self, idx: int) -> T:
        """Get worker. If idx is 0, get local worker."""
        if idx == 0:
            return self.local_worker

        elif idx not in self.remote_workers:
            self.remote_workers[idx] = self._create_worker(idx)

        return self.remote_workers[idx]

    def _working_workers(self) -> Tuple[int, T]:
        """Iterates all current working workers."""
        for worker_id in range(0, self.num_workers):
            yield worker_id, self._get_worker(worker_id)

    def start_loop(self):
        """Setup before running collection loop."""
        pass

    def stop_loop(self):
        """Stop loop, but not terminate entirely."""
        pass

    def terminate(self):
        """Terminate."""
        pass
