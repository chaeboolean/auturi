"""Vector utiility class and functions."""

from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Tuple, TypeVar

import numpy as np

from auturi.tuner.config import ParallelizationConfig

T = TypeVar("T")


class VectorMixin(metaclass=ABCMeta):
    """Mixin used for VectorXX class, including AuturiExecutor.

    Provides utility functions to handle one local and multiple remote
    inner components.

    """

    def __init__(self):
        self._workers = OrderedDict()

    @property
    def num_workers(self):
        return len(self._workers)

    def reconfigure_workers(self, new_num_workers: int, config: ParallelizationConfig):

        old_workers = self._workers
        new_workers = OrderedDict()

        for worker_id in range(new_num_workers):
            if worker_id in old_workers:
                worker = old_workers.pop(worker_id)
            else:
                worker = self._create_worker(worker_id)

            self._reconfigure_worker(worker_id, worker, config)
            new_workers[worker_id] = worker

        for worker_id in old_workers.keys():
            worker = old_workers.pop(worker_id)
            self._terminate_worker(worker_id, worker)

        assert len(old_workers) == 0
        assert len(new_workers) == new_num_workers

        self._workers = new_workers

    @abstractmethod
    def _create_worker(self, worker_id: int) -> T:
        """Create worker."""
        raise NotImplementedError

    @abstractmethod
    def _reconfigure_worker(
        self, worker_id: int, worker: T, config: ParallelizationConfig
    ):
        """Create worker."""
        raise NotImplementedError

    @abstractmethod
    def _terminate_worker(self, worker_id: int, worker: T) -> None:
        """Create worker."""
        raise NotImplementedError

    def get_worker(self, worker_id: int) -> T:
        """Get worker by id."""
        return self._workers[worker_id]

    def workers(self) -> Tuple[int, T]:
        """Iterates all workers."""
        for worker_id, worker in self._workers.items():
            yield worker_id, worker


def aggregate_partial(partial: List[Dict[str, Any]], to_stack=False, to_extend=False):
    """Helper function that aggregates worker's remainings, filtering out empty dict.

    Assume that all partial elements have same keys.

    Args:
        - partial (List[Dict[str, Any]]): Partial rollouts.
        - to_stack (bool): Whether to add additional dimension on element.
        - to_extend (bool): When element of partial is Dict[str, List]

    """
    concat_fn = np.stack if to_stack else np.concatenate
    if to_extend:
        append_fn = lambda list_, elem_: list_.extend(elem_)
    else:
        append_fn = lambda list_, elem_: list_.append(elem_)

    agg_dict = defaultdict(list)
    dones = list(filter(lambda elem: len(elem) > 0, partial))

    if len(dones) > 0:
        keys = list(dones[0].keys())
        for key in keys:
            list_ = []
            for done in dones:
                append_fn(list_, done[key])
                agg_dict[key] = concat_fn(list_)

    return agg_dict
