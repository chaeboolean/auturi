"""Vector utiility class and functions."""

from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict
from typing import Tuple, TypeVar

import numpy as np

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

    def _existing_workers(self) -> Tuple[int, T]:
        """Iterates all current alive workers."""
        yield 0, self._get_worker(0)
        for worker_id, worker in self.remote_workers.items():
            yield worker_id, worker

    def start_loop(self):
        """Setup before running collection loop."""
        pass

    def stop_loop(self):
        """Stop loop, but not terminate entirely."""
        pass

    def terminate(self):
        """Terminate."""
        pass


def aggregate_partial(partial, to_stack=False, to_extend=False):
    """Helper function that aggregates worker's remainings, filtering out empty dict.

    Assume that all partial elements have same keys.

    """
    # if already_agg=True, just append & concat
    # if already_agg=False, unpack first and then append

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
                # print(f"**** ", len(list_), " Try to append, ... ", key)

                append_fn(list_, done[key])
                # print(f"**** ", len(list_))

            try:
                agg_dict[key] = concat_fn(list_)

            except:
                print(
                    f"Given Args = {len(partial)}, {partial[0]['obs'].shape}, {done[key].shape}"
                )

                li = []
                append_fn(li, 3)

                print(f"test--> {li}")

                raise RuntimeError(f"Error!!! {key}, {len(list_)}")

    return agg_dict
