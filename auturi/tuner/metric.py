from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional

from auturi.tuner.config import ParallelizationConfig


class AuturiNotEnoughSampleError(Exception):
    pass


@dataclass(frozen=True)
class AuturiMetric:
    """Metrics that an AuturiActor collects for each iteration.

    Args:
        num_traj (int): number of trajectories collected.
        elapsed (float): second.
    """

    num_traj: int
    elapsed: float

    @property
    def throughput(self):
        return self.num_traj / self.elapsed


class MetricRecorder:
    """Records the feedback of specific config given by AuturiVectorActor."""

    def __init__(self, num_iterate: int):
        self.num_iterate = num_iterate
        self.records = defaultdict(list)

    def _insert_record(
        self, _dict: Dict[Any, Any], config: ParallelizationConfig, metric: AuturiMetric
    ):
        _dict[hash(config)].append(metric)

    def _get_record(self, _dict: Dict[Any, Any], config: ParallelizationConfig):
        return _dict[hash(config)]

    def add(self, config: ParallelizationConfig, metric: AuturiMetric) -> None:
        """Adds given metric to recorder."""
        self._insert_record(self.records, config, metric)

    def get(self, config: ParallelizationConfig, stat="mean") -> AuturiMetric:

        records_ = self._get_record(self.records, config)
        if len(records_) < self.num_iterate:
            raise AuturiNotEnoughSampleError("Not enough samples to make decision.")

        total_num_traj = sum([_metric.num_traj for _metric in records_])
        total_time = sum([_metric.elapsed for _metric in records_])

        if stat == "mean":
            res = AuturiMetric(num_traj=total_num_traj, elapsed=total_time)

        elif stat == "median":
            sorted_time = sorted(
                [(_metric.elapsed, _metric.num_traj) for _metric in records_]
            )           
            median_elem = sorted_time[len(sorted_time) // 2]
            res = (sorted_time, AuturiMetric(num_traj=median_elem[1], elapsed=median_elem[0]))

        else:
            raise NotImplementedError

        self.reset(config)
        return res

    def reset(self, config: Optional[ParallelizationConfig] = None) -> None:
        if config is None:
            self.records.clear()

        else:
            self._get_record(self.records, config).clear()
