from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional

from auturi.tuner.config import TunerConfig


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
        self, _dict: Dict[Any, Any], config: TunerConfig, metric: AuturiMetric
    ):
        _dict[hash(config)].append(metric)

    def _get_record(self, _dict: Dict[Any, Any], config: TunerConfig):
        return _dict[hash(config)]

    def add(self, config: TunerConfig, metric: AuturiMetric) -> None:
        """Adds given metric to recorder."""
        self._insert_record(self.records, config, metric)

    def get(self, config: TunerConfig, stat="mean") -> AuturiMetric:

        records_ = self._get_record(self.records, config)
        if len(records_) < self.num_iterate:
            raise AuturiNotEnoughSampleError("Not enough samples to make decision.")

        total_num_traj = sum([_metric.num_traj for _metric in records_])
        total_time = sum([_metric.elapsed for _metric in records_])

        if stat == "mean":
            res = AuturiMetric(num_traj=total_num_traj, elapsed=total_time)

        else:
            raise NotImplementedError

        self.reset(config)
        return res

    def reset(self, config: Optional[TunerConfig] = None) -> None:
        if config is None:
            self.records.clear()

        else:
            self._get_record(self.records, config).clear()
