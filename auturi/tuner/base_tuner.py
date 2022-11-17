from abc import ABCMeta, abstractmethod
from enum import Enum

from auturi.tuner.config import ParallelizationConfig
from auturi.tuner.metric import AuturiMetric, AuturiNotEnoughSampleError, MetricRecorder


class TunerState:
    """Indicates state of AuturiTuner.

    Ensure that AuturiTuner.next() and AuturiTuner.feedback() is called alternatively.
    """

    class _TunerState(Enum):
        STOPPED = 1
        SEND_CONFIG = 2  # Testing with generated configuration
        RECV_FEEDBACK = 3  # Tuner got result from AuturiVectorActor
        STABLE = 4  # found optimal configuration and runnign with it.

    def __init__(self):
        self.value = self._TunerState.STOPPED

    def send_config(self):
        assert self.value in [self._TunerState.RECV_FEEDBACK, self._TunerState.STOPPED]
        self.value = self._TunerState.SEND_CONFIG

    def recv_feedback(self):
        assert self.value == self._TunerState.SEND_CONFIG
        self.value = self._TunerState.RECV_FEEDBACK


class AuturiTuner(metaclass=ABCMeta):
    def __init__(
        self,
        min_num_env: int,
        max_num_env: int,
        num_collect: int,
        num_iterate: int = 10,
    ):
        """AuturiTuner Initialization.

        Args:
            min_num_env (int): Minimum number of total environments.
            max_num_env (int): Maximum number of total environments.
            num_iterate (int): Keeps the same config for num_iterate times to get consistent throughput.
            num_collect (int): Number of trajectories to collect.

        """
        assert min_num_env <= max_num_env

        self.min_num_env = min_num_env
        self.max_num_env = max_num_env
        self.num_collect = num_collect
        self.num_iterate = num_iterate

        self.recorder = MetricRecorder(num_iterate)
        self.state = TunerState()

        self.curr_config = self._generate_next()

    def next(self) -> ParallelizationConfig:
        """Return next TunerConfig to try."""
        self.state.send_config()
        return self.curr_config

    def feedback(self, metric: AuturiMetric):
        """Record the performance metric of last config."""
        self.recorder.add(self.curr_config, metric)
        try:
            mean_metric = self.recorder.get(self.curr_config, "median")
            self._update_tuner(self.curr_config, mean_metric)
            new_config = self._generate_next()
            self._validate(new_config)
            self.curr_config = new_config

        except AuturiNotEnoughSampleError as e:
            pass

        self.state.recv_feedback()

    @abstractmethod
    def _generate_next(self) -> ParallelizationConfig:
        """Return next TunerConfig to test."""
        pass

    @abstractmethod
    def _update_tuner(
        self, config: ParallelizationConfig, mean_metric: AuturiMetric
    ) -> None:
        """Update Tuner with (config, mean_metric) tuple."""
        pass

    def _validate(self, config: ParallelizationConfig) -> bool:
        assert config.num_envs >= self.min_num_env
        assert config.num_envs <= self.max_num_env
        assert config.num_collect == self.num_collect
