from auturi.tuner.base_tuner import AuturiTuner
from auturi.tuner.config import ParallelizationConfig


def create_tuner_with_config(num_envs: int, config: ParallelizationConfig):
    """Mock tuner that gives given config.

    Args:
        config_list (List[ParallelizationConfig]): Pre-defined ParallelizationConfig.
    """

    class _MockTuner(AuturiTuner):
        def __init__(self):
            super().__init__(num_envs, num_envs, config.num_collect)

        def next(self):
            return config

    return _MockTuner()


__all__ = ["AuturiTuner", "create_tuner_with_config"]
