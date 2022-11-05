from auturi.tuner.base_tuner import AuturiTuner
from auturi.tuner.config import TunerConfig


def create_tuner_with_config(num_envs: int, config: TunerConfig):
    """Mock tuner that gives given config.

    Args:
        config_list (List[TunerConfig]): Pre-defined TunerConfig.
    """

    class _MockTuner(AuturiTuner):
        def __init__(self):
            super().__init__(num_envs, num_envs)

        def next(self):
            return config

    return _MockTuner()


__all__ = ["AuturiTuner", "create_tuner_with_config"]
