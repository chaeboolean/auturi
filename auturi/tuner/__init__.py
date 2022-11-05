from typing import List

from auturi.tuner.config import TunerConfig


class AuturiTuner:
    def __init__(self, min_num_env: int, max_num_env: int):
        """AuturiTuner Initialization.

        Args:
            min_num_env (int): Minimum number of total environments.
            max_num_env (int): Maximum number of total environments.
        """
        assert min_num_env <= max_num_env
        self.min_num_env = min_num_env
        self.max_num_env = max_num_env

        self.mode = "tuning"  # ["tuning" or "finishing"]
        self.recorder = None

    def next(self):
        pass


def create_tuner_with_config(num_envs: int, config_list: List[TunerConfig]):
    """Mock tuner that gives given config.

    Args:
        config_list (List[TunerConfig]): List of pre-defined TunerConfigs.
    """

    class _MockTuner(AuturiTuner):
        def __init__(self):
            self._ctr = 0
            self.configs = config_list
            super().__init__(num_envs, num_envs)

        def next(self):
            next_config = self.configs[self._ctr]
            self._ctr += 1
            return next_config

    return _MockTuner()
