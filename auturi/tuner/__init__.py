from auturi.tuner.base_tuner import AuturiTuner
from auturi.tuner.config import ActorConfig, ParallelizationConfig
from auturi.tuner.metric import AuturiMetric


def create_tuner_with_config(num_envs: int, num_iterate: int, config: ParallelizationConfig):
    """Mock tuner that gives given config.

    Args:
        config_list (List[ParallelizationConfig]): Pre-defined ParallelizationConfig.
    """

    class _MockTuner(AuturiTuner):
        def __init__(self):
            self.cnt = 0
            self.tuning_results = None 
            super().__init__(num_envs, num_envs, config.num_collect, num_iterate)

        @property
        def config(self):
            return config

        def _generate_next(self):
            if self.cnt == 0:
                self.cnt += 1
                return config
            else:
                raise StopIteration()

        def _update_tuner(self, config, res):
            self.tuning_results = res

    return _MockTuner()


__all__ = [
    "AuturiTuner",
    "ParallelizationConfig",
    "AuturiMetric",
    "ActorConfig",
    "create_tuner_with_config",
]
