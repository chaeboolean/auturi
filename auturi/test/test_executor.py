import numpy as np
import pytest

import auturi.test.utils as utils
from auturi.executor import AuturiExecutor
from auturi.executor.config import ActorConfig, TunerConfig
from auturi.tuner import AuturiTuner


def create_executor_with_mock_tuner(num_envs, config_list):
    class _MockTuner(AuturiTuner):
        """Just generate pre-defined TunerConfigs."""

        def __init__(self):
            self._ctr = 0
            self.configs = config_list

        def next_config(self):
            next_config = self.configs[self._ctr]
            self._ctr += 1
            return next_config

    tuner = _MockTuner()

    test_envs, test_policy, model = utils.create_ray_actor_args(num_envs)
    return AuturiExecutor(test_envs, test_policy, tuner), model


def test_executor_basic():
    num_envs = 1
    config = None
    executor, model = create_executor_with_mock_tuner(num_envs, [config])
    executor.run(num_collect=3)

    executor.run()


@pytest.mark.skip
def test_multiple_actor():
    pass


# TODO
@pytest.mark.skip
def test_correctness():
    """Test with gym simulator, to check its correctness."""
    pass
