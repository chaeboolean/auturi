import numpy as np
import pytest

import auturi.test.utils as utils
from auturi.executor.config import ActorConfig, TunerConfig
from auturi.executor.ray.executor import RayExecutor
from auturi.test.utils import check_timeout
from auturi.tuner import AuturiTuner


def create_executor_with_mock_tuner(num_envs, config_list):
    class _MockTuner(AuturiTuner):
        """Just generate pre-defined TunerConfigs."""

        def __init__(self):
            self._ctr = 0
            self.configs = config_list

        def next(self):
            next_config = self.configs[self._ctr]
            self._ctr += 1
            return next_config

    tuner = _MockTuner()

    env_fn, policy_fn, model = utils.create_ray_actor_args(num_envs)
    return RayExecutor(env_fn, policy_fn, tuner), model


def test_executor_basic():
    num_envs = 1
    actor_config = ActorConfig(num_envs)
    config = TunerConfig(1, {0: actor_config})
    executor, model = create_executor_with_mock_tuner(num_envs, [config])

    rollouts, metric = executor.run(model, num_collect=3)
    check_timeout(metric.elapsed, timeout=1.5 * 3)
    assert rollouts["obs"].shape == (3, 5, 2)


def test_multiple_actor():
    num_envs = 1
    actor_config = ActorConfig(num_envs)
    config = TunerConfig(3, {0: actor_config, 1: actor_config, 2: actor_config})
    executor, model = create_executor_with_mock_tuner(num_envs, [config])

    rollouts, metric = executor.run(model, num_collect=3)

    check_timeout(metric.elapsed, timeout=1.5 * 3)
    assert rollouts["obs"].shape == (9, 5, 2)


# TODO
@pytest.mark.skip
def test_correctness():
    """Test with real-world simulator, to check its correctness."""
    pass
