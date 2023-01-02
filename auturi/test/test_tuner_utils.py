import pytest

from auturi.tuner.config import ActorConfig, ParallelizationConfig
from auturi.tuner.metric import AuturiMetric, AuturiNotEnoughSampleError, MetricRecorder


def test_actor_config():
    assert ActorConfig() == ActorConfig()
    assert ActorConfig() != ActorConfig(num_envs=2, batch_size=2, num_collect=100)

    with pytest.raises(AssertionError):
        ActorConfig(num_envs=2, num_parallel=3, num_collect=100)

    with pytest.raises(AssertionError):
        ActorConfig(num_envs=4, num_parallel=3, num_collect=100)

    with pytest.raises(AssertionError):
        ActorConfig(num_envs=4, num_parallel=2, batch_size=1, num_collect=100)

    with pytest.raises(AssertionError):
        ActorConfig(num_envs=0, num_parallel=0, num_collect=100)


def test_tuner_config():
    config_1 = ParallelizationConfig.create(
        [ActorConfig(num_envs=2, batch_size=2, num_collect=100)]
    )
    config_2 = ParallelizationConfig.create(
        [ActorConfig(num_envs=2, batch_size=2, num_collect=100)]
    )
    config_3 = ParallelizationConfig.create(
        [ActorConfig(num_envs=2, num_parallel=2, num_collect=100)]
    )

    assert config_1 == config_2
    assert config_2 != config_3

    with pytest.raises(Exception):
        config_1[0] = ActorConfig(num_envs=2, batch_size=2)


def test_tuner_method():
    config = ParallelizationConfig.create(
        [
            ActorConfig(
                num_envs=2, num_parallel=2, num_policy=2, batch_size=1, num_collect=100
            ),
            ActorConfig(num_envs=2, batch_size=2, num_collect=50),
        ]
    )
    assert config.num_actors == 2
    assert config.num_policy == 3
    assert config.num_collect == 150

    assert config.compute_index_for_actor("num_envs", 0) == 0
    assert config.compute_index_for_actor("num_envs", 1) == 2


def test_metric():
    metric = AuturiMetric(10, 2)

    with pytest.raises(Exception):
        metric.elapsed = 1

    assert metric.throughput == 5


def test_metric_recorder():
    recorder = MetricRecorder(num_iterate=2)

    config_1 = ParallelizationConfig.create(
        [ActorConfig(num_envs=2, batch_size=2, num_collect=100)]
    )
    config_2 = ParallelizationConfig.create(
        [ActorConfig(num_envs=2, batch_size=2, num_collect=100)]
    )  # equals with config_1
    config_3 = ParallelizationConfig.create(
        [ActorConfig(num_envs=2, num_parallel=2, num_collect=100)]
    )

    recorder.add(config_1, AuturiMetric(10, 1))
    recorder.add(config_3, AuturiMetric(10, 1))

    # Not enough error
    with pytest.raises(AuturiNotEnoughSampleError) as e:
        recorder.get(config_1)

    # Add config_2 (== config_1)
    recorder.add(config_2, AuturiMetric(30, 1))

    res = recorder.get(config_1)
    assert res.throughput == 20

    # already reset
    with pytest.raises(AuturiNotEnoughSampleError) as e:
        res = recorder.get(config_1)
