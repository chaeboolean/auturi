import pytest

from auturi.tuner.config import ActorConfig, ParallelizationConfig
from auturi.tuner.grid_search import GridSearchTuner
from auturi.tuner.metric import AuturiMetric


def _configs_to_list(tuner, num=None):
    res = []
    try:
        while True:
            res.append(tuner.next())
            tuner.feedback(AuturiMetric(1, 1))
            if num is not None and len(res) >= num:
                raise StopIteration

    except StopIteration:
        return res


def test_tuner_next_feedback_alternative():
    tuner = GridSearchTuner(2, 2, 1, num_iterate=1000)

    tuner.next()

    with pytest.raises(Exception) as e:
        tuner.next()

    tuner.feedback(AuturiMetric(1, 1))
    tuner.next()
    tuner.feedback(AuturiMetric(1, 1))

    with pytest.raises(Exception) as e:
        tuner.feedback(AuturiMetric(1, 1))

    tuner.next()
    with pytest.raises(Exception) as e:
        tuner.next()


def test_grid_search_tuner_record():
    tuner = GridSearchTuner(2, 2, 1, num_iterate=2)

    conf1 = tuner.next()
    tuner.feedback(AuturiMetric(3, 1))
    conf2 = tuner.next()
    tuner.feedback(AuturiMetric(1, 1))

    conf3 = tuner.next()
    tuner.feedback(AuturiMetric(3, 2))
    conf4 = tuner.next()
    tuner.feedback(AuturiMetric(1, 2))

    conf5 = tuner.next()
    tuner.feedback(AuturiMetric(1, 2))

    assert len(tuner.tuning_results) == 2
    assert tuner.tuning_results[hash(conf1)][1].throughput == 2
    assert tuner.tuning_results[hash(conf3)][1].throughput == 1


def test_tuner_iterate():
    tuner = GridSearchTuner(2, 2, 1, num_iterate=2)
    configs = _configs_to_list(tuner, num=4)
    assert configs[0] == configs[1]
    assert configs[2] == configs[3]
    assert configs[0] != configs[2]

    tuner = GridSearchTuner(32, 32, 32, num_iterate=5)
    configs = _configs_to_list(tuner, num=15)
    assert len(set(configs)) == 3
    assert configs[0] == configs[4]
    assert configs[4] != configs[5]


def test_grid_search_single_env():
    tuner = GridSearchTuner(1, 1, 1, num_iterate=1)
    configs = _configs_to_list(tuner, num=None)
    assert len(configs) == 2

    tuner = GridSearchTuner(
        1, 1, 1, num_iterate=1, validator=lambda x: x.policy_device == "cpu"
    )
    configs = _configs_to_list(tuner, num=None)
    assert len(configs) == 1


# def test_grid_search():
#     tuner = GridSearchTuner(4, 4, 4, num_iterate=1)
#     configs = _configs_to_list(tuner, num=None)

#     assert (
#         ParallelizationConfig.create([ActorConfig(num_envs=4, num_parallel=4, batch_size=2, num_collect=100)])
#         in configs
#     )
#     assert (
#         ParallelizationConfig.create([ActorConfig(num_envs=4, num_parallel=4, batch_size=4, num_collect=100)])
#         in configs
#     )
#     assert (
#         ParallelizationConfig.create([ActorConfig(num_envs=4, num_parallel=4, num_policy=4, num_collect=100)])
#         in configs
#     )

#     assert ParallelizationConfig.create([ActorConfig(num_collect=100)] * 4) in configs

#     assert ParallelizationConfig.create([ActorConfig(num_envs=2, num_parallel=2, num_policy=2, num_collect=100)] * 2) in configs
