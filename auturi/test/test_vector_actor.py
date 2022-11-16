import numpy as np
import pytest

import auturi.test.utils as utils
from auturi.executor import create_executor
from auturi.test.utils import check_timeout
from auturi.tuner import create_tuner_with_config
from auturi.tuner.config import ActorConfig, ParallelizationConfig

VECTOR_BACKEND = "shm"


def create_vector_actor(mode, num_envs, tuner_config):
    env_fns = utils.create_env_fns(num_envs)
    model, policy_cls, policy_kwargs = utils.create_policy_args()
    tuner = create_tuner_with_config(num_envs, tuner_config)
    vector_actor = create_executor(env_fns, policy_cls, policy_kwargs, tuner, mode)

    return vector_actor, model


def mock_reconfigure(vector_actor, tuner_config):
    num_envs = tuner_config.num_envs
    tuner = create_tuner_with_config(num_envs, tuner_config)
    vector_actor.tuner = tuner


def test_executor_basic():
    num_envs = 1
    actor_config = ActorConfig(num_envs, num_collect=3)
    config = ParallelizationConfig.create([actor_config])
    vector_actor, model = create_vector_actor(VECTOR_BACKEND, num_envs, config)

    rollouts, metric = vector_actor.run(model)
    check_timeout(metric.elapsed, timeout=1.5 * 3)

    rollouts, metric = vector_actor.run(model)
    check_timeout(metric.elapsed, timeout=1.5 * 3)

    assert rollouts["obs"].shape == (3, 5, 2)

    assert set([elem.flat[0] for elem in rollouts["obs"]]) == set([1001, 1003, 1006])
    assert set([elem.flat[0] for elem in rollouts["action"]]) == set([1, 2, 3])

    vector_actor.terminate()


def test_two_actors():
    num_envs = 4
    actor_config = ActorConfig(num_envs=2, num_parallel=1, batch_size=2, num_collect=4)
    config = ParallelizationConfig.create([actor_config] * 2)
    vector_actor, model = create_vector_actor(VECTOR_BACKEND, num_envs, config)

    rollouts, metric = vector_actor.run(model)
    check_timeout(metric.elapsed, timeout=2 * 2)
    assert rollouts["obs"].shape == (8, 5, 2)
    assert rollouts["action"].shape == (8, 5, 2)

    assert [elem.flat[0] for elem in rollouts["action"]] == [1, 2] * 4

    vector_actor.terminate()


def test_hetero_actors():
    num_envs = 1 + 2 + 3

    # takes 1.5*3 = 4.5 sec
    actor_config1 = ActorConfig(num_envs=1, num_parallel=1, batch_size=1, num_collect=3)

    # takes 2*2 = 4 sec
    actor_config2 = ActorConfig(num_envs=2, num_parallel=1, batch_size=2, num_collect=4)

    # takes 1.5*3 = 4.5 sec
    actor_config3 = ActorConfig(num_envs=3, num_parallel=3, batch_size=3, num_collect=9)
    config = ParallelizationConfig.create([actor_config1, actor_config2, actor_config3])
    vector_actor, model = create_vector_actor(VECTOR_BACKEND, num_envs, config)

    rollouts, metric = vector_actor.run(model)
    check_timeout(metric.elapsed, timeout=4.5)
    assert rollouts["obs"].shape == (16, 5, 2)

    obs_rollouts = np.array([elem.flat[0] for elem in rollouts["obs"]])
    print(obs_rollouts)
    assert np.all(obs_rollouts[:3] == np.array([1001, 1003, 1006]))
    assert np.all(obs_rollouts[3:7] == np.array([2001, 2003, 3001, 3003]))
    assert np.all(obs_rollouts[7:10] == obs_rollouts[10:13] - 1000)
    assert np.all(obs_rollouts[7:10] == obs_rollouts[13:] - 2000)

    vector_actor.terminate()


def test_reconfigure_inside_single_actor():
    num_envs = 8

    # start with num_policy=2
    actor_config = ActorConfig(
        num_envs=8, num_policy=2, num_parallel=4, batch_size=4, num_collect=16
    )
    config = ParallelizationConfig.create([actor_config])
    vector_actor, model = create_vector_actor(VECTOR_BACKEND, num_envs, config)

    rollouts, metric = vector_actor.run(model)
    action_rollouts = [elem.flat[0] for elem in rollouts["action"]]
    utils.check_timeout(metric.elapsed, timeout=4)
    assert set(action_rollouts) == set([1, 2, 11, 12])

    # increase to num_policy=4
    actor_config = ActorConfig(
        num_envs=8, num_policy=4, num_parallel=8, batch_size=2, num_collect=16
    )
    config = ParallelizationConfig.create([actor_config])
    mock_reconfigure(vector_actor, config)

    rollouts, metric = vector_actor.run(model)
    action_rollouts = [elem.flat[0] for elem in rollouts["action"]]
    utils.check_timeout(metric.elapsed, timeout=3)
    assert np.max(action_rollouts) > 1000
    assert np.min(action_rollouts) == 1

    # decrease to num_policy=1
    actor_config = ActorConfig(
        num_envs=2, num_policy=1, num_parallel=2, batch_size=2, num_collect=4
    )
    config = ParallelizationConfig.create([actor_config])
    mock_reconfigure(vector_actor, config)

    rollouts, metric = vector_actor.run(model)
    action_rollouts = [elem.flat[0] for elem in rollouts["action"]]
    utils.check_timeout(metric.elapsed, timeout=3)

    vector_actor.terminate()


def test_reconfigure_actors():
    num_envs = 8

    # num_actors = 1
    actor_config1 = ActorConfig(
        num_envs=8, num_parallel=4, num_policy=2, batch_size=4, num_collect=16
    )
    config1 = ParallelizationConfig.create([actor_config1])
    vector_actor, model = create_vector_actor(VECTOR_BACKEND, num_envs, config1)
    rollouts1, metric = vector_actor.run(model)
    check_timeout(metric.elapsed, timeout=2 * 2)
    action_rollouts = [elem.flat[0] for elem in rollouts1["action"]]
    assert set(action_rollouts[0::2]) == set([1, 11])
    assert set(action_rollouts[1::2]) == set([2, 12])

    # num_actors = 1
    actor_config2 = ActorConfig(
        num_envs=8, num_parallel=8, num_policy=4, batch_size=2, num_collect=16
    )
    config2 = ParallelizationConfig.create([actor_config2])
    mock_reconfigure(vector_actor, config2)
    rollouts2, metric = vector_actor.run(model)
    check_timeout(metric.elapsed, timeout=1.5 * 2)
    action_rollouts = [elem.flat[0] for elem in rollouts2["action"]]
    assert (len(set(action_rollouts))) == 8

    # num_actors = 8
    actor_config3 = ActorConfig(
        num_envs=1, num_parallel=1, num_policy=1, batch_size=1, num_collect=2
    )
    config3 = ParallelizationConfig.create([actor_config3] * 8)
    mock_reconfigure(vector_actor, config3)

    rollouts3, metric = vector_actor.run(model)
    check_timeout(metric.elapsed, timeout=1.5 * 2)

    # num_actors = 2
    actor_config4 = ActorConfig(
        num_envs=4, num_parallel=1, num_policy=1, batch_size=4, num_collect=8
    )
    config4 = ParallelizationConfig.create([actor_config4] * 2)
    mock_reconfigure(vector_actor, config4)
    rollouts4, metric = vector_actor.run(model)
    check_timeout(metric.elapsed, timeout=3 * 2)
    assert np.all(np.array(rollouts3["obs"]) == np.array(rollouts4["obs"]))

    vector_actor.terminate()


# TODO
@pytest.mark.skip
def test_correctness():
    """Test with real-world simulator, to check its correctness."""
    pass
