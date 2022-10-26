import numpy as np
import pytest

import auturi.test.utils as utils
from auturi.executor.actor import AuturiActor
from auturi.executor.config import ActorConfig
from auturi.test.utils import check_timeout


def create_actor(num_envs):
    test_env_fn, test_policy_fn, model = utils.create_ray_actor_args(num_envs)
    actor = AuturiActor(test_env_fn, test_policy_fn)
    return actor, model


def run_actor(actor, num_collect):
    trajectories, metric = actor.run(num_collect=num_collect)
    action_list = [obs_.flat[0] for obs_ in trajectories["action"]]

    return np.array(action_list), metric.elapsed


def test_single_env_single_policy():
    actor, model = create_actor(num_envs=1)
    actor_config = ActorConfig(num_envs=1, num_policy=1, num_parallel=1, batch_size=1)
    actor.reconfigure(actor_config, model)
    action_list, elapsed = run_actor(actor, num_collect=3)

    assert np.all(action_list == np.array([1, 2, 3]))

    # one step takes 0.5(env[0].step) + 1 (policy)
    check_timeout(elapsed, timeout=1.5 * 3)


def test_serial_env_single_policy():
    actor, model = create_actor(num_envs=4)
    actor_config = ActorConfig(num_envs=2, num_policy=1, num_parallel=1, batch_size=2)
    actor.reconfigure(actor_config, model)

    action_list, elapsed = run_actor(actor, num_collect=6)
    print(action_list, elapsed)

    assert np.all(action_list == np.array([1, 2, 3, 1, 2, 3]))

    # one step takes 0.5(env[0].step) + 0.5(env[1].step)+ 1 (policy)
    check_timeout(elapsed, timeout=2 * 3)


def test_parallel_env_single_policy():
    actor, model = create_actor(num_envs=4)
    actor_config = ActorConfig(num_envs=3, num_policy=1, num_parallel=3, batch_size=3)
    actor.reconfigure(actor_config, model)

    action_list, elapsed = run_actor(actor, num_collect=9)

    assert np.all(action_list == np.array([1, 2, 3, 1, 2, 3, 1, 2, 3]))

    # one step takes 0.5(max(env.step)) + 1 (policy)
    check_timeout(elapsed, timeout=1.5 * 3)


def test_parallel_and_serial_env_single_policy():
    actor, model = create_actor(num_envs=4)

    # Case 1) batch size = 4
    actor_config = ActorConfig(num_envs=4, num_policy=1, num_parallel=2, batch_size=4)
    actor.reconfigure(actor_config, model)
    action_list, elapsed = run_actor(actor, num_collect=4 * 3)

    assert np.all(action_list == np.array([1, 2, 3] * 4))
    # one step takes 0.5(env[0].step) + 0.5(env[1].step)+ 1 (policy)
    check_timeout(elapsed, timeout=2 * 3)

    # Case 2) batch size = 2
    actor_config = ActorConfig(num_envs=4, num_policy=1, num_parallel=2, batch_size=2)
    actor.reconfigure(actor_config, model)
    action_list, elapsed = run_actor(actor, num_collect=2 * 4)

    env_action = action_list[:2], action_list[2:4], action_list[4:6], action_list[6:]
    assert np.all(env_action[0] == env_action[1])
    assert np.all(env_action[2] == env_action[3])
    assert set(env_action[0]).intersection(set(env_action[2])) == set()
    assert set(env_action[0]).union(set(env_action[2])) == set([1, 2, 3, 4])
    assert set(env_action[0]) == set([1, 3]) or set(env_action[0]) == set([2, 4])

    # action time: [0, 1], [1, 2], [2, 3], [3, 4]
    check_timeout(elapsed, timeout=5)


def test_parallel_env_async():
    actor, model = create_actor(num_envs=4)

    actor_config = ActorConfig(num_envs=4, num_policy=1, num_parallel=4, batch_size=2)
    actor.reconfigure(actor_config, model)
    action_list, elapsed = run_actor(actor, num_collect=2 * 4)
    check_timeout(elapsed, timeout=4.5)


def test_parallel_env_multiple_policy():
    actor, model = create_actor(num_envs=4)

    actor_config = ActorConfig(num_envs=4, num_policy=1, num_parallel=4, batch_size=1)
    actor.reconfigure(actor_config, model)
    action_list, elapsed = run_actor(actor, num_collect=4)

    assert set(action_list) == set([1, 2, 3, 4])
    check_timeout(elapsed, timeout=4.5)

    actor_config = ActorConfig(num_envs=4, num_policy=2, num_parallel=4, batch_size=1)
    actor.reconfigure(actor_config, model)
    action_list, elapsed = run_actor(actor, num_collect=4)

    assert set(action_list) == set([1, 2, 11, 12])
    check_timeout(elapsed, timeout=2.5)
