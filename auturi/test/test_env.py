import numpy as np
import pytest

import auturi.test.utils as utils
from auturi.executor.environment import AuturiSerialEnv
from auturi.executor.ray import RayParallelEnv


def _test_env(test_env, num_envs, num_steps, timeout):
    """Test elapsed time and step sequence."""

    actions = np.stack([test_env.action_space.sample()] * num_envs)
    actions.fill(1)
    action_artifacts = np.copy(actions)

    test_env.seed(-1)
    test_env.reset()

    with utils.Timeout(min_sec=timeout - 0.5, max_sec=timeout + 0.5):
        for step in range(num_steps):
            obs = test_env.step([actions, [action_artifacts]])

    agg_obs = test_env.aggregate_rollouts()["obs"]
    agg_obs = np.array([obs_.flat[0] for obs_ in agg_obs])
    return agg_obs


def create_serial_env(num_envs):
    env_fns = utils.create_env_fns(num_envs)
    test_envs = AuturiSerialEnv(0, env_fns)
    test_envs.set_working_env(start_idx=0, num_envs=num_envs)
    return test_envs


def test_serial_env():
    agg_obs = _test_env(create_serial_env(1), num_envs=1, num_steps=3, timeout=0.5 * 3)
    assert np.all(agg_obs == np.array([1001, 1002, 1003]))

    agg_obs = _test_env(
        create_serial_env(2), num_envs=2, num_steps=3, timeout=(0.5 + 0.5) * 3
    )
    assert np.all(agg_obs == np.array([1001, 1002, 1003, 2001, 2002, 2003]))

    agg_obs = _test_env(
        create_serial_env(3), num_envs=3, num_steps=3, timeout=(0.5 + 0.5 + 0.5) * 3
    )
    assert np.all(
        agg_obs == np.array([1001, 1002, 1003, 2001, 2002, 2003, 3001, 3002, 3003])
    )


def test_reconfigure_serial_env():
    serial_env = create_serial_env(3)
    agg_obs = _test_env(serial_env, num_envs=3, num_steps=3, timeout=(1.5) * 3)

    serial_env.set_working_env(0, 1)
    agg_obs = _test_env(serial_env, num_envs=1, num_steps=3, timeout=0.5 * 3)

    serial_env.set_working_env(0, 2)
    agg_obs = _test_env(serial_env, num_envs=2, num_steps=3, timeout=1 * 3)

    serial_env.terminate()


def test_env_aggregate():
    serial_env = create_serial_env(2)
    agg_obs = _test_env(serial_env, num_envs=2, num_steps=3, timeout=1 * 3)
    assert len(agg_obs) == 6

    # Reconfigure
    serial_env.set_working_env(0, 1)
    agg_obs = _test_env(serial_env, num_envs=1, num_steps=3, timeout=0.5 * 3)
    assert len(agg_obs) == 3


def mock_reconfigure(test_env, num_envs, num_parallel):
    class _MockConfig:
        def __init__(self, num_envs, num_parallel):
            self.num_envs = num_envs
            self.num_parallel = num_parallel
            self.batch_size = -1  # dummy

    test_env.reconfigure(_MockConfig(num_envs, num_parallel))


def create_ray_env(num_envs, num_parallel):
    env_fns = utils.create_env_fns(num_envs)
    test_envs = RayParallelEnv(env_fns)
    mock_reconfigure(test_envs, num_envs, num_parallel)
    return test_envs


def test_ray_basic():
    ray_env = create_ray_env(1, 1)
    assert ray_env.num_envs == 1
    assert ray_env.num_workers == 1
    _test_env(ray_env, num_envs=1, num_steps=3, timeout=0.5 * 3)


def test_ray_fully_parallel():
    ray_env = create_ray_env(3, 3)
    assert ray_env.num_envs == 3
    assert ray_env.num_workers == 3
    agg_obs = _test_env(ray_env, num_envs=3, num_steps=3, timeout=0.5 * 3)


def test_ray_serial():
    ray_env = create_ray_env(2, 1)
    assert ray_env.num_envs == 2
    assert ray_env.num_workers == 1
    agg_obs = _test_env(ray_env, num_envs=2, num_steps=3, timeout=(0.5 + 0.5) * 3)


def test_ray_combination():
    ray_env = create_ray_env(4, 2)
    assert ray_env.num_envs == 4
    assert ray_env.num_workers == 2
    agg_obs = _test_env(ray_env, num_envs=4, num_steps=3, timeout=(0.5 + 0.5) * 3)
    print(agg_obs)


def test_ray_reconfigure():
    ray_env = create_ray_env(4, 1)

    mock_reconfigure(ray_env, num_envs=1, num_parallel=1)
    assert ray_env.num_envs == 1
    assert ray_env.num_workers == 1
    _test_env(ray_env, num_envs=1, num_steps=3, timeout=0.5 * 3)

    mock_reconfigure(ray_env, num_envs=2, num_parallel=2)
    assert ray_env.num_envs == 2
    assert ray_env.num_workers == 2
    _test_env(ray_env, num_envs=2, num_steps=3, timeout=0.5 * 3)

    mock_reconfigure(ray_env, num_envs=2, num_parallel=1)
    assert ray_env.num_envs == 2
    assert ray_env.num_workers == 1
    _test_env(ray_env, num_envs=2, num_steps=3, timeout=(0.5 + 0.5) * 3)


def test_ray_rollouts():
    ray_env = create_ray_env(4, 1)
    mock_reconfigure(ray_env, num_envs=4, num_parallel=1)
    agg_obs = _test_env(
        ray_env, num_envs=4, num_steps=2, timeout=(0.5 + 0.5 + 0.5 + 0.5) * 2
    )
    assert len(agg_obs) == 8

    mock_reconfigure(ray_env, num_envs=4, num_parallel=2)
    agg_obs = _test_env(ray_env, num_envs=4, num_steps=2, timeout=(0.5 + 0.5) * 2)
    assert len(agg_obs) == 8

    mock_reconfigure(ray_env, num_envs=4, num_parallel=4)
    agg_obs = _test_env(ray_env, num_envs=4, num_steps=2, timeout=(0.5) * 2)
    assert len(agg_obs) == 8
