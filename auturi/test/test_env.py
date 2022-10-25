import numpy as np
import pytest

import auturi.test.utils as utils
from auturi.typing.environment import AuturiSerialEnv
from auturi.vector.ray_backend import RayParallelEnv


def _test_env(test_env, num_envs, num_steps, timeout):
    """Test elapsed time and step sequence."""

    actions = np.stack([test_env.action_space.sample()] * num_envs)
    actions.fill(1)
    action_artifacts = np.copy(actions)

    obs_list = []
    test_env.seed(-1)

    obs_list.append(test_env.reset())
    with utils.Timeout(min_sec=timeout - 0.5, max_sec=timeout + 0.5):
        for step in range(num_steps):
            obs = test_env.step([actions, [action_artifacts]])
            obs_list.append(obs)

    for step, obs in enumerate(obs_list):
        assert len(obs) == num_envs, print(f"shape == {obs.shape}")
        for env_id in range(num_envs):
            assert np.all(obs[env_id] == env_id * 10 + step)

    return obs_list


def create_serial_env(num_envs):
    env_fns = utils.create_env_fns(num_envs)
    test_envs = AuturiSerialEnv(0, env_fns)
    test_envs.set_working_env(start_idx=0, num_envs=num_envs)
    return test_envs


def test_serial_env():
    _test_env(create_serial_env(1), num_envs=1, num_steps=3, timeout=1 * 3)
    _test_env(create_serial_env(2), num_envs=2, num_steps=3, timeout=(1 + 2) * 3)
    _test_env(create_serial_env(3), num_envs=3, num_steps=3, timeout=(1 + 2 + 3) * 3)


def test_reconfigure_serial_env():
    serial_env = create_serial_env(3)
    _test_env(serial_env, num_envs=3, num_steps=3, timeout=(1 + 2 + 3) * 3)

    serial_env.set_working_env(0, 1)
    _test_env(serial_env, num_envs=1, num_steps=3, timeout=1 * 3)

    serial_env.set_working_env(0, 2)
    _test_env(serial_env, num_envs=2, num_steps=3, timeout=(1 + 2) * 3)

    serial_env.terminate()


def test_serial_env_aggregate():
    serial_env = create_serial_env(2)
    _test_env(serial_env, num_envs=2, num_steps=3, timeout=(1 + 2) * 3)
    rollouts = serial_env.aggregate_rollouts()
    assert len(rollouts["obs"]) == 6
    assert len(rollouts["action"]) == 6

    # Reconfigure
    serial_env.set_working_env(0, 1)
    _test_env(serial_env, num_envs=1, num_steps=3, timeout=1 * 3)
    rollouts = serial_env.aggregate_rollouts()
    assert len(rollouts["obs"]) == 3
    assert len(rollouts["action"]) == 3


def create_ray_env(num_envs, num_parallel):
    env_fns = utils.create_env_fns(num_envs)
    test_envs = RayParallelEnv(env_fns)
    test_envs.reconfigure(num_envs, num_parallel=num_parallel)
    return test_envs


def test_ray_basic():
    ray_env = create_ray_env(1, 1)
    assert ray_env.num_envs == 1
    assert ray_env.num_env_workers == 1
    _test_env(ray_env, num_envs=1, num_steps=3, timeout=1 * 3)


def test_ray_fully_parallel():
    ray_env = create_ray_env(3, 3)
    assert ray_env.num_envs == 3
    assert ray_env.num_env_workers == 3
    _test_env(ray_env, num_envs=3, num_steps=3, timeout=3 * 3)


def test_ray_serial():
    ray_env = create_ray_env(2, 1)
    assert ray_env.num_envs == 2
    assert ray_env.num_env_workers == 1
    _test_env(ray_env, num_envs=2, num_steps=3, timeout=(1 + 2) * 3)


def test_ray_combination():
    ray_env = create_ray_env(4, 2)
    assert ray_env.num_envs == 4
    assert ray_env.num_env_workers == 2
    _test_env(ray_env, num_envs=4, num_steps=3, timeout=(3 + 4) * 3)


def test_ray_reconfigure():
    ray_env = create_ray_env(4, 1)
    ray_env.reconfigure(1, 1)
    _test_env(ray_env, num_envs=1, num_steps=3, timeout=1 * 3)

    ray_env.reconfigure(2, 2)
    _test_env(ray_env, num_envs=2, num_steps=3, timeout=2 * 3)

    ray_env.reconfigure(2, 1)
    _test_env(ray_env, num_envs=2, num_steps=3, timeout=(1 + 2) * 3)


def test_ray_rollouts():
    ray_env = create_ray_env(4, 1)

    _test_env(ray_env, num_envs=4, num_steps=2, timeout=(1 + 2 + 3 + 4) * 2)
    rollouts = ray_env.aggregate_rollouts()
    assert len(rollouts["obs"]) == 8
    assert len(rollouts["action"]) == 8

    ray_env.reconfigure(2, 2)
    _test_env(ray_env, num_envs=2, num_steps=3, timeout=2 * 3)
    rollouts = ray_env.aggregate_rollouts()
    assert len(rollouts["obs"]) == 6
    assert len(rollouts["action"]) == 6


# TODO
def test_correctness():
    """Test with gym simulator, to check its correctness."""
    pass
