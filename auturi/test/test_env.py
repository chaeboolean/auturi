import numpy as np

import auturi.test.utils as utils
from auturi.executor.environment import AuturiSerialEnv, AuturiVectorEnv
from auturi.executor.ray import RayParallelEnv
from auturi.tuner.config import ActorConfig, ParallelizationConfig

VECTOR_BACKEND = "ray"


def create_env(mode, num_envs, num_parallel=-1):
    env_fns = utils.create_env_fns(num_envs)
    if mode == "serial":
        test_envs = AuturiSerialEnv(0, 0, env_fns)
        test_envs.set_working_env(start_idx=0, num_envs=num_envs)

    elif mode == "ray":
        test_envs = RayParallelEnv(0, env_fns)
        mock_reconfigure(test_envs, num_envs, num_parallel)

    return test_envs


def mock_reconfigure(test_env, num_envs, num_parallel):
    config = ActorConfig(
        num_envs=num_envs,
        num_parallel=num_parallel,
        batch_size=num_envs,
        num_collect=100,
    )
    test_env.reconfigure(ParallelizationConfig.create([config]))


def step_env(test_env, num_envs, num_steps, timeout):
    """Test elapsed time and step sequence."""

    actions = np.stack([test_env.action_space.sample()] * num_envs)
    actions.fill(1)
    action_artifacts = [np.copy(actions)]

    test_env.seed(-1)
    test_env.reset()

    with utils.Timeout(min_sec=timeout - 0.5, max_sec=timeout + 0.5):
        if isinstance(test_env, AuturiVectorEnv):
            test_env.start_loop()
        for step in range(num_steps):
            obs = test_env.step(actions, action_artifacts)
        if isinstance(test_env, AuturiVectorEnv):
            test_env.stop_loop()

    agg_obs = test_env.aggregate_rollouts()["obs"]
    agg_obs = np.array([obs_.flat[0] for obs_ in agg_obs])
    return agg_obs


def test_serial_env():
    agg_obs = step_env(
        create_env("serial", 1), num_envs=1, num_steps=3, timeout=0.5 * 3
    )
    assert np.all(agg_obs == np.array([1001, 1002, 1003]))

    agg_obs = step_env(
        create_env("serial", 2), num_envs=2, num_steps=3, timeout=(0.5 + 0.5) * 3
    )
    assert np.all(agg_obs == np.array([1001, 1002, 1003, 2001, 2002, 2003]))

    agg_obs = step_env(
        create_env("serial", 3), num_envs=3, num_steps=3, timeout=(0.5 + 0.5 + 0.5) * 3
    )
    assert np.all(
        agg_obs == np.array([1001, 1002, 1003, 2001, 2002, 2003, 3001, 3002, 3003])
    )


def test_reconfigure_serial_env():
    serial_env = create_env("serial", 3)
    agg_obs = step_env(serial_env, num_envs=3, num_steps=3, timeout=(1.5) * 3)

    serial_env.set_working_env(0, 1)
    agg_obs = step_env(serial_env, num_envs=1, num_steps=3, timeout=0.5 * 3)

    serial_env.set_working_env(0, 2)
    agg_obs = step_env(serial_env, num_envs=2, num_steps=3, timeout=1 * 3)

    serial_env.terminate()


def test_serial_env_aggregate():
    serial_env = create_env("serial", 2)
    agg_obs = step_env(serial_env, num_envs=2, num_steps=3, timeout=1 * 3)
    assert len(agg_obs) == 6

    # Reconfigure
    serial_env.set_working_env(0, 1)
    agg_obs = step_env(serial_env, num_envs=1, num_steps=3, timeout=0.5 * 3)
    assert len(agg_obs) == 3
    serial_env.terminate()


def test_vector_env_basic():
    vector_env = create_env(VECTOR_BACKEND, 1, 1)
    assert vector_env.num_envs == 1
    assert vector_env.num_workers == 1
    step_env(vector_env, num_envs=1, num_steps=3, timeout=0.5 * 3)
    vector_env.terminate()


def test_fully_parallel():
    vector_env = create_env(VECTOR_BACKEND, 3, 3)
    assert vector_env.num_envs == 3
    assert vector_env.num_workers == 3
    agg_obs = step_env(vector_env, num_envs=3, num_steps=3, timeout=0.5 * 3)
    assert max(agg_obs) == 3003
    assert min(agg_obs) == 1001

    vector_env.terminate()


def test_serial():
    vector_env = create_env(VECTOR_BACKEND, 2, 1)
    assert vector_env.num_envs == 2
    assert vector_env.num_workers == 1

    agg_obs = step_env(vector_env, num_envs=2, num_steps=3, timeout=(0.5 + 0.5) * 3)
    print(agg_obs)
    assert max(agg_obs) == 2003
    assert min(agg_obs) == 1001
    vector_env.terminate()


def test_combination():
    vector_env = create_env(VECTOR_BACKEND, 4, 2)
    assert vector_env.num_envs == 4
    assert vector_env.num_workers == 2
    agg_obs = step_env(vector_env, num_envs=4, num_steps=3, timeout=(0.5 + 0.5) * 3)
    assert max(agg_obs) == 4003
    assert min(agg_obs) == 1001
    assert len(agg_obs) == 12
    vector_env.terminate()


def test_reconfigure():
    vector_env = create_env(VECTOR_BACKEND, 4, 2)

    mock_reconfigure(vector_env, num_envs=1, num_parallel=1)
    assert vector_env.num_envs == 1
    assert vector_env.num_workers == 1
    agg_obs1 = step_env(vector_env, num_envs=1, num_steps=3, timeout=0.5 * 3)
    assert np.all(agg_obs1 == np.array([1001, 1002, 1003]))

    mock_reconfigure(vector_env, num_envs=2, num_parallel=2)
    assert vector_env.num_envs == 2
    assert vector_env.num_workers == 2
    agg_obs2 = step_env(vector_env, num_envs=2, num_steps=3, timeout=0.5 * 3)

    mock_reconfigure(vector_env, num_envs=2, num_parallel=1)
    assert vector_env.num_envs == 2
    assert vector_env.num_workers == 1
    agg_obs3 = step_env(vector_env, num_envs=2, num_steps=3, timeout=(0.5 + 0.5) * 3)
    for e1, e2 in zip(agg_obs2, agg_obs3):
        assert e1 == e2

    vector_env.terminate()


def test_rollouts():
    vector_env = create_env(VECTOR_BACKEND, 4, 1)
    mock_reconfigure(vector_env, num_envs=4, num_parallel=1)
    agg_obs = step_env(
        vector_env, num_envs=4, num_steps=2, timeout=(0.5 + 0.5 + 0.5 + 0.5) * 2
    )
    assert len(agg_obs) == 8

    mock_reconfigure(vector_env, num_envs=4, num_parallel=2)
    agg_obs = step_env(vector_env, num_envs=4, num_steps=2, timeout=(0.5 + 0.5) * 2)
    assert len(agg_obs) == 8

    mock_reconfigure(vector_env, num_envs=4, num_parallel=4)
    agg_obs = step_env(vector_env, num_envs=4, num_steps=2, timeout=(0.5) * 2)
    assert len(agg_obs) == 8
    vector_env.terminate()
