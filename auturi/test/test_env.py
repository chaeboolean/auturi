from contextlib import contextmanager

import pytest

import auturi.test.utils as utils
from auturi.executor.environment import AuturiLocalEnv
from auturi.executor.shm.loop import SHMNestedLoopHandler
from auturi.tuner.config import ActorConfig, ParallelizationConfig

VECTOR_BACKEND = "shm"


def dummy_action(num_envs, is_discrete):
    action = utils.make_dummy_action(is_discrete, num_envs)
    return (action, [action])


@contextmanager
def create_local_env_handler(num_envs, num_collect, is_discrete):
    env_fns = utils.create_env_fns(num_envs, is_discrete)
    env_handler = AuturiLocalEnv(0, env_fns)  # actor_id = 0
    config = ActorConfig(
        num_envs=num_envs,
        num_parallel=1,
        batch_size=num_envs,
        num_collect=num_collect,
    )
    env_handler.reconfigure(ParallelizationConfig.create([config]))
    yield env_handler


@contextmanager
def create_vector_env_handler(num_envs, num_parallel, num_collect, is_discrete):
    env_fns = utils.create_env_fns(num_envs, is_discrete)

    # borrow only intialization function
    loop_handler = SHMNestedLoopHandler(0, env_fns, None, None, num_envs, num_collect)
    env_handler = loop_handler._create_env_handler()

    config = ActorConfig(
        num_envs=num_envs,
        num_parallel=num_parallel,
        batch_size=num_envs,
        num_collect=num_collect,
    )
    env_handler.reconfigure(ParallelizationConfig.create([config]))
    env_handler.start_loop()

    yield env_handler

    env_handler.stop_loop()


@pytest.mark.parametrize(
    "num_envs,is_discrete", [(1, True), (1, False), (3, True), (3, False)]
)
def test_local_env(num_envs, is_discrete):
    actions = dummy_action(num_envs, is_discrete)
    with create_local_env_handler(
        num_envs=num_envs, num_collect=3, is_discrete=is_discrete
    ) as env_handler:
        assert env_handler.num_envs == num_envs

        obs = env_handler.poll()
        assert obs.shape == (num_envs, 5, 2)

        env_handler.send_actions(actions)
        obs = env_handler.poll()

        env_handler.send_actions(actions)
        obs = env_handler.poll()

        env_handler.send_actions(actions)
        obs = env_handler.poll()

        assert obs.shape == (num_envs, 5, 2)
        rollouts = env_handler.aggregate_rollouts()
        assert rollouts["obs"].shape == (3 * num_envs, 5, 2)

    env_handler.terminate()


@pytest.mark.parametrize(
    "num_envs,is_discrete", [(1, True), (1, False), (3, True), (3, False)]
)
def test_fully_parallel_vector_env(num_envs, is_discrete):
    actions = dummy_action(num_envs, is_discrete)

    with create_vector_env_handler(
        num_envs=num_envs,
        num_parallel=num_envs,
        num_collect=3 * num_envs,
        is_discrete=is_discrete,
    ) as env_handler:

        assert env_handler.num_envs == num_envs
        obs = env_handler.step(*actions)
        assert obs.shape == (num_envs, 5, 2)

        env_handler.step(*actions)
        env_handler.step(*actions)

    rollouts = env_handler.aggregate_rollouts()
    assert rollouts["obs"].shape == (3 * num_envs, 5, 2)
    env_handler.terminate()


# def test_reconfigure_serial_env():
#     serial_env = create_env("serial", 3)
#     agg_obs = step_env(serial_env, num_envs=3, num_steps=3, timeout=(1.5) * 3)

#     serial_env.set_working_env(0, 1)
#     agg_obs = step_env(serial_env, num_envs=1, num_steps=3, timeout=0.5 * 3)

#     serial_env.set_working_env(0, 2)
#     agg_obs = step_env(serial_env, num_envs=2, num_steps=3, timeout=1 * 3)

#     serial_env.terminate()


# def test_serial_env_aggregate():
#     serial_env = create_env("serial", 2)
#     agg_obs = step_env(serial_env, num_envs=2, num_steps=3, timeout=1 * 3)
#     assert len(agg_obs) == 6

#     # Reconfigure
#     serial_env.set_working_env(0, 1)
#     agg_obs = step_env(serial_env, num_envs=1, num_steps=3, timeout=0.5 * 3)
#     assert len(agg_obs) == 3
#     serial_env.terminate()


# def test_vector_env_basic():
#     vector_env = create_env(VECTOR_BACKEND, 1, 1)
#     assert vector_env.num_envs == 1
#     assert vector_env.num_workers == 1
#     step_env(vector_env, num_envs=1, num_steps=3, timeout=0.5 * 3)

#     # stop and restart
#     step_env(vector_env, num_envs=1, num_steps=3, timeout=0.5 * 3)

#     # stop and restart
#     step_env(vector_env, num_envs=1, num_steps=3, timeout=0.5 * 3)

#     vector_env.terminate()


# def test_fully_parallel():
#     vector_env = create_env(VECTOR_BACKEND, 3, 3)
#     assert vector_env.num_envs == 3
#     assert vector_env.num_workers == 3
#     agg_obs = step_env(vector_env, num_envs=3, num_steps=3, timeout=0.5 * 3)
#     assert max(agg_obs) == 3003
#     assert min(agg_obs) == 1001

#     vector_env.terminate()


# def test_serial():
#     vector_env = create_env(VECTOR_BACKEND, 2, 1)
#     assert vector_env.num_envs == 2
#     assert vector_env.num_workers == 1

#     agg_obs = step_env(vector_env, num_envs=2, num_steps=3, timeout=(0.5 + 0.5) * 3)
#     assert max(agg_obs) == 2003
#     assert min(agg_obs) == 1001
#     vector_env.terminate()


# def test_combination():
#     vector_env = create_env(VECTOR_BACKEND, 4, 2)
#     assert vector_env.num_envs == 4
#     assert vector_env.num_workers == 2
#     agg_obs = step_env(vector_env, num_envs=4, num_steps=3, timeout=(0.5 + 0.5) * 3)
#     assert max(agg_obs) == 4003
#     assert min(agg_obs) == 1001
#     assert len(agg_obs) == 12
#     vector_env.terminate()


# def test_reconfigure():
#     vector_env = create_env(VECTOR_BACKEND, 4, 2)

#     mock_reconfigure(vector_env, num_envs=1, num_parallel=1)

#     assert vector_env.num_envs == 1
#     assert vector_env.num_workers == 1
#     agg_obs1 = step_env(vector_env, num_envs=1, num_steps=3, timeout=0.5 * 3)
#     assert np.all(agg_obs1 == np.array([1001, 1002, 1003]))

#     mock_reconfigure(vector_env, num_envs=2, num_parallel=2)
#     assert vector_env.num_envs == 2
#     assert vector_env.num_workers == 2
#     agg_obs2 = step_env(vector_env, num_envs=2, num_steps=3, timeout=0.5 * 3)

#     mock_reconfigure(vector_env, num_envs=2, num_parallel=1)
#     assert vector_env.num_envs == 2
#     assert vector_env.num_workers == 1
#     agg_obs3 = step_env(vector_env, num_envs=2, num_steps=3, timeout=(0.5 + 0.5) * 3)
#     for e1, e2 in zip(agg_obs2, agg_obs3):
#         assert e1 == e2

#     vector_env.terminate()


# def test_rollouts():
#     vector_env = create_env(VECTOR_BACKEND, 4, 1)
#     mock_reconfigure(vector_env, num_envs=4, num_parallel=1)
#     agg_obs1 = step_env(
#         vector_env, num_envs=4, num_steps=2, timeout=(0.5 + 0.5 + 0.5 + 0.5) * 2
#     )

#     mock_reconfigure(vector_env, num_envs=4, num_parallel=2)
#     agg_obs2 = step_env(vector_env, num_envs=4, num_steps=2, timeout=(0.5 + 0.5) * 2)

#     mock_reconfigure(vector_env, num_envs=4, num_parallel=4)
#     agg_obs3 = step_env(vector_env, num_envs=4, num_steps=2, timeout=(0.5) * 2)
#     vector_env.terminate()

#     # assert all results equal.
#     assert len(agg_obs1) == 8
#     assert np.all(agg_obs1 == agg_obs2)
#     assert np.all(agg_obs1 == agg_obs3)
