import numpy as np
import ray

import auturi.test.utils as utils
from auturi.executor.ray.vector_actor import RayActor
from auturi.test.shm_utils import SHMActorTester
from auturi.tuner.config import ActorConfig, ParallelizationConfig

VECTOR_BACKEND = "shm"


def create_actor(mode, num_envs):
    env_fns = utils.create_env_fns(num_envs)
    model, policy_cls, policy_kwargs = utils.create_policy_args()

    if VECTOR_BACKEND == "ray":
        actor = RayActor(0, env_fns, policy_cls, policy_kwargs)

    elif VECTOR_BACKEND == "shm":
        actor = SHMActorTester.create(
            env_fns, policy_cls=policy_cls, policy_kwargs=policy_kwargs
        )

    return actor, model


def mock_reconfigure(test_actor, actor_config, model):
    config = ParallelizationConfig.create([actor_config])
    test_actor.reconfigure(config, model)


def run_actor(actor):
    trajectories, metric = actor.run()
    if VECTOR_BACKEND == "ray":
        trajectories = ray.get(trajectories)

    action_list = [obs_.flat[0] for obs_ in trajectories["action"]]

    return np.array(action_list), metric.elapsed


def test_single_env_single_policy():
    actor, model = create_actor(VECTOR_BACKEND, num_envs=1)
    actor_config = ActorConfig(
        num_envs=1, num_policy=1, num_parallel=1, batch_size=1, num_collect=3
    )
    mock_reconfigure(actor, actor_config, model)
    action_list, elapsed = run_actor(actor)

    assert np.all(action_list == np.array([1, 2, 3]))

    # one step takes 0.5(env[0].step) + 1 (policy)
    utils.check_timeout(elapsed, timeout=1.5 * 3)
    actor.terminate()


def test_serial_env_single_policy():
    actor, model = create_actor(VECTOR_BACKEND, num_envs=4)
    actor_config = ActorConfig(
        num_envs=2, num_policy=1, num_parallel=1, batch_size=2, num_collect=6
    )
    mock_reconfigure(actor, actor_config, model)

    action_list, elapsed = run_actor(actor)
    print(action_list, elapsed)

    assert np.all(action_list == np.array([1, 2, 3, 1, 2, 3]))

    # one step takes 0.5(env[0].step) + 0.5(env[1].step)+ 1 (policy)
    utils.check_timeout(elapsed, timeout=2 * 3)
    actor.terminate()


def test_parallel_env_single_policy():
    actor, model = create_actor(VECTOR_BACKEND, num_envs=4)
    actor_config = ActorConfig(
        num_envs=3, num_policy=1, num_parallel=3, batch_size=3, num_collect=9
    )
    mock_reconfigure(actor, actor_config, model)

    action_list, elapsed = run_actor(actor)

    assert np.all(action_list == np.array([1, 2, 3, 1, 2, 3, 1, 2, 3]))

    # one step takes 0.5(max(env.step)) + 1 (policy)
    utils.check_timeout(elapsed, timeout=1.5 * 3)
    actor.terminate()


def test_parallel_and_serial_env_single_policy():
    actor, model = create_actor(VECTOR_BACKEND, num_envs=4)

    # Case 1) batch size = 4
    actor_config = ActorConfig(
        num_envs=4, num_policy=1, num_parallel=2, batch_size=4, num_collect=12
    )
    mock_reconfigure(actor, actor_config, model)
    action_list, elapsed = run_actor(actor)

    assert np.all(action_list == np.array([1, 2, 3] * 4))
    # one step takes 0.5(env[0].step) + 0.5(env[1].step)+ 1 (policy)
    utils.check_timeout(elapsed, timeout=2 * 3)

    # Case 2) batch size = 2
    actor_config = ActorConfig(
        num_envs=4, num_policy=1, num_parallel=2, batch_size=2, num_collect=8
    )
    mock_reconfigure(actor, actor_config, model)
    action_list, elapsed = run_actor(actor)

    env_action = action_list[:2], action_list[2:4], action_list[4:6], action_list[6:]
    assert np.all(env_action[0] == env_action[1])
    assert np.all(env_action[2] == env_action[3])
    assert set(env_action[0]).intersection(set(env_action[2])) == set()
    assert set(env_action[0]).union(set(env_action[2])) == set([1, 2, 3, 4])
    assert set(env_action[0]) == set([1, 3]) or set(env_action[0]) == set([2, 4])

    # action time: [0, 1], [1, 2], [2, 3], [3, 4]
    utils.check_timeout(elapsed, timeout=5)
    actor.terminate()


def test_parallel_env_async():
    actor, model = create_actor(VECTOR_BACKEND, num_envs=4)

    actor_config = ActorConfig(
        num_envs=4, num_policy=1, num_parallel=4, batch_size=2, num_collect=8
    )
    mock_reconfigure(actor, actor_config, model)
    action_list, elapsed = run_actor(actor)
    utils.check_timeout(elapsed, timeout=4.5)
    actor.terminate()


def test_parallel_env_multiple_policy():
    actor, model = create_actor(VECTOR_BACKEND, num_envs=4)

    actor_config = ActorConfig(
        num_envs=4, num_policy=1, num_parallel=4, batch_size=1, num_collect=4
    )
    mock_reconfigure(actor, actor_config, model)
    action_list, elapsed = run_actor(
        actor,
    )

    assert set(action_list) == set([1, 2, 3, 4])
    utils.check_timeout(elapsed, timeout=4.5)

    actor_config = ActorConfig(
        num_envs=4, num_policy=2, num_parallel=4, batch_size=1, num_collect=4
    )
    mock_reconfigure(actor, actor_config, model)
    action_list, elapsed = run_actor(actor)

    assert set(action_list) == set([1, 2, 11, 12])
    utils.check_timeout(elapsed, timeout=2.5)
    actor.terminate()


def test_reconfigure_same_config():
    actor, model = create_actor(VECTOR_BACKEND, num_envs=4)

    # start with num_env = 2
    actor_config = ActorConfig(
        num_envs=2, num_policy=1, num_parallel=2, batch_size=1, num_collect=4
    )
    mock_reconfigure(actor, actor_config, model)
    action_list, elapsed = run_actor(actor)
    utils.check_timeout(elapsed, timeout=4.5)
    assert set(action_list) == set([1, 2, 3, 4])

    # run with same config
    mock_reconfigure(actor, actor_config, model)
    action_list, elapsed = run_actor(actor)
    utils.check_timeout(elapsed, timeout=4.5)
    assert set(action_list) == set([1, 2, 3, 4])

    actor.terminate()


def test_reconfigure_envs():
    actor, model = create_actor(VECTOR_BACKEND, num_envs=8)

    # num_serialenvs = 8
    actor_config = ActorConfig(
        num_envs=8, num_policy=1, num_parallel=8, batch_size=8, num_collect=16
    )
    mock_reconfigure(actor, actor_config, model)
    action_list, elapsed = run_actor(actor)
    utils.check_timeout(elapsed, timeout=1.5 * 2)
    assert set(action_list) == set([1, 2])

    # decrease num_serialenvs to 2
    actor_config = ActorConfig(
        num_envs=8, num_policy=1, num_parallel=2, batch_size=4, num_collect=16
    )
    mock_reconfigure(actor, actor_config, model)
    action_list, elapsed = run_actor(actor)
    utils.check_timeout(elapsed, timeout=7)
    assert set(action_list) == set([1, 2, 3, 4])

    # increase num_serialenvs to 4
    actor_config = ActorConfig(
        num_envs=8, num_policy=1, num_parallel=4, batch_size=8, num_collect=16
    )
    mock_reconfigure(actor, actor_config, model)
    action_list, elapsed = run_actor(actor)
    utils.check_timeout(elapsed, timeout=4)

    assert set(action_list) == set([1, 2])
    actor.terminate()


def test_reconfigure_policy():
    actor, model = create_actor(VECTOR_BACKEND, num_envs=8)

    # start with num_policy=2
    actor_config = ActorConfig(
        num_envs=8, num_policy=2, num_parallel=4, batch_size=4, num_collect=16
    )
    mock_reconfigure(actor, actor_config, model)
    action_list, elapsed = run_actor(actor)
    utils.check_timeout(elapsed, timeout=4)
    assert set(action_list) == set([1, 2, 11, 12])

    # increase to num_policy=4
    actor_config = ActorConfig(
        num_envs=8, num_policy=4, num_parallel=8, batch_size=2, num_collect=16
    )
    mock_reconfigure(actor, actor_config, model)
    action_list, elapsed = run_actor(actor)
    utils.check_timeout(elapsed, timeout=3)
    assert np.max(action_list) > 1000
    assert np.min(action_list) == 1

    # increase to num_policy=1
    actor_config = ActorConfig(
        num_envs=2, num_policy=1, num_parallel=2, batch_size=2, num_collect=4
    )
    mock_reconfigure(actor, actor_config, model)
    action_list, elapsed = run_actor(actor)
    utils.check_timeout(elapsed, timeout=3)
    actor.terminate()
