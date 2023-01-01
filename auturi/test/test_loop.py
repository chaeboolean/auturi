from contextlib import contextmanager

import auturi.test.utils as utils
import pytest
from auturi.executor.loop import SimpleLoopHandler
from auturi.executor.shm.loop import SHMMultiLoopHandler, SHMNestedLoopHandler
from auturi.tuner.config import ActorConfig, ParallelizationConfig


@contextmanager
def get_simple_loop_handler(num_envs, num_collect, is_discrete):
    env_fns = utils.create_env_fns(num_envs, is_discrete)
    model, policy_cls, policy_kwargs = utils.create_policy_args(is_discrete)
    loop_handler = SimpleLoopHandler(0, env_fns, policy_cls, policy_kwargs)

    config = ActorConfig(
        num_envs=num_envs,
        num_parallel=1,
        batch_size=num_envs,
        num_collect=num_collect,
    )
    loop_handler.reconfigure(ParallelizationConfig.create([config]), model)
    yield loop_handler

    loop_handler.terminate()


@contextmanager
def get_nested_loop_handler(actor_config: ActorConfig, is_discrete):
    env_fns = utils.create_env_fns(actor_config.num_envs, is_discrete)
    model, policy_cls, policy_kwargs = utils.create_policy_args(is_discrete)
    loop_handler = SHMNestedLoopHandler(
        0,
        env_fns,
        policy_cls,
        policy_kwargs,
        max_num_envs=actor_config.num_envs,
        max_num_policy=actor_config.num_policy,
        num_rollouts=actor_config.num_collect,
    )

    loop_handler.reconfigure(ParallelizationConfig.create([actor_config]), model)
    yield loop_handler

    loop_handler.terminate()


@contextmanager
def get_multi_loop_handler(num_loops, num_collect, num_envs, is_discrete):
    env_fns = utils.create_env_fns(num_envs, is_discrete)
    model, policy_cls, policy_kwargs = utils.create_policy_args(is_discrete)
    loop_handler = SHMMultiLoopHandler(
        env_fns,
        policy_cls,
        policy_kwargs,
        max_num_loop=num_loops,
        num_rollouts=num_collect,
    )

    config = ActorConfig(
        num_envs=num_envs // num_loops,
        num_parallel=1,
        batch_size=num_envs // num_loops,
        num_collect=num_collect // num_loops,
    )
    loop_handler.reconfigure(ParallelizationConfig.create([config] * num_loops), model)
    yield loop_handler

    loop_handler.terminate()


@pytest.mark.parametrize(
    "num_envs,is_discrete", [(1, False), (1, True), (2, False), (2, True)]
)
def test_simple_loop(num_envs, is_discrete):
    with get_simple_loop_handler(num_envs, 4, is_discrete) as loop_handler:
        agg_data, metric = loop_handler.run()

    assert agg_data["obs"].shape == (4, 5, 2)
    if num_envs == 1:
        utils.check_array(agg_data["obs"][:, 0, 0], [1001, 1003, 1006, 1010])
        utils.check_array(agg_data["action"][:, 0], [1, 2, 3, 4])

    else:
        utils.check_array(agg_data["obs"][:, 0, 0], [1001, 1003, 2001, 2003])
        utils.check_array(agg_data["action"][:, 0], [1, 2, 1, 2])

    if is_discrete:
        assert agg_data["action"].shape == (4, 1)
    else:
        assert agg_data["action"].shape == (4, 3)

    print(metric)
    # assert utils.check_timeout(metric.elapsed, 4.5)


@pytest.mark.parametrize("is_discrete", [False, True])
def test_nested_loop_single_env(is_discrete):
    config = ActorConfig(
        num_envs=1,
        num_parallel=1,
        num_policy=1,
        batch_size=1,
        num_collect=3,
    )

    with get_nested_loop_handler(config, is_discrete) as loop_handler:
        agg_data, metric = loop_handler.run()

    assert agg_data["obs"].shape == (3, 5, 2)
    utils.check_array(agg_data["obs"][:, 0, 0], [1001, 1003, 1006])
    utils.check_array(agg_data["action"][:, 0], [1, 2, 3])

    if is_discrete:
        assert agg_data["action"].shape == (3, 1)
    else:
        assert agg_data["action"].shape == (3, 3)


@pytest.mark.parametrize(
    "num_envs,is_discrete", [(2, False), (2, True), (4, False), (4, True)]
)
def test_nested_loop_env_parallel(num_envs, is_discrete):
    config = ActorConfig(
        num_envs=num_envs,
        num_parallel=num_envs,
        num_policy=1,
        batch_size=num_envs,
        num_collect=3 * num_envs,
    )

    with get_nested_loop_handler(config, is_discrete) as loop_handler:
        agg_data, metric = loop_handler.run()

    assert agg_data["obs"].shape == (3 * num_envs, 5, 2)
    obs_answer = []
    for i in range(num_envs):
        offset_ = 1000 * (1 + i)
        obs_answer += [offset_ + 1, offset_ + 3, offset_ + 6]

    utils.check_array(agg_data["obs"][:, 0, 0], obs_answer)
    utils.check_array(agg_data["action"][:, 0], [1, 2, 3] * num_envs)

    if is_discrete:
        assert agg_data["action"].shape == (3 * num_envs, 1)
    else:
        assert agg_data["action"].shape == (3 * num_envs, 3)


@pytest.mark.parametrize("is_discrete", [True, False])
def test_nested_loop_overlap(is_discrete):
    num_envs = 2
    config = ActorConfig(
        num_envs=2,
        num_parallel=2,
        num_policy=2,
        batch_size=1,
        num_collect=6,
    )

    with get_nested_loop_handler(config, is_discrete) as loop_handler:
        agg_data, metric = loop_handler.run()

    assert agg_data["obs"].shape == (6, 5, 2)
    utils.check_array(agg_data["action"][:, 0], [1, 2, 3, 11, 12, 13], keep_order=False)

    if is_discrete:
        assert agg_data["action"].shape == (3 * num_envs, 1)
    else:
        assert agg_data["action"].shape == (3 * num_envs, 3)


@pytest.mark.parametrize(
    "num_envs_per_loop,is_discrete", [(1, False), (1, True), (2, False), (3, True)]
)
def test_multi_loop_single_loop(num_envs_per_loop, is_discrete):
    num_collect = 3 * num_envs_per_loop
    with get_multi_loop_handler(
        1, num_collect, num_envs_per_loop, is_discrete
    ) as loop_handler:
        agg_data, metric = loop_handler.run()

    assert agg_data["obs"].shape == (3 * num_envs_per_loop, 5, 2)
    # utils.check_array(agg_data["action"][:, 0], [1, 2, 3] * num_envs_per_loop, keep_order=False)

    if is_discrete:
        assert agg_data["action"].shape == (3 * num_envs_per_loop, 1)
    else:
        assert agg_data["action"].shape == (3 * num_envs_per_loop, 3)


@pytest.mark.parametrize(
    "num_envs_per_loop,is_discrete", [(1, False), (1, True), (2, False), (3, True)]
)
def test_multi_loop_double_loop(num_envs_per_loop, is_discrete):
    num_collect = 6 * num_envs_per_loop
    num_envs = 2 * num_envs_per_loop
    with get_multi_loop_handler(2, num_collect, num_envs, is_discrete) as loop_handler:
        agg_data, metric = loop_handler.run()

    assert agg_data["obs"].shape == (6 * num_envs_per_loop, 5, 2)
    # utils.check_array(agg_data["action"][:, 0], [1, 2, 3] * num_envs_per_loop, keep_order=False)

    if is_discrete:
        assert agg_data["action"].shape == (6 * num_envs_per_loop, 1)
    else:
        assert agg_data["action"].shape == (6 * num_envs_per_loop, 3)
