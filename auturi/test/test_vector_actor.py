import numpy as np
import pytest

import auturi.test.utils as utils
from auturi.executor import create_executor
from auturi.test.utils import check_timeout
from auturi.tuner import create_tuner_with_config
from auturi.tuner.config import ActorConfig, ParallelizationConfig

VECTOR_BACKEND = "ray"


def create_vector_actor(mode, num_envs, tuner_config):
    env_fns = utils.create_env_fns(num_envs)
    model, policy_cls, policy_kwargs = utils.create_policy_args()
    tuner = create_tuner_with_config(num_envs, tuner_config)
    vector_actor = create_executor(env_fns, policy_cls, policy_kwargs, tuner, mode)

    return vector_actor, model


def test_executor_basic():
    num_envs = 1
    actor_config = ActorConfig(num_envs, num_collect=3)
    config = ParallelizationConfig.create([actor_config])
    vector_actor, model = create_vector_actor(VECTOR_BACKEND, num_envs, config)

    rollouts, metric = vector_actor.run(model)
    check_timeout(metric.elapsed, timeout=1.5 * 3)
    assert rollouts["obs"].shape == (3, 5, 2)


def test_multiple_actors():
    num_envs = 9
    actor_config = ActorConfig(num_envs=3, num_parallel=3, batch_size=3, num_collect=9)
    config = ParallelizationConfig.create([actor_config] * 3)
    vector_actor, model = create_vector_actor(VECTOR_BACKEND, num_envs, config)

    rollouts, metric = vector_actor.run(model)
    check_timeout(metric.elapsed, timeout=1.5 * 3)
    assert rollouts["obs"].shape == (27, 5, 2)

    config = ParallelizationConfig.create([actor_config] * 2)
    vector_actor.tuner = create_tuner_with_config(num_envs, config)

    rollouts, metric = vector_actor.run(model)
    check_timeout(metric.elapsed, timeout=1.5 * 3)
    assert rollouts["obs"].shape == (18, 5, 2)


# TODO
@pytest.mark.skip
def test_correctness():
    """Test with real-world simulator, to check its correctness."""
    pass
