import numpy as np
import pynvml
import pytest
import ray

import auturi.test.utils as utils
from auturi.executor.ray import RayVectorPolicy
from auturi.tuner.config import ActorConfig, ParallelizationConfig

VECTOR_BACKEND = "ray"


def get_cuda_memory(device_index):
    h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    return info.used


def mock_reconfigure(test_policy, num_policy, device, model):
    actor_config = ActorConfig(
        num_envs=100,
        num_parallel=100,
        num_policy=num_policy,
        policy_device=device,
        num_collect=100,
    )
    config = ParallelizationConfig.create([actor_config])
    test_policy.reconfigure(config, model)


def create_policy(mode, num_policy, device):
    model, policy_cls, policy_kwargs = utils.create_policy_args()
    if mode == "ray":
        test_policy = RayVectorPolicy(0, policy_cls, policy_kwargs)

    mock_reconfigure(test_policy, num_policy, device, model)
    return test_policy, model


def step_policy(test_policy, mock_obs, num_steps, timeout):
    action_refs = []
    mock_obs_refs = {0: utils.mock_ray.remote(mock_obs)}

    test_policy.start_loop()
    with utils.Timeout(min_sec=timeout - 0.5, max_sec=timeout + 0.5):
        for step in range(num_steps):
            ref = test_policy.compute_actions(mock_obs_refs, step)
            action_refs.append(ref)
        test_policy.stop_loop()

    return [ray.get(ref)[0].flat[0] for ref in action_refs]


# TODO: Not working as wanted.
@pytest.mark.skip
def test_load_model():
    pynvml.nvmlInit()

    test_policy, model = create_policy(VECTOR_BACKEND, 1, "cpu")
    assert get_cuda_memory(0) == get_cuda_memory(1)

    mock_reconfigure(test_policy, 1, "cuda:0", model)
    assert get_cuda_memory(0) > get_cuda_memory(1)


def test_vector_policy_basic():
    mock_obs = np.ones((1, 5, 2))
    test_policy, model = create_policy(VECTOR_BACKEND, 1, "cpu")

    mock_reconfigure(test_policy, 1, "cpu", model)
    action_refs = step_policy(test_policy, mock_obs, num_steps=3, timeout=3)
    assert np.all(np.array(action_refs) == np.array([1, 2, 3]))


def test_vector_policy_reconfigure():
    mock_obs = np.ones((1, 5, 2))
    test_policy, model = create_policy(VECTOR_BACKEND, 1, "cpu")

    mock_reconfigure(test_policy, 3, "cpu", model)
    assert test_policy.num_workers == 3
    action_refs = step_policy(test_policy, mock_obs, num_steps=9, timeout=3)
    assert max(action_refs) > 100

    mock_reconfigure(test_policy, 2, "cpu", model)
    assert test_policy.num_workers == 2
    action_refs = step_policy(test_policy, mock_obs, num_steps=5, timeout=3)
    assert max(action_refs) < 100  # assert policy2 not worked.

    mock_reconfigure(test_policy, 5, "cpu", model)
    assert test_policy.num_workers == 5
    action_refs = step_policy(test_policy, mock_obs, num_steps=5, timeout=1)
    assert set(action_refs) == set([1, 11, 101, 1001, 10001])
