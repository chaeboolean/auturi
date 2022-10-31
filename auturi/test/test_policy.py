import numpy as np
import pynvml
import pytest
import ray

import auturi.test.utils as utils
from auturi.executor.ray import RayVectorPolicy
from auturi.executor.shm import SHMVectorPolicy

TEST_BACKEND = "shm"


@pytest.fixture
def create_policy():
    vector_cls = RayVectorPolicy if TEST_BACKEND == "ray" else SHMVectorPolicy
    model, policy_cls, policy_kwargs = utils.create_vector_policy()
    vector_policy = vector_cls(policy_cls, policy_kwargs)
    vector_policy.start_loop()
    yield vector_policy, model
    vector_policy.stop_loop()


def get_cuda_memory(device_index):
    h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    return info.used


def reconfigure_mock_config(vector_policy, device, model, num_policy):
    class MockConfig:
        def __init__(self, num_policy, device):
            self.num_policy = num_policy
            self.policy_device = device

    vector_policy.reconfigure(MockConfig(num_policy, device), model)


def step_policy(test_policy, mock_obs, num_steps, timeout):
    action_refs = []
    mock_obs_refs = {0: utils.mock_ray.remote(mock_obs)}
    test_policy.start_loop()
    with utils.Timeout(min_sec=timeout - 0.5, max_sec=timeout + 1.5):
        for step in range(num_steps):
            ref = test_policy.compute_actions(mock_obs_refs, step)
            action_refs.append(ref)

        test_policy.stop_loop()

    return [ray.get(ref)[0].flat[0] for ref in action_refs]


# TODO: Not working as wanted.
@pytest.mark.skip
def test_load_model(create_policy):
    pynvml.nvmlInit()

    vector_policy, model = create_policy
    reconfigure_mock_config(vector_policy, "cpu", model, num_policy=1)
    assert get_cuda_memory(0) == get_cuda_memory(1)

    reconfigure_mock_config(vector_policy, "cuda:0", model, num_policy=1)
    assert get_cuda_memory(0) > get_cuda_memory(1)


def test_vector_policy_basic(create_policy):
    mock_obs = np.ones((5, 2))
    vector_policy, model = create_policy
    reconfigure_mock_config(vector_policy, "cpu", model, num_policy=1)

    action_refs = step_policy(vector_policy, mock_obs, num_steps=3, timeout=3)
    assert np.all(np.array(action_refs) == np.array([1, 2, 3]))


@pytest.mark.skip
def test_vector_policy_reconfigure():
    mock_obs = np.ones((5, 2))
    model, policy_cls, policy_kwargs = utils.create_vector_policy()
    vector_policy = RayVectorPolicy(policy_cls, policy_kwargs)

    reconfigure_mock_config(vector_policy, "cpu", model, num_policy=3)
    assert vector_policy.num_workers == 3
    action_refs = step_policy(vector_policy, mock_obs, 9, 2)
    assert max(action_refs) > 100

    reconfigure_mock_config(vector_policy, "cpu", model, num_policy=2)
    assert vector_policy.num_workers == 2
    action_refs = step_policy(vector_policy, mock_obs, 5, 3)
    assert max(action_refs) < 100  # assert policy2 not worked.

    reconfigure_mock_config(vector_policy, "cpu", model, num_policy=5)
    assert vector_policy.num_workers == 5
    action_refs = step_policy(vector_policy, mock_obs, 5, 1)
    assert set(action_refs) == set([1, 11, 101, 1001, 10001])
