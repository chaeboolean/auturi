import numpy as np
import pynvml
import pytest
import ray
import torch

import auturi.test.utils as utils
from auturi.vector.ray_backend import RayVectorPolicies


def get_cuda_memory(device_index):
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    return info.used


@pytest.fixture
def create_vector_policy():
    model = torch.nn.Linear(10, 10, bias=False)
    model.eval()
    torch.nn.init.uniform_(model.weight, 1, 1)
    return model, RayVectorPolicies(
        policy_cls=utils.DumbPolicy,
        policy_kwargs={},
    )


# TODO: Not working as wanted.
@pytest.mark.skip
def test_load_model(create_vector_policy):
    pynvml.nvmlInit()

    model, vector_policy = create_vector_policy
    assert vector_policy.num_policies == 1
    vector_policy.load_model(model, "cpu")
    assert get_cuda_memory(0) == get_cuda_memory(1)

    vector_policy.reconfigure(3)
    assert vector_policy.num_policies == 3
    vector_policy.load_model(model, "cuda:0")
    assert get_cuda_memory(0) > get_cuda_memory(1)


def step_policy(test_policy, mock_obs, num_steps, timeout):
    action_refs = []
    mock_obs_refs = {0: utils.mock_ray.remote(mock_obs)}
    test_policy.start_loop()
    with utils.Timeout(min_sec=timeout - 0.5, max_sec=timeout + 1.5):
        for step in range(num_steps):
            ref = test_policy.compute_actions(mock_obs_refs, step)
            action_refs.append(ref)

        test_policy.stop_loop()

    return [ray.get(ref).flat[0] for ref in action_refs]


def test_vector_policy_basic(create_vector_policy):
    mock_obs = np.ones(10)
    model, vector_policy = create_vector_policy
    vector_policy.load_model(model, "cpu")

    action_refs = step_policy(vector_policy, mock_obs, num_steps=3, timeout=3)

    assert np.all(np.array(action_refs) == np.array([11, 12, 13]))


def test_vector_policy_reconfigure(create_vector_policy):
    mock_obs = np.ones(10)
    model, vector_policy = create_vector_policy

    vector_policy.reconfigure(3)
    vector_policy.load_model(model, "cpu")
    action_refs = step_policy(vector_policy, mock_obs, 9, 2)
    assert set(action_refs[:3]) == set([11, 21, 31])
    assert set(action_refs[3:6]).issubset(set([12, 22, 32]))
    assert set(action_refs[6:]).issubset(set([13, 23, 33]))

    vector_policy.reconfigure(2)
    vector_policy.load_model(model, "cpu")
    action_refs = step_policy(vector_policy, mock_obs, 5, 3)
    assert set(action_refs[:2]) == set([11, 21])
    assert set(action_refs[2:4]) == set([12, 22])
    assert set(action_refs[4:]).issubset(set([13, 23]))

    vector_policy.reconfigure(5)
    vector_policy.load_model(model, "cpu")
    action_refs = step_policy(vector_policy, mock_obs, 5, 1)
    assert set(action_refs) == set([11, 21, 31, 41, 51])
