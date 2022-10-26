from collections import OrderedDict
from typing import Callable, List

import gym
import numpy as np
import ray


def _flatten_obs(obs, space, stacking_fn: Callable) -> None:
    """Borrowed from Stable-baselines3 SubprocVec implementation."""

    assert isinstance(
        obs, (list, tuple)
    ), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, gym.spaces.Dict):
        assert isinstance(
            space.spaces, OrderedDict
        ), "Dict space must have ordered subspaces"
        assert isinstance(
            obs[0], dict
        ), "non-dict observation for environment with Dict observation space"
        return OrderedDict(
            [(k, stacking_fn([o[k] for o in obs])) for k in space.spaces.keys()]
        )

    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(
            obs[0], tuple
        ), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple(stacking_fn([o[i] for o in obs]) for i in range(obs_len))

    else:
        return stacking_fn(obs)


def clear_pending_list(pending_list):
    """Wait for all remaining elements in pending list, and clear."""
    num_ret_ = len(pending_list)
    ray.wait(list(pending_list.keys()), num_returns=num_ret_)
    pending_list.clear()


def process_ray_env_output(raw_output: List[object], obs_space: gym.Space):
    """Unpack ray object reference and stack to generate np.array."""
    unpack = [ray.get(ref_) for ref_ in raw_output]

    stacking_fn = np.stack if unpack[0].ndim == len(obs_space.shape) else np.concatenate
    return _flatten_obs(unpack, obs_space, stacking_fn=stacking_fn)


@ray.remote
def mock_ray(obj):
    return obj
