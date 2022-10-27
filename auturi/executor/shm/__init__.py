from typing import Any, Callable, Dict, List, Tuple

import numpy as np

import auturi.executor.shm.util as shm_util
from auturi.executor.environment import AuturiEnv, AuturiVectorEnv
from auturi.executor.policy import AuturiVectorPolicy
from auturi.executor.shm.environment import SHMParallelEnv
from auturi.executor.shm.policy import SHMVectorPolicy

# FIXME
MAX_NUM_POLICY = 64
MAX_ROLLOUT_SIZE = 256


def create_shm_actor_args(
    env_fns: List[Callable[[], AuturiEnv]],
    policy_cls: Any,
    policy_kwargs: Dict[str, Any],
) -> Tuple[Callable[[], AuturiVectorEnv], Callable[[], AuturiVectorPolicy]]:

    # Collect sample data
    dummy_env = env_fns[0]()  # Single environment (not serial)
    dummy_policy = policy_cls(**policy_kwargs)
    obs = dummy_env.reset()

    action, action_artifacts = dummy_policy.compute_actions(obs, n_steps=1)
    assert len(action_artifacts) == 1  # FIXME

    dummy_env.step(action, action_artifacts)
    rollouts = dummy_env.aggregate_rollouts()

    # Add basic buffers
    max_num_envs = len(env_fns)
    buffer_sample_dict = {
        "obs": (obs, max_num_envs),
        "action": (action, max_num_envs),
        "artifacts": (action_artifacts[0], max_num_envs),
        "env": (
            np.array([1, 1, 1, 1], dtype=np.int64),
            max_num_envs,
        ),  # cmd, state, data1, data2
        "policy": (
            np.array([1, 1, 1], dtype=np.int32),
            MAX_NUM_POLICY,
        ),  # cmd, state, data
    }

    # Add rollout-related buffers
    for key, rollout in rollouts.items():
        shm_key = "roll_" + key
        assert not isinstance(rollout, np.ndarray) and isinstance(
            rollout[0], np.ndarray
        )
        buffer_sample_dict[shm_key] = (rollout[0], MAX_ROLLOUT_SIZE)

    shm_buffer_dict, shm_buffer_attr_dict = shm_util.create_shm_buffer_from_dict(
        buffer_sample_dict
    )
    print("buffer dict: ", list(shm_buffer_dict.keys()))
    print("shm_buffer_attr_dict dict: ", list(shm_buffer_attr_dict.keys()))

    def create_shm_env(env_fns):
        def _wrap():
            return SHMParallelEnv(env_fns, shm_buffer_dict, shm_buffer_attr_dict)

        return _wrap

    def create_shm_policy(policy_cls, policy_kwargs):
        def _wrap():
            return SHMVectorPolicy(
                policy_cls, policy_kwargs, shm_buffer_dict, shm_buffer_attr_dict
            )

        return _wrap

    return create_shm_env(env_fns), create_shm_policy(policy_cls, policy_kwargs)


__all__ = ["SHMParallelEnv", "SHMVectorPolicy", "create_shm_actor_args"]
