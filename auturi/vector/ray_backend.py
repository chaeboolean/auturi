import math
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import gym
import numpy as np
import ray
import torch.nn as nn

from auturi.typing.environment import AuturiSerialEnv, AuturiVectorEnv
from auturi.typing.policy import AuturiPolicy, AuturiVectorPolicy


def _flatten_obs(obs, space, to_stack=True) -> None:
    """Borrowed from Stable-baselines3 SubprocVec implementation."""

    stacking_fn = np.stack if to_stack else np.concatenate
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


def _clear_pending_list(pending_list):
    """Wait for all remaining elements in pending list, and clear."""
    num_ret_ = len(pending_list)
    ray.wait(list(pending_list.keys()), num_returns=num_ret_)
    pending_list.clear()


def _process_ray_env_output(
    raw_output: List[object], obs_space: gym.Space, to_stack: bool = True
):
    """Unpack ray object reference and stack to generate np.array."""
    unpack = [ray.get(ref_) for ref_ in raw_output]

    return _flatten_obs(unpack, obs_space, to_stack=to_stack)


@ray.remote
def _mock_policy(obj):
    return obj


@ray.remote
class RayEnvWrapper(AuturiSerialEnv):
    """SerialEnv used by RayParallelEnv.
    It inherits step function.

    """

    def step(self, actions_, lid=-1):
        # action_ref here is already np.nd.array
        action, action_artifacts = actions_
        my_action = action[lid : lid + self.num_envs]
        my_artifacts = [elem[lid : lid + self.num_envs] for elem in action_artifacts]

        return super().step([my_action, my_artifacts])


class RayParallelEnv(AuturiVectorEnv):
    """RayParallelVectorEnv that uses Ray as backend."""

    def __init__(self, env_fns: List[Callable]):
        super().__init__(env_fns)
        self.pending_steps = dict()
        self.last_output = dict()

    def _create_env_worker(self, idx):
        return RayEnvWrapper.remote(idx, self.env_fns)

    def _set_working_env(self, wid, remote_env, start_idx, num_envs):
        ref = remote_env.set_working_env.remote(start_idx, num_envs)
        self.pending_steps[ref] = wid

    def reset(self, to_return=True):
        _clear_pending_list(self.pending_steps)
        self.last_output.clear()
        self.pending_steps = {
            env_worker.reset.remote(): wid
            for wid, env_worker in self._working_workers()
        }
        if to_return:
            return _process_ray_env_output(
                list(self.pending_steps.keys()),
                self.observation_space,
                self.num_env_serial <= 1,
            )

    def seed(self, seed: int):
        _clear_pending_list(self.pending_steps)
        for wid, env_worker in self._working_workers():
            ref = env_worker.seed.remote(seed)
            self.pending_steps[ref] = wid

    def poll(self) -> Dict[object, int]:
        num_to_return = math.ceil(self.batch_size / self.num_env_serial)
        assert len(self.pending_steps) >= num_to_return

        done_envs, _ = ray.wait(list(self.pending_steps), num_returns=num_to_return)

        self.last_output = {
            self.pending_steps.pop(done_envs[i]): done_envs[i]  # (wid, step_ref)
            for i in range(num_to_return)
        }

        return self.last_output

    def send_actions(self, action_ref) -> None:
        for lid, wid in enumerate(self.last_output.keys()):
            step_ref_ = self._get_env_worker(wid).step.remote(action_ref, lid)
            self.pending_steps[step_ref_] = wid  # update pending list

    def aggregate_rollouts(self):
        _clear_pending_list(self.pending_steps)
        partial_rollouts = [
            worker.aggregate_rollouts.remote()
            for idx, worker in self._working_workers()
        ]

        dones = ray.get(partial_rollouts)
        dones = list(filter(lambda elem: len(elem) > 0, dones))

        keys = list(dones[0].keys())
        buffer_dict = dict()
        for key in keys:
            li = []
            for done in dones:
                li.append(done[key])
            buffer_dict[key] = np.concatenate(li)

        return buffer_dict

    def start_loop(self):
        self.reset(to_return=False)

    def step(self, actions: Tuple[np.ndarray]):
        """Synchronous step wrapper, just for debugging purpose."""
        assert len(actions[0]) == self.num_envs
        self.batch_size = self.num_envs

        if len(self.last_output) == 0:
            self.last_output = {wid: None for wid, _ in self._working_workers()}

        _clear_pending_list(self.pending_steps)
        self.send_actions(_mock_policy.remote(actions))

        raw_output = self.poll()
        sorted_output = OrderedDict(sorted(raw_output.items()))

        return _process_ray_env_output(
            list(sorted_output.values()),
            self.observation_space,
            self.num_env_serial <= 1,
        )


class RayVectorPolicies(AuturiVectorPolicy):
    def __init__(self, policy_cls, policy_kwargs):
        super().__init__(policy_cls, policy_kwargs)
        self.pending_policies = dict()

    def _create_policy_worker(self, idx: int):
        @ray.remote(num_gpus=0.2)
        class RayPolicyWrapper(self.policy_cls):
            """Wrappers run in separated Ray process."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert hasattr(self, "observation_space")

            def compute_actions(self, obs_refs, n_steps):
                env_obs = _process_ray_env_output(
                    list(obs_refs.values()), self.observation_space
                )
                return super().compute_actions(env_obs, n_steps)

        return RayPolicyWrapper.remote(**self.policy_kwargs, idx=idx)

    def _load_policy_model(
        self, idx: int, policy: AuturiPolicy, model: nn.Module, device: str
    ):
        ref = policy.load_model.remote(model, device)
        self.pending_policies[ref] = idx

    def compute_actions(self, obs_refs: Dict[int, object], n_steps: int):
        free_policies, _ = ray.wait(list(self.pending_policies.keys()))
        policy_id = self.pending_policies.pop(free_policies[0])
        free_policy = self._get_policy_worker(policy_id)

        action_refs = free_policy.compute_actions.remote(obs_refs, n_steps)
        self.pending_policies[action_refs] = policy_id

        return action_refs

    def start_loop(self):
        _clear_pending_list(self.pending_policies)
        for wid, _ in self._working_workers():
            self.pending_policies[_mock_policy.remote(None)] = wid

    def stop_loop(self):
        _clear_pending_list(self.pending_policies)
