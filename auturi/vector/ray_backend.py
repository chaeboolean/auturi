import functools
import time
from collections import OrderedDict
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple, Type,
                    Union)

import gym
import numpy as np
import ray

from auturi.typing.policy import AuturiVectorPolicy
from auturi.typing.simulator import AuturiParallelEnv

# from auturi.typing.auxilary



def _clear_pending_list(pending_list):
    num_ret_ = len(pending_list)
    ray.wait(list(pending_list.keys()), num_returns=num_ret_)
    pending_list.clear()


@ray.remote
class RayEnvWrapper:
    def __init__(self, idx, env_fn):
        self.env_id = idx
        self.env = env_fn()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if done:
            # save final observation where user can get it, then reset
            info["terminal_observation"] = observation
            observation = self.env.reset()
        return (observation, reward, done, info)

    def reset(self):
        return self.env.reset()

    def seed(self, seed):
        self.env.seed(seed)


class RayParallelEnv(AuturiParallelEnv):
    """RayVectorEnv

    Uses Ray as backend
    """

    def _create_env(self, index, env_fn):
        def _wrap_env():
            return RayEnvWrapper.remote(index, env_fn)

        return _wrap_env()

    def _setup(self, dummy_env):
        self.observation_space = dummy_env.observation_space
        self.action_space = dummy_env.action_space
        self.metadata = dummy_env.metadata
        self.pending_steps = dict()

        # print(self.observation_space, self.action_space)

    def reset(self):
        _clear_pending_list(self.pending_steps)

        assert len(self.pending_steps) == 0
        self.pending_steps = {
            env.reset.remote(): eid for eid, env in self.remote_envs.items()
        }

    def seed(self, seed: int):
        self._set_seed({eid: seed + eid for eid in range(self.num_envs)})

    def _set_seed(self, seed_dict: Dict[int, int]):
        futs = []
        for eid, eseed in seed_dict.items():
            futs.append(self.remote_envs[eid].seed.remote(eseed))

        ray.wait(futs, num_returns=len(futs))

    def poll(self, bs: int = -1) -> Dict[object, int]:

        if bs < 0:
            bs = self.num_envs
        assert len(self.pending_steps) >= bs

        done_envs, _ = ray.wait(list(self.pending_steps), num_returns=bs)

        output = {
            self.pending_steps.pop(done_envs[i]): done_envs[i]  # (eid, step_ref)
            for i in range(bs)
        }
        return output

    def send_actions(self, action_dict):
        for eid, action in action_dict.items():
            step_ref_ = self.remote_envs[eid].step.remote(action)
            self.pending_steps[step_ref_] = eid  # update pending list

    def step(self, actions: np.ndarray):
        """For debugging. Synchronous step wrapper."""
        assert len(actions) == self.num_envs
        action_dict = {eid: action_ for eid, action_ in enumerate(actions)}

        _clear_pending_list(self.pending_steps)
        self.send_actions(action_dict)

        raw_output = self.poll(bs=self.num_envs)

        raw_output = OrderedDict(sorted(raw_output.items()))
        return self._process_output(raw_output)

    def _process_output(self, raw_output: Dict[int, object]):
        unpack = [ray.get(ref_) for ref_ in raw_output.values()]
        obs, rews, dones, infos = zip(*unpack)
        return self._flatten_obs(obs), np.stack(rews), np.stack(dones), infos

    def _flatten_obs(self, obs) -> None:
        """Borrowed from Stable-baselines3 SubprocVec implementation."""

        space = self.obs_space

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
                [(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()]
            )
        elif isinstance(space, gym.spaces.Tuple):
            assert isinstance(
                obs[0], tuple
            ), "non-tuple observation for environment with Tuple observation space"
            obs_len = len(space.spaces)
            return tuple(np.stack([o[i] for o in obs]) for i in range(obs_len))
        else:
            return np.stack(obs)


@ray.remote
class RayPolicyWrapper:
    def __init__(self, policy_fn):
        self.policy = policy_fn()

    def reset(self):
        pass

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if done:
            # save final observation where user can get it, then reset
            info["terminal_observation"] = observation
            observation = self.env.reset()
        return (observation, reward, done, info)


class RayVectorPolicies(AuturiVectorPolicy):
    def _create_policy(self, index, policy_fn):
        def _wrap_policy():
            return RayPolicyWrapper.remote(index, policy_fn)

        return _wrap_policy()

    def _setup(self):
        self.pending_policies = dict()
        self.reset()

    def reset(self):
        _clear_pending_list(self.pending_policies)
        self.pending_policies = {
            pol.reset.remote(): pid for pid, pol in self.remote_policies.items()
        }

    def assign_free_server(self, obs_refs: Dict[object, int]):
        free_servers, _ = ray.wait(list(self.pending_policies))
        server_id = self.pending_policies.pop(free_servers[0])
        free_server = self.remote_policies[server_id]

        action_refs = free_server.compute_actions.remote(obs_refs)
        return action_refs, free_server
