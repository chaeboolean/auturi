import functools
import time
from collections import OrderedDict
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple, Type,
                    Union)

import gym
import numpy as np
import ray

from auturi.typing.simulator import AuturiEnv, AuturiParallelEnv
from auturi.typing.policy import AuturiPolicy, AuturiVectorPolicy



def _flatten_obs(obs, space) -> None:
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


def _clear_pending_list(pending_list):
    """Wait for all remaining elements in pending list, and clear."""
    num_ret_ = len(pending_list)
    ray.wait(list(pending_list.keys()), num_returns=num_ret_)
    pending_list.clear()
    

def _process_ray_env_output(raw_output: Dict[int, object], obs_space: gym.Space):
    """Unpack ray object reference and stack to generate np.array."""
    unpack = [ray.get(ref_) for ref_ in raw_output.values()]
    # if len(unpack[0]) == 4:  
    #     obs, rews, dones, infos = zip(*unpack)
    # return _flatten_obs(obs, obs_space), np.stack(rews), np.stack(dones), infos

    return _flatten_obs(unpack, obs_space)


@ray.remote
class RayEnvWrapper(AuturiEnv):
    """Environment to Ray Actor"""
    def __init__(self, idx, env_fn):
        self.env_id = idx
        self.env = env_fn()
        assert isinstance(self.env, AuturiEnv)

    def step(self, action, lid=-1):
        # action_ref here is already np.nd.array
        action_, action_artifacts = action
        my_action = action_[lid]
        my_artifacts = [elem[lid] for elem in action_artifacts]
        observation = self.env.step(my_action, my_artifacts)
        return observation

    def reset(self):
        return self.env.reset()

    def seed(self, seed):
        self.env.seed(seed)
        
    def close(self): 
        self.env.close()
        
    def fetch_rollouts(self):
        return self.env.fetch_rollouts()


class RayParallelEnv(AuturiParallelEnv):
    """RayParallelVectorEnv that uses Ray as backend."""
    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        self.remote_envs = {
            i: RayEnvWrapper.remote(i, env_fn_) for i, env_fn_ in enumerate(env_fns)
        }
        
        super().__init__(env_fns)        
        self.pending_steps = dict()
        self.last_output = {eid: 325465 for eid in range(self.num_envs)}


    def reset(self):
        _clear_pending_list(self.pending_steps)
        self.pending_steps = {
            env.reset.remote(): eid for eid, env in self.remote_envs.items()
        }

    def seed(self, seed: int):
        assert len(self.pending_steps) == 0
        self._set_seed({eid: seed + eid for eid in range(self.num_envs)})


    def _set_seed(self, seed_dict: Dict[int, int]):
        futs = []
        for eid, eseed in seed_dict.items():
            futs.append(self.remote_envs[eid].seed.remote(eseed))

        ray.wait(futs, num_returns=len(futs))

    def _poll(self, bs: int = -1) -> Dict[object, int]:
        assert len(self.pending_steps) >= bs
        done_envs, _ = ray.wait(list(self.pending_steps), num_returns=bs)

        self.last_output = {
            self.pending_steps.pop(done_envs[i]): done_envs[i]  # (eid, step_ref)
            for i in range(bs)
        }
        return self.last_output


    def send_actions(self, action_ref):
        for lid, eid in enumerate(self.last_output.keys()):
            step_ref_ = self.remote_envs[eid].step.remote(action_ref, lid)
            self.pending_steps[step_ref_] = eid  # update pending list
            

    def aggregate_rollouts(self):
        _clear_pending_list(self.pending_steps)
        partial_rollouts = [env.fetch_rollouts.remote() for env in self.remote_envs.values()]
        
        dones = ray.get(partial_rollouts)
        dones = list(filter(lambda elem: len(elem) > 0, dones))

        keys = list(dones[0].keys())
        buffer_dict = dict()
        for key in keys:
            buffer_dict[key] = np.concatenate([done[key] for done in dones])
        
        return buffer_dict        
        

    def step(self, actions: np.ndarray):
        """Synchronous step wrapper, just for debugging purpose."""        

        @ray.remote
        def mock_policy():
            return actions
        
        _clear_pending_list(self.pending_steps)
        self.send_actions(mock_policy.remote())
        raw_output = self.poll(bs=self.num_envs)
        sorted_output = OrderedDict(sorted(raw_output.items()))
        return _process_ray_env_output(sorted_output, self.observation_space)

    def start_loop(self):
        self.reset()



@ray.remote(num_gpus=1)
class RayPolicyWrapper(AuturiPolicy):
    """Wrappers run in separated Ray process."""

    def __init__(self, idx, policy_fn):
        self.init_finish = False
        self.policy_id = idx
        self.policy = policy_fn()
        assert isinstance(self.policy, AuturiPolicy)


    def load_model(self, device="cpu"):
        self.policy.load_model(device)

    def compute_actions(self, obs_refs, n_steps):

        env_obs = _process_ray_env_output(
            obs_refs, self.policy.observation_space
        )
        
        return self.policy.compute_actions(
            env_obs, n_steps
        )
        

class RayVectorPolicies(AuturiVectorPolicy):
    def __init__(self, num_policies: int, policy_fn: Callable):
        super().__init__(num_policies, policy_fn)
        self.remote_policies = {
            i: RayPolicyWrapper.remote(i, policy_fn) for i in range(num_policies)
        }
        
        self.pending_policies = dict()

    def load_model_from_path(self, device="cpu"):
        _clear_pending_list(self.pending_policies)
        self.pending_policies = {
            pol.load_model.remote(device): pid for pid, pol in self.remote_policies.items()
        }


    def start_loop(self, device="cpu"):
        self.load_model_from_path()
        return super().start_loop()

    def assign_free_server(self, obs_refs: Dict[int, object], n_steps: int):
        free_servers, _ = ray.wait(list(self.pending_policies.keys()))
        server_id = self.pending_policies.pop(free_servers[0])
        free_server = self.remote_policies[server_id]

        action_refs = free_server.compute_actions.remote(obs_refs, n_steps)
        self.pending_policies[action_refs] = server_id

        return action_refs
