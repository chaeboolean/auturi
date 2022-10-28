from typing import Dict

import ray
import torch.nn as nn

import auturi.executor.ray.util as util
from auturi.executor.policy import AuturiPolicy, AuturiVectorPolicy


class RayVectorPolicy(AuturiVectorPolicy):
    def __init__(self, policy_cls, policy_kwargs):
        super().__init__(policy_cls, policy_kwargs)
        self.pending_policies = dict()

    def _create_worker(self, idx: int):
        @ray.remote(num_gpus=0.2)
        class RayPolicyWrapper(self.policy_cls):
            """Wrappers run in separated Ray process."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert hasattr(self, "observation_space")

            def compute_actions(self, obs_refs, n_steps):
                env_obs = util.process_ray_env_output(
                    list(obs_refs.values()), self.observation_space
                )
                return super().compute_actions(env_obs, n_steps)

        return RayPolicyWrapper.remote(**self.policy_kwargs)

    def _load_policy_model(
        self, idx: int, policy: AuturiPolicy, model: nn.Module, device: str
    ):
        ref = policy.load_model.remote(model, device)
        self.pending_policies[ref] = idx

    def compute_actions(self, obs_refs: Dict[int, object], n_steps: int):
        free_policies, _ = ray.wait(list(self.pending_policies.keys()))
        policy_id = self.pending_policies.pop(free_policies[0])
        free_policy = self._get_worker(policy_id)
        action_refs = free_policy.compute_actions.remote(obs_refs, n_steps)
        self.pending_policies[action_refs] = policy_id

        return action_refs

    def start_loop(self):
        util.clear_pending_list(self.pending_policies)
        for wid, _ in self._working_workers():
            self.pending_policies[util.mock_ray.remote(None)] = wid

    def stop_loop(self):
        util.clear_pending_list(self.pending_policies)
