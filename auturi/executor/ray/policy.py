from typing import Any, Dict

import ray

import auturi.executor.ray.util as util
from auturi.executor.policy import AuturiPolicy, AuturiPolicyHandler
from auturi.executor.typing import PolicyModel
from auturi.executor.vector_utils import VectorMixin
from auturi.tuner.config import ParallelizationConfig


class RayVectorPolicy(AuturiPolicyHandler, VectorMixin):
    def __init__(self, actor_id, policy_cls, policy_kwargs):
        self.pending_policies = dict()
        AuturiPolicyHandler.__init__(self, actor_id, policy_cls, policy_kwargs)
        VectorMixin.__init__(self)

    def reconfigure(self, config: ParallelizationConfig, model: PolicyModel):
        new_num_workers = config[self.actor_id].num_policy
        device = config[self.actor_id].policy_device
        VectorMixin.reconfigure_workers(
            self, new_num_workers=new_num_workers, model=model, device=device
        )

    def _create_worker(self, worker_id: int):
        @ray.remote(num_gpus=0.0001)
        class RayPolicy(self.policy_cls):
            """Wrappers run in separated Ray process."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert hasattr(self, "observation_space")

            def compute_actions(self, obs_refs, n_steps):
                env_obs = util.process_ray_env_output(
                    list(obs_refs.values()), self.observation_space
                )
                return super().compute_actions(env_obs, n_steps)

        self.policy_kwargs["idx"] = worker_id
        return RayPolicy.remote(**self.policy_kwargs)

    def _reconfigure_worker(
        self, worker_id: int, worker: AuturiPolicy, model: PolicyModel, device: str
    ):
        ref = worker.load_model.remote(model, device)
        self.pending_policies[ref] = worker_id

    def _terminate_worker(self, worker_id: int, worker: Any):
        del worker

    def compute_actions(self, obs_refs: Dict[int, object], n_steps: int):
        free_policies, _ = ray.wait(list(self.pending_policies.keys()))
        policy_id = self.pending_policies.pop(free_policies[0])
        free_policy = self.get_worker(policy_id)
        action_refs = free_policy.compute_actions.remote(obs_refs, n_steps)
        self.pending_policies[action_refs] = policy_id

        return action_refs

    def start_loop(self):
        util.clear_pending_list(self.pending_policies)
        for wid, _ in self.workers():
            self.pending_policies[util.mock_ray.remote(None)] = wid

    def stop_loop(self):
        for wid, policy_worker in self.workers():
            self._load_policy_model(wid, policy_worker, None, "cpu")

        util.clear_pending_list(self.pending_policies)

    def terminate(self):
        util.clear_pending_list(self.pending_policies)
        for worker_id, worker in self.workers():
            self._terminate_worker(worker_id, worker)
