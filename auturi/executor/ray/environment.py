import math
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import numpy as np
import ray

import auturi.executor.ray.util as util
from auturi.executor.environment import AuturiSerialEnv, AuturiVectorEnv
from auturi.executor.vector_utils import aggregate_partial


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

    def _create_worker(self, idx):
        return RayEnvWrapper.remote(idx, self.env_fns)

    def _set_working_env(self, wid, remote_env, start_idx, num_envs):
        ref = remote_env.set_working_env.remote(start_idx, num_envs)
        self.pending_steps[ref] = wid

    def reset(self, to_return=True):
        util.clear_pending_list(self.pending_steps)
        self.last_output.clear()
        self.pending_steps = {
            env_worker.reset.remote(): wid
            for wid, env_worker in self._working_workers()
        }
        if to_return:
            return util.process_ray_env_output(
                list(self.pending_steps.keys()),
                self.observation_space,
            )

    def seed(self, seed: int):
        util.clear_pending_list(self.pending_steps)
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
            step_ref_ = self._get_worker(wid).step.remote(action_ref, lid)
            self.pending_steps[step_ref_] = wid  # update pending list

    def aggregate_rollouts(self):
        util.clear_pending_list(self.pending_steps)
        partial_rollouts = [
            worker.aggregate_rollouts.remote() for _, worker in self._working_workers()
        ]

        partial_rollouts = ray.get(partial_rollouts)
        return aggregate_partial(partial_rollouts, already_agg=True)

    def start_loop(self):
        self.reset(to_return=False)

    def stop_loop(self):
        util.clear_pending_list(self.pending_steps)

    def step(self, actions: Tuple[np.ndarray]):
        """Synchronous step wrapper, just for debugging purpose."""
        assert len(actions[0]) == self.num_envs
        self.batch_size = self.num_envs

        if len(self.last_output) == 0:
            self.last_output = {wid: None for wid, _ in self._working_workers()}

        util.clear_pending_list(self.pending_steps)
        self.send_actions(util.mock_ray.remote(actions))

        raw_output = self.poll()
        sorted_output = OrderedDict(sorted(raw_output.items()))

        return util.process_ray_env_output(
            list(sorted_output.values()),
            self.observation_space,
        )
