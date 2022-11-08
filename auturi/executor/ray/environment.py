from collections import OrderedDict
from typing import Callable, Dict, List

import numpy as np
import ray

import auturi.executor.ray.util as util
from auturi.executor.environment import AuturiSerialEnv, AuturiVectorEnv
from auturi.executor.vector_utils import aggregate_partial
from auturi.tuner.config import ParallelizationConfig


@ray.remote
class RaySerialEnv(AuturiSerialEnv):
    """SerialEnv used by RayParallelEnv.
    It inherits step function.

    """

    def step(self, action_ref, local_id=-1):
        # action_ref here is already unpacked.
        actions, action_artifacts = action_ref
        action_for_this_env = actions[local_id : local_id + self.num_envs]
        my_artifacts = [
            elem[local_id : local_id + self.num_envs] for elem in action_artifacts
        ]

        return super().step(action_for_this_env, my_artifacts)


class RayParallelEnv(AuturiVectorEnv):
    """Implementation of the AuturiVectorEnv for using Ray as a backend.."""

    def __init__(self, actor_id: int, env_fns: List[Callable]):
        super().__init__(actor_id, env_fns)

        # pending_steps: Dict[ray.future, worker_id]
        # Stores the remote env.step() calls, and maps the future with worker_id
        self.pending_steps = dict()

        # last_output: Dict[worker_id, ray.future]
        # Stores the output of lastly called poll(), for following send_action() call.
        self.last_output = dict()

    def _create_worker(self, worker_id: int):
        print("create!!!  - - - - - - -- - - - - --  --")

        return RaySerialEnv.remote(self.actor_id, worker_id, self.env_fns)

    def _reconfigure_worker(
        self, worker_id: int, worker: RaySerialEnv, config: ParallelizationConfig
    ):
        print("reconfigure!!!  - - - - - - -- - - - - --  --")

        actor_env_offset = config.compute_index_for_actor("num_envs", self.actor_id)
        start_idx = actor_env_offset + self.num_env_serial * worker_id
        ref = worker.set_working_env.remote(start_idx, self.num_env_serial)

        self.pending_steps[ref] = worker_id
        print("reconfigure!!!")

    def _terminate_worker(self, worker_id: int, worker: RaySerialEnv):
        del worker

    def reset(self, to_return=True):
        util.clear_pending_list(self.pending_steps)
        self.last_output.clear()
        self.pending_steps = {
            env_worker.reset.remote(): wid for wid, env_worker in self.workers()
        }
        if to_return:
            return util.process_ray_env_output(
                list(self.pending_steps.keys()),
                self.observation_space,
            )

    def seed(self, seed: int):
        util.clear_pending_list(self.pending_steps)
        for wid, env_worker in self.workers():
            ref = env_worker.seed.remote(seed)
            self.pending_steps[ref] = wid

    def poll(self) -> Dict[object, int]:
        assert len(self.pending_steps) >= self.num_worker_to_poll

        done_envs, _ = ray.wait(
            list(self.pending_steps), num_returns=self.num_worker_to_poll
        )

        self.last_output = {
            self.pending_steps.pop(done_envs[i]): done_envs[i]  # (wid, step_ref)
            for i in range(self.num_worker_to_poll)
        }
        return self.last_output

    def send_actions(self, action_ref) -> None:
        for i, wid in enumerate(self.last_output.keys()):
            step_ref_ = self.get_worker(wid).step.remote(action_ref, i)
            self.pending_steps[step_ref_] = wid  # update pending list

    def aggregate_rollouts(self):
        util.clear_pending_list(self.pending_steps)
        partial_rollouts = [
            worker.aggregate_rollouts.remote() for _, worker in self.workers()
        ]

        partial_rollouts = ray.get(partial_rollouts)
        return aggregate_partial(partial_rollouts)

    def start_loop(self):
        util.clear_pending_list(self.pending_steps)
        self.reset(to_return=False)

    def stop_loop(self):
        util.clear_pending_list(self.pending_steps)

    def terminate(self):
        util.clear_pending_list(self.pending_steps)
        for worker_id, worker in self.workers():
            self._terminate_worker(worker_id, worker)

    def step(self, action: np.ndarray, action_artifacts: List[np.ndarray]):
        """Synchronous step wrapper, just for debugging purpose."""
        assert len(action) == self.num_envs
        self.batch_size = self.num_envs

        # When step() is first called.
        if len(self.last_output) == 0:
            self.last_output = {wid: None for wid, _ in self.workers()}

        util.clear_pending_list(self.pending_steps)
        self.send_actions(util.mock_ray.remote((action, action_artifacts)))

        raw_output = self.poll()
        sorted_output = OrderedDict(sorted(raw_output.items()))

        return util.process_ray_env_output(
            list(sorted_output.values()),
            self.observation_space,
        )
