import time
from typing import Any, Dict, List, Tuple

import numpy as np

import auturi.executor.shm.util as util
import auturi.executor.typing as types
from auturi.executor.loop import MultiLoopHandler, NestedLoopHandler, SimpleLoopHandler
from auturi.executor.shm.constant import ActorCommand
from auturi.executor.shm.environment import SHMParallelEnv
from auturi.executor.shm.mp_mixin import SHMProcMixin, SHMVectorMixin
from auturi.executor.shm.policy import SHMVectorPolicy
from auturi.tuner import AuturiMetric, ParallelizationConfig


class SHMNestedLoopHandler(NestedLoopHandler):
    def __init__(
        self,
        loop_id,
        env_fns,
        policy_cls,
        policy_kwargs,
        max_num_envs,
        max_num_policy,
        num_rollouts,
        rollout_buffer_attr=None,
    ):
        super().__init__(loop_id, env_fns, policy_cls, policy_kwargs)
        self.max_num_envs = max_num_envs
        self.max_num_policy = max_num_policy
        self.base_buffers, self.base_buffer_attr = util.create_data_buffer_from_env(
            env_fns[0], max_num_envs
        )
        if rollout_buffer_attr is None:
            (
                self.rollout_buffers,
                self.rollout_buffer_attr,
            ) = util.create_rollout_buffer_from_env(env_fns[0], num_rollouts)

    def _create_env_handler(self):
        return SHMParallelEnv(
            self.loop_id,
            self.env_fns,
            self.base_buffers,
            self.base_buffer_attr,
            self.rollout_buffers,
            self.rollout_buffer_attr,
            self.max_num_envs,
        )

    def _create_policy_handler(self):
        return SHMVectorPolicy(
            self.loop_id,
            self.policy_cls,
            self.policy_kwargs,
            self.base_buffers,
            self.base_buffer_attr,
            self.max_num_policy,
        )

    def terminate(self):
        super().terminate()
        for _, tuple_ in self.base_buffers.items():
            tuple_[0].unlink()

        for _, tuple_ in self.rollout_buffers.items():
            tuple_[0].unlink()


class SimpleLoopProc(SHMProcMixin):
    """Child process that runs a simple loop."""

    def __init__(
        self,
        worker_id,
        cmd_attr_dict,
        env_fns,
        policy_cls,
        policy_kwargs,
        rollout_buffer_attr,
    ):
        self.env_fns = env_fns
        self.policy_cls = policy_cls
        self.policy_kwargs = policy_kwargs

        self.rollout_buffer_attr = rollout_buffer_attr
        self.rollout_buffers = util.set_rollout_buffer_from_attr(rollout_buffer_attr)
        self._start_idx, self._end_idx = -1, -1

        super().__init__(worker_id, cmd_attr_dict=cmd_attr_dict)

    def initialize(self) -> None:
        cls = SimpleLoopHandler
        self._loop_handler = cls(
            self.worker_id, self.env_fns, self.policy_cls, self.policy_kwargs
        )

        SHMProcMixin.initialize(self)

    @property
    def proc_name(self) -> str:
        return f"Loop(id={self.worker_id})"

    def set_command_handlers(self) -> None:
        self.cmd_handler[ActorCommand.RECONFIGURE] = self.reconfigure_handler
        self.cmd_handler[ActorCommand.RUN] = self.run_handler

    def reconfigure_handler(self, cmd: int, _) -> None:
        config = util.convert_buffer_to_config(self._command_buffer[:, 1:])

        # Update rollout buffer index
        self._start_idx = config.compute_index_for_actor("num_collect", self.worker_id)
        self._end_idx = self._start_idx + config[self.worker_id].num_collect

        # TODO: FIXME
        self._loop_handler.num_collect = config[self.worker_id].num_collect

        self._loop_handler.reconfigure(config, model=None)
        self._logger.debug("Reconfigure.. sync done")
        self.reply(cmd)

    def run_handler(self, cmd: int, _) -> None:
        local_rollouts, metric = self._loop_handler.run()
        for key_, trajectories in local_rollouts.items():
            roll_buffer = self.rollout_buffers[key_][1]

            # TODO: stack first
            np.copyto(roll_buffer[self._start_idx : self._end_idx], trajectories)
            # print(key_, type(self.rollout_buffers[key_][0]), \
            #     self.rollout_buffers[key_][0].name, f" after copy to buffer => {roll_buffer[self._start_idx : self._end_idx]}")

        self.reply(cmd)

    def _term_handler(self, cmd: int, data_list: List[int]) -> None:
        self._logger.info("TERM handler is called...")
        self._loop_handler.terminate()
        super()._term_handler(cmd, data_list)


MAX_ACTOR_DATA = 7


class SHMMultiLoopHandler(MultiLoopHandler, SHMVectorMixin):
    def __init__(
        self, env_fns, policy_cls, policy_kwargs, max_num_loop: int, num_rollouts: int
    ):
        MultiLoopHandler.__init__(self, env_fns, policy_cls, policy_kwargs)
        SHMVectorMixin.__init__(self, max_num_loop, MAX_ACTOR_DATA)
        (
            self.rollout_buffers,
            self.rollout_buffer_attr,
        ) = util.create_rollout_buffer_from_env(env_fns[0], num_rollouts)

    @property
    def num_actors(self):
        return self.num_workers

    @property
    def proc_name(self) -> str:
        return "SHMMultiLoopHandler"

    def reconfigure(
        self, config: ParallelizationConfig, model: types.PolicyModel
    ) -> None:
        self._logger.info(f" Reconfigure {config}\n")
        self.num_collect = config.num_collect
        util.copy_config_to_buffer(config, self._command_buffer[:, 1:])
        self.reconfigure_workers(config.num_actors, config=config, model=model)
        self._logger.info("Called VectorMixin.reconfigure_workers")
        self.sync()  # sync

    def _create_worker(self, worker_id: int) -> SimpleLoopProc:
        kwargs = {
            "env_fns": self.env_fns,
            "policy_cls": self.policy_cls,
            "policy_kwargs": self.policy_kwargs,
            "rollout_buffer_attr": self.rollout_buffer_attr,
        }
        return self.init_proc(worker_id, SimpleLoopProc, kwargs)

    def _reconfigure_worker(
        self,
        worker_id: int,
        worker: SimpleLoopProc,
        config: ParallelizationConfig,  # do not need
        model: types.PolicyModel,  # do not need
    ) -> None:
        self.request(
            ActorCommand.RECONFIGURE,
            worker_id=worker_id,
        )

        self._logger.info(f"RECONFIGURE({worker_id})")


    def run(self) -> Tuple[Dict[str, Any], AuturiMetric]:
        self._logger.info(f"\n\n============================RUN\n")

        start_time = time.perf_counter()
        self.request(ActorCommand.RUN)
        self._logger.debug("Set command RUN")

        self.sync()
        rollouts = self._aggregate_rollouts()
        self.sync()
        end_time = time.perf_counter()

        # Aggregate
        return rollouts, AuturiMetric(self.num_collect, end_time - start_time)

    # TODO: Does not copy properly. Why?
    def _aggregate_rollouts(self) -> Dict[str, np.ndarray]:
        ret_dict = dict()
        for key, tuple_ in self.rollout_buffers.items():
            ret_dict[key] = tuple_[1]

        return ret_dict

    def terminate(self):
        self._logger.info("Handler terminates")

        SHMVectorMixin.terminate(self)
        for _, tuple_ in self.rollout_buffers.items():
            tuple_[0].unlink()
