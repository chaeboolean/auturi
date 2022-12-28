from typing import Any, List, Tuple, Dict
import  time

import numpy as np

import auturi.executor.shm.util as util
import auturi.executor.typing as types
from auturi.executor.loop import MultiLoopHandler, NestedLoopHandler, SimpleLoopHandler
from auturi.executor.shm.constant import ActorCommand
from auturi.executor.shm.environment import SHMParallelEnv
from auturi.executor.shm.mp_mixin import SHMProcMixin, SHMVectorMixin
from auturi.executor.shm.policy import SHMVectorPolicy
from auturi.tuner import ParallelizationConfig, AuturiMetric


class SHMNestedLoopHandler(NestedLoopHandler):
    def __init__(self, env_fns, policy_cls, policy_kwargs, max_num_envs, max_rollouts):
        super().__init__(env_fns, policy_cls, policy_kwargs)

        self.base_buffers, self.base_buffer_attr = util.create_data_buffer_from_env(
            env_fns[0], max_num_envs
        )
        (
            self.rollout_buffers,
            self.rollout_buffer_attr,
        ) = util.create_rollout_buffer_from_env(env_fns[0], max_rollouts)

    def _create_env_handler(self):
        return SHMParallelEnv(
            0, self.env_fns, self.base_buffer_attr, self.rollout_buffer_attr
        )

    def _create_policy_handler(self):
        return SHMVectorPolicy(
            0, self.policy_cls, self.policy_kwargs, self.base_buffer_attr
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

        super().__init__(worker_id, cmd_attr_dict=cmd_attr_dict)

    def initialize(self) -> None:
        self._loop_handler = SimpleLoopHandler(
            self.env_fns, self.policy_cls, self.policy_kwargs
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
        self._loop_handler.reconfigure(config, model=None)
        self._logger.debug("Reconfigure.. sync done")
        self.reply(cmd)

    def run_handler(self, cmd: int, _) -> None:
        self._loop_handler.run()
        self.reply(cmd)

    def _term_handler(self, cmd: int, data_list: List[int]) -> None:
        self._loop_handler.terminate()
        super()._term_handler(cmd, data_list)


class SHMMultiLoopHandler(MultiLoopHandler, SHMVectorMixin):
    @property
    def num_actors(self):
        return self.num_workers

    @property
    def proc_name(self) -> str:
        return "SHMMultiLoopHandler"

    def __init__(self, env_fns, policy_cls, policy_kwargs, max_rollouts: int):
        super().__init__(env_fns, policy_cls, policy_kwargs)
        self.rollout_buffer_attr = util.create_rollout_buffer_from_env(
            env_fns[0], max_rollouts
        )

    def reconfigure(
        self, config: ParallelizationConfig, model: types.PolicyModel
    ) -> None:
        self._logger.info(f"\n\n============================reconfigure {config}\n")
        util.copy_config_to_buffer(config, self._command_buffer[:, 1:])
        super().reconfigure(config, model)
        self._wait_cmd_done()
        

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

    def _terminate_worker(self, worker_id: int, worker: SimpleLoopProc) -> None:
        SHMVectorMixin.terminate_single_worker(self, worker_id, worker)
        self._logger.info(f"Join worker={worker_id} pid={worker.pid}")


    def _run(self) -> Tuple[Dict[str, Any], AuturiMetric]:
        self._logger.info(f"\n\n============================RUN\n")
        self.request(ActorCommand.RUN)
        self._logger.debug("Set command RUN")

        start_time = time.perf_counter()
        self._wait_cmd_done()
        end_time = time.perf_counter()

        # Aggregate
        agg_rollouts = self._aggregate_rollouts(self.rollout_size)
        return agg_rollouts, AuturiMetric(
            self.rollout_size, end_time - start_time
        )  # TODO: how to measure time of each worker?

    def _aggregate_rollouts(self, num_collect: int) -> Dict[str, np.ndarray]:
        ret_dict = dict()
        for key, tuple_ in self.rollout_buffers.items():
            # ret_dict[key] = tuple_[1][:num_collect, :]
            ret_dict[key] = tuple_[1]

        return ret_dict

    def terminate(self):
        SHMVectorMixin.terminate_all_workers(self)
        for _, tuple_ in self.rollout_buffers.items():
            tuple_[0].unlink()
