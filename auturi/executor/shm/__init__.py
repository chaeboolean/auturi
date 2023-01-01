from auturi.executor.executor import AuturiExecutor
from auturi.executor.shm.environment import SHMParallelEnv
from auturi.executor.shm.loop import SHMMultiLoopHandler, SHMNestedLoopHandler
from auturi.executor.shm.policy import SHMVectorPolicy

# TODO: Receive as user input
MAX_NUM_POLICY = 8


class SHMExecutor(AuturiExecutor):
    def _create_nested_loop_handler(self) -> SHMNestedLoopHandler:
        return SHMNestedLoopHandler(
            0,
            self.env_fns,
            self.policy_cls,
            self.policy_kwargs,
            max_num_envs=self.tuner.max_num_env,
            max_num_policy=MAX_NUM_POLICY,
            num_rollouts=self.tuner.num_collect,
        )

    def _create_multiple_loop_handler(self) -> SHMMultiLoopHandler:
        return SHMMultiLoopHandler(
            self.env_fns,
            self.policy_cls,
            self.policy_kwargs,
            max_num_loop=self.tuner.max_num_env,
            num_rollouts=self.tuner.num_collect,
        )


__all__ = [
    "SHMParallelEnv",
    "SHMVectorPolicy",
    "SHMExecutor",
]
