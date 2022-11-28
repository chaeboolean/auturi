from typing import List

import auturi.executor.shm.util as util
from auturi.executor.actor import AuturiActor
from auturi.executor.shm.constant import ActorCommand
from auturi.executor.shm.environment import SHMParallelEnv
from auturi.executor.shm.mp_mixin import SHMProcMixin
from auturi.executor.shm.policy import SHMVectorPolicy


class SHMActor(AuturiActor):
    def __init__(
        self,
        actor_id,
        env_fns,
        policy_cls,
        policy_kwargs,
        base_buffer_attr,
        rollout_buffer_attr,
    ):
        self.base_buffer_attr = base_buffer_attr
        self.rollout_buffer_attr = rollout_buffer_attr
        super().__init__(actor_id, env_fns, policy_cls, policy_kwargs)

    def _create_vector_env(self, env_fns):
        return SHMParallelEnv(
            self.actor_id, env_fns, self.base_buffer_attr, self.rollout_buffer_attr
        )

    def _create_vector_policy(self, policy_cls, policy_kwargs):
        return SHMVectorPolicy(
            self.actor_id, policy_cls, policy_kwargs, self.base_buffer_attr
        )

    def sync(self):
        self.vector_policy.sync()
        self.vector_envs.sync()


class SHMActorProc(SHMProcMixin):
    def __init__(
        self,
        worker_id,
        cmd_attr_dict,
        env_fns,
        policy_cls,
        policy_kwargs,
        base_buffer_attr,
        rollout_buffer_attr,
    ):
        self.actor_id = worker_id
        self.env_fns = env_fns
        self.policy_cls = policy_cls
        self.policy_kwargs = policy_kwargs
        self.base_buffer_attr = base_buffer_attr
        self.rollout_buffer_attr = rollout_buffer_attr

        super().__init__(worker_id, cmd_attr_dict=cmd_attr_dict)

    def initialize(self) -> None:
        self.actor = SHMActor(
            self.actor_id,
            self.env_fns,
            self.policy_cls,
            self.policy_kwargs,
            self.base_buffer_attr,
            self.rollout_buffer_attr,
        )

        SHMProcMixin.initialize(self)

    @property
    def proc_name(self):
        return f"Actor(aid={self.actor_id})"

    def set_command_handlers(self):
        self.cmd_handler[ActorCommand.RECONFIGURE] = self.reconfigure_handler
        self.cmd_handler[ActorCommand.RUN] = self.run_handler

    def reconfigure_handler(self, cmd: int, _):
        config = util.convert_buffer_to_config(self._command_buffer[:, 1:])
        self.actor.reconfigure(config, model=None)
        self.actor.sync()
        self._logger.debug("Reconfigure.. sync done")
        self.reply(cmd)

    def run_handler(self, cmd: int, _):
        self.actor.run()
        self.actor.sync()
        self.reply(cmd)

    def _term_handler(self, cmd: int, data_list: List[int]):
        self._logger.info("Got Term signal....")
        self.actor.vector_envs.terminate()
        self.actor.vector_policy.terminate()
        super()._term_handler(cmd, data_list)
