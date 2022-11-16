from auturi.executor.actor import AuturiActor
from auturi.executor.shm.constant import ActorCommand
from auturi.executor.shm.environment import SHMParallelEnv
from auturi.executor.shm.mp_mixin import Request, SHMProcMixin
from auturi.executor.shm.policy import SHMVectorPolicy
from auturi.logger import get_logger

logger = get_logger()


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
        req_queue,
        rep_queue,
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

        super().__init__(worker_id, req_queue=req_queue, rep_queue=rep_queue)

    def initialize(self) -> None:
        self.actor = SHMActor(
            self.actor_id,
            self.env_fns,
            self.policy_cls,
            self.policy_kwargs,
            self.base_buffer_attr,
            self.rollout_buffer_attr,
        )

    @property
    def identifier(self):
        return f"Actor(aid={self.actor_id}): "

    def set_handler_for_command(self):
        self.cmd_handler[ActorCommand.RECONFIGURE] = self.reconfigure_handler
        self.cmd_handler[ActorCommand.RUN] = self.run_handler

    def reconfigure_handler(self, request: Request):
        config = request.data[0]
        self.actor.reconfigure(config, model=None)
        self.actor.sync()
        logger.debug(self.identifier + "Reconfigure.. sync done")
        self.reply(request.cmd)

    def run_handler(self, request: Request):
        self.actor.run()
        self.actor.sync()
        self.reply(request.cmd)

    def _term_handler(self, request: Request):
        logger.info(self.identifier + "Got Term signal....")
        self.actor.vector_envs.terminate()
        self.actor.vector_policy.terminate()
        super()._term_handler(request)
