import numpy as np

from auturi.executor.environment import AuturiSerialEnv
from auturi.executor.shm.constant import EnvCommand
from auturi.executor.shm.mp_mixin import Request, SHMProcLoopMixin
from auturi.executor.shm.util import set_shm_from_attr, wait
from auturi.logger import get_logger

logger = get_logger()


class EnvStateEnum:
    """Indicates single simulator state.
    Initialized to STOPPED.

    """

    STOPPED = 0
    STEP_DONE = 22  # Newly arrived requests
    QUEUED = 23  # Inside Server side waiting queue
    POLICY_DONE = 24  # Processed requests
    POLICY_OFFSET = 30  # offset => ASSIGNED


class SHMEnvProc(SHMProcLoopMixin):
    def __init__(
        self,
        actor_id,
        worker_id,
        req_queue,
        rep_queue,
        env_fns,
        base_buffer_attr,
        rollout_buffer_attr,
    ):
        self.actor_id = actor_id
        self.env_fns = env_fns
        self.base_buffer_attr = base_buffer_attr
        self.rollout_buffer_attr = rollout_buffer_attr

        super().__init__(worker_id, req_queue=req_queue, rep_queue=rep_queue)

    def initialize(self) -> None:
        self.env = AuturiSerialEnv(self.actor_id, self.worker_id, self.env_fns)
        self.__env, self.env_buffer = set_shm_from_attr(self.base_buffer_attr["env"])

        self.__obs, self.obs_buffer = set_shm_from_attr(self.base_buffer_attr["obs"])
        self.__action, self.action_buffer = set_shm_from_attr(
            self.base_buffer_attr["action"]
        )
        self.__artifact, self.artifact_buffer = set_shm_from_attr(
            self.base_buffer_attr["artifact"]
        )

        self.rollout_buffers = dict()
        for key, attr_dict in self.rollout_buffer_attr.items():
            raw_buf, np_buffer = set_shm_from_attr(attr_dict)
            self.rollout_buffers[key] = (raw_buf, np_buffer)

    @property
    def identifier(self):
        return f"EnvProc(aid={self.actor_id}, wid={self.worker_id}): "

    def set_handler_for_command(self):
        self.cmd_handler[EnvCommand.RESET] = self.reset_handler
        self.cmd_handler[EnvCommand.SEED] = self.seed_handler
        self.cmd_handler[EnvCommand.SET_ENV] = self.set_visible_env_handler
        self.cmd_handler[EnvCommand.AGGREGATE] = self.aggregate_handler

    def reset_handler(self, request: Request):
        obs = self.env.reset()
        self.insert_obs_buffer(obs)
        self.reply(request.cmd)

    def seed_handler(self, request: Request):
        self.env.seed(int(request.data[0]))
        self.reply(request.cmd)

    def set_visible_env_handler(self, request: Request):
        self.env.set_working_env(int(request.data[0]), int(request.data[1]))
        self.reply(request.cmd)

    def aggregate_handler(self, request: Request):
        local_rollouts = self.env.aggregate_rollouts()

        for key_, trajectories in local_rollouts.items():
            roll_buffer = self.rollout_buffers[key_][1]

            # TODO: stack first
            np.copyto(roll_buffer[request.data[0] : request.data[1]], trajectories)

        self.reply(request.cmd)

    def _step_loop_once(self, is_first: bool) -> None:
        if is_first:
            assert np.all(
                self._get_env_state() == EnvStateEnum.STOPPED
            ), f"curr={self._get_single_env_state()}"

            logger.debug(self.identifier + "Entered the loop.")

            obs = self.env.reset()
            self.insert_obs_buffer(obs)
            self._set_env_state(EnvStateEnum.STEP_DONE)

        elif np.all(self._get_env_state() == EnvStateEnum.POLICY_DONE):
            action, artifacts_list = self.get_actions()
            obs = self.env.step(action, artifacts_list)
            logger.debug(self.identifier + "Called env.step()")
            logger.info(
                self.identifier + f"env.step({action.flat[0]}) => {obs.flat[0]}"
            )

            self.insert_obs_buffer(obs)
            self._set_env_state(EnvStateEnum.STEP_DONE)

    def insert_obs_buffer(self, obs):
        self.obs_buffer[self.env.start_idx : self.env.end_idx, :] = obs

    def get_actions(self):
        action = self.action_buffer[self.env.start_idx : self.env.end_idx, :]
        artifacts = self.artifact_buffer[self.env.start_idx : self.env.end_idx, :]
        return action, [artifacts]

    def _set_env_state(self, state):
        self.env_buffer[self.env.start_idx : self.env.end_idx] = state

    def _get_env_state(self):
        return self.env_buffer[self.env.start_idx : self.env.end_idx]

    def _wait_to_stop(self):
        # env which policy worker is allocated should finish its env.step()
        cond_ = lambda: np.all(
            np.ma.mask_or(
                (self._get_env_state() == EnvStateEnum.STEP_DONE),
                (self._get_env_state() == EnvStateEnum.QUEUED),
            )
        )
        wait(cond_, self.identifier + f"Wait to set STOP sign.")
        self._set_env_state(EnvStateEnum.STOPPED)
        logger.debug(self.identifier + f"Escaped the loop.")
        super()._wait_to_stop()
