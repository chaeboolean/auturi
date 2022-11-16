from auturi.executor.shm import SHMActor, SHMParallelEnv, create_shm_from_env


class SHMTester:
    """Tester class to test SHMParallelEnv without policy interaction."""

    @classmethod
    def create(cls, env_fns, **kwargs):
        (
            base_buffers,
            base_buffer_attr,
            rollout_buffers,
            rollout_buffer_attr,
        ) = create_shm_from_env(
            env_fns[0], len(env_fns), 100
        )  # 100 is arbitrary max num traj.

        return cls(
            0,
            env_fns=env_fns,
            base_buffer_attr=base_buffer_attr,
            rollout_buffer_attr=rollout_buffer_attr,
            base_buffers=base_buffers,
            rollout_buffers=rollout_buffers,
            **kwargs,
        )

    def set_num_steps(self, num_collect):
        self.num_collect = num_collect

    def _aggregate_from_rollout_buffer(self):
        ret_dict = dict()
        for key, tuple_ in self.rollout_buffers.items():
            ret_dict[key] = tuple_[1][: self.num_collect, :]

        return ret_dict

    def _buffer_clean_up(self):
        for k, v in self.base_buffers.items():
            v[0].unlink()
        for k, v in self.rollout_buffers.items():
            v[0].unlink()


class SHMParallelEnvTester(SHMParallelEnv, SHMTester):
    def __init__(
        self,
        actor_id,
        env_fns,
        base_buffer_attr,
        rollout_buffer_attr,
        base_buffers,
        rollout_buffers,
    ):
        self.base_buffers, self.rollout_buffers = base_buffers, rollout_buffers
        self.num_collect = -1  # used for aggregate rollouts.

        super().__init__(actor_id, env_fns, base_buffer_attr, rollout_buffer_attr)

    def aggregate_rollouts(self):
        super().aggregate_rollouts()  # sync
        return self._aggregate_from_rollout_buffer()

    def terminate(self):
        super().terminate()
        self._buffer_clean_up()


class SHMActorTester(SHMActor, SHMTester):
    """Tester class to test SHMActor without SHMVectorActor."""

    def __init__(
        self,
        actor_id,
        env_fns,
        policy_cls,
        policy_kwargs,
        base_buffer_attr,
        rollout_buffer_attr,
        base_buffers,
        rollout_buffers,
    ):

        self.base_buffers, self.rollout_buffers = base_buffers, rollout_buffers
        self.num_collect = -1  # used for aggregate rollouts.

        super().__init__(
            actor_id,
            env_fns,
            policy_cls,
            policy_kwargs,
            base_buffer_attr,
            rollout_buffer_attr,
        )

    def run(self):
        _, metric = super().run()
        return self._aggregate_from_rollout_buffer(), metric

    def terminate(self):
        super().terminate()
        self._buffer_clean_up()
