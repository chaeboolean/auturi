"""Typing Definition for Auturi Tuner."""
from dataclasses import dataclass
from typing import Callable, List, Optional

from frozendict import frozendict


@dataclass(eq=True, frozen=True, order=True)
class ActorConfig:
    """Configuration for a single actor component.

    Args:
        num_collect (int): number of trajectories that an actor should collect.
        num_envs (int): number of environments that an actor manages.
        num_policy (int): number of policy replica that an actor manages.
        num_parallel (int): number of SerialEnv that runs in parallel.
        batch_size (int): number of observations from a different environment, that a policy consumes for computing an action
        policy_device (str): device on which we call compute_actions the policy network.

    """

    num_envs: int = 1
    num_policy: int = 1
    num_parallel: int = 1
    batch_size: int = 1
    policy_device: str = "cpu"
    num_collect: int = 1

    def __post_init__(self):
        """Validate configurations."""
        # assert self.policy_device in ["cpu", "cuda"]
        assert self.num_parallel > 0

        # SerialEnvs inside the same Actor should have same number of serial envs.
        assert self.num_envs % self.num_parallel == 0

        # If the batch_size > num_envs, deadlock occurs.
        # Else If the product num_policy and batch_size exceeds num_envs,
        # at least one policy is always idle.
        assert self.num_policy * self.batch_size <= self.num_envs

        # SerialEnvs should contain more than one envs.
        num_env_serial = self.num_envs // self.num_parallel
        assert num_env_serial > 0

        # Since num_env_serial is minimum unit of SerialEnv's output,
        # batch_size should exceed num_env_serial and multiple of it.
        assert self.batch_size >= num_env_serial
        assert self.batch_size % num_env_serial == 0

        # At least one SerialEnv should be executed.

        assert self.num_collect >= num_env_serial


@dataclass(eq=True, frozen=True)
class ParallelizationConfig:
    """Parallelization configuration for AuturiExecutor, found by AuturiTuner.

    Args:
        actor_map (frozendict[int, ActorConfig]): maps actor id and actor instance.

    """

    actor_map: frozendict

    @classmethod
    def create(cls, actor_configs: List[ActorConfig]):
        return cls(
            frozendict({idx: config for idx, config in enumerate(actor_configs)}),
        )

    def __getitem__(self, actor_id: int) -> ActorConfig:
        return self.actor_map[actor_id]

    @property
    def num_actors(self):
        return len(self.actor_map)

    @property
    def num_collect(self):
        num_collect = 0
        for _, actor_config in self.actor_map.items():
            num_collect += actor_config.num_collect
        return num_collect

    @property
    def num_envs(self):
        num_envs = 0
        for _, actor_config in self.actor_map.items():
            num_envs += actor_config.num_envs
        return num_envs

    @property
    def num_parallel_envs(self):
        num_env_workers = 0
        for _, actor_config in self.actor_map.items():
            num_env_workers += actor_config.num_parallel
        return num_env_workers

    @property
    def num_policy(self):
        num_policy = 0
        for _, actor_config in self.actor_map.items():
            num_policy += actor_config.num_policy
        return num_policy

    def compute_index_for_actor(self, name: str, actor_id: int) -> int:
        """Return the global index of Env, SerialEnv, or Policy with given actor_id."""
        assert name in ["num_envs", "num_parallel", "num_policy", "num_collect"]
        start_idx = 0
        for idx, actor_config in self.actor_map.items():
            if idx == actor_id:
                return start_idx

            start_idx += getattr(actor_config, name)

        raise IndexError(f"Actor id {actor_id} does not exist.")

    def validate(
        self,
        min_num_envs: int,
        max_num_envs: int,
        max_num_policy: int = 1000,
        validator: Optional[Callable[[ActorConfig], bool]] = None,
    ):
        """Validate TunerConfig.

        Args:
            min_num_envs (int): minimum number of environments.
            max_num_envs (int): maximum number of environments.
            max_num_policy (int, optional): maximum number of policies.
            validator (Callable[[ActorConfig], bool]): Additional user-defined function.
        """

        assert self.num_envs >= min_num_envs and self.num_envs <= max_num_envs
        assert self.num_policy <= max_num_policy

        if validator is None:
            validator = lambda x: True

        for _, actor_config in self.actor_map.items():
            assert validator(actor_config)
