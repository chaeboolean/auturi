"""Typing Definition for Auturi Tuner."""
from dataclasses import dataclass
from typing import Dict


@dataclass
class ActorConfig:
    """Configuration for a single actor component.

    Args:
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

    def __post_init__(self):
        """Validate configurations."""
        # assert self.policy_device in ["cpu", "cuda"]

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


@dataclass
class ParallelizationConfig:
    """Parallelization configuration for AuturiExecutor, found by AuturiTuner.

    Args:
        num_actors (int): number of actors that the executor manages.
        actor_config_map (Dict[int, ActorConfig]): maps actor id and actor instance.

    """

    num_actors: int
    actor_config_map: Dict[int, ActorConfig]

    def __post_init__(self):
        assert self.num_actors > 0
        assert len(self.actor_config_map) == self.num_actors

    def __getitem__(self, actor_id: int):
        return self.actor_config_map[actor_id]


@dataclass
class AuturiMetric:
    """Metrics that an AuturiActor collects every run of a collection loop.

    Args:
        num_trajectories (int): number of trajectories collected.
        elapsed (float): second.
        throughput (float): the number of trajectories per second

    """

    num_trajectories: int
    elapsed: float
    throughput: float = -1

    def __post_init__(self):
        self.throughput = self.num_trajectories / self.elapsed
