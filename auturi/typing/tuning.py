"""Typing Definition for Auturi Tuner."""
from dataclasses import dataclass
from typing import Dict


@dataclass
class ActorConfig:
    """Configuration for a single actor component.
    
    Args:
        num_envs (int): number of environments that an actor manages.
        num_policy (int): number of policy replica that an actor manages.
        num_parallel (int): number of parallel environment worker.
        batch_size (int): number of environments that a policy replica handles at a time.
        policy_device (str): device placement of policy replica.
        
    """
    num_envs: int = 1
    num_policy: int = 1  # number of servers
    num_parallel: int = 1
    batch_size: int = 1
    policy_device: str = "cpu"  # device where server reside

    def __post_init__(self):
        """Validate configurations."""
        assert self.policy_device in ["cpu", "cuda"]
        assert self.num_envs % self.num_parallel == 0
        assert self.num_policy * self.batch_size <= self.num_envs


@dataclass
class TunerConfig:
    """Configuration for AuturiExecutor.
    
    Args:
        num_actors (int): number of actors that the executor manages.
        actor_config_map (Dict[int, ActorConfig]): maps actor id and actor instance.

    """
    num_actors: int
    actor_config_map: Dict[int, ActorConfig]

    def validate(self):
        assert self.num_actors > 0
        assert len(self.actor_config_map) == self.num_actors

@dataclass
class AuturiMetic:
    """Metrics that an AuturiActor collects for each iteration.
    
    Args:
        num_trajectories (int): number of trajectories collected.
        elapsed (float): second.
        throughput (float): througgput

    """
    num_trajectories: int 
    elapsed: float
    throughput: float = -1
    
    def __post_init__(self):
        self.throughput = self.num_trajectories / self.elapsed
