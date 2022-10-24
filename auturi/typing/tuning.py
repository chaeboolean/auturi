"""Typing Definition for Auturi Tuner."""
from dataclasses import dataclass
from typing import Dict


@dataclass
class ActorConfig:
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
    num_actors: int
    actor_config_map: Dict[int, ActorConfig]

    def validate(self):
        assert self.num_actors > 0
        assert len(self.actor_config_map) == self.num_actors
