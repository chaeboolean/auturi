"""Typing Definition for Auturi Tuner."""

from dataclasses import dataclass


@dataclass
class SamplerConfig:
    bs: int  # batch size
    numc: int = 1  # number of servers
    device: str = "cpu"  # device where server reside
    serial: bool = False  # indicates whether to run in serial or not
    policy: str = "fix"  # serving policy

    def validate(self, N):
        if self.serial:
            assert self.bs * self.numc == N
        else:
            assert self.bs * self.numc <= N

        assert self.device in ["cpu", "gpu"]
        assert self.policy in ["fix", "min"]


@dataclass
class StepTime:
    pass


@dataclass
class ServiceTime:
    pass
