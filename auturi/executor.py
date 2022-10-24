from typing import Any, Dict

from auturi.tuner import AuturiTuner
from auturi.typing.actor import AuturiActor, RayAuturiActor, SHMAuturiActor
from auturi.typing.environment import AuturiVectorEnv
from auturi.typing.policy import AuturiVectorPolicy
from auturi.typing.tuning import TunerConfig


class AuturiExecutor:
    """Interacts with Tuner.
    Get configuration from tuner, and change its execution plan.
    One of major components in Auturi System.
    Handles multiple Actors, similar with VectorActor.
    """

    def __init__(
        self,
        vector_env: AuturiVectorEnv,
        vector_policy: AuturiVectorPolicy,
        tuner: AuturiTuner,
        remote_actor_backend: str = "ray",
    ):
        self.vector_env = vector_env
        self.vector_policy = vector_policy
        self.tuner = tuner

        self.local_actor = AuturiActor(vector_env, vector_policy)
        self.remote_actors = dict()

        # self.remote_actor_cls = RayAuturiActor.remote \
        #     if remote_actor_backend == "ray" else SHMAuturiActor

    @property
    def num_actors(self):
        return 1 + len(self.remote_actors)

    def reconfigure(self, next_config: TunerConfig):
        """Adjust executor's component according to tuner-given config.

        Args:
            next_config (TunerConfig): Configurations for tuning.
        """

        # Create actors if needed.
        while next_config.num_actors > self.num_actors:
            new_actor_id = self.num_actors - 1
            self.remote_actors[new_actor_id] = RayAuturiActor.remote(
                self.vector_env, self.vector_policy
            )

        # Set configs for each actor.
        self.local_actor.reconfigure(next_config)
        for remote_actor in self.remote_actors.values():
            remote_actor.remote.reconfigure(next_config)

    def run(self, num_collect: int) -> Dict[str, Any]:
        """Run collection loop with `num_collect` iterations, and return experience trajectories"""

        next_config = self.tuner.step()
        self.reconfigure(next_config)

        self.local_actor.run(num_collect)
        for remote_actor in self.remote_actors.values():
            remote_actor.remote.run(num_collect)
