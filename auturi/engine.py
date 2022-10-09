from auturi.tuner import AuturiTuner
from auturi.typing.policy import AuturiVectorPolicy
from auturi.typing.simulator import AuturiVectorEnv


class AuturiEngine:
    """Interacts with Tuner.
    Get configuration from tuner, and change its execution plan.
    One of major components in Auturi System.
    """

    def __init__(
        self,
        vector_env: AuturiVectorEnv,
        vector_policy: AuturiVectorPolicy,
        tuner: AuturiTuner,
    ):
        self.vector_env = vector_env
        self.vector_policy = vector_policy
        self.tuner = tuner

        self._setup()

    def _setup(self):
        pass

    def begin_collection_loop(self):
        """Called at the beginning of the collection loop."""
        self.vector_env.reset()
        self.vector_policy.reset()

    def finish_collection_loop(self):
        """Called at the beginning of the collection loop."""
        pass

    def run(self):
        # config_changed, config = self.tuner.next_step()

        # change configurations
        # if config_changed:
        #     self.servers.set_device(config.device)
        #     self.servers.set_num_server(config.num_server)

        # Get step-finished simulators
        obs_refs = self.vector_env.poll(bs=1)  # Dict[ObsRef, env_id]

        # Find free server and assign ready envs to it
        action_refs, free_server = self.vector_policy.assign_free_server(obs_refs)

        # send action to remote simulators
        self.vector_env.send_actions(action_refs)
        # self.vector_policy.compute_action_callback(free_server)

        return len(obs_refs)
