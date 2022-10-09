@dataclass
class AuturiStrategy:
    remoteEnvs: AuturiVecEnv = None
    servers: AuturiPolicy = None


class ServingStrategy(AuturiStrategy):
    def run(self):

        # Get step-finished simulators
        ready_env_refs = self.remoteEnvs.poll(self.batch_size)

        # Find free server and assign ready envs to it
        action_refs = self.servers.assign_to_free_server(ready_env_refs)

        # send action to remote simulators
        self.remoteEnvs.send_actions(action_refs)

        return len(ready_env_refs)


class SerialStrategy(AuturiStrategy):
    def run(self):
        pass
