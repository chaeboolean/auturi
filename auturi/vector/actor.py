from auturi.typing.environment import AuturiVectorEnv


class AuturiActor:
    def __init__(self, vector_env: AuturiVectorEnv, policy):
        self.vector_env = vector_env
        self.policy = policy

    def run(self, num_steps):
        cnt = 0
        while cnt < num_steps:
            obs_refs = self.vector_env.poll()

            action_refs = self.policy.compute_actions(obs_refs, cnt)
            self.vector_env.send_actions(action_refs)

            self.num_timesteps += len(obs_refs)
            cnt += len(obs_refs)


class RayVectorActor:
    pass


class SHMVectorActor:
    pass
