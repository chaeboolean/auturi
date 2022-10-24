import torch as th
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import obs_as_tensor

from auturi.adapter.sb3.policy_adapter import SB3PolicyAdapter
from auturi.engine import AuturiEngine
from auturi.tuner import AuturiTuner
from auturi.typing.simulator import AuturiVectorEnv
from auturi.vector.ray_backend import RayVectorPolicies
from auturi.vector.shm_policy import SHMVectorPolicies


class SB3OnPolicyAlgorithmEngine(AuturiEngine):
    def _setup(self, rollout_buffer):
        self.rollout_buffer = rollout_buffer

    def begin_collection_loop(self):
        return super().begin_collection_loop()

    def finish_collection_loop(self):
        """_summary_"""
        # Aggregate buffers from all policy workers.

        last_server = self.vector_policy.get_last_server()
        new_obs = last_server.new_obs
        dones = last_server.last_dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        self.rollout_buffer.compute_returns_and_advantage(
            last_values=values, dones=dones
        )


def wrap_sb3_OnPolicyAlgorithm(
    sb3_algo: OnPolicyAlgorithm, tuner: AuturiTuner, backend: str = "ray"
):
    """Use wrapper like this

    algo = sb3.create_algorithm(configs)
    tuner = auturi.Tuner(**uturi_configs)
    wrap_sb3(algo, tuner)
    algo.learn()

    """
    assert isinstance(sb3_algo, OnPolicyAlgorithm), "Not implemented for other case."

    # create engine with
    assert isinstance(sb3_algo.env, AuturiVectorEnv)

    def policy_creator():
        return SB3PolicyAdapter(
            observation_space=sb3_algo.env.observation_space,
            action_space=sb3_algo.env.action_space,
            model_cls=sb3_algo.policy_class,
            use_sde=sb3_algo.use_sde,
            sde_sample_freq=sb3_algo.sde_sample_freq,
            model_path=sb3_algo.policy_save_path,
        )

    vecpol_cls = RayVectorPolicies if backend == "ray" else SHMVectorPolicies
    vecpol_kwargs = {
        "num_policies": 1,  # tuner.max_policy
        "policy_fn": policy_creator,
    }
    if backend != "ray":
        vecpol_kwargs["shm_config"] = sb3_algo.env.shm_configs

    vector_policy = vecpol_cls(**vecpol_kwargs)

    engine = AuturiEngine(sb3_algo.env, vector_policy, tuner)
    sb3_algo.set_auturi_engine(engine)