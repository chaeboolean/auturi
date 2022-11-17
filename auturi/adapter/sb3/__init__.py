import functools
from functools import partial

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from auturi.adapter.sb3.buffer_utils import insert_as_buffer, process_buffer
from auturi.adapter.sb3.env_adapter import SB3EnvAdapter
from auturi.adapter.sb3.policy_adapter import SB3PolicyAdapter
from auturi.executor import create_executor

SAVE_MODEL_PATH = "/home/ooffordable/auturi/log/model_save.pt"


def _collect_rollouts_auturi(sb3_algo, env, callback, rollout_buffer, n_rollout_steps):

    # Save trained policy network
    sb3_algo.policy.save(sb3_algo._save_model_path)

    num_envs = len(sb3_algo.env_fns)
    agg_rollouts, metric = sb3_algo._auturi_executor.run(model=None)
    #print("**** Rollout ends... = ", agg_rollouts["terminal_obs"].shape)

    process_buffer(agg_rollouts, sb3_algo.policy, sb3_algo.gamma)
    insert_as_buffer(rollout_buffer, agg_rollouts, num_envs)  # TODO
    last_obs = rollout_buffer.observations[-1]
    last_dones = rollout_buffer.episode_starts[-1]

    return last_obs, last_dones


def _create_auturi_env(env_fn, dumb=None):
    return SB3EnvAdapter(env_fn)


def wrap_sb3_OnPolicyAlgorithm(
    sb3_algo: OnPolicyAlgorithm, tuner, backend: str = "ray"
):
    assert isinstance(sb3_algo, OnPolicyAlgorithm), "Not implemented for other case."
    assert hasattr(sb3_algo, "env_fns")
    assert hasattr(sb3_algo, "policy_class")

    auturi_env_fns = [
        partial(_create_auturi_env, env_fn) for env_fn in sb3_algo.env_fns
    ]
    policy_kwargs = {
        "observation_space": sb3_algo.observation_space,
        "action_space": sb3_algo.action_space,
        "model_cls": sb3_algo.policy_class,
        "use_sde": sb3_algo.use_sde,
        "sde_sample_freq": sb3_algo.sde_sample_freq,
        "model_path": SAVE_MODEL_PATH,
    }

    num_envs = len(sb3_algo.env_fns)
    num_collect = num_envs * sb3_algo.n_steps

    executor = create_executor(
        env_fns=auturi_env_fns,
        policy_cls=SB3PolicyAdapter,
        policy_kwargs=policy_kwargs,
        tuner=tuner,
        backend=backend,
    )

    sb3_algo._auturi_executor = executor
    sb3_algo._save_model_path = SAVE_MODEL_PATH
    sb3_algo._collect_rollouts_fn = functools.partial(
        _collect_rollouts_auturi, sb3_algo
    )
    sb3_algo.env.close()
