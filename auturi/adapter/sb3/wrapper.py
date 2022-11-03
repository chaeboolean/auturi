import functools
import time

import numpy as np
import torch as th
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from auturi.adapter.sb3.env_adapter import SB3EnvAdapter
from auturi.adapter.sb3.policy_adapter import SB3PolicyAdapter
from auturi.executor.ray import create_ray_executor

SAVE_MODEL_PATH = "/home/ooffordable/ray_fork/log/model_save.pt"


def process_buffer(agg_buffer, policy, gamma):
    # process reward from terminal observations
    terminal_indices = np.where(agg_buffer["has_terminal_obs"] == True)[0]

    # agg_buffer is totally CPU buffer
    if len(terminal_indices) > 0:
        terminal_obs = agg_buffer["terminal_obs"][terminal_indices]
        terminal_obs = policy.obs_to_tensor(terminal_obs)[0]

        with th.no_grad():
            terminal_value = policy.predict_values(terminal_obs)

        agg_buffer["reward"][terminal_indices] += gamma * (
            terminal_value.cpu().numpy().flatten()
        )


def insert_as_buffer(rollout_buffer, agg_buffer, num_envs):

    # insert to rollout_buffer
    bsize = rollout_buffer.buffer_size
    total_length = bsize * num_envs

    def _truncate_and_reshape(buffer_, add_dim=False, dtype=np.float32):
        # print(
        #     f"_truncate_and_reshape(buffer:{buffer_.shape}) & total_length={total_length} -->"
        # )
        shape_ = (bsize, num_envs, *buffer_.shape[1:]) if add_dim else (bsize, num_envs)
        ret = buffer_[:total_length].reshape(*shape_)
        return ret.astype(dtype)

    # reshape to (k, self.n_envs, obs_size)
    rollout_buffer.observations = _truncate_and_reshape(agg_buffer["obs"], add_dim=True)
    rollout_buffer.actions = _truncate_and_reshape(agg_buffer["action"], add_dim=True)
    rollout_buffer.rewards = _truncate_and_reshape(agg_buffer["reward"], add_dim=False)
    rollout_buffer.episode_starts = _truncate_and_reshape(
        agg_buffer["episode_start"], add_dim=False
    )
    rollout_buffer.values = _truncate_and_reshape(agg_buffer["value"], add_dim=False)
    rollout_buffer.log_probs = _truncate_and_reshape(
        agg_buffer["log_prob"], add_dim=False
    )
    rollout_buffer.pos = rollout_buffer.buffer_size
    rollout_buffer.full = True


def _collect_rollouts_auturi(sb3_algo, env, callback, rollout_buffer, n_rollout_steps):

    # Save trained policy network
    sb3_algo.policy.save(sb3_algo._save_model_path)

    num_envs = len(sb3_algo.env_fns)
    num_collect = n_rollout_steps * num_envs
    agg_rollouts, metric = sb3_algo._auturi_executor._run(num_collect=num_collect)
    print("**** Rollout ends... = ", agg_rollouts["terminal_obs"].shape)

    process_buffer(agg_rollouts, sb3_algo.policy, sb3_algo.gamma)
    insert_as_buffer(rollout_buffer, agg_rollouts, num_envs)  # TODO
    last_obs = rollout_buffer.observations[-1]
    last_dones = rollout_buffer.episode_starts[-1]

    return last_obs, last_dones


def wrap_sb3_OnPolicyAlgorithm(
    sb3_algo: OnPolicyAlgorithm, tuner, backend: str = "ray"
):
    assert isinstance(sb3_algo, OnPolicyAlgorithm), "Not implemented for other case."
    assert hasattr(sb3_algo, "env_fns")
    assert hasattr(sb3_algo, "policy_class")

    auturi_env_fns = [lambda: SB3EnvAdapter(env_fn) for env_fn in sb3_algo.env_fns]
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

    executor = create_ray_executor(
        env_fns=auturi_env_fns,
        policy_cls=SB3PolicyAdapter,
        policy_kwargs=policy_kwargs,
        tuner=tuner,
        max_rollout_size=num_collect,
    )

    sb3_algo._auturi_executor = executor
    sb3_algo._save_model_path = SAVE_MODEL_PATH
    sb3_algo._collect_rollouts_fn = functools.partial(
        _collect_rollouts_auturi, sb3_algo
    )
    sb3_algo.env.close()
