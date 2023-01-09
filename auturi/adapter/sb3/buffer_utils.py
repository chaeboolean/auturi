import numpy as np
import torch as th


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

    # Change shape: (num_collect, *) --> (bsize, num_envs, *)
    def _reshape(buffer_, shape=None):
        shape_ = shape
        if shape_ is None:
            shape_ = (bsize, num_envs, *(buffer_.shape[1:]))
        return buffer_[:total_length].reshape(*shape_)

    # reshape to (k, self.n_envs, obs_size)
    np.copyto(dst=rollout_buffer.observations, src=_reshape(agg_buffer["obs"]))
    np.copyto(dst=rollout_buffer.actions, src=_reshape(agg_buffer["action"], shape=rollout_buffer.actions.shape))
    np.copyto(dst=rollout_buffer.rewards, src=_reshape(agg_buffer["reward"]))
    np.copyto(
        dst=rollout_buffer.values, src=_reshape(agg_buffer["action_artifacts"][:, 0])
    )
    np.copyto(
        dst=rollout_buffer.log_probs, src=_reshape(agg_buffer["action_artifacts"][:, 1])
    )
    np.copyto(
        dst=rollout_buffer.episode_starts, src=_reshape(agg_buffer["episode_start"])
    )

    rollout_buffer.pos = rollout_buffer.buffer_size
    rollout_buffer.full = True
