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

    def _truncate_and_reshape(buffer_, add_dim=False, dtype=np.float32):
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
