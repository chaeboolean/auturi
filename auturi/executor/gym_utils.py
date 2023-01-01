import gym
import numpy as np


def get_action_sample(action_space, batch_size):
    action_sample = action_space.sample()
    if isinstance(action_space, gym.spaces.Discrete):
        sample_to_array = np.array([action_sample] * batch_size)
        return np.expand_dims(sample_to_array, -1)
    else:
        return np.stack([action_sample] * batch_size)
