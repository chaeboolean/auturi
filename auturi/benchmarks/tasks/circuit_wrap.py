import array
import pathlib
from collections import defaultdict

import cloudpickle
import gym
import numpy as np
import tensorflow as tf
from auturi.executor.environment import AuturiEnv
from auturi.executor.policy import AuturiPolicy
from circuit_training.environment.environment import create_circuit_environment
from circuit_training.model import model
from tf_agents.agents.ppo.ppo_policy import PPOPolicy
from tf_agents.train.utils import spec_utils
from tf_agents.utils import nest_utils

root = pathlib.Path(__file__).absolute().parent

NETLIST_FILE = str(root) + "/circuit_assets/ariane/netlist.pb.txt"

INIT_PLACEMENT = str(root) + "/circuit_assets/ariane/initial.plc"

CIRCUIT_OBS_LEN = 539323


def _load_pickled(buffer):
    load_pkl_ = lambda x: cloudpickle.loads(array.array("B", x))
    if buffer.shape[0] == 1:
        return load_pkl_(buffer[0])
    else:
        return nest_utils.stack_nested_arrays([load_pkl_(x) for x in buffer])


def _put_pickled(obs):
    return np.array(bytearray(cloudpickle.dumps(obs)))


class CircuitEnvWrapper(AuturiEnv):
    def __init__(self, task_id: str, rank: int):
        self.env = create_circuit_environment(
            netlist_file=NETLIST_FILE, init_placement=INIT_PLACEMENT, global_seed=3
        )

        self.rank = rank
        self.metadata = None
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(
            low=0, high=0, shape=(CIRCUIT_OBS_LEN,), dtype=np.int8
        )

        self.storage = defaultdict(list)
        self.artifacts_samples = [np.array([True])]

    def step(self, action, artifacts):
        if not isinstance(action, np.ndarray):
            action = np.array([action])

        obs, reward, done, info = self.env.step(action)
        if done:
            obs = self.env.reset()

        pickled_obs = _put_pickled(obs)

        self.storage["action"].append(action)
        self.storage["reward"].append(reward)
        self.storage["done"].append(done)

        return pickled_obs

    def reset(self):
        self.storage.clear()
        obs = self.env.reset()
        return _put_pickled(obs)

    def seed(self, seed):
        self.env.seed(seed + self.rank)

    def aggregate_rollouts(self):
        return self.storage

    def terminate(self):
        self.env.close()


class CircuitPolicyWrapper(AuturiPolicy):
    def __init__(self, task_id: str, idx: int):
        self.device = "cpu"
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        dummy_env = create_circuit_environment(
            netlist_file=NETLIST_FILE, init_placement=INIT_PLACEMENT, global_seed=3
        )

        (
            observation_tensor_spec,
            action_tensor_spec,
            time_step_tensor_spec,
        ) = spec_utils.get_tensor_specs(dummy_env)

        # static_features = single_dummy_env.wrapped_env().get_static_obs()

        grl_shared_net = model.GrlModel(
            observation_tensor_spec,
            action_tensor_spec,
            static_features=None,
            use_model_tpu=False,
        )
        grl_actor_net = model.GrlPolicyModel(
            grl_shared_net, observation_tensor_spec, action_tensor_spec
        )
        grl_value_net = model.GrlValueModel(observation_tensor_spec, grl_shared_net)

        self.policy = PPOPolicy(
            time_step_spec=time_step_tensor_spec,
            action_spec=action_tensor_spec,
            actor_network=grl_actor_net,
            value_network=grl_value_net,
            observation_normalizer=None,
            clip=False,
            collect=True,
        )

        dummy_env.close()

        self.policy_state = None

    def load_model(self, model, device):
        pass

    def compute_actions(self, input, n_steps):
        input = _load_pickled(input)

        bs_ = input.observation["current_node"].shape[0]
        if self.policy_state is None:
            self.policy_state = self.policy.get_initial_state(None)

        action_step = self.policy.action(input, self.policy_state)
        self.policy_state = action_step.state
        action = np.expand_dims(action_step.action.numpy(), -1)

        return action, [np.array([True])]

    def terminate(self):
        del self.policy
