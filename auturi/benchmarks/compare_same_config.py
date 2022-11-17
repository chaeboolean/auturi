import argparse
import functools

from rl_zoo3.exp_manager import ExperimentManager

from auturi.adapter.sb3 import wrap_sb3_OnPolicyAlgorithm
from auturi.tuner import ActorConfig, ParallelizationConfig, create_tuner_with_config
from auturi.tuner.grid_search import GridSearchTuner

import ray
import time

def get_config(args, architecture, device="cpu"):
    n_envs = args.num_envs

    if architecture == "subproc":
        actor_config = ActorConfig(num_envs=n_envs, num_policy=1, num_parallel=n_envs, \
            batch_size=n_envs, policy_device=device, num_collect=args.num_collect)

        return ParallelizationConfig.create([actor_config])
    
    elif architecture == "dummy":
        actor_config = ActorConfig(num_envs=n_envs, num_policy=1, num_parallel=1, \
            batch_size=n_envs, policy_device=device, num_collect=args.num_collect)

        return ParallelizationConfig.create([actor_config])


    elif architecture == "rllib":
        num_actors = args.num_actors
        envs_per_actor = args.num_envs // num_actors
        actor_config = ActorConfig(num_envs=envs_per_actor, num_policy=1, num_parallel=1, \
            batch_size=envs_per_actor, policy_device=device, num_collect=args.num_collect // num_actors)

        return ParallelizationConfig.create([actor_config] * num_actors)

    else:
        raise NotImplementedError

@ray.remote
class RayActor:
    def __init__(self, args):
        self._init = False
        num_actors = args.num_actors
        
        n_envs = args.num_envs // num_actors
        n_steps = args.num_collect // num_actors // n_envs
        
        self.exp_manager, self.model = \
            create_sb3_algorithm(args, n_envs, n_steps, 1, "dummy")

        self._init = True
    def initialized(self):
        while True:
            if self._init:
                return
    
    def run(self):
        self.exp_manager.learn(self.model)
        assert len(self.model.collect_time) == 1
        t = self.model.collect_time[0]
        self.model.collect_time.clear()
        return t
        


def run_rllib(args):
    actors = dict()
    pending = dict()

    for actor_id in range(args.num_actors):
        actors[actor_id] = RayActor.remote(args)
    
    for actor_id, actor in actors.items():
        ref = actor.initialized.remote()
        pending[ref] = actor_id
    
    ray.wait(list(pending.keys()), num_returns=args.num_actors)
    pending.clear()
    
    times = []
    for _ in range(args.num_iteration):
        pending.clear()
        start_time = time.perf_counter()

        for actor_id, actor in actors.items():
            ref = actor.run.remote()
            pending[ref] = actor_id
            
        ray.wait(list(pending.keys()), num_returns=args.num_actors)
        end_time = time.perf_counter()
        times += [end_time - start_time]
        
    return times



def create_sb3_algorithm(args, num_envs, n_steps, num_iteration, vec_cls="dummy"):
    exp_manager = ExperimentManager(
        args=args,
        algo="ppo",
        env_id=args.env,
        log_folder="",
        vec_env_type=vec_cls,
        device="cpu",
        verbose=0,
    )
    

    exp_manager.auturi_num_envs = num_envs
    model, _ = exp_manager.setup_experiment()

    assert model.env.num_envs == num_envs, f"model has {model.env.num_envs}"

    _wrap = functools.partial(exp_manager.create_envs, 1)
    model.env_fns = [_wrap for _ in range(num_envs)]

    model._auturi_iteration = num_iteration
    model._auturi_train_skip = True
    model.n_steps = n_steps

    return exp_manager, model


def run(args):
    print(args)
    # create ExperimentManager with minimum argument.
    num_iteration = args.num_iteration

    n_envs = args.num_envs
    num_collect = args.num_collect

    if args.run_auturi:
        exp_manager, model = create_sb3_algorithm(args, n_envs, num_collect, num_iteration, "dummy")
        auturi_config = get_config(args, architecture=args.running_mode)
        tuner = create_tuner_with_config(
            n_envs, ParallelizationConfig.create([auturi_config])
        )
        wrap_sb3_OnPolicyAlgorithm(model, tuner=tuner, backend="shm")
        
        exp_manager.learn(model)
        model._auturi_executor.terminate()


    if args.running_mode in ["dummy", "subproc"]:
        n_steps = num_collect // n_envs
        exp_manager, model = create_sb3_algorithm(args, n_envs, n_steps, num_iteration, args.running_mode)
        exp_manager.learn(model)
        print(f"SB3 {args.running_mode} (env={args.env}). Collect time: {model.collect_time}")    


    elif args.running_mode == "rllib":
        collect_times = run_rllib(args)
        print(f"RLLib (actors={args.num_actors}) (env={args.env}). Collect time: {collect_times}")

    else:
        raise NotImplementedError    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--running-mode",
        type=str,
        default="auturi",
        choices=["dummy", "subproc", "rllib"],
    )
    parser.add_argument("--env", type=str, default="CartPole-v1", help="environment ID")
    parser.add_argument(
        "--run-auturi", action="store_true", help="skip backprop stage."
    )
    parser.add_argument(
        "--num-envs", type=int, default=4, help="number of trials for each config."
    )
    parser.add_argument(
        "--num-collect", type=int, default=4, help="number of trials for each config."
    )

    parser.add_argument(
        "--num-iteration", type=int, default=3, help="number of trials for each config."
    )

    parser.add_argument(
        "--num-actors", type=int, default=1, help="number of trials for each config."
    )

    args = parser.parse_args()
    run(args)
