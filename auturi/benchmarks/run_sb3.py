import argparse
import functools
import os

from rl_zoo3.exp_manager import ExperimentManager

from auturi.adapter.sb3 import wrap_sb3_OnPolicyAlgorithm
from auturi.tuner import ActorConfig, ParallelizationConfig, create_tuner_with_config
from auturi.tuner.grid_search import GridSearchTuner


def create_sb3_algorithm(args, num_iteration, vec_cls="dummy"):
    exp_manager = ExperimentManager(
        args=args,
        algo="ppo",
        env_id=args.env,
        log_folder="",
        vec_env_type=vec_cls,
        device="cuda:1",
        verbose=0,
    )

    # _wrap = create_envs(exp_manager, 3)
    _wrap = functools.partial(exp_manager.create_envs, 1)
    model, _ = exp_manager.setup_experiment()
    model.env_fns = [_wrap for _ in range(exp_manager.n_envs)]

    model._auturi_iteration = num_iteration
    model._auturi_train_skip = args.skip_update

    return exp_manager, model

def make_single_tuner(num_collect, num_iteration, args):
    dev = "cuda:4" if args.cuda else "cpu"
    subproc_config = ActorConfig(
        num_envs=1,
        num_parallel=1, # ep
        num_policy=1, # pp
        batch_size=1, #bs 
        num_collect=num_collect,
        policy_device=dev,
    )
    tuner_config = ParallelizationConfig.create([subproc_config])
    return create_tuner_with_config(1, num_iteration + 1, tuner_config)


def run(args):
    print(args)
    # create ExperimentManager with minimum argument.
    num_iteration = args.num_iteration
    vec_cls = (
        "dummy" if args.running_mode in ["auturi", "search"] else args.running_mode
    )
    exp_manager, model = create_sb3_algorithm(args, num_iteration, vec_cls)

    n_envs = exp_manager.n_envs
    num_collect = model.n_steps * n_envs

    tuner = make_single_tuner(num_collect, num_iteration, args)
    wrap_sb3_OnPolicyAlgorithm(model, tuner=tuner, backend="shm")

    try:
        exp_manager.learn(model)
    except Exception as e:
        raise e
    
    dev = "cuda" if args.cuda else "cpu"
    outfilename = f"log/{args.env}_{dev}.txt"
    with open(outfilename, "w") as f:
        with open("out+loop.txt") as in_f:
            lines = in_f.readlines()
            for line in lines:
                f.write(line)

        with open("sim+inf.txt") as in_f:
            lines = in_f.readlines()
            for line in lines:
                f.write(line)
                
    os.remove("out+loop.txt")
    os.remove("sim+inf.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--running-mode",
        type=str,
        default="auturi",
        choices=["L"],
    )
    parser.add_argument("--env", type=str, default="CartPole-v1", help="environment ID")
    parser.add_argument(
        "--skip-update", action="store_true", help="skip backprop stage."
    )
    parser.add_argument(
        "--cuda", action="store_true", help="skip backprop stage."
    )

    parser.add_argument(
        "--num-iteration", type=int, default=5, help="number of trials for each config."
    )

    args = parser.parse_args()
    run(args)
