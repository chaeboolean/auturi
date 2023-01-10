from typing import Tuple
import os
import enum
import time

from auturi.tuner.base_tuner import AuturiTuner
from auturi.tuner.config import ActorConfig, ParallelizationConfig
from auturi.common.chrome_profiler import merge_file

from collections import defaultdict

class TuningMode(enum.Enum):
    FIND_BS = 1
    INCR_DEGREE = 2 
    

class GreedyTuner(AuturiTuner):
    """AuturiTuner that launches process to empty space.

    By parsing trace, this tuner compares two kind of waiting time
        - W_pol: Env -> Pol
        - W_env: Pol -> Env
    
    if W_pol > W_env: W_pol launches
    """

    def __init__(
        self,
        min_num_env: int,
        max_num_env: int,
        num_collect: int,
        max_policy_num: int,
        use_gpu: bool, 
        num_iterate: int = 10,
        task_name: str = "", 
        log_path: str = "",
        num_core: int = 64,
    ):

        # turn on tracing option
        os.environ["AUTURI_TRACE"] = "1"
        self._task_name = task_name
        self.use_gpu = use_gpu
        self.max_policy_num = max_policy_num
        self.num_cores = num_core
        
        # counter
        self.cnt = 0
        self.stime = 0

        # Dict[(ep, pp)] = List[Tuple(elapsed time, bs)]
        self.dict_bs = defaultdict(list)

        self._last_best = (1, 1, 1, 9876543221)  # ep, pp, bs, elapsed time
        self.log_path = log_path
        if os.path.isfile(log_path):
            os.remove(log_path)
        
        self._last_config = ParallelizationConfig.create([ActorConfig(
            num_envs=min_num_env,
            num_policy=1,
            num_parallel=1, 
            batch_size=1,
            policy_device="cuda" if self.use_gpu else "cpu",
            num_collect=num_collect,
        )])
        self._init=  False 


        super().__init__(min_num_env, max_num_env, num_collect, num_iterate)

    @property
    def task_name(self):
        return self._task_name

    def write_log(self, message):
        with open(self.log_path, "a") as f:
            f.write(str(message) + "\n")
            

    def _increase_degree(self, config, increase_policy: bool):
        actor_config = config[0]
        incr_deg = (1, 2) if increase_policy else (2, 1)
        incr_str = "Policy" if increase_policy else "Env"
        try:
            next_config = ParallelizationConfig.create([ActorConfig(
                num_envs=actor_config.num_envs,
                num_parallel=actor_config.num_parallel * incr_deg[0], 
                num_policy=actor_config.num_policy * incr_deg[1],
                batch_size=1,
                policy_device=actor_config.policy_device,
                num_collect=actor_config.num_collect,
            )])
            
            if self.use_gpu:
                assert next_config.num_policy <= self.max_policy_num
            assert next_config[0].num_parallel + next_config[0].num_policy <= self.num_cores
            self.write_log(f"Increase {incr_str} degree! => {(_config_to_tuple(next_config))}")

            return next_config
    
        except AssertionError as e:
            self.write_log(f"Increase {incr_str} FAIL.")

            return None

        

    def _check_config(self, config: ParallelizationConfig):
        if self.use_gpu and config.num_policy > self.max_policy_num:
            return False

        if config.num_parallel_envs + config.num_policy > self.num_cores:
            return False
        
        return True

    def terminate_tuner(self):
        self.write_log(f"Search time = {self.stime}, tried {self.cnt} configs")
        self.write_log(f"Best configuration: ({self._last_best[:3]}) ---> {self._last_best[3]} sec")


    def _generate_next(self):
        self.cnt += 1
        actor_config = self._last_config[0]

        # Initialize
        if not self._init:
            self.write_log(f"Init!")
            self._init = True

    
        # Case 1. Batch size is maximum. Increaes degree.
        elif actor_config.batch_size == actor_config.num_envs // actor_config.num_policy:
            
            # Find best batch size
            degree_key = (actor_config.num_parallel, actor_config.num_policy) # (ep, pp)
            sorted_bs = sorted(self.dict_bs[degree_key])[0] # tuple (time, bs, pol exec, env exec)
            
            self.write_log(f"Best batch size for ep={degree_key[0]},pp={degree_key[1]}: {sorted_bs[1]} ---> result = {sorted_bs[0]} sec")
            
            # Stop if previous value is best
            if sorted_bs[0] > self._last_best[3]:
                self.terminate_tuner()
                raise StopIteration
            else:
                self._last_best = (degree_key[0], degree_key[1], sorted_bs[1], sorted_bs[0])
            
            # Pick which to increase degree
            policy_ratio, env_ratio = sorted_bs[2], sorted_bs[3]
            next_config = self._increase_degree(self._last_config, policy_ratio - env_ratio > 0.1)
            if next_config is None:
                self.terminate_tuner()
                raise StopIteration
            else:
                self._last_config = next_config
        
        # Case 2. Search another batch size
        else:
            self._last_config = ParallelizationConfig.create([ActorConfig(
                num_envs=actor_config.num_envs,
                num_policy=actor_config.num_policy,
                num_parallel=actor_config.num_parallel, 
                batch_size=actor_config.batch_size * 2,
                policy_device=actor_config.policy_device,
                num_collect=actor_config.num_collect,
            )])

        return self._last_config


    def _update_tuner(self, config, res):
        
        for elapsed, traj in res[0]:
            self.stime += elapsed
        
        mean_metric = res[1]

        # write trace files
        out_dir, output_name = self.trace_out_name(config)
        policy_exec_ratio, env_exec_ratio = merge_file(out_dir, output_name)

        # write log
        ep, pp, bs = _config_to_tuple(config)
        message = f"\n\n==========\n [{self.cnt}] Config({ep}, {pp}, {bs}): {mean_metric.elapsed} sec\n"
        message += f"Pol: {round(policy_exec_ratio, 2)}, Env: {round(env_exec_ratio, 2)}"
        self.write_log(message)
        print(message)

        # put result to dict_bs
        degree_key = (config[0].num_parallel, config[0].num_policy) # (ep, pp)
        degree_val = (mean_metric.elapsed, config[0].batch_size, policy_exec_ratio, env_exec_ratio)
        self.dict_bs[degree_key].append(degree_val)


    def trace_out_name(self, config):
        ep, pp, bs = _config_to_tuple(config)
        config_str = f"({self.cnt})ep={ep}+pp={pp}+bs={bs}"
        return f"{self.task_name}_env{self.min_num_env}", config_str


def _config_to_tuple(config: ParallelizationConfig) -> Tuple[int]:
    actor_config = config[0]
    return actor_config.num_parallel, actor_config.num_policy, actor_config.batch_size
