from typing import Callable, Optional
import os
import enum

from auturi.tuner.base_tuner import AuturiTuner
from auturi.tuner.config import ActorConfig, ParallelizationConfig
from auturi.common.chrome_profiler import merge_file

from collections import defaultdict

class TuningMode(enum.Enum):
    FIND_BS = 1
    INCR_DEGREE = 2 
    

class LaunchingTuner(AuturiTuner):
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
        num_iterate: int = 10,
        validator: Optional[Callable[[ActorConfig], bool]] = None,
    ):

        # turn on tracing option
        os.environ["AUTURI_TRACE"] = "1"
        validator = (lambda _: True) if validator is None else validator

        # Dict[(ep, pp)] -> optimal (bs, elapsed)
        self.best_bs = defaultdict(list)
        self.mode_ = TuningMode.FIND_BS

        super().__init__(min_num_env, max_num_env, num_collect, num_iterate)

    @property
    def task_name(self):
        return self._task_name

    def _generate_next(self):
        pass

    def _get_next_config(self, bs_key, last_bs, policy_exec_ratio, env_exec_ratio):
        if self.mode_ == TuningMode.INCR_DEGREE:
            if policy_exec_ratio - env_exec_ratio > 0.1:
                plus_value = (1, 0)
            else:
                plus_value = (0, 1)

        else:
            last_best_bs = self.best_bs[bs_key][0]
            if last_best_bs > 1:
                return bs_key, last_best_bs // 2
            

    def _update_tuner(self, config, mean_metric):
        self.tuning_results[hash(config)] = (config, mean_metric)

        print("=" * 20)
        print(config)
        print(f"Mean Result: {mean_metric.elapsed} sec")
        print("=" * 20)

        # write trace files
        out_dir, output_name = self.trace_out_name(config)
        policy_exec_ratio, env_exec_ratio = merge_file(out_dir, output_name)
        

        bs_key = (config[0].num_parallel, config[0].num_policy)

        self.best_bs[bs_key].append(mean_metric.elapsed, config[0].batch_size, mean_metric.elapsed)
        
        else:
            prev_bs, prev_value = self.best_bs[bs_key]
            
            # update value
            if prev_value > mean_metric.elapsed:
                self.best_bs[bs_key] = (config[0].batch_size, mean_metric.elapsed)
                self.mode_ = TuningMode.INCR_DEGREE




    def trace_out_name(self, entire_config):
        config = entire_config[0]
        config_str = f"ep={config.num_parallel}pp={config.num_policy}bs={config.batch_size}"
        return f"{self.task_name}_{self.min_num_env}", config_str
