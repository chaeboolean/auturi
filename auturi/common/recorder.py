import time
from contextlib import contextmanager
from collections import defaultdict
import os

def make_profiler():
    class EmptyProfiler:
        @contextmanager
        def timespan(self, name):
            yield
            
        def dumps(self, filename):
            pass

    class MyProfiler:
        def __init__(self):
            self.ts_dict = defaultdict(list)        
        
        @contextmanager
        def timespan(self, name):
            stime = time.perf_counter()
            yield
            etime = time.perf_counter()
            
            self.ts_dict[name].append(etime - stime)

        def dumps(self, filename):
            with open(filename, "w") as f:
                for k, v in self.ts_dict.items():
                    f.write(f"{k}: {v}\n")

    # Set environment variable AUTURI_TASK_PROFILE to True
    # in order to record 1E1P timesteps each
    
    enable_profile = os.getenv("AUTURI_TASK_PROFILE", False)
    return MyProfiler() if enable_profile else EmptyProfiler()
